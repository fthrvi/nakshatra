#!/usr/bin/env python3
"""PID-1 service supervisor driven only by signed installer-generated unit files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import shlex
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
UNIT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@-]*")
UNIT_MARKER = "# Managed by nakshatra-node install.py"


def _assignments(value: str) -> dict[str, str]:
    result = {}
    for item in shlex.split(value, posix=True):
        if "=" in item:
            name, setting = item.split("=", 1)
            result[name] = setting
    return result


def _env_file(path: Path) -> dict[str, str]:
    result = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return result
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        try:
            parsed = shlex.split(value, posix=True)
            result[name] = parsed[0] if len(parsed) == 1 else value
        except ValueError:
            result[name] = value
    return result


def _command(value: str, env: dict[str, str]) -> list[str]:
    output = []
    for word in shlex.split(value, posix=True):
        exact = re.fullmatch(r"\$([A-Za-z_][A-Za-z0-9_]*)", word)
        if exact:
            setting = env.get(exact.group(1), "")
            if setting:
                output.extend(shlex.split(setting, posix=True))
            continue
        output.append(VARIABLE.sub(lambda match: env.get(match.group(1), ""), word))
    return output


@dataclass
class Unit:
    name: str
    argv: list[str]
    pre: list[list[str]]
    env: dict[str, str]
    restart: str
    fingerprint: str


def parse_unit(path: Path, base_env: dict[str, str] | None = None) -> Unit:
    """Parse the small, deliberate systemd subset emitted by release/install.py."""
    section = ""
    values: dict[str, list[str]] = {}
    directives: list[tuple[str, str]] = []
    raw = path.read_text()
    for source in raw.splitlines():
        line = source.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section != "Service" or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values.setdefault(key, []).append(value)
        directives.append((key, value))
    env = dict(os.environ if base_env is None else base_env)
    # systemd gives EnvironmentFile values precedence over Environment values, independent of
    # where the directives appear in the unit. This is what makes release opt-in env files useful
    # for overriding signed defaults such as the gateway's container-internal BIND address.
    for key, item in directives:
        if key == "Environment":
            env.update(_assignments(item))
    for key, item in directives:
        if key == "EnvironmentFile":
            env.update(_env_file(Path(item.removeprefix("-"))))
    starts = values.get("ExecStart", [])
    if len(starts) != 1:
        raise ValueError(f"{path} must contain exactly one ExecStart")
    argv = _command(starts[0], env)
    if not argv:
        raise ValueError(f"{path} has an empty ExecStart")
    pre = [_command(item, env) for item in values.get("ExecStartPre", [])]
    restart = values.get("Restart", ["on-failure"])[-1]
    if restart not in {"always", "on-failure", "no"}:
        raise ValueError(f"{path} has unsupported Restart={restart}")
    return Unit(path.stem.removesuffix(".service"), argv, pre, env, restart,
                hashlib.sha256(raw.encode()).hexdigest())


@dataclass
class Service:
    unit: Unit
    process: subprocess.Popen | None = None
    failures: int = 0
    next_start: float = 0.0
    started_at: float = 0.0
    restarts: int = 0
    desired: bool = True


@dataclass
class Supervisor:
    prefix: Path
    unit_dir: Path
    socket_path: Path
    refresh: bool = True
    update_interval: float = 3600.0
    backoff_base: float = 1.0
    backoff_cap: float = 30.0
    control: bool = True
    services: dict[str, Service] = field(default_factory=dict)
    managed_names: set[str] = field(default_factory=set)
    stopping: bool = False
    updating: bool = False
    update_ok: bool | None = None
    update_result: str = "never run"
    requests: queue.Queue = field(default_factory=queue.Queue)
    _server: socketserver.UnixStreamServer | None = None
    _update_thread: threading.Thread | None = None
    _update_process: subprocess.Popen | None = None
    _next_update: float = float("inf")
    _ignored_units: set[str] = field(default_factory=set)

    def log(self, message: str) -> None:
        print(f"[supervisor] {message}", flush=True)

    def _installer(self) -> Path | None:
        manifest = self.prefix / "current" / "manifest.json"
        try:
            data = json.loads(manifest.read_text())
            relative = Path(data["installer"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if relative.is_absolute() or ".." in relative.parts:
            return None
        installer = self.prefix / "current" / relative
        return installer if installer.is_file() else None

    @staticmethod
    def _managed_unit(path: Path) -> bool:
        try:
            return UNIT_MARKER in path.read_text()[:1000]
        except OSError:
            return False

    @staticmethod
    def _set_env(path: Path, name: str, value: str | None) -> None:
        try:
            lines = path.read_text().splitlines()
        except FileNotFoundError:
            lines = []
        prefix = f"{name}="
        lines = [line for line in lines if not line.startswith(prefix)]
        if value is not None:
            lines.append(f'{name}="{value}"')
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("\n".join(lines) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)

    def reconcile_container_env(self) -> None:
        """Keep worker gRPC private while exposing only its deliberate P2P listener."""
        home = Path(os.environ.get("HOME", str(self.prefix.parent)))
        p2p_env = home / ".nakshatra" / "p2p.env"
        worker_env = home / ".nakshatra" / "worker.env"
        if os.environ.get("NAK_CONTAINER_WORKER") != "1":
            # The state volume survives role changes. Strip any fixed inbound listener left by a
            # former worker container so consumer gateways remain dial-only.
            for path in (p2p_env, worker_env):
                if path.exists():
                    self._set_env(path, "NAKSHATRA_P2P_WORKER_ARGS", None)
            return
        port_text = os.environ.get("NAK_CONTAINER_P2P_PORT", "4001")
        try:
            port = int(port_text)
        except ValueError as exc:
            raise ValueError("container p2p port must be an integer") from exc
        if not 1 <= port <= 65535 or str(port) != port_text:
            raise ValueError("container p2p port must be between 1 and 65535")
        if worker_env.exists():
            values = _env_file(worker_env)
            worker_port = values.get("PORT", "")
            if not worker_port.isdigit() or not 1 <= int(worker_port) <= 65535:
                raise ValueError("worker env has an invalid PORT")
            self._set_env(worker_env, "LISTEN", "127.0.0.1")
            self._set_env(
                worker_env,
                "NAKSHATRA_P2P_WORKER_ARGS",
                f"-listen /ip4/0.0.0.0/tcp/{port} -inbound=127.0.0.1:{worker_port} "
                "-require-direct-activation",
            )
        if p2p_env.exists() and not worker_env.exists():
            self._set_env(
                p2p_env, "NAKSHATRA_P2P_WORKER_ARGS", f"-listen /ip4/0.0.0.0/tcp/{port}"
            )

    @staticmethod
    def _validated_names(value: object) -> list[str]:
        if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
            raise RuntimeError("signed installer returned an invalid unit list")
        names = []
        for name in value:
            normalized = name.removesuffix(".service")
            if not UNIT_NAME.fullmatch(normalized):
                raise RuntimeError(f"signed installer returned an invalid unit name: {name!r}")
            if normalized not in names:
                names.append(normalized)
        return names

    def generate_units(self) -> list[str]:
        installer = self._installer()
        if installer is None:
            return []
        command = [sys.executable, str(installer), "--prefix", str(self.prefix), "write-units",
                   "--no-systemd", "--include-updater", "--unit-dir", str(self.unit_dir)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
        if result.returncode and "--include-updater" in result.stderr and "unrecognized arguments" in result.stderr:
            # An already-installed release may predate the container updater-reporting contract. Its
            # ordinary services remain safe to run from its returned allowlist, but its updater is not
            # authorized until a newer signed installer explicitly returns that name.
            self.log("signed installer predates updater allowlisting; automatic updates remain disabled")
            result = subprocess.run(
                [item for item in command if item != "--include-updater"],
                capture_output=True, text=True, timeout=120, check=False,
            )
        if result.returncode:
            raise RuntimeError(f"signed installer could not generate units: {result.stderr.strip()[-300:]}")
        return self._validated_names(json.loads(result.stdout.strip().splitlines()[-1]))

    def load_units(self, names: list[str] | None = None, *, activate_new: bool = False) -> None:
        if names is not None:
            names = self._validated_names(names)
            self.managed_names = set(names)
        else:
            names = sorted(self.managed_names)
        service_names = [name for name in names if name != "nak-update"]
        allowed_files = {f"{name}.service" for name in names}
        ignored = sorted(path.name for path in self.unit_dir.glob("*.service") if path.name not in allowed_files)
        for filename in ignored:
            if filename not in self._ignored_units:
                self.log(f"ignoring unit not returned by signed installer: {filename}")
        self._ignored_units.update(ignored)
        incoming = {}
        if "nak-update" in names and (self.unit_dir / "nak-update.service").is_file() \
                and not self._managed_unit(self.unit_dir / "nak-update.service"):
            self.log("ignoring installer-listed unit without managed marker: nak-update.service")
        for name in service_names:
            path = self.unit_dir / f"{name}.service"
            if path.is_file() and self._managed_unit(path):
                incoming[name] = parse_unit(path)
            elif path.is_file():
                self.log(f"ignoring installer-listed unit without managed marker: {path.name}")
        for name in set(self.services) - set(incoming):
            self.stop_service(name)
            del self.services[name]
        for name, unit in incoming.items():
            old = self.services.get(name)
            if old is None:
                self.services[name] = Service(unit=unit, desired=activate_new)
            else:
                # EnvironmentFile contents are runtime state and are deliberately not part of the generated
                # unit text. Always refresh the parsed environment; a following `restart` must see changes made
                # by commands such as `nak worker owner-lease on`.
                old.unit = unit
        if activate_new:
            self.start_ready()

    def reload(self, *, generate: bool = False) -> None:
        self.reconcile_container_env()
        names = self.generate_units() if generate and self.refresh else None
        self.load_units(names)

    def start_service(self, name: str) -> None:
        service = self.services[name]
        if service.process is not None or not service.desired:
            return
        for command in service.unit.pre:
            result = subprocess.run(command, env=service.unit.env, check=False)
            if result.returncode:
                self._schedule(service, result.returncode)
                return
        try:
            service.process = subprocess.Popen(service.unit.argv, env=service.unit.env, start_new_session=True)
        except OSError as exc:
            self.log(f"{name}: start failed: {exc}")
            self._schedule(service, 127)
            return
        service.started_at = time.monotonic()
        self.log(f"{name}: started pid {service.process.pid}")

    def _schedule(self, service: Service, returncode: int) -> None:
        should_restart = service.desired and (
            service.unit.restart == "always" or (service.unit.restart == "on-failure" and returncode != 0)
        )
        service.process = None
        if not should_restart:
            service.next_start = float("inf")
            return
        service.failures += 1
        service.restarts += 1
        delay = min(self.backoff_cap, self.backoff_base * (2 ** (service.failures - 1)))
        service.next_start = time.monotonic() + delay
        self.log(f"{service.unit.name}: exited {returncode}; restart in {delay:g}s")

    def reap(self) -> None:
        now = time.monotonic()
        for service in self.services.values():
            if service.process is None:
                continue
            returncode = service.process.poll()
            if returncode is None:
                if now - service.started_at >= 60:
                    service.failures = 0
                continue
            self._schedule(service, returncode)

    def start_ready(self) -> None:
        now = time.monotonic()
        for name, service in self.services.items():
            if service.desired and service.process is None and now >= service.next_start:
                self.start_service(name)

    def stop_service(self, name: str, timeout: float = 10.0) -> None:
        service = self.services.get(name)
        if service is None:
            return
        service.desired = False
        process = service.process
        if process is None or process.poll() is not None:
            service.process = None
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        service.process = None
        self.log(f"{name}: stopped")

    def _ensure_known(self, names: list[str]) -> None:
        missing = [name for name in names if name not in self.services]
        if missing and self.refresh:
            self.reload(generate=True)
        still_missing = [name for name in names if name not in self.services]
        if still_missing:
            raise ValueError("unknown generated service(s): " + ", ".join(still_missing))

    def _run_update(self) -> None:
        try:
            update_path = self.unit_dir / "nak-update.service"
            if "nak-update" not in self.managed_names or not self._managed_unit(update_path):
                raise RuntimeError("ignoring updater without installer managed-unit marker")
            unit = parse_unit(update_path)
            self._update_process = subprocess.Popen(
                unit.argv, env=unit.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True,
            )
            stdout, stderr = self._update_process.communicate()
            returncode = self._update_process.returncode
            detail = (stdout or stderr).strip().splitlines()
            self.update_result = (detail[-1] if detail else f"exit {returncode}")[-500:]
            if returncode:
                self.update_ok = False
                self.log(f"update failed: {self.update_result}")
            else:
                self.update_ok = True
                self.log(f"update: {self.update_result}")
        except Exception as exc:  # update failures must not take down healthy services
            self.update_ok = False
            self.update_result = f"failed: {exc}"
            self.log(self.update_result)
        finally:
            self._update_process = None
            if not self.stopping:
                try:
                    self.reload(generate=True)
                except Exception as exc:
                    self.log(f"post-update unit reload failed: {exc}")
            self.updating = False

    def begin_update(self) -> None:
        if self.updating or "nak-update" not in self.managed_names \
                or not self._managed_unit(self.unit_dir / "nak-update.service"):
            return
        self.updating = True
        self.update_ok = None
        self._update_thread = threading.Thread(target=self._run_update, name="nak-update", daemon=True)
        self._update_thread.start()

    def status(self) -> dict:
        return {
            "ok": True,
            "updating": self.updating,
            "update_ok": self.update_ok,
            "update_result": self.update_result,
            "services": {
                name: {
                    "active": bool(service.process and service.process.poll() is None),
                    "pid": service.process.pid if service.process and service.process.poll() is None else 0,
                    "restarts": service.restarts,
                }
                for name, service in sorted(self.services.items())
            },
        }

    def handle(self, request: dict) -> dict:
        action = request.get("action")
        try:
            if action == "status":
                return self.status()
            if action == "update":
                self.begin_update()
                return {"ok": self.updating, "error": "no installed updater" if not self.updating else ""}
            if action == "reload":
                self.reload(generate=True)
                return {"ok": True}
            if action in {"enable", "disable", "restart"}:
                names = request.get("services") or []
                if not all(isinstance(name, str) and name for name in names):
                    raise ValueError("invalid service list")
                if action != "disable":
                    self._ensure_known(names)
                for name in names:
                    if action == "disable":
                        self.stop_service(name)
                    elif action == "restart":
                        self.stop_service(name)
                        self.services[name].desired = True
                        self.services[name].next_start = 0
                        self.start_service(name)
                    else:
                        self.services[name].desired = True
                        self.services[name].next_start = 0
                        if request.get("now"):
                            self.start_service(name)
                return {"ok": True}
            name = str(request.get("service", ""))
            service = self.services.get(name)
            active = bool(service and service.process and service.process.poll() is None)
            if action == "is-active":
                return {"ok": active, "returncode": 0 if active else 3,
                        "stdout": "active" if active else "inactive"}
            if action == "is-enabled":
                enabled = bool(service and service.desired)
                return {"ok": enabled, "returncode": 0 if enabled else 1,
                        "stdout": "enabled" if enabled else "disabled"}
            if action == "show":
                pid = service.process.pid if active else 0
                restarts = service.restarts if service else 0
                return {"ok": bool(service), "returncode": 0 if service else 1,
                        "stdout": f"ActiveState={'active' if active else 'inactive'}\nMainPID={pid}\nNRestarts={restarts}"}
            raise ValueError(f"unsupported action: {action!r}")
        except Exception as exc:
            return {"ok": False, "returncode": 1, "error": str(exc)}

    def _start_socket(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        supervisor = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                try:
                    request = json.loads(self.rfile.readline(65536))
                    reply = queue.Queue(maxsize=1)
                    supervisor.requests.put((request, reply))
                    response = reply.get(timeout=180)
                except Exception as exc:
                    response = {"ok": False, "returncode": 1, "error": str(exc)}
                self.wfile.write(json.dumps(response, separators=(",", ":")).encode() + b"\n")

        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True

        self._server = Server(str(self.socket_path), Handler)
        os.chmod(self.socket_path, 0o600)
        threading.Thread(target=self._server.serve_forever, name="control", daemon=True).start()

    def shutdown(self) -> None:
        self.stopping = True
        update = self._update_process
        if update is not None and update.poll() is None:
            try:
                os.killpg(update.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                update.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(update.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                update.wait(timeout=5)
        if self._update_thread is not None and self._update_thread.is_alive():
            self._update_thread.join(timeout=2)
        for name in list(self.services):
            self.stop_service(name)
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        self.socket_path.unlink(missing_ok=True)

    def run(self) -> int:
        if self.control:
            self._start_socket()
        runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp/run"))
        runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(runtime_dir, 0o700)
        try:
            self.unit_dir.mkdir(parents=True, exist_ok=True)
            self.reconcile_container_env()
            names = self.generate_units() if self.refresh else None
            self.load_units(names, activate_new=True)
        except Exception as exc:
            self.log(f"initial unit load deferred: {exc}")
        if self.update_interval > 0:
            self._next_update = time.monotonic() + self.update_interval
        while not self.stopping:
            try:
                request, reply = self.requests.get(timeout=0.1)
            except queue.Empty:
                pass
            else:
                reply.put(self.handle(request))
            self.reap()
            self.start_ready()
            if time.monotonic() >= self._next_update:
                self.begin_update()
                self._next_update = time.monotonic() + self.update_interval
        self.shutdown()
        return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path,
                        default=Path(os.environ.get("NAK_NODE_PREFIX", Path.home() / ".nakshatra-node")))
    parser.add_argument("--unit-dir", type=Path, default=Path.home() / ".config" / "systemd" / "user")
    parser.add_argument("--socket", type=Path,
                        default=Path(os.environ.get("NAK_SUPERVISOR_SOCKET", "/tmp/nakshatra-supervisor.sock")))
    parser.add_argument("--no-refresh", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-control", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--update-interval", type=float,
                        default=float(os.environ.get("NAK_UPDATE_INTERVAL_S", "3600")))
    args = parser.parse_args(argv)
    supervisor = Supervisor(
        args.prefix, args.unit_dir, args.socket, refresh=not args.no_refresh,
        update_interval=args.update_interval,
        backoff_base=float(os.environ.get("NAK_SUPERVISOR_BACKOFF_BASE_S", "1")),
        backoff_cap=float(os.environ.get("NAK_SUPERVISOR_BACKOFF_CAP_S", "30")),
        control=not args.no_control,
    )

    def stop(_signum, _frame):
        supervisor.stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main())
