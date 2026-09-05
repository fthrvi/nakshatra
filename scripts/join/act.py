"""The acting half. Everything here has a side effect on the machine.

⚠️⚠️ THIS IS THE DANGEROUS FILE, AND IT IS SMALL ON PURPOSE. Every function here runs a
process, writes a file, or opens a socket. The DECISIONS all live in the phases, which are
pure — so this file contains no policy at all: it does what it is told and reports what
happened. If you find yourself adding an `if` about whether something is acceptable, it
belongs in a phase.

⚠️ Nothing here raises on a failed action. A download that fails, a daemon that will not
start, a probe that times out are all OBSERVATIONS — they come back as facts and a phase
decides what they mean. An exception here would bypass the phase that exists to interpret it.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import socket as _socket
import urllib.parse
import signal
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List


def _resolved_ips_are_safe(url: str) -> tuple[bool, str]:
    """⚠️⚠️ THE PURE CHECK LOOKS AT THE STRING; THIS LOOKS AT WHERE IT GOES. `pkgurl` rejects
    an IP literal in a private range and cannot do more — resolution is I/O. So a hostname
    that RESOLVES to 169.254.169.254 sailed through, and DNS rebinding makes that trivial to
    arrange. Resolve here, at the last moment before the dial, and refuse if ANY answer is
    private, loopback, link-local or reserved. Every A/AAAA record — an attacker controls the
    order they come back in."""
    try:
        host = urllib.parse.urlsplit(url).hostname
        if not host:
            return False, "no host in url"
        infos = _socket.getaddrinfo(host, None, proto=_socket.IPPROTO_TCP)
    except (OSError, ValueError) as e:
        return False, f"could not resolve {host!r}: {e}"
    bad = []
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
                or ip.is_reserved or ip.is_unspecified):
            bad.append(str(ip))
    if bad:
        return False, f"{host} resolves to a non-public address: {', '.join(bad)}"
    return True, ""


def download(argv: List[str], dest: str, *, url: str = "") -> Dict[str, Any]:
    """Run a download plan. Returns what happened — never judges it."""
    if url:
        ok, why = _resolved_ips_are_safe(url)
        if not ok:
            return {"download_exit_code": 126, "download_stderr": f"refused: {why}"}
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=3600)
        code, err = p.returncode, (p.stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        code, err = 124, "timed out"
    except Exception as e:                                   # noqa: BLE001
        code, err = 127, f"{type(e).__name__}: {e}"
    out: Dict[str, Any] = {"download_exit_code": code, "download_stderr": err}
    try:
        out["downloaded_bytes"] = os.path.getsize(dest)
        out["downloaded_sha256"] = sha256_file(dest)
    except OSError:
        # ⚠️ Absent, not zero. A missing file is "we did not observe a size", and reporting 0
        # would look like a real empty download to the phase.
        pass
    return out


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """⚠️ Streamed in 1 MiB chunks. Reading a multi-gigabyte slice whole to hash it OOMs the
    very box that is trying to join, and the failure presents as a hardware fault."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def start_daemon(argv: List[str], log_path: str) -> Dict[str, Any]:
    """Launch the worker. ⚠️ `argv` is a LIST and is never joined into a shell string — the
    slice path in it came from a package URL a stranger may control, and a path containing a
    space or a semicolon would become a different command."""
    try:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        log = open(log_path, "ab")
        p = subprocess.Popen(argv, stdout=log, stderr=log, start_new_session=True)
        return {"daemon_pid": p.pid, "daemon_log": log_path}
    except Exception as e:                                   # noqa: BLE001
        return {"daemon_pid": None, "daemon_start_error": f"{type(e).__name__}: {e}"}


def poll_health(url: str, attempts: int, delay_fn, pid: int | None = None) -> List[Dict[str, Any]]:
    """Poll until a readiness judge is satisfied. Returns the WHOLE poll history, because
    `daemon_ready` needs consecutive successes: ⚠️ one 200 is not ready — a server that binds
    the port before loading weights answers once and then stalls for minutes."""
    polls: List[Dict[str, Any]] = []
    for i in range(1, attempts + 1):
        rec: Dict[str, Any] = {"at": time.time()}
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                rec["http_status"] = r.status
        except urllib.error.HTTPError as e:
            rec["http_status"] = e.code
        except Exception as e:                               # noqa: BLE001
            rec["http_status"] = None
            rec["error"] = f"{type(e).__name__}: {e}"
        if pid is not None:
            rec["process_exited"] = not _alive(pid)
        polls.append(rec)
        if i < attempts:
            time.sleep(max(0.0, float(delay_fn(i))))
    return polls


def _alive(pid: int) -> bool:
    """⚠️ signal 0 checks existence without touching the process."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def probe(url: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    """One tiny inference, to prove the node serves rather than merely listens."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read().decode("utf-8", "replace")
        return {"probe_body": text,
                "answered_probe_ms": (time.perf_counter() - t0) * 1000.0}
    except Exception as e:                                   # noqa: BLE001
        return {"probe_body": "", "probe_error": f"{type(e).__name__}: {e}",
                "answered_probe_ms": None}


def fetch_join_info(coordinator: str, timeout: int = 15) -> Dict[str, Any]:
    """Ask the coordinator what a joining node needs: the package URL and its version.

    ⚠️ WHY THIS EXISTS. The `--code` path decoded the join code and then went straight to
    `admit`, which needs `package_url`, `node_version` and `coordinator_version` — none of
    which anything supplied. Production failed at phase 1 with "missing package_url", while
    the end-to-end test passed because its fixture preloaded all six phases' facts. A test
    that passes only with a fixture the real path never sees is a test of the fixture.

    Returns the facts on success; on ANY failure returns nothing, so admit refuses with
    "not observed" — which is the honest answer when the coordinator cannot be reached.
    """
    try:
        base = coordinator.rstrip("/")
        ok, why = _resolved_ips_are_safe(base)
        if not ok:
            return {"join_info_error": f"coordinator refused: {why}"}
        req = urllib.request.Request(base + "/v1/join-info",
                                     headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:                                   # noqa: BLE001
        return {"join_info_error": f"{type(e).__name__}: {e}"}
    out: Dict[str, Any] = {}
    if isinstance(data, dict):
        if isinstance(data.get("package_url"), str):
            out["package_url"] = data["package_url"]
        if isinstance(data.get("version"), str):
            out["coordinator_version"] = data["version"]
        if isinstance(data.get("node_id"), str):
            out["node_id"] = data["node_id"]
    return out


def stop_daemon(pid: int | None) -> Dict[str, Any]:
    """⚠️ Used when a join FAILS after the daemon started. Leaving a half-joined node serving
    is worse than not joining: it answers, it is unregistered, and it will never be paid."""
    if not isinstance(pid, int) or pid <= 0 or not _alive(pid):
        return {"daemon_stopped": False}
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(50):
            if not _alive(pid):
                return {"daemon_stopped": True}
            time.sleep(0.1)
        os.kill(pid, signal.SIGKILL)
        return {"daemon_stopped": True, "daemon_killed": True}
    except OSError as e:
        return {"daemon_stopped": False, "daemon_stop_error": str(e)}
