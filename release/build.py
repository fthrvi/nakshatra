#!/usr/bin/env python3
"""build.py — make one signed Nakshatra node release.

    python release/build.py --version 0.1.0 --channel canary \
        --ref nakshatra=<commit-or-branch> --ref sthambha=<commit-or-branch> \
        --release-key ~/.nakshatra-release/release-TEST.key --out ~/nakshatra-dist

Output, under <out>/<channel>/<version>/:
    nakshatra-<commit12>.tar.gz   git archive plus static sidecar built from that exact commit
    sthambha-<commit12>.tar.gz    git archive of the Sthambha package at that commit
    requirements.lock             hash-locked dependencies (uv pip compile --generate-hashes)
    uv                            the uv binary that builds the environment on the node
    manifest.json                 every file's sha256 + commits + version, SIGNED by the release key
and <out>/<channel>/latest.json pointing at the version (also signed), which is what updaters poll.

Nothing here is hand-built on a node: nodes only download, verify and unpack these files.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import releasekit as rk  # noqa: E402

LATEST_TTL_S = 60 * 86400        # the signed "latest" pointer is refused after this (republish to extend)
PINNED_GO_VERSION = "go1.25.7"
SPEC = json.loads((Path(__file__).resolve().parent / "spec.json").read_text())
_PINNED_REQUIREMENT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^;\s]+)$")
_PACKAGE_SIGNER_SCHEMA = "nak-package-signers/1"
_ED25519_PUBLIC_KEY = re.compile(r"^[0-9a-f]{64}$")
_PACKAGE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_PACKAGE_REQUIREMENTS = {"gpu_vram_bytes", "cpu_ram_bytes", "disk_free_bytes"}


def _requirement_pins(requirements: list[str], profile: str) -> dict[str, str]:
    """Return normalized package -> exact pin for one profile's direct requirements."""
    pins = {}
    for requirement in requirements:
        match = _PINNED_REQUIREMENT.fullmatch(requirement.strip())
        if not match:
            raise RuntimeError(
                f"profile {profile!r} requirement {requirement!r} is not an exact name==version pin"
            )
        name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        if name in pins and pins[name] != match.group(2):
            raise RuntimeError(f"profile {profile!r} pins {name!r} more than once")
        pins[name] = match.group(2)
    return pins


def _validate_profile_subsumption(profiles: dict) -> None:
    """Prove every declared superset contains the subset's direct pins."""
    pins = {
        name: _requirement_pins(profile.get("requirements", []), name)
        for name, profile in profiles.items()
    }
    for subset_name, subset in profiles.items():
        satisfied_by = subset.get("satisfied_by", [])
        if not isinstance(satisfied_by, list) or any(not isinstance(name, str) for name in satisfied_by):
            raise RuntimeError(f"profile {subset_name!r} satisfied_by must be a list of profile names")
        for superset_name in satisfied_by:
            if superset_name == subset_name or superset_name not in profiles:
                raise RuntimeError(
                    f"profile {subset_name!r} names invalid satisfying profile {superset_name!r}"
                )
            superset_pins = pins[superset_name]
            for package, pin in pins[subset_name].items():
                if superset_pins.get(package) != pin:
                    got = superset_pins.get(package, "missing")
                    raise RuntimeError(
                        f"profile {subset_name!r} claims {superset_name!r} satisfies it, but "
                        f"{package} is pinned {pin!r} in the subset and {got!r} in the superset"
                    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _package_signers(resolved: list[tuple]) -> dict | None:
    """Load package trust/catalog policy from the exact component commit being released."""
    source = SPEC.get("package_signers")
    if source is None:
        return None
    if not isinstance(source, dict) or set(source) != {"component", "path"}:
        raise RuntimeError("package_signers must name exactly one component and path")
    try:
        _comp, repo, commit, _out = next(
            item for item in resolved if item[0]["name"] == source["component"])
    except (KeyError, StopIteration) as e:
        raise RuntimeError("package_signers names an unknown component") from e
    path = source.get("path")
    if not isinstance(path, str) or not path or path.startswith("/") or ".." in Path(path).parts:
        raise RuntimeError("package_signers path must be a safe repository-relative path")
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{commit}:{path}"], text=True)
        policy = json.loads(raw)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        raise RuntimeError("could not load package_signers from the release commit") from e
    if not isinstance(policy, dict) or policy.get("schema") != _PACKAGE_SIGNER_SCHEMA:
        raise RuntimeError(f"package_signers schema must be {_PACKAGE_SIGNER_SCHEMA!r}")
    keys = policy.get("keys")
    if not isinstance(keys, list) or not keys or any(
            not isinstance(key, str) or not _ED25519_PUBLIC_KEY.fullmatch(key) for key in keys):
        raise RuntimeError("package_signers keys must be a non-empty list of lowercase Ed25519 hex keys")
    if len(keys) != len(set(keys)):
        raise RuntimeError("package_signers keys must not contain duplicates")
    packages = policy.get("packages", {})
    if not isinstance(packages, dict):
        raise RuntimeError("package_signers packages must be an object")
    normalized_packages = {}
    for package_id, definition in packages.items():
        if not isinstance(package_id, str) or not _PACKAGE_ID.fullmatch(package_id):
            raise RuntimeError(f"package_signers has invalid package id {package_id!r}")
        if not isinstance(definition, dict):
            raise RuntimeError(f"package_signers package {package_id!r} must be an object")
        requirements = definition.get("requirements")
        if not isinstance(requirements, dict) or set(requirements) != _PACKAGE_REQUIREMENTS:
            raise RuntimeError(
                f"package_signers package {package_id!r} requirements must name "
                f"{sorted(_PACKAGE_REQUIREMENTS)}")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in requirements.values()):
            raise RuntimeError(
                f"package_signers package {package_id!r} requirements must be positive byte counts")
        layers = definition.get("layers")
        basis = definition.get("requirements_basis")
        if isinstance(layers, bool) or not isinstance(layers, int) or layers <= 0:
            raise RuntimeError(f"package_signers package {package_id!r} layers must be positive")
        if not isinstance(basis, str) or not basis.strip():
            raise RuntimeError(
                f"package_signers package {package_id!r} requirements_basis must explain the estimate")
        normalized_packages[package_id] = {
            "layers": layers,
            "requirements": {key: requirements[key] for key in sorted(requirements)},
            "requirements_basis": basis,
        }
    result = {"schema": _PACKAGE_SIGNER_SCHEMA, "keys": list(keys)}
    if normalized_packages:
        result["packages"] = normalized_packages
    return result


def _archive(repo: Path, commit: str, paths: list, out: Path, overlays=()) -> None:
    # git archive is deterministic for a commit; gzip -n drops the timestamp so the file is too.
    tar = subprocess.check_output(["git", "-C", str(repo), "archive", "--format=tar", commit, *paths])
    if overlays:
        buf = io.BytesIO(tar)
        commit_ts = int(_git(repo, "show", "-s", "--format=%ct", commit))
        with tarfile.open(fileobj=buf, mode="a") as tf:
            for archive_path, source, mode in overlays:
                data = Path(source).read_bytes()
                info = tarfile.TarInfo(archive_path)
                info.size, info.mode, info.mtime = len(data), mode, commit_ts
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tf.addfile(info, io.BytesIO(data))
        tar = buf.getvalue()
    gz = subprocess.run(["gzip", "-n", "-9"], input=tar, stdout=subprocess.PIPE, check=True).stdout
    out.write_bytes(gz)


def _hermetic_go_env(root: Path) -> dict[str, str]:
    """The complete Go build environment: no ambient GOENV/GOFLAGS/modfile/toolchain tuning leaks in."""
    cache = root / "go-cache"
    modcache = root / "go-mod-cache"
    gopath = root / "go-path"
    for path in (cache, modcache, gopath):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", str(Path.home())),
        "GOCACHE": str(cache),
        "GOMODCACHE": str(modcache),
        "GOPATH": str(gopath),
        "GOENV": "off",
        "GOFLAGS": "-mod=readonly",
        "GOAMD64": "v1",
        "CGO_ENABLED": "0",
        "GOOS": "linux",
        "GOARCH": "amd64",
        "GOTOOLCHAIN": "local",
    }


def _build_go_binary(repo: Path, commit: str, source: str, out: Path, go_bin: Path) -> str:
    """Compile from the exact archived commit, never from a possibly dirty checkout."""
    archive = subprocess.check_output(["git", "-C", str(repo), "archive", "--format=tar", commit, source])
    with tempfile.TemporaryDirectory(prefix="nak-go-build-") as td:
        root = Path(td)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
            tf.extractall(root, filter="data")
        env = _hermetic_go_env(root)
        version = subprocess.run([str(go_bin), "version"], env=env, capture_output=True, text=True, check=True)
        fields = version.stdout.strip().split()
        if len(fields) < 3 or fields[:2] != ["go", "version"] or fields[2] != PINNED_GO_VERSION:
            raise RuntimeError(f"Go toolchain is {version.stdout.strip()!r}; require {PINNED_GO_VERSION}")
        # The source is an archive of the already-recorded commit, with no .git directory.
        subprocess.run([str(go_bin), "build", "-buildvcs=false", "-trimpath", "-o", str(out), "."],
                       cwd=root / source, env=env, check=True)
        return fields[2]


def build(version: str, channel: str, refs: dict, key_path: Path, out_root: Path,
          uv_bin: Path, now: int | None = None, go_bin: Path | None = None) -> Path:
    now = int(now if now is not None else time.time())
    profile_specs = SPEC.get("profiles") or {}
    _validate_profile_subsumption(profile_specs)
    out = out_root / channel / version
    if out.exists():
        raise SystemExit(f"{out} already exists; versions are immutable")
    out.mkdir(parents=True)
    resolved = []
    for comp in SPEC["components"]:
        repo = Path(os.path.expanduser(comp["repo"]))
        commit = _git(repo, "rev-parse", f"{refs.get(comp['name'], comp.get('default_ref', 'HEAD'))}^{{commit}}")
        f = out / f"{comp['name']}-{commit[:12]}.tar.gz"
        resolved.append((comp, repo, commit, f))
    components, artifacts, overlays, toolchains = [], [], {}, {}
    with tempfile.TemporaryDirectory(prefix="nak-release-binaries-") as td:
        for binary in SPEC.get("binaries", []):
            comp, repo, commit, _ = next(r for r in resolved if r[0]["name"] == binary["component"])
            built = Path(td) / binary["file"]
            go_version = _build_go_binary(repo, commit, binary["source"], built,
                                          Path(go_bin or shutil.which("go") or "go"))
            toolchains["go"] = {"version": go_version}
            install_path = binary.get("install_path", f"bin/{binary['file']}")
            overlays.setdefault(comp["name"], []).append((install_path, built, 0o755))
            artifacts.append({"name": binary["name"], "path": install_path,
                              "sha256": rk.sha256_file(built), "mode": "0755",
                              "component": comp["name"], "commit": commit})
        for comp, repo, commit, f in resolved:
            _archive(repo, commit, comp["paths"], f, overlays.get(comp["name"], ()))
            components.append({"name": comp["name"], "commit": commit, "file": f.name,
                               "sha256": rk.sha256_file(f), "pythonpath": comp["pythonpath"]})
    # The two installer files, standalone, from the SAME commit as the tarball: what a newcomer's
    # pasted join line downloads and checks against the sha256 printed in their invite.
    bootstrap = {}
    for comp, built in zip(SPEC["components"], components):
        repo = Path(os.path.expanduser(comp["repo"]))
        for fname in ("install.py", "releasekit.py"):
            r = subprocess.run(["git", "-C", str(repo), "show", f"{built['commit']}:release/{fname}"],
                               capture_output=True)
            if r.returncode == 0:
                (out / fname).write_bytes(r.stdout)
                bootstrap[fname] = rk.sha256_file(out / fname)
        if bootstrap:
            break
    req_in = out / "requirements.in"
    req_in.write_text("\n".join(SPEC["requirements"]) + "\n")
    lock = out / "requirements.lock"
    subprocess.run([str(uv_bin), "pip", "compile", str(req_in), "--generate-hashes", "--quiet", "--no-header", "--no-annotate",
                    "--python-version", SPEC["python"], "-o", str(lock)], check=True)
    req_in.unlink()
    # Optional PROFILES (e.g. "inference"): each is the base requirements PLUS its own, resolved TOGETHER
    # into one hash-locked file, so an opted-in node installs a single consistent environment instead of
    # layering a second resolution on top of the first.
    profiles = {}
    for pname, prof in profile_specs.items():
        p_in = out / f"requirements-{pname}.in"
        p_in.write_text("\n".join(SPEC["requirements"] + prof["requirements"]) + "\n")
        p_lock = out / f"requirements-{pname}.lock"
        subprocess.run([str(uv_bin), "pip", "compile", str(p_in), "--generate-hashes", "--quiet", "--no-header", "--no-annotate",
                        "--python-version", SPEC["python"], "-o", str(p_lock)], check=True)
        p_in.unlink()
        profiles[pname] = {"lock": {"file": p_lock.name, "sha256": rk.sha256_file(p_lock)},
                           "requires": prof["requires"], "description": prof.get("description", ""),
                           "satisfied_by": list(prof.get("satisfied_by", []))}
    shutil.copy2(uv_bin, out / "uv")
    uv_version = subprocess.check_output([str(uv_bin), "--version"], text=True).split()[1]
    manifest = {"schema": rk.MANIFEST_SCHEMA, "name": SPEC["name"], "version": version, "channel": channel,
                "created": now, "python": SPEC["python"], "compat_major": int(version.split(".")[0]),
                "components": components,
                "artifacts": artifacts,
                "lock": {"file": lock.name, "sha256": rk.sha256_file(lock)},
                "uv": {"file": "uv", "sha256": rk.sha256_file(out / "uv"), "version": uv_version},
                "services": SPEC["services"], "health": SPEC["health"],
                "toolchains": toolchains,
                "installer": SPEC.get("installer", ""), "commands": SPEC.get("commands", {}),
                "bootstrap": bootstrap, "profiles": profiles}
    package_signers = _package_signers(resolved)
    if package_signers is not None:
        manifest["package_signers"] = package_signers
    priv_hex = key_path.read_text().strip()
    signed = rk.sign(manifest, priv_hex)
    (out / "manifest.json").write_text(json.dumps(signed, indent=1, sort_keys=True))
    # expires_at: a host can't replay an old pointer forever; a release must be republished within it.
    latest = rk.sign({"schema": "nak-release-latest/1", "channel": channel, "version": version,
                      "manifest_sha256": rk.sha256_file(out / "manifest.json"), "created": now,
                      "expires_at": now + LATEST_TTL_S}, priv_hex)
    (out_root / channel / "latest.json").write_text(json.dumps(latest, indent=1, sort_keys=True))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", required=True)
    ap.add_argument("--channel", default="canary", choices=["canary", "stable"])
    ap.add_argument("--ref", action="append", default=[], help="component=ref (repeatable)")
    ap.add_argument("--release-key", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path.home() / "nakshatra-dist")
    ap.add_argument("--uv", type=Path, default=Path(shutil.which("uv") or "uv"))
    ap.add_argument("--go", type=Path, default=Path(shutil.which("go") or "go"))
    a = ap.parse_args(argv)
    refs = dict(r.split("=", 1) for r in a.ref)
    out = build(a.version, a.channel, refs, a.release_key, a.out, a.uv, go_bin=a.go)
    print(f"built {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
