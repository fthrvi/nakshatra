#!/usr/bin/env python3
"""build.py — make one signed Nakshatra node release.

    python release/build.py --version 0.1.0 --channel canary \
        --ref nakshatra=<commit-or-branch> --ref sthambha=<commit-or-branch> \
        --release-key ~/.nakshatra-release/release-TEST.key --out ~/nakshatra-dist

Output, under <out>/<channel>/<version>/:
    nakshatra-<commit12>.tar.gz   git archive of the node's Nakshatra paths at that commit
    sthambha-<commit12>.tar.gz    git archive of the Sthambha package at that commit
    requirements.lock             hash-locked dependencies (uv pip compile --generate-hashes)
    uv                            the uv binary that builds the environment on the node
    manifest.json                 every file's sha256 + commits + version, SIGNED by the release key
and <out>/<channel>/latest.json pointing at the version (also signed), which is what updaters poll.

Nothing here is hand-built on a node: nodes only download, verify and unpack these files.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import releasekit as rk  # noqa: E402

SPEC = json.loads((Path(__file__).resolve().parent / "spec.json").read_text())


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _archive(repo: Path, commit: str, paths: list, out: Path) -> None:
    # git archive is deterministic for a commit; gzip -n drops the timestamp so the file is too.
    tar = subprocess.check_output(["git", "-C", str(repo), "archive", "--format=tar", commit, *paths])
    gz = subprocess.run(["gzip", "-n", "-9"], input=tar, stdout=subprocess.PIPE, check=True).stdout
    out.write_bytes(gz)


def build(version: str, channel: str, refs: dict, key_path: Path, out_root: Path,
          uv_bin: Path, now: int | None = None) -> Path:
    now = int(now if now is not None else time.time())
    out = out_root / channel / version
    if out.exists():
        raise SystemExit(f"{out} already exists; versions are immutable")
    out.mkdir(parents=True)
    components = []
    for comp in SPEC["components"]:
        repo = Path(os.path.expanduser(comp["repo"]))
        commit = _git(repo, "rev-parse", f"{refs.get(comp['name'], comp.get('default_ref', 'HEAD'))}^{{commit}}")
        f = out / f"{comp['name']}-{commit[:12]}.tar.gz"
        _archive(repo, commit, comp["paths"], f)
        components.append({"name": comp["name"], "commit": commit, "file": f.name,
                           "sha256": rk.sha256_file(f), "pythonpath": comp["pythonpath"]})
    req_in = out / "requirements.in"
    req_in.write_text("\n".join(SPEC["requirements"]) + "\n")
    lock = out / "requirements.lock"
    subprocess.run([str(uv_bin), "pip", "compile", str(req_in), "--generate-hashes", "--quiet",
                    "--python-version", SPEC["python"], "-o", str(lock)], check=True)
    req_in.unlink()
    shutil.copy2(uv_bin, out / "uv")
    uv_version = subprocess.check_output([str(uv_bin), "--version"], text=True).split()[1]
    manifest = {"schema": rk.MANIFEST_SCHEMA, "name": SPEC["name"], "version": version, "channel": channel,
                "created": now, "python": SPEC["python"], "compat_major": int(version.split(".")[0]),
                "components": components,
                "lock": {"file": lock.name, "sha256": rk.sha256_file(lock)},
                "uv": {"file": "uv", "sha256": rk.sha256_file(out / "uv"), "version": uv_version},
                "services": SPEC["services"], "health": SPEC["health"],
                "installer": SPEC.get("installer", "")}
    priv_hex = key_path.read_text().strip()
    signed = rk.sign(manifest, priv_hex)
    (out / "manifest.json").write_text(json.dumps(signed, indent=1, sort_keys=True))
    latest = rk.sign({"schema": "nak-release-latest/1", "channel": channel, "version": version,
                      "manifest_sha256": rk.sha256_file(out / "manifest.json"), "created": now}, priv_hex)
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
    a = ap.parse_args(argv)
    refs = dict(r.split("=", 1) for r in a.ref)
    out = build(a.version, a.channel, refs, a.release_key, a.out, a.uv)
    print(f"built {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
