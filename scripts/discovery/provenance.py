"""Build-provenance fingerprint — the IMMUTABLE engine-build identity of a node,
the companion axis to drift_gauge.py.

Two different questions, two different fingerprints:
  • drift_gauge  → "what does this node COMPUTE?"  (behavioral: the greedy token
    sequence; the bit-determinism gate that decides whether two nodes may chain.)
  • provenance   → "which BUILD is actually running?"  (the engine binary + code
    version loaded on this node.)

They are deliberately separate. A drift-class can stay identical across rebuilds
that don't flip any argmax — so the behavioral gauge alone cannot tell you the
engine was swapped, updated, or tampered with. Provenance changes the instant the
binary bytes or the code SHA change, even when behavior is byte-identical. So:

  drift-class  answers  "may we chain bit-deterministically?"   (admission gate)
  provenance   answers  "is the engine running the one we expect?"  (integrity / audit / pinning)

This is the missing half of a node's identity: not just *how it behaves* but
*what it is built from*. A node advertises BOTH; provenance is NOT a chaining gate
(drift_compatible owns that) — it is the identity & integrity record, used to pin
an expected build and alert if the running build ever silently changes.

Anchored on the SHA-256 of the daemon binary — the bytes actually executing, present
even on an ad-hoc build whose compile stamps read "unknown" — plus the stamps the
daemon prints on --version (nakshatra code SHA, build host, build timestamp).

Pure + dependency-light: hashlib for the hashes, subprocess only to read --version.
This module owns the contract, not the inference — a worker, a CLI probe, and a test
all agree here on what "the same build" means.
"""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from typing import Optional

PROV_VERSION = 1
UNKNOWN = "unknown"


@dataclass(frozen=True)
class BuildProvenance:
    """A node's build-provenance id — which engine build is running."""
    prov_version: int
    engine_sha256: str   # sha256 of the daemon binary — the bedrock, always present
    code_sha: str        # nakshatra git SHA stamped at build (or "unknown")
    build_host: str      # host that built it (or "unknown")
    built_at: str        # compile timestamp (or "unknown")
    fingerprint: str     # hex sha256 over the canonical payload below

    def matches(self, other: "BuildProvenance") -> bool:
        """True iff the two nodes are running the SAME build. Different
        prov_version is never comparable (returns False)."""
        return (self.prov_version == other.prov_version
                and self.fingerprint == other.fingerprint)

    def short(self) -> str:
        return f"prov{self.prov_version}:{self.fingerprint[:12]}"

    def wire(self) -> str:
        """Self-contained, exact-comparable id for advertising to peers
        (version-tagged full fingerprint). Compare with simple string equality."""
        return f"prov{self.prov_version}:{self.fingerprint}"

    def describe(self) -> str:
        return (f"engine={self.engine_sha256[:12]} code={self.code_sha} "
                f"host={self.build_host} built={self.built_at} → {self.short()}")


def fingerprint_from_facts(engine_sha256: str, code_sha: Optional[str],
                           build_host: Optional[str], built_at: Optional[str],
                           prov_version: int = PROV_VERSION) -> BuildProvenance:
    """Build the provenance fingerprint from concrete build facts. The engine
    binary hash is the load-bearing input; the stamps refine it. Any stamp may be
    'unknown' (an ad-hoc build) — the fingerprint is still well-defined and stable."""
    code_sha = code_sha or UNKNOWN
    build_host = build_host or UNKNOWN
    built_at = built_at or UNKNOWN
    payload = f"{prov_version}\n{engine_sha256}\n{code_sha}\n{build_host}\n{built_at}"
    fp = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return BuildProvenance(prov_version=prov_version, engine_sha256=engine_sha256,
                           code_sha=code_sha, build_host=build_host,
                           built_at=built_at, fingerprint=fp)


def sha256_file(path: str, _bufsize: int = 1 << 20) -> str:
    """Stream-hash a (possibly large) binary without loading it into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_daemon_version(version_output: str) -> dict:
    """Parse the daemon's --version stamps:

        nakshatra-fabric-worker
          sha        <code_sha>
          built_on   <host>
          built_at   <ts>

    Returns {code_sha, build_host, built_at}, each 'unknown' if its line is
    absent. Tolerant of extra/leading lines and surrounding whitespace."""
    out = {"code_sha": UNKNOWN, "build_host": UNKNOWN, "built_at": UNKNOWN}
    for line in version_output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        key, val = parts[0], parts[1].strip()
        if key == "sha":
            out["code_sha"] = val
        elif key == "built_on":
            out["build_host"] = val
        elif key == "built_at":
            out["built_at"] = val
    return out


def provenance_from_daemon(daemon_bin: str, *, version_timeout: float = 10.0) -> BuildProvenance:
    """One-call: hash the daemon binary + read its --version stamps → provenance.
    Fail-soft on the --version read (stamps fall back to 'unknown') because the
    binary hash alone is a valid, immutable provenance anchor."""
    engine = sha256_file(daemon_bin)
    try:
        proc = subprocess.run([daemon_bin, "--version"], capture_output=True,
                              text=True, timeout=version_timeout)
        stamps = parse_daemon_version(proc.stdout)
    except Exception:
        stamps = {"code_sha": UNKNOWN, "build_host": UNKNOWN, "built_at": UNKNOWN}
    return fingerprint_from_facts(engine, stamps["code_sha"], stamps["build_host"],
                                  stamps["built_at"])


def provenance_changed(pinned: BuildProvenance, observed: BuildProvenance) -> bool:
    """True iff the running build differs from the pinned/expected one — i.e. the
    engine has been updated, swapped, or tampered with since we recorded it. For
    integrity ALERTING, never for chaining admission (drift_compatible owns that)."""
    return not pinned.matches(observed)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <daemon_bin>\n"
                 f"  prints this node's build-provenance fingerprint for the daemon binary")
    prov = provenance_from_daemon(sys.argv[1])
    print(prov.describe())
