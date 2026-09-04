"""Join phase 3: is the slice we fetched the one we asked for?

The download already happened; its RESULT is in `facts`. This decides what the result means.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# ⚠️ Precedence: needs_operator > fatal > transient. Retrying a full disk is the worst of the
# wrong answers — it burns the window and the disk is still full.
_CLASSES = (
    ("needs_operator", ("no space left", "disk quota exceeded", "permission denied",
                        "command not found", "unable to locate package", "read-only file system")),
    ("fatal", ("404", "no such file", "not found", "unsupported architecture",
               "checksum mismatch", "certificate verify failed")),
    ("transient", ("timeout", "timed out", "connection reset", "temporary failure",
                   "503", "502", "try again", "network is unreachable")),
)
_ADVICE = {"needs_operator": "a person must act; retrying will not help",
           "fatal": "the request itself is wrong and will fail identically",
           "transient": "retry may help"}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _classify(stderr: str) -> str:
    """⚠️ Let the STDERR decide, never the exit code. A nonzero exit says only that it
    failed; the stderr says whether trying again could help. Defaulting to `fatal` on a
    nonzero exit is what made a plain timeout unretryable, twice."""
    low = (stderr or "").lower()
    for kind, phrases in _CLASSES:
        if any(p in low for p in phrases):
            return kind
    return "fatal"


def acquire(facts: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    if not isinstance(facts, dict):
        return False, ["facts is not a dict — nothing was observed"], {}
    problems: List[str] = []

    code = facts.get("download_exit_code")
    if not isinstance(code, int) or isinstance(code, bool):
        problems.append("download_exit_code not observed")
    elif code != 0:
        kind = _classify(facts.get("download_stderr") if isinstance(facts.get("download_stderr"), str) else "")
        problems.append(f"{kind}: {_ADVICE[kind]} (download exited {code})")
        return False, problems, {}

    exp_sha = facts.get("expected_sha256")
    if not isinstance(exp_sha, str) or not _HEX64.match(exp_sha):
        # ⚠️ Name the EXPECTATION as malformed, not the file. The file may be perfect; it is
        # the caller that is broken, and blaming the download sends the operator downstream.
        problems.append("expected_sha256 is malformed (not 64 lowercase hex) — the "
                        "expectation is wrong, not necessarily the file")

    got_b, exp_b = facts.get("downloaded_bytes"), facts.get("expected_bytes")
    size_ok = True
    if isinstance(got_b, int) and isinstance(exp_b, int) and not isinstance(got_b, bool):
        if exp_b > 0 and got_b != exp_b:
            # ⚠️ Size BEFORE hash: a truncated multi-gigabyte download is the common case and
            # one comparison settles it, where hashing costs minutes the operator is watching.
            size_ok = False
            problems.append(f"size mismatch: expected {exp_b} bytes, got {got_b} — "
                            "the download is incomplete")
    elif "downloaded_bytes" not in facts or "expected_bytes" not in facts:
        problems.append("downloaded_bytes/expected_bytes not observed")

    got_sha = facts.get("downloaded_sha256")
    if size_ok and isinstance(exp_sha, str) and _HEX64.match(exp_sha):
        if not isinstance(got_sha, str):
            problems.append("downloaded_sha256 not observed")
        elif got_sha.lower() != exp_sha:
            problems.append("hash mismatch — DELETE the file; a corrupt slice left on disk "
                            "is picked up by the next run and looks like a fresh download")

    if problems:
        return False, problems, {}
    return True, [], {"slice_verified": True, "slice_bytes": got_b}
