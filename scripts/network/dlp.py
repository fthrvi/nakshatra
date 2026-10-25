"""The node's outbound floor: refuse text carrying a recognisable secret or a bulk encoded payload.

Same formats as Prithvi's web egress DLP (mind/web_tools.py). Shared by the daemon and by signed
records (suspension reasons) that leave the node, so every path that sends text checks it."""
from __future__ import annotations

import re as _re

_SECRET_FMT = _re.compile(
    r"(-----BEGIN |ssh-rsa\s|AKIA[0-9A-Z]{16}|"
    r"gh[posru]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"sk-ant-[A-Za-z0-9_-]{20,}|sk-proj-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,}|nsec1[a-z0-9]{20,}|"
    r"bcn_[a-f0-9]{16,}|prithvi-canary-[A-Za-z0-9]{8,}|nki1\.)")
_BLOB = _re.compile(r"[A-Za-z0-9+/=_-]{128,}")


def outbound_problem(text: str) -> str:
    """A reason to refuse sending this text, or ''. (An invite code counts: invites are handed over
    by people, never forwarded by an agent.)"""
    if _SECRET_FMT.search(text):
        return "message contains a secret-shaped token; not sending it"
    if _BLOB.search(text):
        return "message contains a long encoded blob; not sending it"
    return ""
