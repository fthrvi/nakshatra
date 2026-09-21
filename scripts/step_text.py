"""Escaping for the per-token `[chain] step N: id=<id> '<text>'` lines client.py prints.

Why: a generated token can contain a newline (`):\n`, `\n    `, `\n\n`) - which is most tokens in code. Printed raw, one
step spans several output lines, and the serve's line-based stream parser silently DROPS any step that does not fit on
one line: every newline and every token ending in one vanished from the streamed reply. Measured 2026-09-21 through the
blackwell+ijru chain: the non-streaming reply was correct, the streamed one (which Aider always uses) came back as
`def snake_to_camel(s    \"\"\"Convert ...` - no `):`, no newlines - and every edit was garbage.

One token = one line: backslash, newline and carriage return are escaped by the client and unescaped by the serve.
Pure, dependency-free, so both sides (and the tests) can import it.
"""
from __future__ import annotations

import re

_UNESCAPE = {"n": "\n", "r": "\r", "\\": "\\"}
_ESCAPED = re.compile(r"\\(.)", re.DOTALL)


def escape_step_text(text: str) -> str:
    """The token text as ONE physical line (backslash first, so the escapes themselves round-trip)."""
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")


def unescape_step_text(text: str) -> str:
    """Inverse of escape_step_text. Single pass, so an escaped backslash followed by `n` stays a backslash and an `n`."""
    return _ESCAPED.sub(lambda m: _UNESCAPE.get(m.group(1), m.group(0)), text)
