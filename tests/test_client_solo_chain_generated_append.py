"""Regression test for a bug shipped and caught the same day (2026-09-12): client.py's new
solo-chain branch (see client/solo-chain-response) computed next_id correctly but never called
generated.append(next_id) — every OTHER branch (push mode, the multi-worker "last worker" step)
does. Step 1 of a solo chain therefore "succeeded" with an empty output, and step 2 crashed with
IndexError: list index out of range at `input_tokens = [generated[-1]]`, live, in production,
because main()'s decode loop has no unit-test seam to catch it in isolation first.

This can't be a real behavioral test without a large fake-gRPC harness for main() itself (out of
scope for a one-line fix) — it inspects the ACTUAL source of client.main() and asserts the solo
branch appends to `generated` in the same shape the other two branches do, so a future edit that
drops the append again fails this test immediately instead of waiting for a live 502."""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import client as cli  # noqa: E402


def test_solo_branch_appends_next_id_to_generated():
    src = inspect.getsource(cli.main)
    m = re.search(
        r"elif len\(sorted_stubs\) == 1:.*?(?=\n                else:)",
        src, re.DOTALL,
    )
    assert m is not None, "could not find the solo-chain branch in client.main() — did it move or get renamed?"
    solo_branch = m.group(0)
    assert "next_id = struct.unpack" in solo_branch, "solo branch should compute next_id"
    assert "generated.append(next_id)" in solo_branch, (
        "solo branch computes next_id but never appends it to `generated` — this is exactly "
        "the bug that shipped 2026-09-12: step 1 'succeeds' silently with no output, step 2 "
        "crashes with IndexError on generated[-1]"
    )
