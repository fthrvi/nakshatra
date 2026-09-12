"""Tests for "solo" worker mode (2026-09-12, placement lane).

A whole-model worker holding every layer — what placement.py's route-whole planner emits
(assignment_from_plan: mode="solo" when n==1) once NKS_SMART_PLACEMENT can actually see live
telemetry (see reference_nakshatra_placement_identity_split in trisul memory). Never exercised
before that fix landed, so nothing here — worker.py's own argparse, the WorkerServicer branches
that key off mode, or the C++ daemon's mode parser — had ever accepted the string "solo" at all.
This file locks in the Python-side half of that fix; the daemon side is a separate C++ change in
llama.cpp's worker_daemon.cpp (not testable from here — daemon binary tests are the smoke scripts).
"""
from __future__ import annotations

import collections as _c
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import worker  # noqa: E402


def test_mode_argparse_accepts_solo():
    """worker.py builds its real --mode parser inline inside main(), not a separately-callable
    function — inspect the ACTUAL source of main() rather than a parser built fresh here, so this
    fails if the real choices=[...] line ever regresses instead of just checking a duplicate."""
    import inspect
    import re
    src = inspect.getsource(worker.main)
    m = re.search(r'add_argument\("--mode".*?choices=\[([^\]]+)\]', src)
    assert m is not None, "could not find the --mode argparse definition in worker.main()"
    choices = [c.strip().strip('"\'') for c in m.group(1).split(",")]
    assert choices == ["first", "middle", "last", "solo"], choices


class _DaemonStub:
    def __init__(self):
        self.recent_rpc_ms = _c.deque(maxlen=20)

    def info(self):
        return dict(layer_start=0, layer_end=32, n_embd=4096, has_token_embd=True,
                    has_lm_head=True, n_vocab=128256)


def _servicer(mode: str) -> "worker.WorkerServicer":
    return worker.WorkerServicer(daemon=_DaemonStub(), mode=mode,
                                 layer_start=0, layer_end=32, model_id="test")


def test_info_reports_both_embd_and_lm_head_for_solo():
    """The bug this pins down: has_token_embd/has_lm_head were `mode == "first"` / `mode ==
    "last"` — for mode="solo" that's False/False, telling the client this worker holds
    NEITHER end of the model, which is backwards for a worker holding the WHOLE thing."""
    resp = _servicer("solo").Info(None, None)
    assert resp.has_token_embd is True
    assert resp.has_lm_head is True


def test_info_first_and_last_unchanged():
    """Solo must not have widened first/last themselves — regression guard."""
    first = _servicer("first").Info(None, None)
    assert first.has_token_embd is True
    assert first.has_lm_head is False

    last = _servicer("last").Info(None, None)
    assert last.has_token_embd is False
    assert last.has_lm_head is True


def test_middle_still_reports_neither():
    mid = _servicer("middle").Info(None, None)
    assert mid.has_token_embd is False
    assert mid.has_lm_head is False
