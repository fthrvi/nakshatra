"""Which backend a worker REGISTERS with Sthambha (worker.decide_registered_backend).

A newer llama.cpp daemon build is quiet on a successful load: no `offloaded N/M layers` line at all. The old check
read that as "0 layers offloaded" and downgraded a 12 GB RTX 5070 to a CPU node in the registry (2026-09-21). Silence
is UNVERIFIED, not zero; only a positive `offloaded 0/N` proves the binary lacks the backend.
"""
import importlib.util
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_scripts))
_spec = importlib.util.spec_from_file_location("worker_under_test", _scripts / "worker.py")
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)

decide = worker.decide_registered_backend


def _status(n_offloaded, total):
    return {"n_offloaded": n_offloaded, "total_layers": total, "uses_gpu": n_offloaded > 0}


def test_a_positive_offload_report_keeps_the_gpu_backend_and_is_verified():
    assert decide(12, "cuda", _status(48, 48)) == ("cuda", True)


def test_a_quiet_daemon_keeps_the_declaration_but_is_flagged_unverified():
    """The live case: successful CUDA load, daemon logs nothing about offload."""
    assert decide(12, "cuda", _status(0, 0)) == ("cuda", False)


def test_a_positive_zero_report_downgrades_to_cpu():
    """The daemon really printed `offloaded 0/48`: the binary lacks the backend, so registering it as CUDA would lie."""
    assert decide(12, "cuda", _status(0, 48)) == ("cpu", True)


def test_nothing_to_verify_when_no_gpu_was_declared():
    assert decide(0, "cuda", _status(0, 0)) == ("cuda", True)
    assert decide(12, "cpu", _status(0, 0)) == ("cpu", True)
