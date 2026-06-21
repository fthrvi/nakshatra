"""ChainLifecycle.warm() — pre-warm-on-model-select, tested without GPU/network.

Proves: warm() summons the chain ahead of demand (cold-start off the critical
path), is idempotent when already hot, does NOT count as an in-flight request
(so the reaper still governs it), returns False on summon timeout, and a real
begin() after a warm() reuses the summon (no double cold-start)."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import serve_lifecycle as sl
import slice_warm


class _Ctrl:
    """Fake ChainController: ready iff started and `ready` flag set."""
    def __init__(self, ready=True):
        self.ready = ready
        self.starts = 0
        self.stops = 0
    def start(self):
        self.starts += 1
    def stop(self):
        self.stops += 1
        self.ready_started = False
    def is_ready(self):
        return self.ready and self.starts > 0


def _lc(ctrl, **kw):
    kw.setdefault("idle_grace_s", 600.0)
    kw.setdefault("start_timeout_s", 5.0)
    kw.setdefault("poll_s", 0.01)
    return sl.ChainLifecycle(ctrl, **kw)


def test_warm_summons_and_marks_up():
    c = _Ctrl(ready=True)
    lc = _lc(c)
    assert lc.warm() is True
    assert c.starts == 1
    assert lc._up is True
    # pre-warm must NOT register as an in-flight request
    assert lc._active == 0


def test_warm_idempotent_when_already_hot():
    c = _Ctrl(ready=True)
    lc = _lc(c)
    assert lc.warm() is True
    assert lc.warm() is True
    assert c.starts == 1  # no re-summon when already hot


def test_warm_times_out_when_never_ready():
    c = _Ctrl(ready=False)
    lc = _lc(c, start_timeout_s=0.05)
    assert lc.warm() is False
    assert lc._up is False


def test_begin_after_warm_reuses_summon():
    c = _Ctrl(ready=True)
    lc = _lc(c)
    assert lc.warm() is True
    lc.begin()                 # real request — should not cold-start again
    assert c.starts == 1       # reused the warm summon
    assert lc._active == 1
    lc.end()
    assert lc._active == 0


def test_warm_resets_idle_clock():
    c = _Ctrl(ready=True)
    lc = _lc(c)
    lc._last_active = 0.0
    lc.warm()
    assert lc._last_active > 0.0   # reaper grace window starts fresh


def test_warm_warms_slice_paths(tmp_path):
    # warm() should page-cache the model's local slices before summoning workers.
    f = tmp_path / "model@abc-L0-16.gguf"
    f.write_bytes(os.urandom(3 << 20))
    c = _Ctrl(ready=True)
    lc = sl.ChainLifecycle(c, warm_paths=[str(f)], start_timeout_s=5, poll_s=0.01)
    assert lc.warm() is True
    frac = slice_warm.resident_fraction(str(f))
    assert frac == -1.0 or frac > 0.9   # slice is hot in page cache after warm


def test_warm_paths_missing_file_does_not_break_summon(tmp_path):
    # a stale/missing slice path must never break the summon (best-effort warm).
    c = _Ctrl(ready=True)
    lc = sl.ChainLifecycle(c, warm_paths=[str(tmp_path / "gone.gguf")],
                           start_timeout_s=5, poll_s=0.01)
    assert lc.warm() is True
    assert c.starts == 1


def test_ensure_fn_runs_before_summon_and_sets_warm_paths(tmp_path):
    # summon = fetch-if-absent (ensure_fn) → warm → start. ensure_fn's returned
    # paths become the warm set, and it runs before the controller starts.
    order = []
    f = tmp_path / "fetched@h-L0-16.gguf"
    f.write_bytes(b"GGUF" + b"\x00" * 1024)

    class _OrderCtrl(_Ctrl):
        def start(self):
            order.append("start")
            super().start()

    def ensure():
        order.append("ensure")
        return [str(f)]

    c = _OrderCtrl(ready=True)
    lc = sl.ChainLifecycle(c, ensure_fn=ensure, start_timeout_s=5, poll_s=0.01)
    assert lc.warm() is True
    assert order == ["ensure", "start"]      # fetch happened before summon
    assert lc.warm_paths == [str(f)]         # adopted the fetched path


def test_ensure_fn_failure_does_not_break_summon():
    def boom():
        raise RuntimeError("all sources down")
    c = _Ctrl(ready=True)
    lc = sl.ChainLifecycle(c, ensure_fn=boom, start_timeout_s=5, poll_s=0.01)
    assert lc.warm() is True                  # degrades gracefully
    assert c.starts == 1
