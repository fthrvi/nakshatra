"""ChainLifecycle.warm() — pre-warm-on-model-select, tested without GPU/network.

Proves: warm() summons the chain ahead of demand (cold-start off the critical
path), is idempotent when already hot, does NOT count as an in-flight request
(so the reaper still governs it), returns False on summon timeout, and a real
begin() after a warm() reuses the summon (no double cold-start)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import serve_lifecycle as sl


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
