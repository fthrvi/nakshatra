"""A chain must NOT start on a node whose Sthambha lease was refused (a mesh_compute GPU job holds it).

Reviewer finding @cec2270: PillarLeaseClient.lease() retried a 409 three times, logged, returned None; begin()/warm() then started
the chain anyway - two GPU workloads on one card is what hard-crashed blackwell on 2026-09-29. Only an EXPLICIT refusal (409 with
`node_conflicts`) fails closed; unreachable pillar / 5xx / plain 409 / no lease client stay fail-open.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import serve_lifecycle as sl  # noqa: E402

CONFLICT = {"error": "node conflict", "node_conflicts": [{"node": "blackwell", "holder": "job:train-42"}]}


class _Ctl(sl.ChainController):
    def __init__(self):
        self.starts = 0
        self.stops = 0
        self._up = False

    def start(self):
        self.starts += 1
        self._up = True

    def stop(self):
        self.stops += 1
        self._up = False

    def is_ready(self):
        return self._up


def _client(responses):
    """Real PillarLeaseClient with _call scripted: responses is a list of (status, body), last repeats."""
    logs = []
    c = sl.PillarLeaseClient("http://x", "m", "k", b"", log=logs.append)
    c.calls = []
    seq = list(responses)

    def _call(method, path, body=None, timeout=10.0):
        c.calls.append((method, path))
        return seq.pop(0) if len(seq) > 1 else seq[0]
    c._call = _call
    c.logs = logs
    return c


def _lc(ctl, client, logs=None):
    lc = sl.ChainLifecycle(ctl, log=(logs.append if logs is not None else (lambda *_: None)), start_timeout_s=5, poll_s=0.01)
    lc.lease_client = client
    return lc


# ── the client ───────────────────────────────────────────────────────
def test_conflict_409_is_a_refusal_not_retried():
    c = _client([(409, CONFLICT)])
    assert c.lease() is None
    assert len(c.calls) == 1, "a refusal is not transient: no retries"
    assert c.last_refusal["node_conflicts"] == CONFLICT["node_conflicts"]


def test_plain_409_keeps_the_old_retry_semantics():
    c = _client([(409, {"error": "chain not ready"})])
    assert c.lease(retries=3) is None
    assert len(c.calls) == 3
    assert c.last_refusal is None


def test_a_later_success_clears_the_refusal():
    c = _client([(409, CONFLICT), (200, {"lease_id": "L1"})])
    assert c.lease() is None and c.last_refusal
    assert c.lease()["lease_id"] == "L1"
    assert c.last_refusal is None


# ── begin() ──────────────────────────────────────────────────────────
def test_begin_refused_starts_nothing_and_raises_naming_the_holder():
    ctl = _Ctl()
    lc = _lc(ctl, _client([(409, CONFLICT)]))
    try:
        lc.begin()
        raise AssertionError("expected ChainRefused")
    except sl.ChainRefused as e:
        assert "blackwell" in str(e) and "job:train-42" in str(e)
    assert ctl.starts == 0 and ctl.stops == 0
    assert lc._active == 0 and lc._summoning is False


def test_begin_pillar_unreachable_fails_open_and_logs():
    ctl = _Ctl()
    logs = []
    lc = _lc(ctl, _client([(0, {"error": "connection refused"})]), logs)
    lc.begin()
    assert ctl.starts == 1
    assert any("fail-open" in l for l in logs), logs


def test_begin_pillar_5xx_fails_open():
    ctl = _Ctl()
    lc = _lc(ctl, _client([(503, {"error": "down"})]))
    lc.begin()
    assert ctl.starts == 1


def test_begin_plain_409_fails_open():
    ctl = _Ctl()
    lc = _lc(ctl, _client([(409, {"error": "chain not ready"})]))
    lc.begin()
    assert ctl.starts == 1


def test_begin_without_a_lease_client_starts():
    ctl = _Ctl()
    lc = _lc(ctl, None)
    lc.begin()
    assert ctl.starts == 1


# ── warm() ───────────────────────────────────────────────────────────
def test_warm_refused_starts_nothing():
    ctl = _Ctl()
    lc = _lc(ctl, _client([(409, CONFLICT)]))
    try:
        lc.warm()
        raise AssertionError("expected ChainRefused")
    except sl.ChainRefused as e:
        assert "job:train-42" in str(e)
    assert ctl.starts == 0


def test_warm_unreachable_pillar_still_warms():
    ctl = _Ctl()
    lc = _lc(ctl, _client([(0, {})]))
    assert lc.warm() is True and ctl.starts == 1


# ── mid-serve loss ───────────────────────────────────────────────────
def test_renew_404_then_refused_does_not_kill_a_running_chain():
    ctl = _Ctl()
    ctl._up = True
    c = _client([(200, {"lease_id": "L1"})])
    c.lease()
    c._call = lambda m, p, b=None, timeout=10.0: (404, {}) if "renew" in p else (409, CONFLICT)
    lc = _lc(ctl, c)
    lc._up = True
    lc._ready_until = time.monotonic() + 60
    lc.begin()                       # chain already up: serves, no raise, no stop
    assert ctl.starts == 0 and ctl.stops == 0
    assert c.lease_id is None and any("lost" in l for l in c.logs)


def test_renew_404_relases_and_recovers():
    c = _client([(200, {"lease_id": "L1"})])
    c.lease()
    seq = {"n": 0}

    def _call(m, p, b=None, timeout=10.0):
        if "renew" in p:
            return 404, {}
        return 200, {"lease_id": "L2"}
    c._call = _call
    assert c.renew()["lease_id"] == "L2" and c.lease_id == "L2"
