"""ChainLifecycle: readiness cache, single-flight summon, and reaping that cannot be lost.

Found by the 2026-09-21 Fable+astra consult of the scale-to-zero chain:
  * begin() re-probed the chain on EVERY request (~1 s to a far node) -> cache a positive answer;
  * two overlapping cold requests each launched the workers -> one summon, the other waits;
  * a /lease that failed (409 on a cold chain) left NO lease id, and is_expired() is False for "no lease" -> the chain was never
    reaped and held its GPUs (blackwell's coder could not load) forever -> fall back to the local idle clock, and take the lease
    once the chain is up;
  * a reap decided outside the lock could stop workers under a request that arrived meanwhile;
  * a request longer than the pillar's lease grace let the lease lapse under it -> renew while busy.
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import serve_lifecycle as sl  # noqa: E402


class _Ctl(sl.ChainController):
    def __init__(self, ready=True, start_makes_ready_after=0.0):
        self.ready = ready
        self.ready_after = start_makes_ready_after
        self._started_at = None
        self.starts = 0
        self.stops = 0
        self.is_ready_calls = 0

    def start(self):
        self.starts += 1
        self._started_at = time.monotonic()

    def stop(self):
        self.stops += 1
        self.ready = False
        self._started_at = None

    def is_ready(self):
        self.is_ready_calls += 1
        if self._started_at is not None:
            return time.monotonic() - self._started_at >= self.ready_after
        return self.ready


class _Lease:
    def __init__(self, first_lease_ok=True, expired=True, expire_delay=0.0):
        self.lease_id = None
        self.first_ok = first_lease_ok
        self.expired = expired
        self.expire_delay = expire_delay
        self.lease_calls = 0
        self.renews = 0

    def lease(self, retries=3):
        self.lease_calls += 1
        if self.lease_calls == 1 and not self.first_ok:
            return None
        self.lease_id = "L1"
        return {"lease_id": "L1"}  # no idle_grace_s: the pillar's grace would override the tiny one the test sets

    def renew(self):
        self.renews += 1
        return self.lease() if not self.lease_id else {}

    def is_expired(self):
        if not self.lease_id:
            return False  # exactly what PillarLeaseClient.is_expired does for "no lease"
        time.sleep(self.expire_delay)
        return self.expired


def _lc(ctl, **kw):
    kw.setdefault("start_timeout_s", 5)
    kw.setdefault("poll_s", 0.01)
    return sl.ChainLifecycle(ctl, log=lambda *_: None, **kw)


def _wait(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


# -- readiness cache ---------------------------------------------------------------------------------------------------

def test_a_chain_that_just_answered_is_not_re_probed_on_every_request():
    ctl = _Ctl(ready=True)
    lc = _lc(ctl, ready_cache_s=30)
    lc._up = True
    for _ in range(5):
        lc.begin()
        lc.end()
    assert ctl.is_ready_calls == 1


def test_ready_cache_can_be_disabled():
    ctl = _Ctl(ready=True)
    lc = _lc(ctl, ready_cache_s=0)
    lc._up = True
    for _ in range(3):
        lc.begin()
        lc.end()
    assert ctl.is_ready_calls == 3


def test_a_cached_ready_does_not_survive_a_reap():
    ctl = _Ctl(ready=True)
    lc = _lc(ctl, ready_cache_s=30, idle_grace_s=0.05)
    lc._up = True
    lc.begin()
    lc.end()
    lc.start_reaper()
    assert _wait(lambda: ctl.stops == 1)
    lc.stop_reaper()
    assert lc._up is False and lc._ready_until == 0.0


# -- single-flight summon ----------------------------------------------------------------------------------------------

def test_overlapping_cold_requests_summon_the_chain_once():
    ctl = _Ctl(ready=False, start_makes_ready_after=0.3)
    lc = _lc(ctl)
    errs = []

    def _req():
        try:
            lc.begin()
            lc.end()
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    threads = [threading.Thread(target=_req) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not errs and ctl.starts == 1, (errs, ctl.starts)


def test_a_failed_summon_does_not_wedge_later_requests():
    class _Boom(_Ctl):
        def start(self):
            self.attempts = getattr(self, "attempts", 0) + 1
            if self.attempts == 1:
                raise RuntimeError("ssh failed")
            super().start()

    ctl = _Boom(ready=False, start_makes_ready_after=0.0)
    lc = _lc(ctl)
    try:
        lc.begin()
    except RuntimeError:
        pass
    lc.begin()  # would deadlock/skip the summon if _summoning were left True
    assert ctl.attempts == 2 and ctl.starts == 1


# -- reaping -----------------------------------------------------------------------------------------------------------

def test_a_chain_with_no_pillar_lease_is_still_reaped_on_the_local_idle_clock():
    """The never-reaped bug: lease() failed, so lease_id stayed None and is_expired() was never even asked."""
    ctl = _Ctl(ready=True)
    lease = _Lease(first_lease_ok=False)
    lease.lease = lambda retries=3: None  # the pillar never grants one
    lc = _lc(ctl, lease_client=lease, idle_grace_s=0.05)
    lc._up = True
    lc.start_reaper()
    assert _wait(lambda: ctl.stops == 1)
    lc.stop_reaper()


def test_a_held_lease_that_has_not_expired_still_blocks_the_reap():
    """Control: another consumer holds the lease - the pillar decides, exactly as before."""
    ctl = _Ctl(ready=True)
    lease = _Lease(expired=False)
    lease.lease_id = "L1"
    lc = _lc(ctl, lease_client=lease, idle_grace_s=0.05)
    lc._up = True
    lc.start_reaper()
    time.sleep(0.4)
    lc.stop_reaper()
    assert ctl.stops == 0


def test_a_held_lease_that_expired_is_reaped():
    ctl = _Ctl(ready=True)
    lease = _Lease(expired=True)
    lease.lease_id = "L1"
    lc = _lc(ctl, lease_client=lease, idle_grace_s=0.05)
    lc._up = True
    lc.start_reaper()
    assert _wait(lambda: ctl.stops == 1)
    lc.stop_reaper()


def test_the_lease_is_taken_once_the_cold_chain_is_up():
    """begin() leases BEFORE summoning; a cold chain answers 409. After the summon the lease is taken so the pillar knows."""
    ctl = _Ctl(ready=False, start_makes_ready_after=0.05)
    lease = _Lease(first_lease_ok=False)
    lc = _lc(ctl, lease_client=lease)
    lc.begin()
    assert lease.lease_id == "L1", "no lease after summon"


def test_a_request_arriving_while_the_reaper_asks_the_pillar_is_not_reaped_under():
    ctl = _Ctl(ready=True)
    lease = _Lease(expired=True, expire_delay=0.4)  # the pillar takes its time to answer
    lease.lease_id = "L1"
    lc = _lc(ctl, lease_client=lease, idle_grace_s=0.05, ready_cache_s=30)
    lc._up = True
    lc.start_reaper()
    time.sleep(0.15)   # the reaper has decided "due" and is now waiting on is_expired()
    lc.begin()         # a request lands in that window and stays in flight
    time.sleep(0.6)
    lc.stop_reaper()
    assert ctl.stops == 0, "reaped a chain a request was using"
    lc.end()


def test_the_lease_is_renewed_while_a_request_is_in_flight():
    ctl = _Ctl(ready=True)
    lease = _Lease()
    lease.lease_id = "L1"
    lc = _lc(ctl, lease_client=lease, idle_grace_s=0.05, ready_cache_s=30)
    lc._up = True
    lc.begin()           # one renew from begin()
    before = lease.renews
    lc.start_reaper()
    assert _wait(lambda: lease.renews > before + 1)
    lc.stop_reaper()
    lc.end()


# -- a node that refuses to start fails the request AT ONCE ------------------------------------------------------------------

class _Ssh(sl.RemoteSshController):
    def __init__(self, workers, rcs):
        super().__init__(workers, log=lambda *_: None)
        self.rcs = dict(rcs)
        self.calls = []

    def _ssh(self, uh, remote_cmd, timeout=30.0):
        self.calls.append((uh, remote_cmd))
        return self.rcs.get(remote_cmd, 0)


def _rw(name, launch, **kw):
    return sl.RemoteWorker(name=name, ssh=name, launch=launch, probe=("127.0.0.1", 1), stop_match="x", stop=f"stop-{name}", **kw)


def test_a_refused_launch_raises_and_the_rest_of_the_chain_is_not_launched():
    c = _Ssh([_rw("a", "start-a", launch_must_succeed=True), _rw("b", "start-b", launch_must_succeed=True)], {"start-a": 1})
    try:
        c.start()
        raise AssertionError("expected ChainRefused")
    except sl.ChainRefused as e:
        assert "a on a refused" in str(e)
    assert [cmd for _, cmd in c.calls] == ["start-a"], c.calls


def test_a_nonzero_launch_is_still_ignored_for_workers_that_did_not_opt_in():
    """Other configs (roster/unconscious) may return spurious non-zero from a nohup launch: behaviour unchanged for them."""
    c = _Ssh([_rw("a", "start-a"), _rw("b", "start-b")], {"start-a": 1})
    c.start()
    assert [cmd for _, cmd in c.calls] == ["start-a", "start-b"]


def test_begin_fails_at_once_on_a_refusal_and_returns_what_started():
    c = _Ssh([_rw("a", "start-a", launch_must_succeed=True)], {"start-a": 75})
    c.is_ready = lambda: False
    lc = _lc(c, start_timeout_s=60)   # would wait a minute if the refusal were treated as "still coming up"
    t0 = time.monotonic()
    try:
        lc.begin()
        raise AssertionError("expected ChainRefused")
    except sl.ChainRefused:
        pass
    assert time.monotonic() - t0 < 5
    assert ("a", "stop-a") in c.calls, "the refused summon must stop whatever had started"
    assert lc._summoning is False and lc._active == 0


def test_the_json_loader_reads_launch_must_succeed(tmp_path):
    import json
    p = tmp_path / "l.json"
    p.write_text(json.dumps({"remote_workers": [
        {"name": "a", "ssh": "h", "launch": "x", "probe": "1.2.3.4:5", "stop": "y", "launch_must_succeed": True},
        {"name": "b", "ssh": "h", "launch": "x", "probe": "1.2.3.4:6", "stop": "y"}]}))
    a, b = sl._remote_workers_from_json(str(p))
    assert a.launch_must_succeed is True and b.launch_must_succeed is False
