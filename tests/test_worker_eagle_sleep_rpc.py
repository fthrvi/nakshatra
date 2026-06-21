"""Worker servicer routing for the 2026-06-21 additive RPCs:
  - Forward(eagle_hidden=True) -> daemon CMD_EAGLE_HIDDEN (5), raw f32 passthrough
  - Sleep()                    -> daemon CMD_SLEEP (6)
  - Wake()                     -> daemon CMD_WAKE  (7), returns wake_seconds
  - Info() advertises "sleep_wake" + "eagle_hidden"
A fake daemon records (cmd, payload) and returns canned bytes — no GPU/subprocess.
"""
import struct
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import worker
import nakshatra_pb2 as pb


class FakeDaemon:
    def __init__(self, n_embd=4):
        import collections
        self.n_embd = n_embd
        self.recent_rpc_ms = collections.deque(maxlen=20)
        self.calls = []
    def info(self):
        return dict(layer_start=0, layer_end=2, n_embd=self.n_embd,
                    has_token_embd=True, has_lm_head=False, n_vocab=10)
    def call(self, cmd, n_tokens, payload, start_pos=0, flags=0):
        self.calls.append((cmd, n_tokens, payload, start_pos, flags))
        if cmd == worker.CMD_EAGLE_HIDDEN:
            # result_type=3 prefix + n_tokens*3*n_embd float32
            nf = n_tokens * 3 * self.n_embd
            return 0, struct.pack("<I", 3) + struct.pack(f"<{nf}f", *([0.5] * nf))
        return 0, b""   # SLEEP/WAKE: status 0, empty body


class FakeCtx:
    def __init__(self): self.code = None; self.details = None
    def set_code(self, c): self.code = c
    def set_details(self, d): self.details = d
    def invocation_metadata(self): return ()
    def peer(self): return "ipv4:127.0.0.1:0"


def _servicer(daemon, mode="first"):
    return worker.WorkerServicer(daemon=daemon, mode=mode,
                                 layer_start=0, layer_end=2, model_id="test")


def test_forward_eagle_hidden_routes_to_cmd5():
    d = FakeDaemon(n_embd=4)
    s = _servicer(d)
    n = 3
    req = pb.ForwardRequest(hidden_in=struct.pack(f"<{n}i", 1, 2, 3),
                            batch=1, n_tokens=n, has_token_ids=True,
                            eagle_hidden=True)
    resp = s.Forward(req, FakeCtx())
    assert d.calls and d.calls[0][0] == worker.CMD_EAGLE_HIDDEN
    # passthrough strips the 4-byte rtype prefix → raw n*3*n_embd float32
    assert len(resp.hidden_out) == n * 3 * 4 * 4   # n_tokens*3*n_embd*sizeof(f32)


def test_sleep_routes_to_cmd6():
    d = FakeDaemon()
    _servicer(d).Sleep(pb.SleepRequest(), FakeCtx())
    assert d.calls[-1][0] == worker.CMD_SLEEP


def test_wake_routes_to_cmd7_and_reports_seconds():
    d = FakeDaemon()
    resp = _servicer(d).Wake(pb.WakeRequest(), FakeCtx())
    assert d.calls[-1][0] == worker.CMD_WAKE
    assert resp.wake_seconds >= 0.0


def test_info_advertises_new_capabilities():
    caps = _servicer(FakeDaemon()).Info(pb.InfoRequest(), FakeCtx()).protocol_capabilities
    assert "sleep_wake" in caps and "eagle_hidden" in caps


def test_eagle_hidden_requires_token_ids():
    d = FakeDaemon(n_embd=4)
    s = _servicer(d)
    n = 2
    # wrong-size payload (not n*4 token bytes) → client error, no daemon call
    req = pb.ForwardRequest(hidden_in=b"\x00" * 7, batch=1, n_tokens=n,
                            has_token_ids=True, eagle_hidden=True)
    ctx = FakeCtx()
    s.Forward(req, ctx)
    assert ctx.code is not None       # INVALID_ARGUMENT set
    assert not any(c[0] == worker.CMD_EAGLE_HIDDEN for c in d.calls)
