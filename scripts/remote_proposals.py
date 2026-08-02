"""remote_proposals — bounded external token proposals: WAN speculative decoding
WITHOUT splitting the model.

Mesh-LLM adoption #1 (see INBOX.md 2026-07-29, infra lane ride). This is the WAN-regime
complement to `async_pipeline.py`'s LAN-negative finding (2026-07-29 GPU-mesh proof:
sequential 4.58 tok/s vs async 1.04 tok/s — the RTT there was too small to hide, and the
client-side draft continuation burned CPU for nothing). Over a WAN hop the RTT is real and
worth hiding, but this repo's doctrine is ROUTE WHOLE MODELS, DON'T SPLIT — chain-splitting
a model across a WAN link (à la Shard) multiplies that RTT by the pipeline depth. Bounded
external token proposals sidestep the tradeoff entirely: a remote node runs the WHOLE
verifier model (never split), a local node runs a small same-tokenizer draft, and the wire
carries only proposal chunks and accept/correct responses — a few ints per round-trip,
not activations or layers.

THE CORRECTNESS ORACLE (same shape as speculative.py / async_pipeline.py): the final output
must be BYTE-IDENTICAL to what the verifier model alone would generate greedily, for ANY
draft quality (perfect, adversarial, or anything between). The invariant that guarantees
it:

    Every committed token is either (a) a draft token the verifier's own greedy argmax
    independently agreed with, or (b) the verifier's own greedy argmax used as a
    correction. The draft never contributes a token the verifier didn't itself produce.

Because `VerifierSession.submit()` always re-derives its `cur` and `cursor` from what it
JUST committed (never from what the draft assumed), a bad draft only costs speed — it can
never corrupt the stream. This mirrors speculative.py's `accept()` exactly; reimplemented
here (not imported) to keep this module self-contained and dependency-free for the WAN lane
— it must be mountable on a remote node without pulling in the LAN chain-splitting stack.

THE PIECES
----------
  VerifierSession   — server-side: wraps an abstract per-position greedy oracle
                       `verify_fn(tokens, start_pos) -> argmaxes` (in production this is a
                       real whole-model forward pass with a KV cache; here it's anything
                       satisfying the contract, so no GPU/model is needed to test the
                       protocol). Maintains the committed prefix + cursor itself — the
                       BOUNDED WINDOW and the cursor-rewind-on-mispredict live here.
  proposal_loop     — client-side: draft proposes K tokens, submits them, commits what
                       came back, repeats until max_tokens or an EOS id is emitted.
  serve_verifier /
  http_submit       — a thin, OPTIONAL stdlib (http.server + urllib) loopback JSON
                       transport, so a later lane can mount a VerifierSession on
                       nakshatra_serve without inventing a new wire format first. Kept
                       minimal on purpose — this is a seam, not the real transport.

STATUS: protocol core is complete and unit-tested here against fake deterministic
verifiers/drafts (no GPU, no model weights, no live services). Mounting a real whole-model
verifier behind `serve_verifier` and running it over an actual WAN hop is the remaining ask
— see docs/findings/remote-proposals.md.
"""
from __future__ import annotations

import http.client
import json
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, List, Optional, Sequence, Tuple

# Wiring flag for the future client.py / nakshatra_serve integration. This module does not
# read it itself — there is nothing to gate yet, the module is inert (no side effects) until
# something imports and calls it. Declared here, default OFF, purely so the convention is
# discoverable (`grep NKS_REMOTE_PROPOSALS`) before the real wiring lane sets it.
NKS_REMOTE_PROPOSALS = os.environ.get("NKS_REMOTE_PROPOSALS", "0") == "1"

# VerifyFn: (tokens, start_pos) -> per-position greedy argmaxes, len(argmaxes) == len(tokens).
# tokens[0] is always the caller's last TRULY-committed token (`cur`); tokens[1:] is the
# proposal. argmaxes[i] is the verifier's greedy next token after seeing tokens[0..i]. A real
# implementation truncates its KV cache to `start_pos` before evaluating (the rewind primitive
# — same role as speculative.py's KV_TRUNCATE / async_pipeline.py's start_pos trim) so a prior
# round's rejected tail never leaks into this round's context.
VerifyFn = Callable[[List[int], int], List[int]]

# SubmitFn: what proposal_loop calls each round — either VerifierSession.submit directly
# (in-process / test) or http_submit bound to a URL (over the wire).
SubmitFn = Callable[[Sequence[int]], Tuple[int, Optional[int], int]]


class VerifierSession:
    """Server-side session over an abstract whole-model greedy verifier.

    Maintains the committed prefix and the KV cursor. `submit()` is the ONLY mutating
    method — everything else is derived. No model or GPU involved: `verify_fn` is injected
    (production: a real forward pass; tests: a fake deterministic oracle).
    """

    def __init__(self, verify_fn: VerifyFn, prompt_tokens: Sequence[int], max_window: int = 16):
        if not prompt_tokens:
            raise ValueError("prompt_tokens must be non-empty (need a seed `cur` token)")
        if max_window < 1:
            raise ValueError("max_window must be >= 1")
        self.verify_fn = verify_fn
        self.max_window = max_window
        self.prompt: List[int] = [int(t) for t in prompt_tokens]
        self.generated: List[int] = []
        self._cur = self.prompt[-1]
        self.cursor = len(self.prompt) - 1     # position of `_cur` in the full sequence

    @property
    def committed(self) -> List[int]:
        """The full sequence so far: prompt + everything accepted/corrected to date."""
        return self.prompt + self.generated

    def submit(self, proposal: Sequence[int]) -> Tuple[int, Optional[int], int]:
        """Verify one proposal chunk. Returns (n_accepted, correction_token, cursor).

        n_accepted        : how many leading proposal tokens the verifier's own greedy
                             argmax agreed with (0..len(proposal)).
        correction_token   : the verifier's own token that got committed alongside the
                              accepted prefix — either the correction at the first
                              mismatch, or (if every proposed token was accepted) the
                              bonus token past the end of the proposal. None ONLY means
                              the proposal was REJECTED by the bounded window below (no
                              verification ran, no state changed) — it never means "no
                              correction needed"; some token is always committed on a
                              successful submit, exactly like plain greedy decode always
                              advances by at least one token.
        cursor              : the new KV position. Advances by exactly what was KEPT
                              (n_accepted + 1), never by len(proposal) — this IS the
                              rewind: a real verifier truncates its KV to this cursor on
                              the *next* call's start_pos, discarding whatever it may have
                              speculatively computed past the mismatch.

        BOUNDED WINDOW: an oversized proposal is REJECTED, not raised — over a real WAN
        link this is a routine, expected protocol event (the caller guessed too big a K),
        not a bug, so it is signalled through the normal return shape (0, None, unchanged
        cursor) rather than an exception. The caller (proposal_loop) shrinks K and retries.
        An EMPTY proposal is a caller bug (there is nothing to verify), and that DOES raise.
        """
        proposal = [int(t) for t in proposal]
        if len(proposal) == 0:
            raise ValueError("proposal must contain at least one token")
        if len(proposal) > self.max_window:
            return 0, None, self.cursor
        batch = [self._cur] + proposal
        target = [int(t) for t in self.verify_fn(batch, self.cursor)]
        if len(target) != len(batch):
            raise ValueError(
                f"verify_fn contract violation: expected {len(batch)} argmaxes for "
                f"{len(batch)} input tokens, got {len(target)}"
            )
        n_accepted = 0
        for i, d in enumerate(proposal):
            if target[i] != d:
                break
            n_accepted += 1
        correction = target[n_accepted]        # always valid: len(target) == len(proposal)+1
        newly = proposal[:n_accepted] + [correction]
        self.generated.extend(newly)
        self._cur = correction
        self.cursor += len(newly)
        return n_accepted, correction, self.cursor


def proposal_loop(
    *,
    draft_propose: Callable[[Sequence[int], int], Sequence[int]],
    submit: SubmitFn,
    prompt_tokens: Sequence[int],
    k: int,
    max_tokens: int,
    eos_ids: Sequence[int] = (),
    on_token: Optional[Callable[[int], None]] = None,
    min_k: int = 1,
) -> List[int]:
    """Client-side loop: draft proposes K tokens, submit() verifies, commit, repeat.

    draft_propose(context, k) -> up to k draft token ids following `context` (the client's
                 own running copy of the committed sequence — it must be kept in lockstep
                 with what the verifier actually committed, which is exactly what the loop
                 below does: it only ever appends what `submit` returned, never what the
                 draft assumed).
    submit(proposal) -> (n_accepted, correction_token, cursor) — VerifierSession.submit
                 in-process, or http_submit bound to a remote URL. Opaque to this loop by
                 design: it doesn't matter whether the verifier is local or across a WAN.

    CORRECTNESS INVARIANT: regardless of draft quality, the returned token list is
    BYTE-IDENTICAL to what repeatedly calling the verifier alone (one token at a time,
    proposal=[]) would produce — because every committed token traces back to the
    verifier's own greedy argmax over the true committed prefix (see module docstring).
    A bad draft only changes how many rounds it takes, never what comes out.

    On a BOUNDED-WINDOW rejection (correction_token is None) the loop halves k and retries
    — no tokens are lost, no state advances, it just asks for less next time.
    """
    if k < min_k:
        raise ValueError(f"k ({k}) must be >= min_k ({min_k})")
    context: List[int] = list(prompt_tokens)
    generated: List[int] = []
    eos = set(eos_ids)
    cur_k = k
    while len(generated) < max_tokens:
        remaining = max_tokens - len(generated)
        proposal = list(draft_propose(context, min(cur_k, remaining)))[:cur_k]
        if not proposal:
            proposal = [context[-1]]     # degenerate: still must submit >=1 token
        n_accepted, correction, _cursor = submit(proposal)
        if correction is None:
            # bounded-window rejection: nothing committed, nothing lost — ask smaller.
            if cur_k <= min_k:
                raise RuntimeError(
                    f"verifier rejected a proposal of the minimum size ({min_k}) — "
                    "the session's max_window is smaller than min_k"
                )
            cur_k = max(min_k, cur_k // 2)
            continue
        cur_k = k       # reset to the requested depth once a round succeeds
        committed = proposal[:n_accepted] + [correction]
        for t in committed:
            if len(generated) >= max_tokens:
                break
            generated.append(t)
            context.append(t)
            if on_token is not None:
                on_token(t)
            if t in eos:
                return generated
    return generated


# ─────────────────────────────────────────────────────────────────────────────────────────
# Transport seam: stdlib-only loopback JSON, optional. Mounts a VerifierSession behind
# POST /submit so proposal_loop can run over http_submit exactly like it runs in-process —
# proves the protocol survives serialization before any real network/model is involved. A
# later lane can replace this with the real nakshatra_serve surface without touching the
# session or loop logic above.
# ─────────────────────────────────────────────────────────────────────────────────────────

def serve_verifier(session: "VerifierSession", host: str = "127.0.0.1", port: int = 0):
    """Return a ThreadingHTTPServer exposing `session` at POST /submit. Caller runs
    `.serve_forever()` (or in a thread) and `.shutdown()` — same calling convention as
    slice_server.serve()."""
    lock = threading.Lock()     # one session, serialize concurrent HTTP submits onto it

    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.1 keeps the socket open between requests. The default 1.0 closes
        # after every response, which forces the client to re-handshake and makes
        # connection pooling on the far side pointless. Every response path here
        # sends an explicit Content-Length, which is what 1.1 requires.
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass  # quiet by default, matches slice_server.py

        def do_POST(self):
            if self.path != "/submit":
                return self.send_error(404, "not found")
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                proposal = [int(t) for t in body.get("proposal", [])]
            except (ValueError, TypeError, json.JSONDecodeError):
                return self.send_error(400, "bad request")
            try:
                with lock:
                    n_accepted, correction, cursor = session.submit(proposal)
            except ValueError as e:
                # The HTTP reason-phrase is latin-1 by spec, and our own error
                # strings contain em-dashes — so send_error(400, str(e)) raised
                # UnicodeEncodeError *inside the error handler*, killed the
                # connection, and the client saw RemoteDisconnected with no clue
                # what went wrong. An error path that destroys the error is worse
                # than no error path. Reason phrase stays ASCII; detail goes in
                # the body where it belongs. (2026-08-01, first live WAN run.)
                detail = str(e).encode("ascii", "replace").decode("ascii")
                body = json.dumps({"error": detail}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            payload = json.dumps({
                "n_accepted": n_accepted,
                "correction_token": correction,
                "cursor": cursor,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return ThreadingHTTPServer((host, port), Handler)


#: one live connection per peer. See http_submit.
_CONNS: dict = {}
_CONNS_LOCK = threading.Lock()


def close_connections() -> None:
    """Drop every pooled connection (tests, shutdown, peer changed address)."""
    with _CONNS_LOCK:
        for c in _CONNS.values():
            try:
                c.close()
            except Exception:
                pass
        _CONNS.clear()


def http_submit(peer: str, proposal: Sequence[int], timeout: float = 5.0) -> Tuple[int, Optional[int], int]:
    """Client-side counterpart to serve_verifier. `peer` is "host:port" (no scheme),
    matching slice_fetch's peer-address convention.

    KEEPS THE CONNECTION OPEN. The original used urllib.urlopen, which opens a
    fresh TCP connection per call — so every proposal round paid a handshake AND
    the request: TWO round trips, not one, on a protocol whose entire purpose is
    to economise round trips.

    Measured 2026-08-01 on a real WAN link, fitting ms-per-token against RTT:
    the proposals arm paid **0.967 ms of RTT per token** where one round trip per
    2.67-token round predicts 0.375, and two predicts 0.749. The fit also
    produced a NEGATIVE intercept — physically impossible, and the clue that the
    model was misspecified rather than the technique being weak. The measured
    split-vs-proposals crossover of ~110 ms RTT was therefore substantially an
    artifact of a urllib default.

    Retries once on a dropped connection: a pooled socket can be closed by the
    peer between calls, and that is routine, not an error."""
    body = json.dumps({"proposal": [int(t) for t in proposal]}).encode()
    host, _, port = peer.rpartition(":")
    for attempt in (0, 1):
        with _CONNS_LOCK:
            conn = _CONNS.get(peer)
            if conn is None:
                conn = http.client.HTTPConnection(host or "127.0.0.1", int(port), timeout=timeout)
                _CONNS[peer] = conn
        try:
            conn.request("POST", "/submit", body=body,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = json.loads(resp.read())
            break
        except Exception:
            with _CONNS_LOCK:
                try:
                    conn.close()
                except Exception:
                    pass
                _CONNS.pop(peer, None)
            if attempt == 1:
                raise
    correction = data["correction_token"]
    return int(data["n_accepted"]), (None if correction is None else int(correction)), int(data["cursor"])


# ─────────────────────────────────────────────────────────────────────────────────────────
# Self-test: prove the protocol is correct WITHOUT any GPU/network/model weights.
#
# A Markov oracle (next token depends only on the last token — same trick test_speculative.py
# uses) stands in for the whole-model verifier, so a fake verify_fn needs no history beyond
# what VerifierSession already threads through `batch`. Drafts of varying quality are checked
# against a plain sequential reference; a loopback HTTP round-trip proves the wire seam.
# ─────────────────────────────────────────────────────────────────────────────────────────
def _selftest() -> int:
    import time

    SUCC = {t: t + 1 for t in range(100, 120)}
    EOS_ID = 999
    SUCC[120] = EOS_ID
    SUCC[EOS_ID] = EOS_ID
    oracle = lambda tok: SUCC.get(tok, EOS_ID)   # noqa: E731

    def verify_fn(tokens, start_pos):
        return [oracle(t) for t in tokens]

    def sequential_reference(first_tok, n):
        out, cur = [], first_tok
        for _ in range(n):
            cur = oracle(cur)
            out.append(cur)
            if cur == EOS_ID:
                break
        return out

    def make_draft(mode):
        # Only the FIRST wrong token in a proposal matters for correctness (verify() stops
        # accepting at the first mismatch) — so `last` can just track the true oracle chain
        # throughout; whether a given position is deliberately wrong only affects speed.
        def propose(context, k):
            out, last = [], context[-1]
            for i in range(k):
                wrong = mode == "adversarial" or (mode == "mixed" and i % 2 == 1)
                out.append(-1 if wrong else oracle(last))
                last = oracle(last)
            return out
        return propose

    PROMPT = [100]
    N = 15
    ok = True

    for mode in ("perfect", "adversarial", "mixed"):
        session = VerifierSession(verify_fn, PROMPT, max_window=8)
        out = proposal_loop(
            draft_propose=make_draft(mode), submit=session.submit,
            prompt_tokens=PROMPT, k=4, max_tokens=N, eos_ids={EOS_ID},
        )
        ref = sequential_reference(PROMPT[-1], N)
        if out != ref:
            print(f"FAIL[{mode}]: proposal_loop != sequential reference\n  got={out}\n  want={ref}")
            ok = False
        else:
            print(f"PASS[{mode}]: {len(out)} tokens, byte-identical to sequential verifier-only")

    # loopback HTTP round-trip
    session = VerifierSession(verify_fn, PROMPT, max_window=8)
    httpd = serve_verifier(session, host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        time.sleep(0.05)
        out = proposal_loop(
            draft_propose=make_draft("mixed"),
            submit=lambda p: http_submit(f"127.0.0.1:{port}", p),
            prompt_tokens=PROMPT, k=4, max_tokens=N, eos_ids={EOS_ID},
        )
        ref = sequential_reference(PROMPT[-1], N)
        if out != ref:
            print(f"FAIL[http]: {out} != {ref}")
            ok = False
        else:
            print(f"PASS[http]: loopback HTTP round-trip byte-identical ({len(out)} tokens)")
    finally:
        httpd.shutdown()
        thread.join(timeout=2)

    # bounded-window rejection is a real, handled event, not a crash
    session = VerifierSession(verify_fn, PROMPT, max_window=2)
    n_acc, corr, cur = session.submit([1, 2, 3])
    if corr is not None or n_acc != 0 or cur != session.cursor:
        print(f"FAIL[window]: oversized proposal was not cleanly rejected: {(n_acc, corr, cur)}")
        ok = False
    else:
        print("PASS[window]: oversized proposal rejected without state change")

    if ok:
        print(f"PASS: NKS_REMOTE_PROPOSALS={NKS_REMOTE_PROPOSALS} (module inert either way)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
