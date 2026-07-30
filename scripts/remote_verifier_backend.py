"""remote_verifier_backend.py — the REAL whole-model verify_fn for
remote_proposals.VerifierSession, backed by llama.cpp, CPU-ONLY by
construction (n_gpu_layers=0 is hardcoded, never overridable from env) so it
can never compete with Prithvi's conscious/voice models for VRAM — see
docs/findings/remote-proposals.md, "Remaining ask" #1/#2.

This is the ONE seam `nakshatra_serve.py` mounts behind `NKS_REMOTE_PROPOSALS=1`
(`_maybe_start_proposals_server()`): build a `LlamaVerifier`, wrap it in a
`remote_proposals.VerifierSession`, and serve it with
`remote_proposals.serve_verifier` in a daemon thread.

`llama_cpp` is imported LAZILY — only inside `LlamaVerifier.__init__`, and
only when no fake `llama=` handle is injected — so importing this module (or
`nakshatra_serve` with the flag unset) never requires it installed.

THE ADAPTER CONTRACT (see remote_proposals.VerifyFn / VerifierSession.submit's
docstring): given `tokens` (tokens[0] = the already-committed `cur` sitting at
position `start_pos`, tokens[1:] = the proposal) and `start_pos` (cur's
position in the full sequence), return one greedy-argmax per input token —
argmaxes[i] is the model's greedy next-token prediction having seen
tokens[0..i]. A real verifier's KV cache must reflect EXACTLY the prefix up to
`start_pos` before evaluating — this is the rewind primitive
`VerifierSession.submit()`'s cursor arithmetic assumes: on a mispredict, the
caller's NEXT `start_pos` is smaller than what this adapter last cached, and
the adapter must roll its KV back to match before decoding the new batch,
discarding whatever it speculatively computed past the mismatch.

Modeled directly on `speculative.py`'s `DraftModel` (`llama._ctx.kv_cache_seq_rm`
+ `logits_all=True` `.scores`) — same primitive, this module computes ALL K+1
argmaxes per call (one per input token) instead of one token at a time.
"""
from __future__ import annotations

import threading
from typing import Callable, List, Optional, Sequence


class VerifierVocabError(ValueError):
    """Raised when a construction-time vocab-size hint is given and doesn't
    match the loaded GGUF — a cheap guardrail against silently-wrong argmaxes
    from a mismatched draft/verifier tokenizer pair (the module docstring's
    "vocab must match the draft family" requirement)."""


class LlamaVerifier:
    """Wraps one llama.cpp whole-model handle as a remote_proposals.VerifyFn.

    CPU-ONLY (n_gpu_layers=0, hardcoded — see module docstring). Stateful: owns
    ONE running KV cache, rewound to `start_pos` on demand — the caller
    (VerifierSession) is the source of truth for where the cursor is; this
    adapter never advances the cursor on its own, it only ever reacts to the
    `start_pos` it's handed.
    """

    def __init__(self, model_path: Optional[str] = None, *, n_ctx: int = 4096,
                seed: int = 0, verbose: bool = False,
                expect_vocab_size: Optional[int] = None,
                llama: Optional[object] = None):
        """`llama` lets a caller (tests) inject an already-constructed
        llama.cpp-shaped handle — must expose `.n_tokens` (settable int),
        `.eval(tokens)`, `.scores` (indexable by absolute position),
        `._ctx.kv_cache_seq_rm(seq_id, p0, p1)`, `.n_vocab()`, `.token_bos()`,
        `.close()` — the exact surface this class touches, and nothing more.
        `tests/test_remote_verifier_backend.py`'s `FakeLlama` is the
        deterministic stand-in this proves the contract against, with no
        llama_cpp import and no real GGUF anywhere in the test. Production
        callers pass `model_path` and leave `llama` None."""
        if llama is not None:
            self._llama = llama
            self.model_path = model_path or "<injected llama handle>"
        else:
            if not model_path:
                raise ValueError(
                    "LlamaVerifier needs model_path (or an injected `llama=`)")
            from llama_cpp import Llama   # lazy: only a real construction needs it
            # logits_all=True is REQUIRED: llama-cpp-python only fills .scores
            # per-position when this is set (same requirement as
            # speculative.DraftModel) — without it every argmax below reads
            # garbage instead of the batch's real per-position logits.
            self._llama = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=0,
                                logits_all=True, seed=seed, verbose=verbose)
            self.model_path = model_path
        if expect_vocab_size is not None and self._llama.n_vocab() != expect_vocab_size:
            got = self._llama.n_vocab()
            self.close()
            raise VerifierVocabError(
                f"{self.model_path}: vocab size {got} != expected "
                f"{expect_vocab_size} (draft/verifier tokenizer mismatch)")
        self._lock = threading.Lock()   # one Llama handle, serialize concurrent verify()s

    def bos_token(self) -> int:
        return int(self._llama.token_bos())

    def __call__(self, tokens: Sequence[int], start_pos: int) -> List[int]:
        """Satisfies remote_proposals.VerifyFn's call shape directly, so a
        LlamaVerifier instance can be passed as `verify_fn=` to
        VerifierSession without a wrapper lambda."""
        return self.verify(tokens, start_pos)

    def verify(self, tokens: Sequence[int], start_pos: int) -> List[int]:
        tokens = [int(t) for t in tokens]
        if not tokens:
            raise ValueError("verify() needs at least one token")
        with self._lock:
            llama = self._llama
            n_cached = int(llama.n_tokens)
            if start_pos < n_cached:
                # Rewind: discard whatever this adapter over-computed past the
                # caller's true cursor (a prior round's rejected tail) — same
                # primitive as DraftModel.propose's LCP rollback.
                llama._ctx.kv_cache_seq_rm(-1, start_pos, -1)
                llama.n_tokens = start_pos
            elif start_pos > n_cached:
                raise ValueError(
                    f"verify_fn start_pos={start_pos} is ahead of this adapter's "
                    f"cached KV ({n_cached}) — the caller must present a "
                    f"contiguous cursor (no gaps); see VerifierSession.submit's "
                    f"cursor arithmetic")
            base = start_pos          # position of tokens[0]'s row once evaluated
            llama.eval(tokens)
            import numpy as np        # lazy, mirrors speculative.DraftModel
            return [int(np.argmax(llama.scores[base + i])) for i in range(len(tokens))]

    def close(self) -> None:
        try:
            self._llama.close()
        except Exception:
            pass


def make_verifier_session(model_path: Optional[str] = None, *, n_ctx: int = 4096,
                          seed: int = 0, verbose: bool = False,
                          expect_vocab_size: Optional[int] = None,
                          prompt_tokens: Optional[Sequence[int]] = None,
                          max_window: int = 16, llama: Optional[object] = None,
                          verifier: Optional[LlamaVerifier] = None):
    """Build one (LlamaVerifier, VerifierSession) pair.

    `prompt_tokens` seeds the session's committed prefix; defaults to
    `[verifier.bos_token()]` when omitted — VerifierSession requires a
    non-empty seed (see its docstring). This mount runs ONE continuous
    generation stream per process (matching the thin serve_verifier/
    http_submit loopback transport this rides on — remote_proposals.py's own
    module docstring calls it "a seam, not the real transport"). A
    per-request session pool is the remaining ask, not built here — see
    docs/findings/remote-proposals.md.

    `verifier` lets a caller supply an already-built LlamaVerifier (real or
    fake-backed) instead of constructing one from `model_path`."""
    import remote_proposals
    if verifier is None:
        verifier = LlamaVerifier(model_path, n_ctx=n_ctx, seed=seed, verbose=verbose,
                                 expect_vocab_size=expect_vocab_size, llama=llama)
    seed_tokens = list(prompt_tokens) if prompt_tokens else [verifier.bos_token()]
    session = remote_proposals.VerifierSession(verifier, seed_tokens, max_window=max_window)
    return verifier, session


def start_proposals_server(model_path: Optional[str] = None, *, host: str = "127.0.0.1",
                           port: int = 11601, n_ctx: int = 4096, seed: int = 0,
                           verbose: bool = False, expect_vocab_size: Optional[int] = None,
                           prompt_tokens: Optional[Sequence[int]] = None,
                           max_window: int = 16, llama: Optional[object] = None,
                           verifier: Optional[LlamaVerifier] = None,
                           log: Callable[[str], None] = print):
    """Build the verifier + session, mount it behind remote_proposals.serve_verifier,
    and run the HTTP server in a daemon thread.

    Returns `(httpd, thread, verifier, session)` — the caller
    (`nakshatra_serve._maybe_start_proposals_server()`) keeps the return value
    alive for the process lifetime. Nothing here manages its own shutdown —
    matches `serve_verifier`'s own calling convention: the caller owns
    `.shutdown()`."""
    import remote_proposals
    verifier, session = make_verifier_session(
        model_path, n_ctx=n_ctx, seed=seed, verbose=verbose,
        expect_vocab_size=expect_vocab_size, prompt_tokens=prompt_tokens,
        max_window=max_window, llama=llama, verifier=verifier)
    httpd = remote_proposals.serve_verifier(session, host=host, port=port)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True,
                              name="nks-remote-proposals-verifier")
    thread.start()
    log(f"[remote-proposals] verifier serving on {httpd.server_address[0]}:"
        f"{httpd.server_address[1]} (gguf={model_path}, CPU-only, "
        f"max_window={max_window})")
    return httpd, thread, verifier, session
