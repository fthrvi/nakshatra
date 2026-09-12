#!/usr/bin/env python3
"""remote_client.py — draft-side CLI for NKS_REMOTE_PROPOSALS (WAN speculative decoding
without splitting the model; see remote_proposals.py's module docstring for the protocol
and its correctness oracle).

This is the missing client half. The server half already exists and is real:
`remote_verifier_backend.LlamaVerifier` + `remote_proposals.serve_verifier`, mounted by
`nakshatra_serve.py` behind `NKS_REMOTE_PROPOSALS=1`. This script is what a coordinator runs
to actually USE a remote verifier: load a small local draft model (reusing
`speculative.DraftModel`'s incremental-KV `.propose()` unchanged — no reason to
reimplement the longest-common-prefix rollback here), tokenize a prompt, and drive
`remote_proposals.proposal_loop()` against the verifier over `http_submit`.

Standalone tool, not gated by an env var — `NKS_REMOTE_PROPOSALS` (server-side) has nothing
to do with whether this script runs; it just needs a verifier already listening.

Tokenizer: same approach as client.py's `tokenize_local()` for the non-daemon path — a
`Llama(vocab_only=True)` load. Kept independent of the draft model's own (full-weights)
`Llama` handle rather than reaching into `DraftModel`'s private `_llama` — vocab_only loads
are cheap (client.py: ~0.3-0.5s) and this keeps the two modules decoupled.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).parent))

from remote_proposals import http_submit, proposal_loop  # noqa: E402
from speculative import DraftModel  # noqa: E402


def load_tokenizer(model_path: str):
    """A vocab_only Llama handle for tokenize()/detokenize() — same pattern as
    client.py's tokenize_local() non-daemon branch. `model_path` is the DRAFT model's
    GGUF: the draft and the remote verifier must share a tokenizer/vocab (module
    doctrine, same requirement speculative.py's DraftModel carries), so tokenizing
    against the draft's own file is tokenizing against the verifier's vocab too.
    """
    from llama_cpp import Llama  # lazy: only when actually tokenizing, not at import time
    return Llama(model_path=model_path, vocab_only=True, verbose=False)


def make_draft_propose(draft: DraftModel):
    """Adapter from DraftModel.propose(prefix_tokens, k) -> List[int] to
    proposal_loop's draft_propose(context, k) -> Sequence[int]. The signatures already
    line up exactly (context IS prefix_tokens), so this is a passthrough, not a rewrite
    of the LCP-rollback logic — kept as a named function anyway so a future draft
    backend (e.g. eagle_speculative.EagleDraft) can be swapped in behind the same seam.
    """
    def draft_propose(context: Sequence[int], k: int) -> List[int]:
        return draft.propose(context, k)
    return draft_propose


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Draft-side client for NKS_REMOTE_PROPOSALS: local small draft model "
                    "+ a remote whole-model verifier over HTTP, byte-identical to the "
                    "verifier's own greedy output (see remote_proposals.py).")
    ap.add_argument("--draft-model-path", type=str, required=True,
                    help="full GGUF of a small, same-tokenizer-as-verifier draft model "
                         "(loaded via speculative.DraftModel)")
    ap.add_argument("--verifier-url", type=str, required=True,
                    help="host:port of a remote node running "
                         "remote_verifier_backend.start_proposals_server "
                         "(a leading http:// or https:// scheme, if present, is stripped)")
    prompt_src = ap.add_mutually_exclusive_group(required=True)
    prompt_src.add_argument("--prompt", type=str, help="prompt text")
    prompt_src.add_argument("--prompt-file", type=str, help="path to a file containing the prompt")
    ap.add_argument("--max-tokens", "-n", type=int, default=128,
                    help="max tokens to generate (default: 128)")
    ap.add_argument("--k", type=int, default=4,
                    help="proposal chunk size (default: 4, matches client.py's --draft-max)")
    ap.add_argument("--eos-id", type=int, default=None,
                    help="stop token id; defaults to the draft/verifier tokenizer's own EOS")
    ap.add_argument("--timeout", type=float, default=5.0,
                    help="per-round HTTP timeout in seconds (default: 5.0)")
    return ap


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text()
    return args.prompt


def normalize_verifier_url(url: str) -> str:
    """http_submit's `peer` is "host:port", no scheme (slice_fetch's convention) — strip
    one if the caller pasted a full URL."""
    for scheme in ("http://", "https://"):
        if url.startswith(scheme):
            return url[len(scheme):].rstrip("/")
    return url


def run(args: argparse.Namespace) -> str:
    prompt = resolve_prompt(args)
    verifier = normalize_verifier_url(args.verifier_url)

    tokenizer = load_tokenizer(args.draft_model_path)
    prompt_tokens = list(tokenizer.tokenize(prompt.encode("utf-8"), add_bos=True, special=True))

    eos_id = args.eos_id if args.eos_id is not None else int(tokenizer.token_eos())

    draft = DraftModel(args.draft_model_path)
    print(f"[remote-proposals] ON: draft={args.draft_model_path} verifier={verifier} K={args.k}",
          flush=True)
    try:
        generated = proposal_loop(
            draft_propose=make_draft_propose(draft),
            submit=lambda proposal: http_submit(verifier, proposal, timeout=args.timeout),
            prompt_tokens=prompt_tokens,
            k=args.k,
            max_tokens=args.max_tokens,
            eos_ids={eos_id},
        )
    finally:
        draft.close()

    text = tokenizer.detokenize(generated).decode("utf-8", errors="replace")
    return text


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    text = run(args)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
