#!/usr/bin/env python3
"""M4 step 4+5 chain test: validate the patched llama.cpp can do a two-worker
chain in a single process.

  Worker A loads w0.gguf (layers [0,14), has token_embd, no lm_head).
    Input:  prompt tokens via batch.token.
    Output: residual stream after layer 13, read via llama_get_embeddings.

  Worker B loads wlast.gguf (layers [14,28), has lm_head, has token_embd
            for tied-embed fallback).
    Input:  worker A's hidden state via batch.embd (no batch.token).
    Output: logits via llama_get_logits_ith.

We then argmax the logits and compare to a single-machine reference top-1
token. Both produce 'Paris' if the chain is correct.

This is the v0.1 acceptance-test shape, one process, one machine. Once it
passes, M5's gRPC version is just "two processes on the same host with
TCP between them."
"""
import ctypes
import struct
import sys
import time

import numpy as np
import llama_cpp as ll

W0_PATH    = "/tmp/cuts/w0_v2.gguf"
WLAST_PATH = "/tmp/cuts/wlast_v2.gguf"
PROMPT     = b"The capital of France is"


def load_model_for_partial(path):
    """Load a sub-GGUF model and a context appropriate for partial-decode use."""
    mp = ll.llama_model_default_params()
    mp.n_gpu_layers = 0
    model = ll.llama_model_load_from_file(path.encode(), mp)
    if not model:
        sys.exit(f"failed to load {path}")
    cp = ll.llama_context_default_params()
    cp.n_ctx   = 256
    cp.n_batch = 64
    cp.embeddings = True   # enable embedding output (so llama_get_embeddings is populated)
    init = getattr(ll, "llama_init_from_model", None) or ll.llama_new_context_with_model
    ctx = init(model, cp)
    if not ctx:
        sys.exit(f"failed to init context for {path}")
    return model, ctx


def tokenize(model, prompt):
    vocab = ll.llama_model_get_vocab(model)
    buf = (ll.llama_token * 64)()
    n = ll.llama_tokenize(vocab, prompt, len(prompt), buf, 64, True, False)
    if n < 0:
        sys.exit(f"tokenize failed: {n}")
    return [buf[i] for i in range(n)]


def decode_with_tokens(ctx, token_ids):
    n = len(token_ids)
    batch = ll.llama_batch_init(n, 0, 1)   # embd=0 means token-input mode
    batch.n_tokens = n
    for i, t in enumerate(token_ids):
        batch.token[i] = t
        batch.pos[i] = i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = 0
        batch.logits[i] = 1 if i == n - 1 else 0
    rc = ll.llama_decode(ctx, batch)
    ll.llama_batch_free(batch)
    return rc


def decode_with_embd(ctx, hidden_in_np, n_tokens, n_embd):
    """hidden_in_np: float32 array of shape [n_tokens * n_embd]."""
    batch = ll.llama_batch_init(n_tokens, n_embd, 1)
    batch.n_tokens = n_tokens
    # Fill embd: ctypes float pointer, n_tokens * n_embd floats
    ptr = ctypes.cast(batch.embd, ctypes.POINTER(ctypes.c_float))
    for i in range(n_tokens * n_embd):
        ptr[i] = float(hidden_in_np[i])
    for i in range(n_tokens):
        batch.pos[i] = i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = 0
        batch.logits[i] = 1 if i == n_tokens - 1 else 0
    rc = ll.llama_decode(ctx, batch)
    ll.llama_batch_free(batch)
    return rc


def get_hidden(ctx, n_tokens, n_embd):
    """Read llama_get_embeddings_ith for the last token, return numpy fp32."""
    p = ll.llama_get_embeddings_ith(ctx, -1)
    if not p:
        return None
    arr = np.ctypeslib.as_array(ctypes.cast(p, ctypes.POINTER(ctypes.c_float)),
                                shape=(n_embd,)).copy()
    return arr


def get_all_hidden(ctx, n_tokens, n_embd):
    """Read all-tokens hidden state via llama_get_embeddings (n_tokens * n_embd)."""
    p = ll.llama_get_embeddings(ctx)
    if not p:
        return None
    arr = np.ctypeslib.as_array(ctypes.cast(p, ctypes.POINTER(ctypes.c_float)),
                                shape=(n_tokens * n_embd,)).copy()
    return arr


def argmax_logits(ctx, n_vocab):
    p = ll.llama_get_logits_ith(ctx, -1)
    if not p:
        return -1
    arr = np.ctypeslib.as_array(ctypes.cast(p, ctypes.POINTER(ctypes.c_float)),
                                shape=(n_vocab,))
    return int(np.argmax(arr))


def main():
    ll.llama_backend_init()

    print(f"[chain] loading worker A: {W0_PATH}")
    t0 = time.time()
    model_a, ctx_a = load_model_for_partial(W0_PATH)
    print(f"[chain] worker A loaded in {time.time()-t0:.1f}s")

    n_embd = int(ll.llama_model_n_embd(model_a))
    print(f"[chain] hidden_size = {n_embd}")

    tokens = tokenize(model_a, PROMPT)
    print(f"[chain] {len(tokens)} prompt tokens: {tokens}")

    print(f"[chain] worker A: decode with tokens")
    rc = decode_with_tokens(ctx_a, tokens)
    print(f"[chain] worker A decode rc={rc}")
    if rc != 0:
        sys.exit(2)

    hidden_all = get_all_hidden(ctx_a, len(tokens), n_embd)
    print(f"[chain] worker A hidden_all: {None if hidden_all is None else hidden_all.shape} sample={hidden_all[:4] if hidden_all is not None else None}")

    print(f"[chain] loading worker B: {WLAST_PATH}")
    t0 = time.time()
    model_b, ctx_b = load_model_for_partial(WLAST_PATH)
    print(f"[chain] worker B loaded in {time.time()-t0:.1f}s")

    n_vocab = int(ll.llama_model_n_vocab(model_b)) if hasattr(ll, "llama_model_n_vocab") else int(ll.llama_n_vocab(model_b))

    print(f"[chain] worker B: decode with hidden_in via batch.embd")
    rc = decode_with_embd(ctx_b, hidden_all, len(tokens), n_embd)
    print(f"[chain] worker B decode rc={rc}")
    if rc != 0:
        sys.exit(3)

    top = argmax_logits(ctx_b, n_vocab)
    print(f"[chain] argmax logit token id = {top}")
    print(f"TOPTOK_CHAIN {top}")


if __name__ == "__main__":
    main()
