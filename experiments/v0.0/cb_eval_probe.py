#!/usr/bin/env python3
"""Phase 0b prerequisite — cb_eval ergonomics probe for llama-cpp-python.

Goal: confirm that a Python callback registered as ggml_backend_sched_eval_callback
fires during llama_decode, can read tensor metadata (name/shape), and can read the
underlying tensor bytes. Write-back is exercised in a follow-up.

If this probe succeeds, Phase 0b proceeds in Python. If it fails (callback doesn't
fire, or tensor->data is unreadable), fall back to a C++ implementation.
"""
import ctypes
import os
import sys
from pathlib import Path

import llama_cpp as ll

MODEL = "/Users/MentoringInstitute/models/qwen3-draft/Qwen3-1.7B-Q4_K_M.gguf"

# Minimal ggml_tensor mirror — only the fields we need to read (name, type, ne, data).
# Layout matches ggml.h as of llama.cpp recent (build 8142). Brittle to upstream
# struct churn; revisit if upstream reorders fields.
GGML_MAX_DIMS = 4
GGML_MAX_OP_PARAMS = 64
GGML_MAX_SRC = 10
GGML_MAX_NAME = 64


class GgmlTensor(ctypes.Structure):
    pass


GgmlTensor._fields_ = [
    ("type",       ctypes.c_int),
    ("buffer",     ctypes.c_void_p),
    ("ne",         ctypes.c_int64 * GGML_MAX_DIMS),
    ("nb",         ctypes.c_size_t * GGML_MAX_DIMS),
    ("op",         ctypes.c_int),
    ("op_params",  ctypes.c_int32 * (GGML_MAX_OP_PARAMS // 4)),
    ("flags",      ctypes.c_int32),
    ("src",        ctypes.c_void_p * GGML_MAX_SRC),
    ("view_src",   ctypes.c_void_p),
    ("view_offs",  ctypes.c_size_t),
    ("data",       ctypes.c_void_p),
    ("name",       ctypes.c_char * GGML_MAX_NAME),
    ("extra",      ctypes.c_void_p),
    ("padding",    ctypes.c_char * 8),
]


def main():
    if not os.path.exists(MODEL):
        sys.exit(f"model not found: {MODEL}")

    print(f"[init] llama_backend_init")
    ll.llama_backend_init()

    print(f"[load] {MODEL}")
    mparams = ll.llama_model_default_params()
    mparams.n_gpu_layers = 0
    model = ll.llama_model_load_from_file(MODEL.encode(), mparams)
    if not model:
        sys.exit("model load failed")

    vocab = ll.llama_model_get_vocab(model) if hasattr(ll, "llama_model_get_vocab") else model

    stats = {
        "asks": 0,
        "evals": 0,
        "first_names": [],
        "first_with_data": [],
        "with_zero_data_ptr": 0,
        "exceptions": 0,
    }

    CB_TYPE = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p)

    def cb_intercept(tensor_voidp, ask, user_data):
        try:
            if ask:
                stats["asks"] += 1
                return True
            stats["evals"] += 1
            t = ctypes.cast(tensor_voidp, ctypes.POINTER(GgmlTensor)).contents
            name = t.name.decode("utf-8", errors="replace").rstrip("\x00")
            if stats["evals"] <= 5:
                stats["first_names"].append(name)
            if t.data:
                if stats["evals"] <= 5:
                    stats["first_with_data"].append(name)
            else:
                stats["with_zero_data_ptr"] += 1
            return False
        except Exception as e:
            stats["exceptions"] += 1
            if stats["exceptions"] <= 3:
                print(f"[cb exc #{stats['exceptions']}] {type(e).__name__}: {e}", flush=True)
            return False

    cb_ptr = CB_TYPE(cb_intercept)

    print("[ctx] init with cb_eval set")
    cparams = ll.llama_context_default_params()
    cparams.n_ctx = 128
    cparams.n_batch = 32
    cparams.cb_eval = cb_ptr
    cparams.cb_eval_user_data = None

    init_ctx = getattr(ll, "llama_init_from_model", None) or ll.llama_new_context_with_model
    ctx = init_ctx(model, cparams)
    if not ctx:
        sys.exit("context init failed")

    prompt = b"Hello"
    tok_buf = (ll.llama_token * 16)()
    n_tok = ll.llama_tokenize(vocab, prompt, len(prompt), tok_buf, 16, True, False)
    if n_tok < 0:
        sys.exit(f"tokenize failed: {n_tok}")
    print(f"[tok] tokens={[tok_buf[i] for i in range(n_tok)]} ({n_tok} tokens)")

    batch = ll.llama_batch_init(n_tok, 0, 1)
    batch.n_tokens = n_tok
    for i in range(n_tok):
        batch.token[i] = tok_buf[i]
        batch.pos[i] = i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = 0
        batch.logits[i] = 1 if i == n_tok - 1 else 0

    print("[decode] running")
    rc = ll.llama_decode(ctx, batch)
    print(f"[decode] rc={rc}")

    print()
    print("=== cb_eval probe results ===")
    print(f"  ask-mode invocations:     {stats['asks']}")
    print(f"  eval-mode invocations:    {stats['evals']}")
    print(f"  first eval names:         {stats['first_names']}")
    print(f"  first names with data:    {stats['first_with_data']}")
    print(f"  evals with NULL data ptr: {stats['with_zero_data_ptr']}")
    print(f"  callback exceptions:      {stats['exceptions']}")

    ll.llama_batch_free(batch)
    ll.llama_free(ctx)
    if hasattr(ll, "llama_model_free"):
        ll.llama_model_free(model)
    else:
        ll.llama_free_model(model)
    ll.llama_backend_free()

    if stats["evals"] == 0:
        print("\n[VERDICT] callback NEVER fired with ask=False --> Python path NOT viable. Fall back to C++.")
        sys.exit(2)
    if stats["exceptions"] > 0:
        print(f"\n[VERDICT] callback fired but raised {stats['exceptions']} exceptions --> ggml_tensor struct may be misaligned. Inspect.")
        sys.exit(3)
    if not stats["first_with_data"]:
        print("\n[VERDICT] callback fired but tensor->data was always NULL --> Python path NOT viable for write-back.")
        sys.exit(4)
    print("\n[VERDICT] callback fires AND tensor->data is readable. Python path is viable.")


if __name__ == "__main__":
    main()
