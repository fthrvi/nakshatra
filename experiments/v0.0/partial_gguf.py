#!/usr/bin/env python3
"""Phase 0a — Partial-GGUF load test.

Reads a source GGUF, writes a sub-GGUF that:
  * copies all metadata KVs unchanged (block_count is NOT modified)
  * keeps token_embd.* + rope_freqs.* + blk.[0, KEEP) tensors
  * drops blk.[KEEP, total) and the output head (output.weight, output_norm.weight)

The sub-GGUF therefore *claims* the original block_count in metadata but is
missing tensors for blk.KEEP..total-1. We then attempt to load it with
llama-cli to observe whether the loader rejects it.

This operationalises path-a-vs-path-b-memo.md §1.6: the central claim
(no usable partial-model load path in llama.cpp) is what this experiment
falsifies or confirms.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType


def field_value(field):
    """Best-effort decode of a GGUFReader field into a Python value."""
    types = field.types
    primary = types[0]

    if primary == GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode("utf-8", errors="replace")

    if primary == GGUFValueType.ARRAY:
        elem_type = types[1] if len(types) > 1 else None
        if elem_type == GGUFValueType.STRING:
            return [bytes(field.parts[i]).decode("utf-8", errors="replace") for i in field.data]
        return [field.parts[i].tolist()[0] if hasattr(field.parts[i], "tolist") else field.parts[i][0]
                for i in field.data]

    return field.parts[field.data[0]].tolist()[0] if hasattr(field.parts[field.data[0]], "tolist") else field.parts[field.data[0]][0]


TYPED_ADDERS = {
    GGUFValueType.UINT8:   "add_uint8",
    GGUFValueType.INT8:    "add_int8",
    GGUFValueType.UINT16:  "add_uint16",
    GGUFValueType.INT16:   "add_int16",
    GGUFValueType.UINT32:  "add_uint32",
    GGUFValueType.INT32:   "add_int32",
    GGUFValueType.FLOAT32: "add_float32",
    GGUFValueType.BOOL:    "add_bool",
    GGUFValueType.STRING:  "add_string",
    GGUFValueType.UINT64:  "add_uint64",
    GGUFValueType.INT64:   "add_int64",
    GGUFValueType.FLOAT64: "add_float64",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--keep", type=int, default=20,
                    help="number of leading blocks to retain (default 20)")
    ap.add_argument("--keep-output", action="store_true",
                    help="retain output.weight and output_norm.weight (default: drop them)")
    args = ap.parse_args()

    print(f"[reading]  {args.src}", flush=True)
    r = GGUFReader(str(args.src))

    arch = "llama"
    block_count = None
    for f in r.fields.values():
        if f.name == "general.architecture":
            arch = field_value(f)
        if f.name.endswith(".block_count"):
            block_count = int(field_value(f))
    print(f"[arch]     {arch}", flush=True)
    print(f"[blocks]   src={block_count}, keeping=[0, {args.keep})", flush=True)

    KEEP = args.keep
    drop_top = set() if args.keep_output else {"output.weight", "output_norm.weight"}

    def keep_tensor(name: str) -> bool:
        if name in drop_top:
            return False
        if name.startswith("blk."):
            try:
                n = int(name.split(".")[1])
                return n < KEEP
            except (IndexError, ValueError):
                return True
        return True

    keep_t = [t for t in r.tensors if keep_tensor(t.name)]
    drop_t = [t.name for t in r.tensors if not keep_tensor(t.name)]
    print(f"[tensors]  src={len(r.tensors)}  keeping={len(keep_t)}  dropping={len(drop_t)}", flush=True)
    print(f"[kept first 6]    {[t.name for t in keep_t[:6]]}", flush=True)
    print(f"[dropped first 4] {drop_t[:4]}", flush=True)
    print(f"[dropped last 4]  {drop_t[-4:]}", flush=True)

    print(f"[writing]  {args.dst}", flush=True)
    w = GGUFWriter(str(args.dst), arch=arch)

    skip_kv = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
               "general.architecture"}
    copied = 0
    skipped = []
    for f in r.fields.values():
        if f.name in skip_kv:
            continue
        primary = f.types[0]

        try:
            if primary == GGUFValueType.ARRAY:
                elem_type = f.types[1] if len(f.types) > 1 else None
                vals = field_value(f)
                if elem_type == GGUFValueType.STRING:
                    w.add_array(f.name, vals)
                else:
                    w.add_array(f.name, vals)
            elif primary in TYPED_ADDERS:
                method = getattr(w, TYPED_ADDERS[primary])
                method(f.name, field_value(f))
            else:
                skipped.append((f.name, str(primary)))
                continue
            copied += 1
        except Exception as e:
            skipped.append((f.name, f"{type(e).__name__}: {e}"))

    print(f"[kvs]      copied={copied}  skipped={len(skipped)}", flush=True)
    if skipped:
        for name, reason in skipped[:10]:
            print(f"           SKIP {name} :: {reason}", flush=True)

    for i, t in enumerate(keep_t):
        w.add_tensor(t.name, np.array(t.data), raw_dtype=t.tensor_type)
        if i < 3 or i == len(keep_t) - 1:
            print(f"[tensor]   added {t.name} {t.shape} {t.tensor_type}", flush=True)

    print("[finalize]", flush=True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"[done]     wrote {args.dst}", flush=True)


if __name__ == "__main__":
    main()
