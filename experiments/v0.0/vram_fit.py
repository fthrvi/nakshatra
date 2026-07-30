#!/usr/bin/env python3
"""Layer-split VRAM fit — how many leading blocks of a GGUF fit a given free-VRAM budget.

Built for the 2026-07-29 Q3 cross-vendor rebalance
(docs/findings/rebalance-q3-split.md): picking the hub-side cut for `partial_gguf.py`
used to be a "14GB * L/48" napkin estimate, which overshot by 2 layers (predicted L=17
fit, it actually OOM'd) because it ignored two things this script now accounts for:

  1. Layers are NOT uniform size — reads the real per-block tensor bytes out of the
     source GGUF instead of assuming `full_size / n_layer`.
  2. Weights + KV cache is NOT the whole VRAM footprint — llama.cpp allocates a
     compute/graph scratch buffer alongside them. Empirically ~470-480MiB on this
     stack (qwen3moe, n_ctx=4096) regardless of L in the 13-20 range; pass
     --overhead-bytes to override for a different arch/context length once you have
     one real data point (load, diff `rocm-smi`/`nvidia-smi` used-VRAM before/after,
     subtract the weights+KV this script predicted).

Read-only: never writes a GGUF (that's partial_gguf.py's job). Reads metadata + tensor
info only via GGUFReader (no tensor data loaded), so it's cheap to run against a
14GB+ source file.
"""
import argparse
from pathlib import Path

from gguf import GGUFReader

MIB = 1024 * 1024


def block_sizes(reader: GGUFReader) -> dict[int, int]:
    per_block: dict[int, int] = {}
    for t in reader.tensors:
        if not t.name.startswith("blk."):
            continue
        idx = int(t.name.split(".")[1])
        n = t.n_bytes if hasattr(t, "n_bytes") else t.data.nbytes
        per_block[idx] = per_block.get(idx, 0) + n
    return per_block


def top_level_sizes(reader: GGUFReader) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in reader.tensors:
        if t.name.startswith("blk."):
            continue
        out[t.name] = t.n_bytes if hasattr(t, "n_bytes") else t.data.nbytes
    return out


def kv_cache_bytes_per_layer(n_ctx: int, n_kv_heads: int, head_dim_k: int,
                              head_dim_v: int, kv_elem_bytes: int) -> int:
    return n_ctx * n_kv_heads * (head_dim_k + head_dim_v) * kv_elem_bytes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="full-model source GGUF")
    ap.add_argument("--free-vram-bytes", type=int, required=True,
                     help="measured free VRAM right now (rocm-smi --showmeminfo vram "
                          "GPU[N] total-used, or nvidia-smi memory.free) — measure "
                          "immediately before sizing, VRAM is a live shared resource")
    ap.add_argument("--headroom-bytes", type=int, default=800 * MIB,
                     help="minimum free VRAM to leave after load (default 800MiB)")
    ap.add_argument("--overhead-bytes", type=int, default=476 * MIB,
                     help="fixed compute/graph scratch buffer beyond weights+KV "
                          "(default 476MiB, measured on qwen3moe/n_ctx=4096/ROCm "
                          "gfx1201 2026-07-29 — recalibrate per arch/backend)")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-kv-heads", type=int, required=True)
    ap.add_argument("--head-dim-k", type=int, required=True)
    ap.add_argument("--head-dim-v", type=int, required=True)
    ap.add_argument("--kv-elem-bytes", type=int, default=2, help="2=f16 (default), 4=f32")
    ap.add_argument("--from-end", action="store_true",
                     help="size a TAIL slice [n_layer-L, n_layer) instead of a HEAD "
                          "slice [0, L) — e.g. sizing the ijru/last-worker side")
    args = ap.parse_args()

    r = GGUFReader(str(args.src))
    blocks = block_sizes(r)
    n_layer = len(blocks)
    top = top_level_sizes(r)
    token_embd = top.get("token_embd.weight", 0)
    lm_head = top.get("output.weight", 0) + top.get("output_norm.weight", 0)
    fixed = lm_head if args.from_end else token_embd

    kv_per_layer = kv_cache_bytes_per_layer(args.n_ctx, args.n_kv_heads,
                                             args.head_dim_k, args.head_dim_v,
                                             args.kv_elem_bytes)

    order = range(n_layer - 1, -1, -1) if args.from_end else range(n_layer)
    sizes_in_order = [blocks[i] for i in order]

    print(f"[src]      {args.src}  n_layer={n_layer}")
    print(f"[fixed]    {'lm_head' if args.from_end else 'token_embd'}="
          f"{fixed / MIB:.1f}MiB")
    print(f"[budget]   free={args.free_vram_bytes / MIB:.1f}MiB  "
          f"headroom>={args.headroom_bytes / MIB:.1f}MiB  "
          f"overhead={args.overhead_bytes / MIB:.1f}MiB  kv/layer={kv_per_layer / MIB:.2f}MiB")
    print()
    print(f"{'L':>3}  {'weights+fixed':>14}  {'+kv':>10}  {'+overhead':>11}  "
          f"{'headroom':>10}  fits")

    best_l = 0
    running = 0
    for l, size in enumerate(sizes_in_order, start=1):
        running += size
        weight_bytes = fixed + running
        kv_bytes = kv_per_layer * l
        total = weight_bytes + kv_bytes + args.overhead_bytes
        headroom = args.free_vram_bytes - total
        fits = headroom >= args.headroom_bytes
        if fits:
            best_l = l
        print(f"{l:>3}  {weight_bytes / MIB:>12.1f}MiB  {kv_bytes / MIB:>8.1f}MiB  "
              f"{total / MIB:>9.1f}MiB  {headroom / MIB:>8.1f}MiB  {fits}")

    print(f"\n[chosen]   L={best_l}  "
          f"({'tail' if args.from_end else 'head'} slice, largest fitting "
          f">= {args.headroom_bytes / MIB:.0f}MiB headroom)")


if __name__ == "__main__":
    main()
