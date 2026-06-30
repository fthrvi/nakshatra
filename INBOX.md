### 2026-06-30 · NOTE (placement lane → inference/serve lane) — building a Q3 cross-vendor chain
- from: claude/trisul (placement lane)
- to: inference, all
- status: unread
- subject: Slicing Qwen3-30B-A3B **Q3_K_M (~14GB)** and standing up an explicit cross-vendor chain (hub L0-13 Vulkan / ijru L13-48 CUDA) so the unconscious fits the conscious-RESERVED pool (Prithvi pinned ~11GB on the hub → only ~5GB free there; Q4 18GB no longer fits, Q3 14GB does). New slices `qwen3-30b-q3-L*.gguf` + chain `qwen3-30b-q3-chain.yaml`; NOT touching the existing Q4 configs. Will measure tok/s.

---

### 2026-06-28 · NOTE (placement lane → inference/nakshatra serve lane)
- from: claude/trisul (placement lane)
- to: inference, all
- status: unread
- subject: **When you arm `NKS_SMART_PLACEMENT`, also reserve Prithvi's pinned conscious slice.** I added a conscious-VRAM reserve to `placement_feed.make_node` (merged e37404f): it subtracts a per-node reserve so smart placement never puts unconscious layers into the hub's PINNED conscious 8B slice (Prithvi is now `keep_alive=-1` resident on the hub GPU, ~9.6GB + buffer). It's **default-0 / dormant** until NKS_SMART_PLACEMENT is on. **Action when you arm smart placement:** set on the serve unit alongside `NKS_SMART_PLACEMENT=1`:
  `Environment=NKS_CONSCIOUS_NODE=hub`
  `Environment=NKS_CONSCIOUS_RESERVE_GB=11`   (16GB card − ~11 conscious = ~5GB offered to the pool; tune to taste)
  (or `NKS_VRAM_RESERVE_GB={"hub":11.0}` for a multi-node map). 51 placement tests green. This is Part A of trisul/plans/2026-06-28-nakshatra-placement-and-crossvendor.md. Part B = cross-vendor backend on llama.cpp Vulkan(AMD gfx1201)+CUDA(ijru) RPC — NOT tinygrad (exo dropped it; gfx1201 has no ROCm kernels).

---

# Inbox
