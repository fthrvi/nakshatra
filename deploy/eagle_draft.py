"""EagleDraft — the trained EAGLE-3 head as a speculative-decode draft proposer.

Reuses EAGLE's own EAGLE-3 inference Model (~/EAGLE/eagle/model/cnets.py) — the
exact twin of our trained head — so the draft recurrence is EAGLE's tested code,
not a reimplementation. Pairs with the worker daemon's `cmd=5 EAGLE_HIDDEN`
(returns the target's hidden_states[0,1,2] = [n_tokens, 3*n_embd]).

Status (CPU-tested 2026-06-21): load_head ✅, forward ✅, topK_genrate tree ✅,
linear K-chain propose ✅. REMAINING: wire propose() to the LIVE cmd=5 worker
connection + into scripts/speculative.py's spec loop, then validate acceptance
≈ training acc (~0.60).
"""
import os, sys, glob, torch
sys.path.insert(0, os.path.expanduser("~/EAGLE"))


def load_head(ckpt: str, base: str, config_path: str):
    """Load a trained head_step*.pt into EAGLE's EAGLE-3 inference Model."""
    from eagle.model import cnets as infer_cnets
    from eagle.model.configs import EConfig
    cfg = EConfig.from_pretrained(config_path)
    model = infer_cnets.Model(cfg, load_emb=True, path=base)
    miss = model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    model.eval()
    return model, miss, cfg


def propose(model, hidden3, input_ids, k: int):
    """Propose a LINEAR chain of k draft tokens (full-vocab ids) from the target's
    cmd=5 hidden states.
      hidden3   : [1, S, 3*n_embd]  — the target's concatenated hidden_states[0,1,2]
      input_ids : [1, S+1]          — prefix tokens incl. the freshly-sampled one
                  (topK_genrate drops the first; len must be hidden_seq + 1)
    Returns the k drafted token ids (excludes the leading sample_token).
    """
    model.top_k = 1; model.depth = k; model.total_tokens = k   # linear chain
    model.init_tree()
    out = model.topK_genrate(hidden3, input_ids, head=model.lm_head, logits_processor=None)
    chain = out[0].view(-1).tolist()
    return chain[1:k + 1]   # drop the leading sample_token


if __name__ == "__main__":
    base = os.path.expanduser("~/prithvi-target")
    cfgp = os.path.expanduser("~/EAGLE/eagle/traineagle3/config.json")
    ckpt = sorted(glob.glob(os.path.expanduser("~/eagle-out/head_step*.pt")))[-1]
    model, miss, cfg = load_head(ckpt, base, cfgp)
    bad = [k for k in miss.missing_keys if any(k.startswith(t) for t in ("midlayer","fc","lm_head","norm"))]
    print("load:", "OK" if not bad else f"MISSING {bad}")
    H = cfg.hidden_size
    chain = propose(model, torch.randn(1,6,3*H)*0.02, torch.tensor([[1,2,3,4,5,6,7]]), k=5)
    print("propose(k=5) →", chain, "(dummy hidden; real cmd=5 gives real tokens)")
