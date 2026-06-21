"""EagleDraft — load a trained EAGLE-3 head into EAGLE's own inference Model and
(eventually) propose draft tokens from the target's hidden_states[0,1,2] (cmd=5).

Reuses ~/EAGLE/eagle/model/cnets.py (the EAGLE-3 inference twin of our trained
head: midlayer + fc(hidden*3) + lm_head(draft_vocab) + d2t/t2d), so we DON'T
reinvent the draft recurrence. This module: (1) load_head — load head_step*.pt
into the inference Model (the foundation, CPU-testable); the full topK_genrate
chain-draft + cmd=5 wiring + acceptance gate is the focused completion.
"""
import os, sys, glob, torch
sys.path.insert(0, os.path.expanduser("~/EAGLE"))

def load_head(ckpt: str, base: str, config_path: str):
    from eagle.model import cnets as infer_cnets
    from eagle.model.configs import EConfig
    cfg = EConfig.from_pretrained(config_path)
    model = infer_cnets.Model(cfg, load_emb=True, path=base)
    sd = torch.load(ckpt, map_location="cpu")
    miss = model.load_state_dict(sd, strict=False)
    model.eval()
    return model, miss, cfg

if __name__ == "__main__":
    base = os.path.expanduser("~/prithvi-target")
    cfgp = os.path.expanduser("~/EAGLE/eagle/traineagle3/config.json")
    ckpt = sorted(glob.glob(os.path.expanduser("~/eagle-out/head_step*.pt")))[-1]
    print("loading", ckpt.split("/")[-1], "into EAGLE-3 inference Model…", flush=True)
    model, miss, cfg = load_head(ckpt, base, cfgp)
    trained = {"midlayer","fc","lm_head","norm"}
    miss_trained = [k for k in miss.missing_keys if any(k.startswith(t) for t in trained)]
    print("  missing(trained):", miss_trained or "NONE (all head weights loaded)")
    print("  missing(other, expected e.g. embed/rotary):", len(miss.missing_keys)-len(miss_trained))
    print("  unexpected:", miss.unexpected_keys or "none")
    print("  fc:", tuple(model.fc.weight.shape), "lm_head:", tuple(model.lm_head.weight.shape),
          "d2t:", tuple(model.d2t.shape))
    print("  VERDICT:", "LOADER OK — trained head loads into EAGLE-3 inference Model" if not miss_trained else "FAIL: trained weights missing")
