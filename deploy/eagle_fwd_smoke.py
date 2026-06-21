import os, sys, glob, torch
sys.path.insert(0, os.path.expanduser("~/EAGLE"))
sys.path.insert(0, os.path.expanduser("~"))   # for eagle_draft
from eagle_draft import load_head
base=os.path.expanduser("~/prithvi-target")
cfgp=os.path.expanduser("~/EAGLE/eagle/traineagle3/config.json")
ckpt=sorted(glob.glob(os.path.expanduser("~/eagle-out/head_step*.pt")))[-1]
model, miss, cfg = load_head(ckpt, base, cfgp)
H = cfg.hidden_size
# cmd=5 gives [n_tokens, 3*n_embd]; batch it to [1, seq, 3*H]
seq=4
hidden = torch.randn(1, seq, 3*H, dtype=torch.float32) * 0.02
ids = torch.tensor([[1,2,3,4]], dtype=torch.long)
try:
    out = model(hidden, input_ids=ids, use_cache=True)
    oh = out[0] if isinstance(out,(tuple,list)) else out
    print("FORWARD OK — out hidden shape:", tuple(oh.shape), "(expect [1,seq,%d])"%H)
    # project last hidden through the draft lm_head → draft-vocab logits → a token
    logits = model.lm_head(model.norm(oh[:,-1]))
    tok_draft = int(logits.argmax(-1))
    tok_full = int(model.d2t[tok_draft]) + tok_draft   # draft-vocab id -> full vocab via d2t
    print("draft logits:", tuple(logits.shape), "→ draft_tok", tok_draft, "→ full_tok", tok_full)
    print("VERDICT: EagleDraft forward path WORKS (consumes cmd=5-shaped hidden, emits a draft token)")
except Exception as e:
    import traceback; traceback.print_exc(); print("FORWARD FAILED:", e)
