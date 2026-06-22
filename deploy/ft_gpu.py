"""GPU serving-hidden retrain of the EAGLE head (finetune_full + CUDA).

load_head() returns the model on CPU; finetune_full.py never moved it to the GPU,
so it ground on CPU. This version puts model + inputs on CUDA → epochs in seconds.
Loads the latest head_step*.pt as base, fine-tunes on the 500-sample cmd=5 (serving)
hidden cache, holds out 10%, saves best-held-out head to head_serving_hidden.pt.
"""
import os, sys, glob, torch
sys.path.insert(0, os.path.expanduser("~/EAGLE")); sys.path.insert(0, os.path.expanduser("~"))
from eagle_draft import load_head
BASE = os.path.expanduser("~/prithvi-target")
CFG = os.path.expanduser("~/EAGLE/eagle/traineagle3/config.json")
CK = sorted(glob.glob(os.path.expanduser("~/eagle-out/head_step*.pt")))[-1]
CACHE = os.path.expanduser("~/cmd5_hidden_cache_500.pt")
OUT = os.path.expanduser("~/eagle-out/head_serving_hidden.pt")
EPOCHS = int(os.environ.get("EPOCHS", "20"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEV} ckpt={os.path.basename(CK)}", flush=True)
cache = torch.load(CACHE, map_location="cpu")
split = max(1, len(cache) // 10); test = cache[:split]; train = cache[split:]
print(f"train={len(train)} heldout={len(test)} epochs={EPOCHS}", flush=True)
model, _, cfg = load_head(CK, BASE, CFG); DV, V = cfg.draft_vocab_size, cfg.vocab_size
model = model.to(DEV)
full2draft = torch.full((V,), -1, dtype=torch.long)
for di in range(DV):
    full2draft[di + int(model.d2t[di])] = di
full2draft = full2draft.to(DEV)
for m in (model.midlayer, model.fc, model.lm_head, model.norm):
    for p in m.parameters():
        p.requires_grad = True
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
ce = torch.nn.CrossEntropyLoss()


def run(data, train_mode):
    hits = tot = 0; ls = 0.0; nb = 0
    torch.set_grad_enabled(train_mode)
    for s in data:
        hid = s["hidden3"].float()[None].to(DEV); ids = s["input_ids"][None].to(DEV)
        td = full2draft[s["target"].to(DEV)]; v = td >= 0
        if v.sum() < 2:
            continue
        out = model(hid, input_ids=ids, use_cache=False)
        oh = out[0] if isinstance(out, (tuple, list)) else out
        lg = model.lm_head(model.norm(oh))[0][v]; la = td[v]
        if train_mode:
            loss = ce(lg, la)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); ls += loss.item(); nb += 1
        hits += int((lg.argmax(-1) == la).sum()); tot += int(v.sum())
    return hits / max(tot, 1), ls / max(nb, 1)


print(f"  heldout BEFORE: {run(test, False)[0]:.3f}", flush=True)
best = 0.0
for e in range(EPOCHS):
    tr, _ = run(train, True); ho, _ = run(test, False)
    print(f"  epoch {e}: train={tr:.3f} heldout={ho:.3f}", flush=True)
    if ho > best:
        best = ho
        sd = {k: v.detach().half().cpu() for k, v in model.named_parameters() if v.requires_grad}
        sd["d2t"] = model.d2t.cpu(); sd["t2d"] = model.t2d.cpu()
        torch.save(sd, OUT)
print(f"DONE: best heldout acceptance={best:.3f} -> saved {OUT}", flush=True)
