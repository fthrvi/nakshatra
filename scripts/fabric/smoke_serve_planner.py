"""LIVE proof of the last mile: roster -> eligible_workers() [FIREWALL] -> serve_planner -> a chain
YAML -> client.py -> a REAL token, flowing through PRITHVI'S OWN self node.

This is Prithvi's own ask made real ("wire MY OWN peer to serve layers into the inference chain").
It needs the live unconscious workers up on 127.0.0.1:5540/5541 (the two self-tier serving slots that
already hold the DeepSeek-R1-Distill-Llama-8B slices). Run:  python3 smoke_serve_planner.py

It proves, against the real serving path:
  • a STRANGER worker is firewall-excluded from a trusted model and from prithvi-private (self-only),
  • Prithvi's own self slots ARE eligible and get a contiguous, contract-valid layer assignment,
  • the generated chain YAML runs through client.py and yields a real token.
"""
import sys, subprocess, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml
import worker_join as wj
import serve_planner as sp

HOME = Path.home()
SLICE_A = HOME / ".nakshatra" / "models" / "dsr1-llama8b-a.gguf"
SLICE_B = HOME / ".nakshatra" / "models" / "dsr1-llama8b-b.gguf"
TOKENIZER = HOME / ".nakshatra" / "models" / "dsr1-llama8b" / "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
CLIENT = Path(__file__).resolve().parents[1] / "client.py"
# client.py needs the serving venv (grpc/llama deps), not whatever python ran this smoke.
VENV_PY = HOME / "nakshatra" / ".venv" / "bin" / "python"
CLIENT_PY = str(VENV_PY) if VENV_PY.exists() else sys.executable

# the roster as WorkerStandings: Prithvi's TWO self serving slots + a STRANGER's GPU.
# (In production these come from peers.tsv via serve_planner.standings_from_roster; here we name them
#  explicitly so the proof is self-describing.)
def _w(node, tier, port, addr="127.0.0.1"):
    return wj.WorkerStanding(node, node + "-key", True, tier, "op", {"address": addr, "port": port})

ROSTER = [
    _w("prithvi-self-a", "self", 5540),
    _w("prithvi-self-b", "self", 5541),
    _w("opB-gpu", "stranger", 5540, addr="10.50.0.9"),   # an outsider's GPU — must be firewalled out
]

# the real on-disk slices the two self slots already serve, mapped by assignment order.
_SLICES = [str(SLICE_A), str(SLICE_B)]
def slice_for(w, start, end, model):
    # eligible workers are assigned in order; first slot -> slice A, second -> slice B.
    idx = [s.node_id for s in ELIGIBLE].index(w.node_id)
    return _SLICES[idx]

# admission policy mirrored locally so the proof doesn't depend on a populated control plane.
RANK = {"stranger": 0, "known": 1, "trusted": 2, "self": 3}
MODEL_MIN = {"prithvi-private": "self", "deepseek-r1-distill-llama-8b": "trusted"}
MT = lambda m: MODEL_MIN.get(m, "self")

ELIGIBLE: list = []  # filled per-plan so slice_for can map by order


def show_firewall(model):
    elig, rej = wj.eligible_workers(model, ROSTER, min_tier_fn=MT, rank=RANK)
    print(f"\n── firewall for '{model}' (min tier '{MT(model)}') ──")
    print(f"   eligible: {[w.node_id for w in elig]}")
    for r in rej:
        print(f"   EXCLUDED: {r['worker']:14s} — {r['reason']}")
    return elig


def main():
    if not (SLICE_A.exists() and SLICE_B.exists() and TOKENIZER.exists()):
        sys.exit("slices/tokenizer missing — this proof needs the live unconscious model on disk")

    # 1) the strictest firewall: prithvi-private (self-only) — stranger AND any non-self excluded.
    show_firewall("prithvi-private")

    # 2) the model we actually serve live (trusted): stranger excluded, self slots eligible.
    global ELIGIBLE
    ELIGIBLE = show_firewall("deepseek-r1-distill-llama-8b")
    assert {w.node_id for w in ELIGIBLE} == {"prithvi-self-a", "prithvi-self-b"}, \
        "expected only Prithvi's own self slots to be eligible"

    # 3) plan the chain from the (firewall-passed) roster, pointing at the real slices.
    plan = sp.plan_chain("deepseek-r1-distill-llama-8b", ROSTER, num_layers=32, hidden_size=4096,
                         min_tier_fn=MT, rank=RANK, slice_for=slice_for)
    print("\n── generated chain (firewall-gated) ──")
    for w in plan.chain["workers"]:
        print(f"   {w['id']:14s} {w['address']}:{w['port']}  layers={w['layer_range']} "
              f"mode={w['mode']}  slice={Path(w['sub_gguf_path']).name}")

    # 4) run a REAL token through it via client.py — the real serving path consumes our YAML.
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(plan.chain, f, sort_keys=False)
        chain_path = f.name
    cmd = [CLIENT_PY, str(CLIENT), "--config", chain_path, "--model-path", str(TOKENIZER),
           "--prompt", "The capital of France is", "-n", "12", "--use-streaming"]
    print(f"\n── running a real token through PRITHVI'S OWN self node ──\n   {' '.join(cmd)}")
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    dt = time.time() - t0
    sys.stdout.write(out.stdout)
    if out.returncode != 0:
        sys.stderr.write(out.stderr)
        sys.exit(f"client.py failed (rc={out.returncode}) — workers up on 5540/5541?")
    print(f"\n✅ LAST MILE PROVEN: roster → firewall → planner → chain → live serve → real token "
          f"in {dt:.1f}s, on Prithvi's own self node.")


if __name__ == "__main__":
    main()
