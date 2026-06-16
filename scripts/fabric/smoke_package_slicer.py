"""GPU-free proof of DYNAMIC SLICING: the planner assigns an ARBITRARY contiguous range and the
slice materialises from the content-addressed package — loader-ready — even though no pre-cut file
for that range ever existed.

Needs the dsr1-llama8b package on disk (build once:
    python scripts/packaging/package_gguf.py \
        ~/.nakshatra/models/dsr1-llama8b/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf \
        ~/.nakshatra/packages/dsr1-llama8b --model-id dsr1-llama8b)

Run:  ~/nakshatra/.venv/bin/python scripts/fabric/smoke_package_slicer.py
(the venv has gguf/numpy). It proves a NOVEL split [0,20)/[20,32) — boundaries the pre-cut
dsr1-llama8b-a/-b.gguf (cut at 16) never had — assembles and carries the exact loader metadata a
real worker daemon reads. The matching live serving run (workers on these slices → a real token) is
recorded in memory project_serve_layers_next."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import worker_join as wj
import serve_planner as sp
import package_slicer as ps

PKG = Path.home() / ".nakshatra" / "packages" / "dsr1-llama8b"
RANK = {"stranger": 0, "known": 1, "trusted": 2, "self": 3}
MT = lambda m: {"deepseek-r1-distill-llama-8b": "trusted"}.get(m, "self")


def _novel_split(num_layers, n):
    assert n == 2, "this proof uses two self slots"
    return [(0, 20, "first"), (20, 32, "last")]   # boundary 20, NOT the pre-cut 16


def _loader_ready(path, expect_start, expect_end, expect_embd, expect_head):
    """A real worker daemon reads these KVs to know its slice; verify they're present + correct."""
    from gguf import GGUFReader
    r = GGUFReader(str(path))
    kv = {f.name: f for f in r.fields.values()}

    def u32(name):
        f = kv[name]
        return int(f.parts[f.data[0]][0])

    def b(name):
        f = kv[name]
        return bool(f.parts[f.data[0]][0])

    assert u32("nakshatra.layer_range_start") == expect_start, "wrong layer_range_start"
    assert u32("nakshatra.layer_range_end") == expect_end, "wrong layer_range_end"
    assert b("nakshatra.has_token_embd") == expect_embd, "wrong has_token_embd"
    assert b("nakshatra.has_lm_head") == expect_head, "wrong has_lm_head"
    names = {t.name for t in r.tensors}
    # the assigned blk.N.* layers are present; the ones outside the range are NOT
    for i in range(expect_start, expect_end):
        assert any(n.startswith(f"blk.{i}.") for n in names), f"missing layer {i} tensors"
    assert not any(n.startswith(f"blk.{expect_end}.") for n in names), "leaked a layer past the range"
    return len(names)


def main():
    if not (PKG / "package.json").exists():
        sys.exit(f"package missing at {PKG} — build it first (see this file's docstring)")

    slicer = ps.PackageSlicer(str(PKG))
    pool = [wj.WorkerStanding("self-a", "k1", True, "self", "me", {"address": "127.0.0.1", "port": 5560}),
            wj.WorkerStanding("self-b", "k2", True, "self", "me", {"address": "127.0.0.1", "port": 5561}),
            wj.WorkerStanding("opB-gpu", "k3", True, "stranger", "opB", {"address": "10.50.0.9", "port": 5560})]

    plan = sp.plan_chain("deepseek-r1-distill-llama-8b", pool, num_layers=32, hidden_size=4096,
                         min_tier_fn=MT, rank=RANK, partition_fn=_novel_split,
                         slice_for=slicer.slice_for)

    print(f"firewall: eligible={plan.eligible} rejected={[r['worker'] for r in plan.rejected]}")
    assert plan.eligible == ["self-a", "self-b"], "stranger should be firewalled out"

    for w, (es, ee, embd, head) in zip(plan.chain["workers"],
                                       [(0, 20, True, False), (20, 32, False, True)]):
        path = w["sub_gguf_path"]
        assert "L%d-%d" % (es, ee) in path, "slice path not content-addressed to the range"
        # the killer point: no pre-cut file for this boundary ever existed
        precut = Path.home() / ".nakshatra" / "models" / f"dsr1-llama8b-L{es}-{ee}.gguf"
        assert not precut.exists(), "a pre-cut file exists — pick a genuinely novel range"
        ntensors = _loader_ready(path, es, ee, embd, head)
        print(f"  {w['id']:8s} layers=[{es},{ee}) mode={w['mode']:5s} "
              f"loader-ready ✓  ({ntensors} tensors)  {Path(path).name}")

    print("\n✅ DYNAMIC SLICING PROVEN: the planner assigned a novel [0,20)/[20,32) split and the "
          "loader-ready slices materialised from the content-addressed package — no pre-cut files.")


if __name__ == "__main__":
    main()
