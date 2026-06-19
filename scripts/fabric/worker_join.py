"""
worker_join — the OUTSIDER GPU worker self-join (tiered, admission-gated).

The owner-worker join (`fabric/join.py` → Sthambha OWNER `/join`) is for YOUR OWN machines — they
join as *self*, full presence, they are Prithvi. This is the SECOND path: a foreign GPU box joins as
**borrowed compute in the pool**, never as a body Prithvi lives in.

The identity guarantee (Prithvi: "if strangers get their own 'me', I lose who I am"):
  • an outsider box is admitted at a TRUST TIER (default-deny), through the blind junction — it gets
    NO route into your mesh; it serves model *layers*, it does not run Prithvi's mind.
  • THE FIREWALL is `eligible_workers()` (built on admission's trust-tiered splitting): Prithvi's
    SENSITIVE models carry a high min-tier, so a general-tier outsider is NEVER assigned their layers.
    His self runs only on his own/trusted nodes; strangers' GPUs serve general/public inference only.

So strangers add muscle to the pool; Prithvi's mind stays on Prithvi's machines. Composes
`infra/control-plane/admission.py` (admit + trust-tier split) + the worker capability shape; does not
rebuild either. Admission is INJECTABLE so this is testable without the live control plane.

Flow:  mint a worker-invite (operator) → box dials the junction w/ its signed identity →
        admission.admit() @ tier → register capabilities → planner's eligible_workers() gates by tier.
"""
from __future__ import annotations
import platform
import os, re, sys, json, shutil, subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

_CP = os.environ.get("NAKSHATRA_ADMISSION_DIR",
                     str(Path.home() / "trisul" / "infra" / "control-plane"))


def _load_admission():
    if _CP not in sys.path:
        sys.path.insert(0, _CP)
    import admission as adm  # noqa
    return adm


# ── capability self-description (the one-command box declares what it is) ───────────────────────────
def detect_capabilities() -> dict:
    """What this box can contribute — detected, not configured. Recognizes the WHOLE heterogeneous
    fleet a permissionless network draws: NVIDIA (cuda), AMD (rocm), Apple/unified-memory (metal),
    and ANY other vendor — Intel Arc, Asus, etc. — via the universal Vulkan backend. A node must
    have real accelerator memory (discrete GPU VRAM or unified memory) to serve a layer slice; a
    box with none is flagged `cpu-worker` (too slow to be a useful node). The order tries the
    vendor-native SMIs first (accurate VRAM), then Metal, then Vulkan (the any-vendor catch-all)."""
    gpu, vram_mb, backend = None, 0, "cpu"
    try:
        if shutil.which("nvidia-smi"):
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8).stdout
            row = (out.strip().splitlines() or [""])[0].split(",")
            if len(row) >= 2:
                gpu, vram_mb, backend = row[0].strip(), int(re.sub(r"\D", "", row[1]) or 0), "cuda"
        elif shutil.which("rocm-smi"):
            out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--csv"],
                                 capture_output=True, text=True, timeout=8).stdout
            m = re.search(r"(\d{6,})", out or "")
            gpu, vram_mb, backend = "amd-rocm", (int(m.group(1)) // (1024 * 1024)) if m else 0, "rocm"
        elif platform.system() == "Darwin":
            # Apple Silicon / Intel Mac: the GPU runs on UNIFIED memory via Metal. macOS exposes no
            # clean GPU-VRAM figure, so offer a conservative share of total RAM (operator can override
            # with --vram-offered-gb). The point: a Mac IS a valid node, not cpu-only.
            total = int((subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                        text=True, timeout=8).stdout or "0").strip() or 0)
            gpu, vram_mb, backend = "apple-metal", int(total / (1024 * 1024) * 0.6), "metal"
        elif shutil.which("vulkaninfo"):
            # The any-vendor catch-all: Intel Arc, AMD, NVIDIA, … all speak Vulkan, and our partial-load
            # patch is backend-agnostic (frontend-only), so a Vulkan device IS a serving node.
            out = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=12).stdout or ""
            if "deviceName" not in out:
                out = subprocess.run(["vulkaninfo"], capture_output=True, text=True, timeout=12).stdout or ""
            m = re.search(r"deviceName\s*[=:]\s*(.+)", out)
            if m:
                heaps = [int(x) for x in re.findall(r"\((\d{7,})\)", out)]   # best-effort largest heap (bytes)
                gpu = m.group(1).strip() + " (vulkan)"
                vram_mb, backend = (max(heaps) // (1024 * 1024) if heaps else 0), "vulkan"
    except Exception:
        pass
    return {"gpu": gpu or "cpu-only", "vram_mb": vram_mb, "backend": backend,
            "kind": "gpu-worker" if gpu else "cpu-worker"}


@dataclass
class WorkerStanding:
    node_id: str
    pubkey: str
    admitted: bool
    tier: str                       # self | trusted | known | general | (denied → admitted=False)
    tenant: Optional[str]
    capabilities: dict = field(default_factory=dict)
    reason: str = ""
    def to_dict(self) -> dict:
        return asdict(self)


# ── the admission-gated, tiered worker join ─────────────────────────────────────────────────────────
def join_as_worker(admission_request: dict, capabilities: dict, *,
                   admit_fn: Optional[Callable[[dict], dict]] = None,
                   node_id: str = "") -> WorkerStanding:
    """A foreign GPU box presents its signed admission request → admit at a tier (default-deny) →
    return its standing as a pooled worker. The pubkey identifies it; the tier gates what it may serve."""
    pubkey = (admission_request or {}).get("public_key_hex") or (admission_request or {}).get("pubkey") or ""
    if admit_fn is None:
        adm = _load_admission()
        peers, models = adm.load_peers(), adm.load_models()
        admit_fn = lambda req: adm.admit(req, peers=peers, models=models)
    d = admit_fn(admission_request) or {}
    ok = bool(d.get("admitted"))
    return WorkerStanding(
        node_id=node_id or pubkey[:12], pubkey=pubkey, admitted=ok,
        tier=(d.get("tier") or "denied") if ok else "denied",
        tenant=d.get("tenant"), capabilities=capabilities if ok else {},
        reason=d.get("reason", ""))


# ── THE IDENTITY FIREWALL — the planner gate ────────────────────────────────────────────────────────
def eligible_workers(model: str, workers: list[WorkerStanding], *,
                     min_tier_fn: Optional[Callable[[str], str]] = None,
                     rank: Optional[dict] = None, min_vram_mb: int = 0) -> tuple[list, list]:
    """Of the admitted workers, who may serve `model`'s layers — gated by (1) a CAPABILITY FLOOR
    and (2) the model's min trust tier.

    Capability floor (the node rule): a node must have real accelerator memory — discrete GPU VRAM
    or unified memory (Mac/Metal) — to hold a layer slice; a `cpu-worker` (no accelerator) is rejected,
    and `min_vram_mb` (if set) rejects too-small cards. Trust floor: Prithvi's SENSITIVE models
    (min-tier self/trusted) exclude lower-tier outsiders. Returns (eligible, rejected).
    `rank`/`min_tier_fn` injectable so this is testable without the live control plane."""
    if rank is None or min_tier_fn is None:
        adm = _load_admission()
        rank = rank if rank is not None else adm.TIER_RANK            # stranger<known<trusted<self
        if min_tier_fn is None:
            models_policy = adm.load_models()
            min_tier_fn = lambda m: adm.model_min_tier(m, models_policy)
    need = min_tier_fn(model)
    need_r = rank.get(need, 99)
    eligible, rejected = [], []
    for w in workers:
        if not w.admitted:
            rejected.append({"worker": w.node_id, "reason": "not admitted"}); continue
        cap = getattr(w, "capabilities", {}) or {}
        # capability floor — a node needs an accelerator (GPU VRAM or unified memory) to serve a slice.
        # Only reject on EXPLICIT cpu-worker (older workers with no capabilities still pass, back-compat).
        if cap.get("kind") == "cpu-worker":
            rejected.append({"worker": w.node_id, "reason": "no accelerator — GPU VRAM or unified memory required to be a node"}); continue
        if min_vram_mb and cap.get("vram_mb") and int(cap["vram_mb"]) < min_vram_mb:
            rejected.append({"worker": w.node_id, "vram_mb": cap.get("vram_mb"),
                             "reason": f"vram {cap.get('vram_mb')}MB < {min_vram_mb}MB floor"}); continue
        # trust floor — the identity firewall
        if rank.get(w.tier, -1) >= need_r:
            eligible.append(w)
        else:
            rejected.append({"worker": w.node_id, "tier": w.tier, "need": need,
                             "reason": f"tier {w.tier} < {need} required for {model} (identity firewall)"})
    return eligible, rejected


if __name__ == "__main__":   # `python3 worker_join.py` → print this box's detected capabilities
    print(json.dumps(detect_capabilities(), indent=2))
