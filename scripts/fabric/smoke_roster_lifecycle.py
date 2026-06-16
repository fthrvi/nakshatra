"""LIVE proof of slice 4 part 1: the from_roster chain is AUTONOMOUSLY summoned.

No hand-launched workers this time. A ChainLifecycle(RosterWorkerController) does it:
    lifecycle.begin()  → planner (firewall) decides the assignment → launches THIS box's workers,
                         each SELF-PROVISIONING its slice from the package → blocks until ready
    client.py over the generated chain → a real token
    controller.stop()  → reaped (scale-to-zero), GPU freed.

Needs the dsr1-llama8b package on disk (see smoke_package_slicer.py). Stops the live unconscious
workers for VRAM and restores them. Run: ~/nakshatra/.venv/bin/python scripts/fabric/smoke_roster_lifecycle.py
"""
import os, sys, time, tempfile, shutil, subprocess
from pathlib import Path

NK = Path.home() / "nakshatra"
SCRIPTS = NK / "scripts"
sys.path.insert(0, str(SCRIPTS))
PKG = Path.home() / ".nakshatra" / "packages" / "dsr1-llama8b"
TOKENIZER = Path.home() / ".nakshatra" / "models" / "dsr1-llama8b" / "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
MODEL = "deepseek-r1-distill-llama-8b"
WORKER_ENV = {"NAKSHATRA_TLS_REQUIRED": "false", "NAKSHATRA_AUTH_REQUIRED": "false",
              "NAKSHATRA_REFUSE_UNREGISTERED_PEERS": "false"}


def main():
    if not (PKG / "package.json").exists():
        sys.exit(f"package missing at {PKG} — build it first (see smoke_package_slicer.py)")
    tmp = Path(tempfile.mkdtemp(prefix="nks-roster-lc-"))
    stopped_live = False
    lc = None
    try:
        # temp roster: 2 self slots on this box (the controller will summon them) + a stranger.
        peers = tmp / "peers.tsv"
        peers.write_text(
            "pk-a\troster-self-a\tme\tself\thome\t127.0.0.1:5560\n"
            "pk-b\troster-self-b\tme\tself\thome\t127.0.0.1:5561\n"
            "pk-x\topB-gpu\topB\tstranger\topB\t10.50.0.9:5560\n")
        models_pol = tmp / "models.tsv"
        models_pol.write_text(f"{MODEL}\ttrusted\n")
        os.environ.update(WORKER_ENV)
        os.environ["MESH_PEERS"] = str(peers)
        os.environ["MESH_MODELS"] = str(models_pol)

        # free VRAM
        subprocess.run(["systemctl", "--user", "stop",
                        "nakshatra-unconscious-worker@a.service",
                        "nakshatra-unconscious-worker@b.service"])
        stopped_live = True
        time.sleep(2)

        import serve_lifecycle as sl
        spec = sl.RosterWorkerSpec(model_id=MODEL, hidden_size=4096, num_layers=32,
                                   package_location=str(PKG), python_bin=sys.executable,
                                   scripts_dir=str(SCRIPTS), slice_dir=str(tmp),
                                   worker_env=WORKER_ENV)
        controller = sl.RosterWorkerController(spec, log=print)
        lc = sl.ChainLifecycle(controller, idle_grace_s=3.0, start_timeout_s=90.0, log=print)

        # AUTONOMOUS summon — begin() runs the planner + launches the self-provisioning workers.
        print("── lifecycle.begin(): autonomously summoning the from_roster chain ──")
        lc.begin()
        print(f"✓ chain ready (autonomously summoned); generated {controller.chain_yaml}")
        import yaml
        chain = yaml.safe_load(open(controller.chain_yaml))
        ids = [w["id"] for w in chain["workers"]]
        print(f"  workers: {ids}  (stranger firewalled out: {'opB-gpu' not in ids})")
        assert "opB-gpu" not in ids

        # a real token over the autonomously-summoned chain
        out = subprocess.run(
            [sys.executable, str(SCRIPTS / "client.py"), "--config", controller.chain_yaml,
             "--model-path", str(TOKENIZER), "--prompt", "The capital of France is",
             "-n", "12", "--use-streaming"],
            capture_output=True, text=True, timeout=180, env={**os.environ})
        gen = [l for l in out.stdout.splitlines() if l.startswith("[chain] full:")]
        print(f"  {gen[-1] if gen else '(no full line)'}  rc={out.returncode}")
        assert out.returncode == 0, out.stderr[-400:]
        lc.end()

        # scale-to-zero: reap and confirm the GPU is freed
        print("── reaping (scale-to-zero) ──")
        controller.stop()
        time.sleep(2)
        still = subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout
        assert ":5560 " not in still and ":5561 " not in still, "workers did not reap"
        print("✓ workers reaped — ports 5560/5561 freed")
        print("\n✅ SLICE 4 (autonomous launch) PROVEN: ChainLifecycle summoned the from_roster "
              "chain on demand (firewall-gated, self-provisioned), served a real token, and reaped it.")
    finally:
        try:
            if lc is not None:
                lc.controller.stop()
        except Exception:
            pass
        if stopped_live:
            subprocess.run(["systemctl", "--user", "start",
                            "nakshatra-unconscious-worker@a.service",
                            "nakshatra-unconscious-worker@b.service"])
            print("✓ restored the live unconscious workers")
        for k in ("MESH_PEERS", "MESH_MODELS"):
            os.environ.pop(k, None)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
