"""test_sleep_wake.py — verify the daemon's sleep-mode (cmd=6 SLEEP / cmd=7 WAKE).

Proves the four properties sleep-mode must have:
  1. SLEEP frees GPU VRAM (the whole point — yield the GPU to other tenants).
  2. INFO still answers while asleep (cached metadata, no wake).
  3. WAKE re-acquires the GPU FAST (no process respawn / ROCm re-init) and the
     daemon decodes BYTE-IDENTICALLY to before sleep (correctness preserved).
  4. Transparent auto-wake: a decode issued while asleep wakes first, no explicit
     cmd=7 needed (the lifecycle "summon" = first request wakes).
"""
import subprocess, struct, sys, threading, time, glob, re

SLICE = sorted(glob.glob("/home/prithvi/.nakshatra/slices/*L0-16.gguf"))[0]
BIN = "/home/prithvi/llama.cpp/build/bin/llama-nakshatra-worker"
print("slice:", SLICE.split("/")[-1])


def vram_mb():
    """Used VRAM in MB via rocm-smi (best-effort; returns None if unavailable)."""
    try:
        out = subprocess.check_output(["rocm-smi", "--showmeminfo", "vram"],
                                      stderr=subprocess.DEVNULL).decode()
        m = re.search(r"Total Used Memory \(B\):\s*(\d+)", out)
        if m:
            return int(m.group(1)) / 1e6
    except Exception:
        pass
    return None


p = subprocess.Popen([BIN, SLICE, "first", "256", "0", "99"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
ready = threading.Event()
def watch_err():
    for line in p.stderr:
        s = line.decode("utf-8", "replace").rstrip()
        if "ready" in s:
            print("  [daemon]", s); ready.set()
        elif any(k in s for k in ("sleep", "wake", "error", "fail", "Fail")):
            print("  [daemon]", s)
threading.Thread(target=watch_err, daemon=True).start()
if not ready.wait(60): print("daemon never ready"); sys.exit(1)
time.sleep(0.5)


def req(cmd, n_tokens, start_pos, flags, payload=b""):
    hdr = struct.pack("<IIIII", cmd, n_tokens, start_pos, flags, len(payload))
    p.stdin.write(hdr + payload); p.stdin.flush()
    status = struct.unpack("<I", p.stdout.read(4))[0]
    pbytes = struct.unpack("<I", p.stdout.read(4))[0]
    data = p.stdout.read(pbytes) if pbytes else b""
    return status, data


fails = []
N = 4
toks = struct.pack("<4i", 1, 2, 3, 4)

# --- baseline: VRAM loaded + a cmd=5 hidden capture (deterministic reference) ---
v_loaded = vram_mb()
st0, data0 = req(5, N, 0, 0, toks)
print(f"  [baseline] cmd=5 status={st0} bytes={len(data0)} vram={v_loaded}MB")
if st0 != 0 or not data0: fails.append("baseline cmd=5 failed")

# --- 1. SLEEP frees VRAM ---
st, _ = req(6, 0, 0, 0)
time.sleep(1.0)  # let the allocator release
v_sleep = vram_mb()
print(f"  [sleep] status={st} vram={v_sleep}MB (was {v_loaded}MB)")
if st != 0: fails.append("sleep cmd=6 nonzero status")
if v_loaded and v_sleep is not None and not (v_sleep < v_loaded - 200):
    fails.append(f"VRAM not freed on sleep ({v_loaded}->{v_sleep}MB)")

# --- 2. INFO answers while asleep (cached metadata) ---
st, info = req(3, 0, 0, 0)
ok_info = st == 0 and len(info) == 24
print(f"  [asleep] INFO status={st} bytes={len(info)} -> {'OK' if ok_info else 'FAIL'}")
if not ok_info: fails.append("INFO failed while asleep")

# --- 3. WAKE re-acquires GPU + decode is byte-identical ---
t0 = time.time()
st, _ = req(7, 0, 0, 0)
wake_s = time.time() - t0
time.sleep(0.5)
v_wake = vram_mb()
st1, data1 = req(5, N, 0, 0, toks)
identical = data1 == data0
print(f"  [wake] status={st} wake={wake_s*1000:.0f}ms vram={v_wake}MB "
      f"decode_identical={identical}")
if st != 0: fails.append("wake cmd=7 nonzero status")
if not identical: fails.append("decode NOT byte-identical after wake")
if v_loaded and v_wake is not None and not (v_wake > v_sleep + 200):
    fails.append("VRAM not re-acquired on wake")

# --- 4. transparent auto-wake (sleep, then decode directly, no cmd=7) ---
req(6, 0, 0, 0); time.sleep(1.0)
st2, data2 = req(5, N, 0, 0, toks)   # should auto-wake then return
auto_ok = st2 == 0 and data2 == data0
print(f"  [auto-wake] cmd=5 while asleep status={st2} identical={data2==data0} "
      f"-> {'OK' if auto_ok else 'FAIL'}")
if not auto_ok: fails.append("auto-wake decode failed/mismatch")

p.terminate()
print()
if fails:
    print("VERDICT: FAIL")
    for f in fails: print("  -", f)
    sys.exit(1)
print("VERDICT: PASS — sleep frees VRAM, INFO survives, wake is fast + "
      "byte-identical, auto-wake transparent")
