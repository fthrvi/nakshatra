"""Minimal repro: whole-model daemon, prefill (keep=0) then keep_kv decode.
Captures ALL daemon stderr to pinpoint the 'failed to initialize batch' cause."""
import subprocess, struct, sys, threading, time
GGUF = sys.argv[1]
BIN = "/home/prithviraj/llama.cpp/build-cuda/bin/llama-nakshatra-worker"
NCTX = sys.argv[2] if len(sys.argv) > 2 else "512"
p = subprocess.Popen([BIN, GGUF, "first", NCTX, "0", "99"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
ready = threading.Event()
def watch():
    for line in p.stderr:
        s = line.decode("utf-8", "replace").rstrip()
        print("  [ERR]", s, flush=True)
        if "ready" in s: ready.set()
threading.Thread(target=watch, daemon=True).start()
if not ready.wait(180): sys.exit("never ready")
time.sleep(0.5)
def req(cmd, n, sp, fl, payload=b""):
    p.stdin.write(struct.pack("<IIIII", cmd, n, sp, fl, len(payload)) + payload); p.stdin.flush()
    st = struct.unpack("<I", p.stdout.read(4))[0]; pl = struct.unpack("<I", p.stdout.read(4))[0]
    d = p.stdout.read(pl) if pl else b""
    return st, d
# EXACT measurement preamble: cmd=4 truncate-to-0, then 12-tok prefill keep=0
st, _ = req(4, 0, 0, 0, struct.pack("<I", 0))
print(f"cmd=4 truncate-0: st={st}", flush=True)
toks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]   # 12 tokens like the prompt
st, _ = req(1, len(toks), 0, 0, struct.pack(f"<{len(toks)}i", *toks))
print(f"PREFILL 12-tok keep=0: st={st}", flush=True)
time.sleep(0.3)
st, d = req(1, 1, len(toks), 0x1, struct.pack("<i", 130))   # keep_kv decode at pos 12
print(f"KEEP_KV decode at pos {len(toks)}: st={st}", flush=True)
time.sleep(0.3)
# cmd=5 on scratch seq 1 (the draft path), then a keep_kv on seq 0 again
st, d = req(5, len(toks), 0, 0, struct.pack(f"<{len(toks)}i", *toks))
print(f"cmd=5 (seq1 draft hidden): st={st} bytes={len(d)}", flush=True)
st, d = req(1, 1, len(toks) + 1, 0x1, struct.pack("<i", 140))
print(f"KEEP_KV after cmd5 at pos {len(toks)+1}: st={st}", flush=True)
p.terminate()
