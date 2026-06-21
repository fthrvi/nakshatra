import subprocess, struct, sys, threading, time, glob

SLICE = sorted(glob.glob("/home/prithvi/.nakshatra/slices/*L0-16.gguf"))[0]
BIN = "/home/prithvi/llama.cpp/build/bin/llama-nakshatra-worker"
print("slice:", SLICE.split("/")[-1])

p = subprocess.Popen([BIN, SLICE, "first", "256", "0", "99"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

ready = threading.Event()
def watch_err():
    for line in p.stderr:
        s=line.decode("utf-8","replace").rstrip()
        if "ready" in s: 
            print("  [daemon]", s); ready.set()
        elif "error" in s.lower() or "fail" in s.lower():
            print("  [daemon-err]", s)
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

# INFO (cmd=3)
st, data = req(3, 0, 0, 0)
info = struct.unpack("<6i", data)
n_embd = info[2]
print(f"  INFO: status={st} n_embd={n_embd} n_vocab={info[5]}")

# EAGLE_HIDDEN (cmd=5) on 4 tokens, cold (keep_kv off)
N=4
toks = struct.pack("<4i", 1, 2, 3, 4)
st, data = req(5, N, 0, 0, toks)
rtype = struct.unpack("<I", data[:4])[0]
nfloats = (len(data)-4)//4
expect = N*3*n_embd
print(f"  cmd=5: status={st} result_type={rtype} floats={nfloats} expect={expect} match={nfloats==expect}")
import math
sample = struct.unpack("<5f", data[4:24])
print(f"  first 5 hidden floats: {[round(x,4) for x in sample]}  allzero={all(x==0 for x in sample)}")
print("  VERDICT:", "PASS — cmd=5 returns n_tokens*3*n_embd real hidden states" if (st==0 and rtype==3 and nfloats==expect and not all(x==0 for x in sample)) else "FAIL")
p.terminate()
