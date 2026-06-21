"""test_eagle_kv_isolation.py — the correctness property the live EAGLE swap needs.

The draft's cmd=5 (EAGLE_HIDDEN) and the serving/verify cmd=1 share ONE daemon.
cmd=5 must run on a scratch KV sequence (1) so it NEVER perturbs the verify KV
(seq 0). Proof: a keep_kv decode sequence on seq 0 must produce BYTE-IDENTICAL
output whether or not cmd=5 calls are interleaved between its steps. If cmd=5
leaked into seq 0 (clear/append), the interleaved run would diverge.
"""
import subprocess, struct, sys, threading, time, glob

SLICE = sorted(glob.glob("/home/prithvi/.nakshatra/slices/*L0-16.gguf"))[0]
BIN = "/home/prithvi/llama.cpp/build/bin/llama-nakshatra-worker"
print("slice:", SLICE.split("/")[-1])

p = subprocess.Popen([BIN, SLICE, "first", "256", "0", "99"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
ready = threading.Event()
def watch():
    for line in p.stderr:
        s = line.decode("utf-8", "replace").rstrip()
        if "ready" in s: ready.set()
threading.Thread(target=watch, daemon=True).start()
if not ready.wait(60): print("daemon never ready"); sys.exit(1)
time.sleep(0.5)


def req(cmd, n_tokens, start_pos, flags, payload=b""):
    hdr = struct.pack("<IIIII", cmd, n_tokens, start_pos, flags, len(payload))
    p.stdin.write(hdr + payload); p.stdin.flush()
    status = struct.unpack("<I", p.stdout.read(4))[0]
    pbytes = struct.unpack("<I", p.stdout.read(4))[0]
    data = p.stdout.read(pbytes) if pbytes else b""
    return status, data


KEEP = 0x1
PREFILL = [10, 20, 30, 40]
NEXT = [50]


def seq0_run(interleave_eagle: bool):
    # cold prefill on seq 0 (keep_kv off)
    st, _ = req(1, len(PREFILL), 0, 0, struct.pack(f"<{len(PREFILL)}i", *PREFILL))
    assert st == 0, "prefill failed"
    if interleave_eagle:
        # draft hidden on scratch seq 1 — must NOT touch seq 0
        st5, d5 = req(5, len(PREFILL), 0, 0, struct.pack(f"<{len(PREFILL)}i", *PREFILL))
        assert st5 == 0 and len(d5) > 4, "cmd=5 failed during interleave"
    # keep_kv decode of the next token at start_pos=len(PREFILL)
    st, d = req(1, len(NEXT), len(PREFILL), KEEP, struct.pack(f"<{len(NEXT)}i", *NEXT))
    assert st == 0, "keep_kv decode failed"
    return d


fails = []
clean = seq0_run(interleave_eagle=False)
# fresh daemon state for the interleaved run: cold prefill resets seq 0 anyway
interleaved = seq0_run(interleave_eagle=True)

identical = clean == interleaved
print(f"  seq-0 decode bytes: clean={len(clean)} interleaved={len(interleaved)} "
      f"identical={identical}")
if not identical:
    fails.append("seq-0 decode CHANGED when cmd=5 interleaved — KV leak!")

# sanity: cmd=5 itself returns a valid non-zero hidden3
st5, d5 = req(5, len(PREFILL), 0, 0, struct.pack(f"<{len(PREFILL)}i", *PREFILL))
rtype = struct.unpack("<I", d5[:4])[0] if len(d5) >= 4 else -1
nfloats = (len(d5) - 4) // 4
sample = struct.unpack("<5f", d5[4:24]) if len(d5) >= 24 else (0,)*5
ok5 = st5 == 0 and rtype == 3 and nfloats > 0 and not all(x == 0 for x in sample)
print(f"  cmd=5 hidden3: status={st5} rtype={rtype} floats={nfloats} "
      f"nonzero={not all(x==0 for x in sample)} -> {'OK' if ok5 else 'FAIL'}")
if not ok5: fails.append("cmd=5 hidden3 invalid")

p.terminate()
print()
if fails:
    print("VERDICT: FAIL"); [print("  -", f) for f in fails]; sys.exit(1)
print("VERDICT: PASS — cmd=5 (scratch seq 1) leaves the verify KV (seq 0) "
      "byte-identical; the live EAGLE draft swap is KV-safe.")
