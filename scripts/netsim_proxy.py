#!/usr/bin/env python3
"""
netsim_proxy — a userspace WAN simulator for decentralized-inference dev.

Injects controllable one-way LATENCY (and optional bandwidth cap) into a TCP
link, with NO sudo/tc/netem needed. Point a worker's chain coord at this proxy
to make a same-LAN box behave like a cross-continent node, so we can develop +
benchmark WAN algorithms (speculative decode, activation compression, async
pipelining) under real, repeatable conditions before deploying to distant boxes.

    python3 netsim_proxy.py --listen 127.0.0.1:5572 --upstream 10.0.0.227:5570 \
        --delay-ms 80 [--rate-mbit 50]

delay-ms is ONE-WAY, so a request->response round trip incurs ~2x (e.g. 80 -> ~160ms RTT).
"""
import argparse
import asyncio
import random
import time


async def _pump(reader, writer, delay_s, bytes_per_s, jitter_s=0.0):
    # Inject latency PER BURST (one delay per logical request/response turn), not
    # per TCP chunk — so payload SIZE doesn't change the latency (that's the whole
    # point: model propagation RTT, not bandwidth). A burst = data arriving after a
    # quiet gap; chunks within a burst stream at full speed once the "wire" is open.
    GAP = 0.004  # s of quiet that marks a new burst (a new turn over the link)
    last = 0.0
    loop = asyncio.get_event_loop()
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            now = loop.time()
            if delay_s > 0 and (now - last) > GAP:
                d = delay_s + (random.uniform(-jitter_s, jitter_s) if jitter_s else 0.0)
                await asyncio.sleep(max(0.0, d))    # one-way propagation (+jitter) this turn
            if bytes_per_s:
                await asyncio.sleep(len(data) / bytes_per_s)   # optional bandwidth cap
            writer.write(data)
            await writer.drain()
            last = loop.time()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="127.0.0.1:5572")
    ap.add_argument("--upstream", default="10.0.0.227:5570")
    ap.add_argument("--delay-ms", type=float, default=80.0, help="ONE-WAY delay (RTT ~= 2x)")
    ap.add_argument("--jitter-ms", type=float, default=0.0, help="ONE-WAY random jitter +/-")
    ap.add_argument("--rate-mbit", type=float, default=0.0, help="0 = unlimited")
    args = ap.parse_args()
    lh, lp = args.listen.split(":"); uh, up = args.upstream.split(":")
    delay_s = args.delay_ms / 1000.0
    jitter_s = args.jitter_ms / 1000.0
    bps = args.rate_mbit * 1e6 / 8 if args.rate_mbit else 0

    async def handle(cr, cw):
        try:
            ur, uw = await asyncio.open_connection(uh, int(up))
        except Exception:
            cw.close(); return
        await asyncio.gather(_pump(cr, uw, delay_s, bps, jitter_s),
                             _pump(ur, cw, delay_s, bps, jitter_s))

    async def run():
        srv = await asyncio.start_server(handle, lh, int(lp))
        print(f"[netsim] {args.listen} -> {args.upstream} | one-way {args.delay_ms}ms "
              f"(RTT~{2*args.delay_ms:.0f}ms){' | '+str(args.rate_mbit)+'mbit' if bps else ''}",
              flush=True)
        async with srv:
            await srv.serve_forever()
    asyncio.run(run())


if __name__ == "__main__":
    main()
