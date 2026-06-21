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
import time


async def _pump(reader, writer, delay_s, bytes_per_s):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            if bytes_per_s:
                await asyncio.sleep(len(data) / bytes_per_s)  # crude bandwidth cap
            writer.write(data)
            await writer.drain()
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
    ap.add_argument("--rate-mbit", type=float, default=0.0, help="0 = unlimited")
    args = ap.parse_args()
    lh, lp = args.listen.split(":"); uh, up = args.upstream.split(":")
    delay_s = args.delay_ms / 1000.0
    bps = args.rate_mbit * 1e6 / 8 if args.rate_mbit else 0

    async def handle(cr, cw):
        try:
            ur, uw = await asyncio.open_connection(uh, int(up))
        except Exception:
            cw.close(); return
        await asyncio.gather(_pump(cr, uw, delay_s, bps), _pump(ur, cw, delay_s, bps))

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
