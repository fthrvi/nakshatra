# Branches (nakshatra)

Multi-session repo. Before branching/merging, read this. Name `sector/short-task`.
On create add a row (status=active, owner). On merge flip to `merged→<sha>`.

| branch | sector | description | base | status | owner |
|---|---|---|---|---|---|
| inference/spec-decode-pipeline | inference | wire speculative decoding into the distributed decode loop (slice 1 of the shard-gap speed stack); flag-gated, default OFF | main | merged→6a31658 | claude/trisul (specdec lane) |
| inference/spec-decode-speedup | inference | incremental draft KV (O(n²)→O(n) draft, 1.76× faster); spec-decode #6 speedup | main | merged→6a31658 | claude/trisul (specdec lane) |
| inference/rtt-topology-order | inference | RTT-aware pipeline ordering (speed-stack finding #11): order the chain by measured inter-node latency; pure module + planner seam, no GPU | main | merged→fcc012c | claude/trisul (specdec lane) |
| inference/run-receipts | inference | verifiable per-run receipts of a distributed inference (speed-stack finding #20): distinct workers + measured timing + output hash + engine sha; pure module + tests, no GPU | main | merged→fcc012c | claude/trisul (specdec lane) |
| inference/edge-supervision | inference | fail-fast per-edge error context + health (speed-stack #17): classify dropped/timeout/refused, structured EdgeError, per-edge health tracker (also feeds #11 RTT); pure module + tests | main | merged→fcc012c | claude/trisul (specdec lane) |
| inference/placement-real-data | inference | Virtual-Environment engine Phase 0→1: placement_feed (live-telemetry→placement.Node bridge) + placement.plan wired into serve_planner.plan_chain via place_fn + NKS_SMART_PLACEMENT fires through from_roster (route-whole on measured capacity) + signed (Sthambha-Ed25519) pillar telemetry fetch. 23 tests; flag default-OFF = no behavior change. | main | merged→f5f2e18 | claude/trisul (virtual-env lane) |
| inference/conscious-vram-reserve | inference | Reserve Prithvi's PINNED conscious slice from the placement pool: placement_feed.make_node subtracts a per-node conscious reserve (NKS_CONSCIOUS_NODE+NKS_CONSCIOUS_RESERVE_GB or NKS_VRAM_RESERVE_GB json) so smart placement never puts unconscious layers into the hub's conscious VRAM. Default 0 = no change. +5 tests (51 placement green). | main | active | claude/trisul (placement lane) |
| serving/openai-think-split | serving | /v1 think-tag split (reasoning_content) per-model + multi-address --bind (localhost+WG mesh exposure of the OpenAI facade); 8 new tests, 51 green | main | merged→218a5d3 | claude/trisul (market-leverage lane) |
| inference/async-pipelining | inference | Async pipelining (Shard's 2.94→16.6 tok/s technique, the ONE gap vs Shard): fill the pipeline bubble by keeping N verify-chunks in flight via speculative continuation + misprediction flush (reuses the daemon's start_pos/keep_kv KV-rewind). New self-contained scheduler `scripts/async_pipeline.py`, unit-proven output==sequential greedy + real 4/4 stage fill. NOT yet wired into client.py — gated NKS_ASYNC_PIPELINE, needs live multi-box KV-rewind verification (inference lane's step). | main | active | claude/trisul (async-pipeline lane) |
