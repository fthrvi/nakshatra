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
