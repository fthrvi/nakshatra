# Branches (nakshatra)

Multi-session repo. Before branching/merging, read this. Name `sector/short-task`.
On create add a row (status=active, owner). On merge flip to `merged→<sha>`.

| branch | sector | description | base | status | owner |
|---|---|---|---|---|---|
| inference/spec-decode-pipeline | inference | wire speculative decoding into the distributed decode loop (slice 1 of the shard-gap speed stack); flag-gated, default OFF | main | merged→dfef839 | claude/trisul (specdec lane) |
| inference/rtt-topology-order | inference | RTT-aware pipeline ordering (speed-stack finding #11): order the chain by measured inter-node latency; pure module + planner seam, no GPU | main | merged→fcc012c | claude/trisul (specdec lane) |
| inference/run-receipts | inference | verifiable per-run receipts of a distributed inference (speed-stack finding #20): distinct workers + measured timing + output hash + engine sha; pure module + tests, no GPU | main | merged→fcc012c | claude/trisul (specdec lane) |
| inference/edge-supervision | inference | fail-fast per-edge error context + health (speed-stack #17): classify dropped/timeout/refused, structured EdgeError, per-edge health tracker (also feeds #11 RTT); pure module + tests | main | merged→fcc012c | claude/trisul (specdec lane) |
