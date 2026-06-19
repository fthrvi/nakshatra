# Branches (nakshatra)

Multi-session repo. Before branching/merging, read this. Name `sector/short-task`.
On create add a row (status=active, owner). On merge flip to `merged→<sha>`.

| branch | sector | description | base | status | owner |
|---|---|---|---|---|---|
| inference/spec-decode-pipeline | inference | wire speculative decoding into the distributed decode loop (slice 1 of the shard-gap speed stack); flag-gated, default OFF | main | active | claude/trisul (specdec lane) |
| inference/rtt-topology-order | inference | RTT-aware pipeline ordering (speed-stack finding #11): order the chain by measured inter-node latency; pure module + planner seam, no GPU | main | active | claude/trisul (specdec lane) |
