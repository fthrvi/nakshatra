# Nysa — Session Log

**Date:** 2026-05-14 (started 2026-05-13, continued into 14)
**Purpose:** Conversation log to load into the next session as context. Read this before picking up Sthambha work.

---

## Topic arc

A free-form architectural brainstorm about how Nakshatra workers communicate, why the current shape is the way it is, and whether it's time to start building Sthambha. Three big ideas surfaced and got captured into design docs. One strategic decision (start Sthambha now, planner first) is on the table.

---

## Conversation flow

### Exchange 1 — "Can we hook all the GPUs as one rather than running separate processes? And what about a virtual environment inside each GPU, combined as one?"

**My read:** two distinct ideas bundled together.

- **"One logical GPU"** exists as NVLink (NVIDIA) / Infinity Fabric (AMD) *inside a chassis*, and as InfiniBand + GPUDirect RDMA *across machines in an HPC fabric*. Over Tailscale/Ethernet on this cluster, you can't — GPU HBM bandwidth is ~1 TB/s, the network is ~1 Gb/s, that's a 1000× gap. `rCUDA` and SSI hypervisors fake it but collapse under inference's memory bandwidth demands.
- **"Combined VE / unified container"** is what Ray, PyTorch Distributed, DeepSpeed, vLLM-distributed, Horovod do. They give you a unified Python process that *appears* to span GPUs. But under the hood they still split tensors and ship bytes between machines — it's an orchestration layer, not a memory fabric. You haven't changed *what* moves; only *who calls the move*.
- **Why Nakshatra is shaped the way it is:** the only thing cheap enough to move per token over Ethernet is the hidden-state activation (a few MB). Weights and KV cache must stay pinned. That forces pipeline-parallel — each machine owns a layer range, activations stream forward, KV stays local. Tensor-parallel (split a single layer across GPUs) needs NVLink-class bandwidth = won't work on this cluster.
- **Future opening:** if a node arrives with multiple GPUs joined by NVLink/PCIe, you can do tensor-parallel within the node and pipeline-parallel across nodes. That's a Sthambha planner concern.

### Exchange 2 — "Send this Sthambha-related idea to its memory or idea md file"

**Action taken:** appended new §8 "Intra-node fast-link aware planning (future)" to `~/sthambha/docs/layer-split-planner.md`. Renumbered the old §8 (Suggested next steps) to §9.

The new section captures: registry field for `intra_node_links`, two-level planning (outer pipeline-parallel across nodes, inner tensor-parallel within a fast-linked GPU group), `ChainSlot.tensor_parallel_group` extension, why deferred (today's cluster is mostly 1-GPU-per-node so it'd be dead code), and a cross-link to `[[moe_cpu_beats_metal_finding]]` because both are "the planner needs to look beyond raw VRAM."

### Exchange 3 — "I don't know how I want to make the tunnel environment, maybe a virtual simulation, maybe a tunnel where inference is light-speed fast"

**My read:** pushed back on the "light-speed" framing — electrons in copper / photons in fiber are already at ~2/3 c, propagation at LAN distances is <1ms. The actual bottleneck is **per-token overhead**: gRPC handshake, serialization, kernel context switches, GPU sync waits. Goal isn't "make the wire faster" — it's "make the wire feel invisible."

- **"Tunnel environment"** — Tailscale already IS a tunnel. What might be on top of it: kernel-bypass networking (DPDK/io_uring/RDMA), speculative decoding (biggest practical win, no new networking required), activation prefetching with pipeline overlap.
- **"Virtual simulation"** — the useful version is a *cluster simulator* that mocks workers so you can test chain plans, drift, and the planner without booting real GPUs. Cheap to build, big payoff for design iteration. Distinct from the physics-defying version.

### Exchange 4 — "Can't we check the path of a token, and when it's outputted from the model, our network creates a wormhole that teleports it to the other side? If it's our network we made the rules"

**My read:** the wormhole framing is a *good* intuition pump. Made real: **bytes leave Worker A's GPU memory and appear in Worker B's GPU memory with no host RAM hop, no Python, no protobuf, no TLS, no TCP, no per-token negotiation.** That's GPUDirect RDMA + NCCL/RCCL/UCX + a fixed-schema raw protocol. The receiving GPU kernel is *already running, waiting on that memory address*. About as close to teleportation as physics allows.

**Two wormholes** in pipeline-parallel autoregressive inference:
- **Forward:** hidden state activation, ~MB, worker_i → worker_{i+1}, every step (the bandwidth-critical one)
- **Feedback:** sampled token, 4 bytes, worker_K → worker_1, every generated token (the latency-critical one)

**Cluster-specific catches I named:**
- Tailscale is a userspace overlay → blocks true GPUDirect RDMA (RDMA wants kernel-level or raw Ethernet)
- Metal on Macs has no GPUDirect equivalent → the Macs will always pay the host-bounce cost cross-machine

**Achievable wins on today's cluster, ranked:** (1) pre-launched receiver kernels + pinned ring buffers, (2) drop protobuf for hot path, (3) long-lived bidirectional gRPC stream, (4) speculative decoding (orthogonal but biggest perceived-latency win).

### Exchange 5 — "We don't even have to use the Tailnet, we make this in our network. The only thing is we need a process for computers to first join"

**My read:** exactly the right Sthambha-shaped instinct. The pillar already owns identity/trust; extending the join handshake to also negotiate **the transport** is the natural move.

**The join handshake (control plane):**
1. Bootstrap connect (slow path, e.g., HTTPS to pillar — used only at join)
2. Capability declaration (transports supported, NIC info, GPU-to-NIC path, public endpoint, kernel-bypass support)
3. Identity issuance + per-pair shared symmetric keys (pillar distributes, peers don't negotiate keys with each other)
4. Transport negotiation (pick highest-ranked mutual transport, capped by topology mode)
5. Neighbor assignment (forward + backward, from chain plan)
6. Direct dial (peer opens raw socket to neighbors, pre-allocates pinned GPU ring buffer)

**After step 6, pillar is OUT of the data path.** Same shape as routers — control plane negotiates, data plane doesn't ask permission again.

**Three topology modes** the fabric needs to handle:
- **Mode A — All-LAN:** drop Tailscale entirely, raw UDP / RoCE
- **Mode B — Mixed (today's reality):** lab Macs are one LAN island, prithvi is another; cross-internet hop needs *something* — could be **bare WireGuard keyed by the pillar** (slimmer than Tailscale, no NAT-traversal/coordination layer)
- **Mode C — Public:** future Nakshatra opens to untrusted peers, encryption mandatory

**Prior art** so we don't reinvent: libp2p (Petals uses this), NCCL/RCCL bootstrap, UCX, Ray + plasma store, bare WireGuard.

### Exchange 6 — "Yes start it at ~/sthambha/docs/network-fabric.md"

**Action taken:** wrote `~/sthambha/docs/network-fabric.md` (11 sections):
1. Problem
2. The wormhole, named precisely (forward + feedback)
3. Topology reality check (Modes A/B/C)
4. The join handshake (control plane)
5. The data plane (Nakshatra-owned, designed against this fabric)
6. API surface (pillar endpoints + worker `FabricBackend`)
7. Integration with the rest of the stack
8. Open questions (7 items)
9. Prior art
10. Suggested next steps
11. What this is NOT

Also created memory pointer `~/.claude/.../memory/sthambha_network_fabric_sketch.md` and added line to `MEMORY.md` directly under the planner-sketch line.

The doc explicitly says: **build planner first, fabric second.** Fabric is multi-week and gated on hardware (RDMA NICs) we don't have. Planner is ~2-3 days for v0.

### Exchange 7 — "Do you think it's time to start working on Sthambha before we fix the issues here on Nakshatra?"

**My read: yes, start Sthambha — specifically the planner, not the fabric.**

**Why now is right:**
- v0.5 design-lock just landed (commit e78cc18). Natural milestone boundary.
- Two big Sthambha sketches are warm in your head (planner, fabric). Design rot starts the moment you walk away.
- The Nakshatra issues from memory are mostly **hardware-determined ceilings, not architectural bugs**: Metal non-determinism (Metal property, can't fix in Nakshatra), prithvi ROCm hot-restart wedge (needs reboot, no software fix), MoE cross-machine coherence drift (model-class limitation). Pouring more Nakshatra effort into these doesn't change the ceiling.
- The actual operator pain is **manual cluster bring-up** (hand-editing YAML, scp'ing slices). That's exactly what the planner solves.

**The tradeoff I flagged for redirect:** Sthambha planner depends on Nakshatra's auto-fetch + pillar-served `/chain` being solid. Per Phase G/H/I/J memory those *are* shipped. But — see "Open thread" below — there's at least one fixable Nakshatra issue (`metal_last_buffer_assert_finding.md`, added to memory mid-conversation) that IS blocking 5-machine 70B all-GPU runs. That one is code-fixable (libggml-metal patch) and may deserve to land before Sthambha pulls focus.

**Concrete recommendation:** start the planner per its own §10 next-steps. ~2-3 days for v0.

I asked which Nakshatra issues you had in mind — that's the open question this session ended on.

---

## Artifacts created in this session

| File | Status | Purpose |
|---|---|---|
| `~/sthambha/docs/layer-split-planner.md` | UPDATED — added §8, renumbered §9 | Intra-node fast-link aware planning (future extension) |
| `~/sthambha/docs/network-fabric.md` | CREATED (11 sections) | Drop-Tailscale-from-data-path design — pillar-brokered join handshake + per-peer transport negotiation + raw UDP / RDMA wormholes |
| `~/.claude/.../memory/sthambha_network_fabric_sketch.md` | CREATED | Memory pointer to the fabric doc |
| `MEMORY.md` (memory index) | UPDATED — added fabric line below planner line | Index entry for the fabric memory pointer |
| `nysa.md` (this file) | CREATED | Session log for next-session handoff |

No code was written. All design / docs work.

---

## Strategic decision pending your call

**Question:** start Sthambha planner now, or land the Nakshatra Metal-last buffer-assert fix first?

- **Start Sthambha planner:** removes operator pain, ~2-3 days, doesn't depend on the Metal-last fix (planner is L3, runs on Pi/pillar host).
- **Fix Metal-last first:** unlocks 5-machine 70B all-GPU runs, which is the natural test target for the planner once it ships. Per `metal_last_buffer_assert_finding.md`, fix is "patch/bump libggml-metal" — unknown effort, could be small or large depending on whether it's a simple alignment fix or a deeper Apple-side bug.

My read: if the Metal-last fix is small (a day or less), do it first so the planner has a real 5-machine target to validate against. If it's an unknown-depth bug, start the planner in parallel and let the bishwa-last-on-CPU workaround stay in place.

You have better signal than me on the bug depth — that's why I'm flagging the decision rather than picking it.

---

## Open threads to pick up next session

1. **The Sthambha-vs-Nakshatra prioritization question above.** Which Nakshatra issues did you have in mind when you asked? I was mostly thinking of the hardware-determined ceilings; the metal-last buffer assert (added to memory during this conversation) is the one fixable bug I'm aware of.

2. **If starting the planner:** §10 of `~/sthambha/docs/layer-split-planner.md` is the resume-here checklist. First step is locking/amending §7 open questions before writing code.

3. **If fixing Metal-last buffer assert first:** see `docs/findings/metal-last-buffer-set-tensor.md` in nakshatra (per the memory entry). Issue is `inp_out_ids` not page-aligned for Apple's `newBufferWithBytesNoCopy`.

4. **The two wormhole concepts (§2 of network-fabric.md)** — forward vs feedback — are worth keeping in mind even before fabric work starts. They affect how the planner thinks about chain topology (e.g., placing the last worker close to the first to minimize feedback-wormhole RTT).

5. **No commits made this session** beyond what was already on `main` at start (e78cc18 "v0.5 design lock"). The new untracked file from the start of session (`scripts/cluster_l3370b_5machine.yaml`) is still untracked — wasn't touched.

---

## Relevant context already in memory (skim if helpful)

- `[[three_project_architecture]]` — Sthambha (L3) + Nakshatra (L2) + Prithvi (L4); read before any L3 work
- `[[pillar_soul_backup]]` — pillar's dual purpose (L3 + Prithvi resurrection layer)
- `[[sthambha_planner_sketch]]` — sister doc to the fabric work
- `[[sthambha_network_fabric_sketch]]` — the doc just created
- `[[phase_GHIJ_2026-05-08]]` — pillar-served /chain plan + gap detection live; auto-fetch shipped
- `[[metal_last_buffer_assert_finding]]` — the one fixable Nakshatra bug currently blocking 5-machine 70B all-GPU
- `[[moe_cross_machine_degrades_coherence]]` — multi-Mac MoE produces garbage, dense models survive
- `[[metal_nondeterminism_finding]]` — Metal is non-deterministic; v0.5 §7 "bit-identical" is unsatisfiable on Metal
