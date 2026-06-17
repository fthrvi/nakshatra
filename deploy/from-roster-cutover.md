# Unconscious from_roster cutover (slice 4 capstone)

The live unconscious (`nakshatra-unconscious.service`, `prithvi-unconscious` on :11599 — what
`think_deeper` escalates to) is served via **from_roster**: the chain is generated from a roster at
request time (firewall-gated `serve_planner`), slices self-provisioned from a content-addressed
package (`PackageSlicer`), workers autonomously summoned + scale-to-zero (`RosterWorkerController`).

## Local operational state (under ~/.nakshatra, NOT git — carries on-disk paths)
- `unconscious-roster.tsv` — the hub's 2 self GPU slots (127.0.0.1:5540/5541), tier=self.
- `packages.yaml` — `prithvi-unconscious -> ~/.nakshatra/packages/dsr1-llama8b`.
- `serve_models.unconscious-roster.yaml` — the from_roster model entry (hidden_size 4096, num_layers 32).
- `packages/dsr1-llama8b/` — the content-addressed package (revision 13ba667a), built via
  `scripts/packaging/package_gguf.py <full.gguf> ~/.nakshatra/packages/dsr1-llama8b --model-id dsr1-llama8b`.

## Apply
Drop-in `~/.config/systemd/user/nakshatra-unconscious.service.d/70-from-roster.conf` (vendored here)
overrides MODELS + MESH_PEERS, clears the static `@a/@b` UNITS/PROBES, sets `NAKSHATRA_LIFECYCLE_ROSTER_*`.
`systemctl --user daemon-reload && systemctl --user restart nakshatra-unconscious.service`.

## Rollback (instant)
`rm ~/.config/systemd/user/nakshatra-unconscious.service.d/70-from-roster.conf`
`systemctl --user daemon-reload && systemctl --user restart nakshatra-unconscious.service`
→ reverts to the static `serve_models.unconscious.yaml` chain + the `@a/@b` units (both kept intact).
