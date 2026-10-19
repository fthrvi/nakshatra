# M2 node-side report — 2026-10-18

Implementation commit: `505343c` (`net: add Neuron task escrow`)

## What I built

- Added the opt-in `NAKSHATRA_ESCROW=neuron` backend. It signs the fixed-price task-hold contract with the node Ed25519 key, requires ledger enforce mode, requires pinned HTTPS, and reads the same ledger URL/SPKI configuration as the gateway (environment first, `gateway.env` fallback). The existing local `ledger` backend remains the default.
- Kept the signer cap check at post, but deferred the Neuron `/spend/task/authorize` call until assignment so the hold binds the winning contact card's `ledger_keyid`. The task hash is the idempotent `client_nonce`, and the encrypted node store durably records the authorization and returned hold id.
- Implemented worker pay-first verification on assignment and submit. An exact open hold must match task hash, reward amount, and the worker's own ledger key; missing, mismatched, refunded, or unreachable holds remain visibly unfunded and cannot be submitted.
- Wired accept to `/spend/task/pay`, reject/expiry to `/spend/task/refund`, and the existing reconciliation loop to retry pending settlement exactly once. Local state is not marked paid until the ledger reports paid. Task rendering exposes funded/unfunded/paid/refunded (and payment-pending) state.
- Added optional encrypted `ledger_keyid` to contact cards. `nak card set --paid` is permitted only on the Neuron backend and fills the key from the node identity; there is no manual key argument and no MCP mutation tool.
- Enabled proposal-only `agent.msg` budgets for `intent=offer|ask`, unit `credits`, and integer amounts `1..10^12`; they are displayed to receivers and never move funds.
- Added `nak tasks` as a list alias and `nak escrow status <task>`. Existing task CLI/MCP reward paths now render credit units and funding state; MCP refuses to describe an unfunded assignment as workable.
- Added a shared fake ledger with conservation checks plus two-node coverage for pay, refund, expiry, mismatch refusal, unpaid-card refusal, over-cap refusal, crash reconciliation, and ledger-down retry. Added signed-route/SPKI tests for the real Neuron adapter and encryption/budget matrices.
- Did not touch installer/updater code, `INBOX.md`, the real node, a signer socket, systemd, or a real ledger.

## Tests: before vs after

Same command (temporary HOME and required Sthambha PYTHONPATH), before: **135 passed, 1 failed**. After: **146 passed, 1 failed**. The one failure is identical before and after: `test_schema_migration_is_idempotent_on_a_forgejo_main_store` assumes `forgejo/main` has no M1 `agent_messages` table, but the current moving ref already contains it.

Expanded final M2 command, excluding only that known baseline assertion: **157 passed, 1 deselected** in 36.93s. Python compilation checks also passed for every changed runtime module.

## Open questions / remaining validation

- The Neuron server side was being built in parallel, so this branch was tested against the frozen contract and an injected contract-faithful fake ledger, not a live server. A two-node live canary against the completed server remains the integration/cutover check.
- The unrelated moving-`forgejo/main` migration test should eventually pin a historical pre-M1 commit (or build its legacy schema explicitly) so the full focused command is green independently of remote branch movement.
