# Payment-bound worker admission for Nakshatra, v2

**Status:** design only, not implemented

**Date:** 2026-10-15

**Supersedes:** v1 of this document

**Scope:** close the payment-attribution blocker without changing genuine
roster-peer economics, and ship the required peer-session isolation in the same
worker release

This revision incorporates every finding in `~/wt/spendauth/review.md` and
`~/wt/rv2-ccall.md`. The central change is deliberately simpler than v1:
authorization moves money into a real internal escrow account, and the worker
learns the authoritative scope only by claiming an opaque token online through
the SPKI-pinned ledger front. There is no offline spend credential.

## 1. Decisions (Biswa, 2026-10-15)

1. **Hold representation:** each hold is an internal ledger account named
   `__spend_hold__:<hold_id>`. Authorization transfers the maximum billable
   amount payer -> hold, settlement transfers actual usage hold -> worker, and
   release transfers unused value hold -> payer. The design adds no
   `active_reserved` calculation or edits to unrelated debit paths.
2. **Credential:** the gateway receives an opaque, cryptographically random
   online claim token. A worker must claim it online through the SPKI-pinned
   public front and durably persist the ledger's response before GPU or KV
   work. The response, not the token, supplies the payer, immutable worker
   account snapshot, `billable` flag, caps, price, session, model, span, and
   deadlines. There is no ledger-signed offline authorization and no use of the
   hub node key as spend authority.
3. **Paid-v1 transport:** paid sessions use direct consumer -> stage calls
   only. A worker rejects `chain`, `next_server`, `pushed`, `rpc_push`, and
   fabric bridging for a paid session. The official client cannot weaken this
   worker-side rule.
4. **Expiry:** execution ends 15 minutes after authorization. Unclaimed entries
   are releasable then; claimed entries retain their remaining escrow and may
   settle for a further 24 hours. The gateway releases output only when the
   hold is exactly `state=settled` and every stage actually used by the run has
   a final receipt.
5. **Prompt pricing:** v1 charges prompt token-layers and generated
   token-layers. Authorization escrows the worst case for both.
6. **Earnings:** the seven-day maturity clock starts only when the hold reaches
   a terminal state. Withdrawal is permitted strictly after seven full days.
   Immature or disputable paid earnings cannot fund another paid hold.
   Disputes are accepted through maturity. Withdrawals and S6 reconciliation
   consume one shared economic-receipt view.
7. **Peer exemption:** genuine peer-originated work remains exempt. A paid
   consumer session forwarded by a peer is never exempt: paid origin is
   transitive. Paid v1 removes the ambiguity by forbidding forwarding.
8. **Rollout:** consumer identity resolution and spend enforcement ship in one
   atomic, default-off worker release. Readiness refuses consumer admission
   unless the ledger URL, SPKI pin, durable store, and claim capability all
   pass. The canary uses a real consumer-role key. Rollback disables consumer
   resolution first.
9. **Aggregate receipt:** the gateway's aggregate receipt is audit-only.
   Legacy `/settle` rejects v2, `hold_id`, and embedded usage receipts. Only a
   strict, single-worker receipt submitted to `/spend/settle` moves balances.
10. **Final receipt:** exactly one zero-delta `nonfinal -> final` transition is
    allowed. `receipt_id` is a domain-separated hash over the receipt fields
    excluding `receipt_id` and the signature.
11. **Authority boundary:** hub authority is acceptable on devnet only.
    Mainnet requires a separate authority; its exact form is a later Biswa
    decision.
12. **Session isolation:** the peer-safe ownership, idempotency, KV-residency,
    stream-lifetime, and truncate-capability fixes in section 7 ship with the
    payment worker release, not as an earlier consumer-enabling change.

## 2. Required properties and non-goals

The implementation must provide all of the following:

- No consumer call reaches KV state or the daemon until that worker has an
  online-claimed, locally durable hold entry.
- Authorization is collateralized immediately by a zero-sum transfer. A later
  withdrawal, dispute, transfer, legacy settlement, or concurrent hold cannot
  consume the escrow.
- A claim is bound to the authenticated payer key, one session, one model, one
  worker key and immutable account snapshot, one layer span, one billing mode,
  and bounded prompt/output cost.
- Every accepted economic delta is exactly one hold debit and, when billable,
  one worker credit of the same amount. Replays cannot produce another delta.
- A worker can finish and be paid after a ledger partition only if it claimed
  before the execution deadline and settles by the settlement deadline.
- No successful strict response is released while a used stage is missing a
  final receipt or the hold is `open`, `closing`, `expired`, `cancelled`, or
  `disputed`.
- Existing peer-originated calls remain economically exempt, but they receive
  the stronger session-isolation semantics in section 7.
- Paid and legacy receipt namespaces cannot overlap.

This is economic authorization, not proof of useful computation. A compromised
worker key can lie about usage up to its entry cap. Immutable attribution,
bounded escrow, the dispute window, and delayed withdrawal contain that fraud;
TEE or verifiable inference is a later system.

Paid v1 supports greedy direct execution only. Consumer speculative/all-logits,
EAGLE, async speculative execution, worker push, and fabric execution are out
of scope. Stateful consumer `Forward` is also out of scope: paid v1 uses direct
`Inference` streams; a consumer `Forward` is accepted only if it is explicitly
stateless/cold, directly claimed, and billed under the same rules. Peers retain
supported modes subject to section 7.

## 3. Trust, identity, and authority

### 3.1 Existing transport and caller authentication

- Ledger HTTP requests continue to use `Sthambha-Ed25519` envelopes. The
  authorize signer is the payer; the claim and normal settle signer is the
  worker. A payer may relay an unmodified worker-signed usage receipt.
- All spend routes are reached through the existing HTTPS public front with a
  configured SPKI pin. Workers fail closed on URL, TLS, SPKI, response-shape,
  or signer-authentication failure.
- gRPC requests continue to prove the caller node key. The worker compares the
  authenticated key to `payer_keyid` returned by claim; a caller-supplied payer
  or account name has no authority.
- Worker identities are full lowercase 32-byte Ed25519 public keys. Labels and
  truncated node IDs are display-only.

### 3.2 What the opaque token does and does not prove

`POST /spend/authorize` returns at least 256 bits of random, base64url-encoded
token material. The token has no parseable claims and no signature. Possession
alone cannot start work: `/spend/claim` also authenticates the listed worker,
checks the hold and deadlines, and atomically claims that worker entry.

The claim response is authoritative because it arrives over the SPKI-pinned
front. Before touching KV or the daemon, the worker commits the complete
response and its response digest to its uid-owned durable admission database.
At minimum the persisted response contains:

- `hold_id`, token digest, claim instance, and worker key;
- `payer_keyid` and `payer_account_at_authorize`;
- `worker_account_at_authorize` and `billable`;
- session, model, billing mode, exact layer span, and price multiplier;
- prompt-token, output-token, token-layer, and cost caps;
- `execution_deadline` and `settlement_deadline`;
- the ledger response schema version and server time.

No Nakshatra node key signs spend authorization. Worker node keys still sign
their own usage receipts, which is a bounded earning claim rather than power to
create or enlarge a hold. On devnet the hub still operates the ledger database
and front, so it is the economic authority in fact. Mainnet must move that
authority to a distinct principal, multisig, or on-chain rule after Biswa's
later decision; changing only a signing key would not remove database trust.

### 3.3 Peer versus paid origin

Identity resolution returns an authenticated key and its role set; code must
use `ROLE_CONSUMER`, never a literal `"consumer"`. A peer call is exempt only
when it begins as genuine peer work. Once a session is marked paid, that origin
is sticky across every internal representation and cannot be cleared by a peer
hop. Paid v1 forbids such hops entirely and workers reject any forwarding field
or pushed/fabric entry point before daemon work.

Role overlap does not erase paid origin. Presenting a claim token, reusing a
paid session ID, or carrying paid-origin context selects the paid path even if
the caller key is also in the peer roster. A future forwarded paid protocol
must authenticate the original payer end-to-end and make every downstream
worker claim its own entry; it is not a compatible extension of v1.

## 4. Ledger model and lifecycle

### 4.1 Authorization is an escrow transfer

The authorize request contains no payer, account, price, cost, or `billable`
field. It contains a client nonce, session, model/mode, prompt and output caps,
and the planned workers with exact spans. Under one `BEGIN IMMEDIATE`
transaction, the ledger:

1. derives `payer_keyid`, payer account, and tier from the verified request and
   one roster snapshot;
2. resolves every worker key and snapshots
   `worker_account_at_authorize`;
3. sets `billable=false` exactly when policy makes that stage free (initially,
   payer and worker account are the same), otherwise `true`;
4. computes every entry cap using section 5 and sums billable caps;
5. computes paid-hold capacity after excluding restricted immature/disputable
   earnings and enforces the payer's tier limit;
6. creates internal account `__spend_hold__:<hold_id>` and atomically transfers
   the total cap payer -> hold using the ordinary journal/zero-sum machinery;
7. inserts the hold and immutable worker-entry snapshots; and
8. returns the opaque token, retaining only its digest plus a short-lived,
   encrypted idempotency response so an identical retry can recover the same
   token without storing plaintext at rest.

`(payer_keyid, client_nonce)` and `(payer_keyid, session_id)` are unique. An
identical retry is idempotent; changing any request field conflicts. Internal
hold accounts are not addressable by balance, transfer, withdrawal, generic
settlement, or user APIs.

There is no metadata-only encumbrance and no `active_reserved`. Existing debit
paths do not need reservation awareness because the funds have already left
the payer. The only new capacity rule outside escrow movement is that a new
paid hold must subtract restricted paid earnings as described in section 6.
Concretely, authorization requires
`balance - restricted_paid_earnings + tier_limit >= hold_max_cost` before the
payer -> hold transfer.

### 4.2 Claims

The gateway passes the opaque token through an inherited file descriptor or a
mode-0600 file, never argv, logs, metrics, or an environment dump. Each planned
stage receives it on its first direct call.

Before GPU/KV work, a worker sends the token plus a durable random
`claim_instance` to `/spend/claim` using its signed HTTP identity. The ledger
hashes the token at rest, finds the hold, verifies the worker entry, payer,
state, and execution deadline, and atomically claims that entry. A retry with
the same worker and instance returns the same response; another instance is
`409`. The response includes all fields listed in section 3.2.

The worker first fsyncs the claim response and then creates the in-memory
session. A crash between ledger claim and local commit burns the session rather
than running unrecorded work. After restart, the worker retries durable receipts
but does not silently resurrect compute from an ambiguous session.

### 4.3 Deadlines and hold states

The ledger assigns:

```text
execution_deadline  = authorized_at + 15 minutes
settlement_deadline = execution_deadline + 24 hours
```

Policy may reduce the execution deadline but does not extend it per caller.
There is no silent renewal.

- Before `execution_deadline`, listed workers may claim and claimed workers may
  execute within cap.
- At `execution_deadline`, new claims and new GPU steps stop. Unclaimed or
  explicitly abandoned entry caps transfer hold -> payer. Every claimed entry's
  remaining cap stays in the hold.
- Through `settlement_deadline`, cumulative receipts for work completed by the
  execution deadline remain payable. Release requests cannot take claimed
  collateral ahead of those receipts.
- When each used/claimed entry is final and every unused entry is abandoned,
  actual unused escrow transfers hold -> payer and the hold becomes `settled`.
- At `settlement_deadline`, unresolved remaining escrow transfers hold ->
  payer, unresolved entries close, and the hold becomes `expired`. Late
  receipts are rejected. `expired` is never output-releasable.
- A wholly unclaimed hold may be cancelled and refunded immediately. After the
  first claim, payer release only records `release_requested`.

Transitions and transfers use the same transaction. Closed internal hold
accounts must have zero balance; a nonzero closed account is an invariant
failure, not a sweeper detail.

### 4.4 Used-stage finality and output release

The ledger treats an entry as used once claimed or once any usage receipt is
accepted. The gateway also retains the exact planned chain and marks entries it
never invoked as abandoned. For a successful generation it must see every
invoked stage in status with `final=true`, the exact worker key/span snapshot,
and a compatible terminal reason.

The strict gateway buffers all generated output. It releases that output only
if `/spend/status` reports exactly `state="settled"`, all used stages final, no
entry disputed, and the charged amount/schema match its plan. Any other state,
unknown shape, timeout, or missing stage produces a failure with no generated
text. A direct client may stream output, but disconnecting cannot reverse
already settled usage.

## 5. Metering, receipts, and settlement

### 5.1 Prompt and generation pricing

For each entry:

```text
layers = layer_end - layer_start
max_token_layers = (max_prompt_tokens + max_output_tokens) * layers
entry_max_cost = 0                                      if billable=false
entry_max_cost = max_token_layers * price_multiplier   if billable=true
hold_max_cost = sum(entry_max_cost)
```

Actual cumulative cost uses actual prompt and generated counts:

```text
actual_token_layers =
    (cumulative_prompt_tokens + cumulative_generated_tokens) * layers
actual_cost = actual_token_layers * price_multiplier
```

The first successful step records the actual prompt length exactly once.
Subsequent greedy steps add generated tokens only. A reset or replay cannot
charge another prompt under the same session; a different prompt digest under
the same step/session is rejected. Workers derive token counts from validated
request/tensor dimensions and reject over-cap, batched post-prefill, unsupported
mode, or arithmetic overflow before daemon work.

`billable=false` is part of the immutable claim response. Such an entry still
claims, meters, finalizes, and participates in gateway finality, but its cap and
economic deltas are zero. Settlement never re-resolves the roster or changes
the snapshot after authorization.

### 5.2 Direct-only worker admission

For a paid call, the worker rejects before KV/daemon access if any of the
following is present or selected: `chain`, `next_server`, `pushed`, `rpc_push`,
fabric bridging, an indirect downstream caller, speculative/all-logits, EAGLE,
async speculation, or stateful consumer `Forward`. The paid client opens one
authenticated direct stream to every stage and relays activations itself.

The first frame carries the opaque token and stable `session_id`; later frames
are authorized by the durable claimed-session record and authenticated stream.
Each unique successful step is journaled locally before its response is
released. Duplicate `(payer_keyid, session_id, step_id)` plus identical digest
may return an eligible cached response and adds no meter. Different bytes for
the same step ID are rejected.

### 5.3 Strict per-worker usage receipt

Each stage emits a cumulative object with a strict schema, including:

```json
{
  "v": 2,
  "kind": "nakshatra-worker-usage",
  "receipt_id": "sha256:<hex>",
  "hold_id": "...",
  "claim_response_sha256": "...",
  "payer_keyid": "<64 hex>",
  "payer_account_at_authorize": "...",
  "worker_keyid": "<64 hex>",
  "worker_account_at_authorize": "...",
  "billable": true,
  "session_id": "...",
  "model_id": "...",
  "billing_mode": "greedy-direct-v1",
  "layer_start": 0,
  "layer_end": 14,
  "sequence": 3,
  "cumulative_prompt_tokens": 127,
  "cumulative_generated_tokens": 37,
  "cumulative_token_layers": 2296,
  "cumulative_cost": 2296,
  "work_completed_at": 1792080060,
  "final": false,
  "termination": "checkpoint",
  "output_sha256": ""
}
```

Define `receipt_id` as:

```text
SHA-256(
  "nakshatra-spend-usage-receipt-v2\0" ||
  canonical_json(receipt fields excluding receipt_id and signature)
)
```

The worker signature covers the canonical object including the computed
`receipt_id`; the signature is transported separately. Unknown fields, floats,
booleans in integer positions, invalid bounds, and alternate canonicalizations
are rejected. The ledger recomputes every derived counter and cost.

Normally both sequence and cumulative counters increase. Exactly once per
entry, the ledger also permits `sequence = previous + 1` with unchanged
counters/cost when and only when the prior receipt is nonfinal and the new
receipt changes it to `final=true`. No receipt is accepted after final.

Receipts checkpoint at a bounded interval and finalize on normal close,
cancellation, daemon error, idle timeout, or execution deadline. They enter a
mode-0600 durable outbox before submission.

### 5.4 Economic settlement and namespace separation

Only `POST /spend/settle` accepts a v2 usage receipt, and each request contains
exactly one worker receipt plus its signature. In one transaction the ledger:

1. verifies the relayer and worker signature;
2. matches every immutable field to the claimed entry;
3. enforces sequence/finality and both deadlines;
4. recomputes cumulative prompt/output token-layers and cost;
5. calculates the new delta and proves cumulative charge is within entry cap;
6. transfers the delta `__spend_hold__:<hold_id>` ->
   `worker_account_at_authorize` when billable, or records zero otherwise;
7. inserts the strict receipt and paired journal linkage idempotently; and
8. records provisional earnings metadata for the shared economic view.

The unique economic key is `(hold_id, worker_keyid, sequence)` plus monotonic
cumulative state. Replaying an identical receipt returns its result; a different
receipt at the same sequence, a regression, or any post-final receipt conflicts.

Legacy `/settle` becomes strict v1: it rejects unknown fields and any request
with `v != 1`, `hold_id`, v2 kind, or embedded usage receipts. `/spend/settle`
rejects aggregate receipts and all legacy shapes. The gateway may produce a v2
aggregate run receipt for audit/export, but no ledger route accepts it for an
economic mutation. This prevents a per-worker settlement from being charged a
second time through the legacy endpoint.

## 6. Earnings maturity, disputes, and the shared economic view

Every paid usage delta creates an economic-receipt row containing its journal
links, `funded`, amount, worker account snapshot, `earned_at`, terminal anchor,
`matures_at`, and dispute state. Until the hold is terminal, the earning is
funded but provisional and fully restricted. When the hold first becomes
terminal (`settled`, `expired`, `cancelled`, or `disputed`), the ledger sets one
immutable `hold_terminal_at` and:

```text
matures_at = hold_terminal_at + 7 * 24 hours
```

Withdrawal is allowed only when `now > matures_at`, the row remains funded,
and it is not disputed or clawed back. A dispute is allowed while
`now <= matures_at`; winning disputes apply equal-and-opposite entries exactly
once and preserve receipt history. Thus there is no withdrawal/dispute race at
the boundary.

`economic_receipts` is one normalized database view (or one equivalently shared
query module) over legacy funded receipts and v2 spend usage. Both withdrawal
eligibility and S6 use this exact view and the same maturity/dispute predicates.
No second implementation of the seven-day rule is permitted.

When authorizing a new paid hold, the ledger computes the payer's restricted
paid earnings from this view and excludes them from paid-hold capacity.
Immature, still-disputable, unfunded, or disputed earnings therefore cannot be
re-escrowed for paid compute. No `active_reserved` mechanism is introduced.

## 7. Peer-safe session isolation shipped with payment

The payment release must include the peer isolation work below. These rules
apply even when the caller is an economically exempt peer, because peer bypass
must not imply KV or replay bypass.

### 7.1 Principal, idempotency, and truncate ownership

- The logical session principal is `(authenticated_keyid, session_id)`, never
  textual `session_id` alone. Session ownership, served-session state,
  idempotency, truncate authority, and receipt attribution use that tuple.
- Another key presenting the same session ID is refused without releasing or
  mutating the current owner.
- `TruncateKV` reaches the daemon only when authenticated key and session equal
  the active owner. Consumer stateful `Forward` remains denied.
- Add capability `truncate_session_owner`. New clients never issue an
  ownership-sensitive truncate to a worker lacking it. For an old client, a
  new worker may infer the session only when the authenticated peer is the sole
  active owner; otherwise it fails closed. A new-client field sent to an old
  worker being protobuf-preserved is not evidence of enforcement.

### 7.2 Cached peer-push streams cannot cross sessions

The current address-keyed bidi push stream must not survive into another
logical session. The required v2 behavior is to key the downstream stream by
the owning `(keyid, session_id)` and close it with that upstream session. A new
session creates a new downstream stream and performs a cold first step. Do not
silently transition an address-cached stream between principals.

Acceptance requires two consecutive sessions through a three-stage peer push
chain. Both must decode, the second must use a new downstream stream, and no
`session_id must remain stable` error may occur. Paid v1 still rejects push;
this test protects the exempt peer path.

### 7.3 KV residency is separate from idempotency

Track the daemon's resident KV as `(keyid, session_id, kv_epoch)` independently
from the response/idempotency cache. Every cold decode that replaces KV advances
the epoch. A cached response is eligible to continue stateful execution only
when its principal and recorded epoch still own resident KV.

The same-key A-B-A case is mandatory: after A0, a cold B0 replaces KV; an A0
retry must not return a cache hit that licenses A1 against B's KV. It returns a
typed `KV_RESIDENCY_LOST`/cold-replay requirement, and only a cold replay may
re-establish A. The no-intervening-session retry remains idempotent.

### 7.4 Handoffs avoid redundant or unsafe truncation

A cold `Inference` or stateless `Forward` lets its decode reset KV and does not
make a synchronous truncate round trip first. Cross-key stateful `Forward` is
refused rather than clearing KV and then decoding with `keep_kv=true` against
empty state. Both decisions use the residency epoch from section 7.3.

An alternating-peer regression must show no pre-truncate for cold calls, no
cross-principal state reuse, and no additional daemon mutation on rejection.

### 7.5 Landing boundaries from the peer review

- The hardened release-trust implementation is independent and may land on its
  own: `current` is confined to exactly `releases/<version>`, directory-FD and
  `O_NOFOLLOW` checks protect manifests, and versions older than installer state
  are refused.
- Do not cherry-pick reviewed mixed commit `e6bc4fb` wholesale; its simulated
  landing is conflicted and it depends on deferred consumer code. Port its final
  release-trust change separately. Build peer isolation and payment from the
  current worker base while preserving `_check_grpc_auth()` returning the
  verified key ID.
- Consumer roles, reader projection, consumer `Forward` policy, and consumer
  tests do not become live before the atomic payment/isolation release.
  Nakshatra `e28d831`, Sthambha `1fb1293`/`21eea35`, and consumer `/readers`
  distribution remain unshipped until that release passes its gate.
- Put peer-only regressions in the existing KV/push suites, not in a
  consumer-caller test file. Replace literal role comparisons with constants.

## 8. API, schema, and component changes

### 8.1 Neuron ledger

Add the following internal state:

- `spend_holds`: hold/token digest, payer snapshot, session/model/mode, caps,
  deadlines, state, release request, terminal time, and escrow account name;
- `spend_hold_workers`: worker key/account snapshot, span, `billable`, caps,
  claim instance/response digest, cumulative meters/cost, finality, and final
  receipt;
- `spend_usage_receipts`: the strict receipt, signature, accepted delta, and
  journal links, unique by ID and entry sequence;
- paid economic-receipt metadata feeding the shared view; and
- internal-account policy that makes `__spend_hold__:*` unreachable to generic
  user operations.

Core methods are `authorize_spend`, `claim_spend_worker`,
`settle_spend_usage`, `abandon_spend_worker`, `request_spend_release`,
`spend_status`, and `expire_spend_holds`. Every balance mutation remains under
the existing `_tx` zero-sum assertion. There is no `active_reserved` helper.

Routes are:

| Route | Authentication | Strict result |
|---|---|---|
| `POST /spend/authorize` | payer | opaque token, hold ID, caps, deadlines |
| `POST /spend/claim` | listed worker | authoritative immutable claim response |
| `POST /spend/settle` | worker; payer may relay intact receipt | accepted delta, cumulative charge, state |
| `POST /spend/abandon` | payer | marks an unclaimed/uninvoked entry unused |
| `POST /spend/release` | payer | immediate only when wholly unclaimed; otherwise request |
| `GET /spend/status?hold_id=...` | payer or listed worker | state, escrow, per-entry claim/finality/charge |

Unknown fields and unknown response shapes are errors. Token material is never
returned by status/history. Errors distinguish invalid authentication, scope,
conflict, execution expiry, settlement expiry, and dependency unavailability;
the gateway treats all unrecognized failures as closed.

Existing `/balance`, `/can_consume`, and `/history` may expose additive escrow,
restricted-earning, and available values. `can_consume` remains advisory only.
Legacy `/settle` gets the strict namespace checks in section 5.4.

### 8.2 S6 reconciliation extensions

S6 snapshots the new hold, entry, usage-receipt, economic-view, internal-account,
and linked journal rows. In addition to global zero-sum it verifies all six
review requirements:

1. every accepted usage delta has exactly one worker credit and exactly one
   attributable payer debit through that payer-funded hold (or an explicit
   zero billable delta), with no second direct payer debit;
2. journal amounts equal independently recomputed hold pricing;
3. cumulative entry charge never exceeds its immutable cap;
4. internal spend-hold balances equal all unreleased liabilities;
5. closed or expired holds have no orphan balance; and
6. v1 and v2 economic namespaces never overlap.

It also proves the shared economic view agrees with receipt funding,
terminal-time maturity, disputes, withdrawals, and journal links. A symmetric
double debit/credit must fail even though `SUM(balance)==0` still passes.

### 8.3 Nakshatra protocol and worker

Add bounded fields for `session_id`, `step_id`, and opaque `spend_claim_token`
to the first paid `Inference` frame and to the explicitly stateless paid
`Forward` request. `TruncateRequest` gains `session_id`; `InfoResponse` advertises
at least:

```text
spend_online_claim_v1
paid_direct_greedy_v1
prompt_token_layers_v1
usage_receipt_v2
truncate_session_owner
kv_residency_epoch_v1
```

Do not define an offline authorization protobuf. Old peers ignore additive
fields. Paid clients require the complete capability set; protobuf wire
compatibility alone is insufficient.

The worker adds one durable SQLite store for claimed response snapshots,
session/step digests, meter state, KV principal/epoch linkage, receipts, and the
settlement outbox. Its atomic default-off feature gate controls both consumer
resolution and enforcement. Readiness is false, and consumer identities are
not installed into the resolver, unless all of these pass:

- configured HTTPS ledger URL and successful SPKI validation;
- writable, recoverable, uid-owned durable store;
- authenticated claim-capability probe with the expected strict schema; and
- worker key/model/span configuration sufficient to match a claim response.

No flag combination may resolve a consumer but route it through peer/no-payment
handlers. Peer isolation is compiled and tested in the same release artifact.

### 8.4 Gateway, client, and audit receipt

The gateway/client extracts a read-only chain planner so the worker keys and
spans authorized are exactly those dialed. It tokenizes before authorization,
reserves worst-case prompt plus output work, transports the token through a
protected FD/file, opens direct calls to every stage, uses stable step IDs, and
never selects push/fabric for a paid session.

The gateway requests abandonment/release for entries not invoked, continues to
buffer output, and polls strict status through settlement. It releases output
only under section 4.4. Its aggregate receipt embeds exact worker receipts for
audit, export, and user evidence, but is visibly marked `economic=false` and is
never sent to either settle endpoint.

### 8.5 Trisul VPS nginx front

Expose only exact allowlisted paths for the six routes above under `/ledger`.
Each location restricts its HTTP method, requires the existing authorization
header, strips only `/ledger`, preserves the signed bare upstream path and
query, applies existing body/time/rate limits, and proxies only through the
loopback tunnel. Do not expose generic `/ledger/spend/`, `/balances`,
`/transfer`, internal account names, or an arbitrary ledger prefix.

Deployment remains default-off: locations may be installed before activation,
but upstream spend routing stays disabled/closed until the loopback service and
canary are ready. Workers and gateways use the same SPKI-pinned base.

## 9. Phased BUILD PLAN

Every slice below is independently testable and behaviorally default-off. The
order is intentional; a later repo may develop against fakes, but no consumer
admission is enabled until the final cross-repo gate.

### Phase N — Neuron

**N1. Escrow domain and pricing (default off).** Add internal hold accounts,
hold/entry tables, immutable account snapshots, prompt+output pricing, lifecycle
transitions, and pure transaction methods. Do not register HTTP routes.

Acceptance: concurrent authorizations cannot exceed eligible capacity; each
authorize/settle/release journal remains zero-sum; disputes and concurrent
withdrawals cannot undercollateralize a hold; payer-owned entries stay zero;
closed holds have zero internal balance.

**N2. Strict receipts, maturity, and S6 (default off).** Add single-worker v2
settlement, strict legacy-v1 rejection rules, terminal-anchored seven-day
maturity, the shared economic view, dispute deadlines, and all S6 extensions.

Acceptance: aggregate/v2 attempts on `/settle` fail; v1 attempts on
`/spend/settle` fail; receipt replay is idempotent; symmetric double accounting
is detected; withdrawal is refused at/before maturity and allowed strictly
after; restricted earnings cannot authorize a paid hold.

**N3. Loopback spend API (feature flag off).** Add authorize, claim, settle,
abandon, release, status, expiry sweeper, strict schema limits, and an
authenticated claim-capability readiness probe. No public front yet.

Acceptance: opaque-token claim binds the correct worker and returns/persists
the snapshot; wrong signer/instance/expiry fails; execution versus settlement
deadlines behave exactly as section 4.3; a 24-hour fault-clock test pays a
claimed receipt and rejects it after grace.

### Phase W — Nakshatra worker

**W0. Release-trust port (independent, default behavior unchanged).** Port the
already reviewed final release-trust implementation and its tests without the
mixed consumer commit.

Acceptance: outside-prefix, signed rollback, manifest-symlink, and swap-race
reproductions pass.

**W1. Atomic payment + peer-isolation worker release (single default-off
gate).** From the current worker base, add online claim persistence, direct-only
paid admission, prompt/output metering, receipts/outbox, readiness, role-safe
resolver wiring, and every section 7 isolation fix. Consumer resolution and
enforcement are one release and one switch; neither ships active alone.

Acceptance: every payment rejection has daemon-call count zero; paid
chain/`next_server`/push/fabric inputs fail; claim survives restart; meter and
outbox replay exactly once; two-session three-stage peer push passes; same-key
A-B-A returns typed cold replay after KV replacement; truncate capability
mixed-fleet tests pass; alternating cold peers add no pre-truncate; all existing
peer inference/signing tests remain green.

### Phase G — Nakshatra gateway/client

**G1. Exact planner and direct paid client (default off).** Share chain planning,
negotiate the complete capability set, authorize prompt+output caps, pass the
opaque token securely, and relay activations through direct consumer -> stage
calls only.

Acceptance: planned and dialed key/span sets are byte-equal; token is absent
from argv/env/logs; any capability or topology change abandons/releases without
compute; official-client attempts to select push/fabric are rejected locally
and would also be rejected by the worker.

**G2. Strict gateway finality and audit receipt (default off).** Add buffering,
abandon/release, status polling, all-used-stage final checks, and the audit-only
aggregate receipt.

Acceptance: output appears only for exact `settled`; `expired`, partial-final,
missing-stage, dispute, unknown schema, or status outage releases no text; the
aggregate cannot be submitted economically; direct custom clients are still
charged by workers without gateway receipt output.

### Phase T — Trisul VPS nginx front

**T1. Exact spend locations (deployed closed/default off).** Add only the six
allowlisted paths and existing tunnel/SPKI policy, with the upstream activation
kept closed until N3, W1, and G2 pass.

Acceptance: method matrix, missing-auth rejection, signed path/query
preservation, body/rate bounds, and 404s for generic spend, balances, transfer,
and internal accounts all pass from outside the VPS.

### Phase C — Cross-repo canary and cutover

**C1. Devnet canary (still fleet-default off).** Enable one model and worker set
for a real consumer-role key—not a peer impersonating a consumer. Exercise both
the gateway and a direct custom consumer, prompt-heavy runs, cancellation,
gateway death, ledger loss after claim, settlement during grace, replay,
dispute, and maturity with a fault clock.

Acceptance: packet/log evidence shows no unclaimed consumer request reaches a
daemon; every used stage is paid/final; output release matches strict status;
hold accounts reconcile to liabilities; S6 passes all six extensions; legacy
peer behavior remains unchanged.

**C2. Atomic enablement.** Enable the worker's consumer resolution/enforcement
gate only after its readiness is green, then enable gateway routing for that
capability-filtered worker set. Expand gradually. Remove the old
`can_consume -> run -> settle` consumer critical path only after the canary.

Rollback always disables consumer resolution/routing first. Only then may the
gateway, front, or payment code roll back. A rollback must never leave consumer
resolution active on a worker without spend enforcement.

## 10. Consolidated acceptance and adversarial matrix

The repo slices above must collectively cover these cases:

| Attack/failure | Required result |
|---|---|
| Consumer omits or invents a token | Online claim fails; zero daemon calls. |
| Consumer pays first worker and requests downstream push | First worker rejects forwarding before work. |
| Peer attempts to forward paid origin | Paid context remains sticky; v1 forwarding is rejected. |
| Prompt-heavy, one-output sessions | Prompt token-layers are escrowed and charged. |
| Ledger fails before claim | Admission fails closed. |
| Ledger fails after claim | Work stays within cap/deadline; durable receipt may settle during 24-hour grace. |
| Grace expires before final receipt | Remaining escrow returns to payer; hold is `expired`; no gateway output. |
| Gateway submits aggregate to legacy `/settle` | Strict schema rejection; no balance change. |
| Roster owner changes after authorization | Settlement credits the snapshotted worker account and keeps snapshotted billability. |
| Last checkpoint has all usage, then close | One next-sequence zero-delta final receipt succeeds exactly once. |
| Worker invents receipt IDs | Domain hash and entry cumulative cap prevent extra value. |
| Worker earns then immediately opens a hold | Restricted economic view excludes the immature earning. |
| Withdraw races dispute at seven days | Dispute allowed through maturity; withdrawal only strictly after it. |
| Symmetric double charge preserves zero-sum | S6 receipt/journal linkage fails. |
| Cached push stream handles a second peer session | New session gets a new downstream stream and succeeds. |
| Same peer key runs A-B-A | A cache cannot continue after B replaces KV; typed cold replay is required. |
| New/old truncate peers mix | Capability gate or sole-owner inference; never unenforced truncate. |
| Alternating peer cold calls | Decode resets KV with no pre-truncate; stateful cross-key reuse is refused. |

## 11. Operational defaults

- Execution deadline: 15 minutes.
- Claimed-entry settlement grace: 24 hours after execution deadline.
- Worker admission requires enough remaining execution time for at least one
  bounded step; the exact minimum is policy, not caller-controlled.
- Usage checkpoint: every 8 generated tokens or 15 seconds, whichever comes
  first, plus prompt completion and every terminal event.
- Prompt and output caps are bounded independently of protobuf integer ranges.
- Claim/session/outbox DB: uid-owned, mode 0600, WAL with durability settings
  tested under process kill; backups must not expose live tokens.
- Logs and metrics contain counts, states, and units—not prompts, activations,
  token IDs, claim tokens, complete payer IDs, or raw signed receipts.
- Token plaintext is never stored at rest: the ledger retains a digest for
  claims and may retain a short-lived encrypted authorize response solely for
  idempotent delivery. Its sealing key is not accepted as spend authority.
- Devnet may use hub-operated authority. Mainnet remains blocked on Biswa's
  separate-authority decision.

These defaults do not weaken existing holds when configuration changes. The
ledger's persisted claim response and immutable hold snapshots govern each
run. No consumer-reader publication or paid admission is safe before the
atomic W1 + G2 + N3 + T1 canary gate is complete.
