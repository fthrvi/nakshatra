# Broker-held Solana pay wallet (M4c part 2c)

Status: design for unification-lane review. Devnet only.

This note replaces the earlier idea that `custody=broker` merely gated a key that stayed on the
node. The wallet key moves into the existing approval broker as a third Ed25519 principal. There is
no unlock operation, unlock token, or signing session.

## 1. Goal and non-goals

The goal is to let the node renew its own delegate, grant a starter delegate, and request or execute
an escrow withdrawal without leaving the Solana wallet seed readable by the node user. Every wallet
signature requires one short-lived broker approval over one exact legacy Solana message. The broker,
not the caller, decodes that message, derives the approval text, and signs the same bytes it decoded.

This preserves the split already implemented by the payment design: the wallet is the money root,
while the hot node key opens channels and signs cumulative vouchers
(`docs/design/2026-10-26-noncustodial-payments.md:378-403`). The owner revisions remain controlling:
the user pays fees, withdrawals use request then delayed execute, and mainnet is not enabled
(`docs/design/2026-10-26-noncustodial-payments.md:452-476`).

Non-goals for this slice are mainnet, a generic Solana signer, session unlock, fee sponsorship,
moving the node key, changing the OpenAI-compatible service, or adding new on-chain instructions.
Deposits, delegate revoke, escrow close, escrow creation, and the wallet-authority fallback for old
channel-close requests are not silently admitted: they remain file-custody-only or fail closed until
their exact transaction templates receive a separate review.

## 2. Principal and one-time key import

Each node gets one principal named `nakshatra-wallet:<node>`. `<node>` is the canonical lowercase
node key id already known at install time. Its broker-owned wallet record pins:

- the principal and node public key;
- `wallet_id`, equal to the base58 wallet public key in `wallet.pub`;
- the wallet Ed25519 public key;
- the devnet genesis hash, deployed `nakshatra_pay` program id, classic SPL mint, and all ceiling
  values from the root-owned broker environment.

The dynamic principal must authenticate as itself before enqueue, as the release client currently
checks `nakshatra-release` before asking for an approval (`release/broker_signer.py:162-183`). It must
not share the person or release principal credential.

The staged broker installer gains `--import-nak-wallet`. It follows the existing transactional
person-key import, which prepares one principal addition before touching authority state and keeps a
rollback record (`/home/prithvi/trisul@forgejo/main:infra/pillar-agent/deploy/prithvi-broker/install.sh:1352-1405`).
The wallet import does the following:

1. Refuse unless the staged package digest, broker pre-state, source ownership/mode, `wallet.pub`,
   `custody=file`, node id, and broker policy are all valid. Derive the public key inside the trusted
   import helper and require a byte-for-byte match with `wallet.pub`; never print seed bytes.
2. Refuse a different pre-existing broker wallet key or registry row. An identical already-imported
   key is an idempotent resume, not a rotation.
3. Copy the existing raw 32-byte `wallet.ed25519` into broker-owned key storage, install the
   `nakshatra-wallet:<node>` principal credential and wallet policy, and self-test a fixed harmless
   message entirely inside the sealed import transaction.
4. Start the candidate broker and verify the public key, principal, decoder, and rollback path before
   retiring the source. No live Solana transaction is signed or sent during import.
5. Only after verification, atomically replace the custody marker with `broker`, remove the node
   user's seed file, and verify it is no longer readable through every installer-declared source.
   A failure before that boundary restores `custody=file` and the original source. A failure after it
   leaves the broker stopped or restores the complete old state; it never leaves two active copies.

After success, the wallet key material visible to the node is exactly `wallet.pub` plus
`custody=broker`. The existing public-only loader already supports that shape and returns no seed
for broker custody (`scripts/network/pay_wallet.py:289-316`). `nak pay backup` and `nak pay export`
remain unavailable: those commands intentionally load and disclose the file seed
(`scripts/network/nak.py:1230-1253`). Import is custody migration, not key generation or rotation.

The rehearsal harness adds mode `import-nak-wallet`; today it accepts only person and release modes
(`/home/prithvi/trisul@forgejo/main:infra/pillar-agent/deploy/prithvi-broker/rehearsal/run.sh:26-31`).
The new mode uses throwaway material in the network-disabled sealed container, proves source
retirement, public projection, idempotent resume, and full rollback, and never mounts the real home.
That matches the current harness boundary
(`/home/prithvi/trisul@forgejo/main:infra/pillar-agent/deploy/prithvi-broker/rehearsal/README.md:16-18,43-53`).

## 3. Sign request and response

The client first calls `/principal/v1`, then uses the existing enqueue/result flow. Its enqueue body is:

```json
{
  "what": "nakshatra.wallet.v1",
  "tier": "high",
  "review": {"description": "Solana payment transaction"},
  "exact_content": {
    "v": 1,
    "wallet_id": "<base58 wallet public key>",
    "message_b64": "<canonical base64 of raw legacy Solana message bytes>"
  },
  "ttl": 60
}
```

`review.description` is a fixed label and is never displayed as transaction meaning. The approval
screen is rendered from the decoded `exact_content.message_b64` and server-side wallet policy only.
The closed schema admits no caller summary, operation name, amount text, cluster label, account
label, or decoded-field overrides.

After status becomes `approved`, the client sends the same `request_id` and byte-identical
`exact_content` to `POST /wallet-sign/v1`. The response mirrors the person/release endpoints:

```json
{
  "request_id": "<16 lowercase hex>",
  "wallet_id": "<same wallet id>",
  "message_sha256": "<64 lowercase hex>",
  "signatures": ["<128 lowercase hex Ed25519 signature>"],
  "items_b64": ["<the exact message bytes>"]
}
```

The client rejects a changed request id, wallet id, digest, item count, message, or signature, and
verifies the signature with `wallet.pub` before assembling the transaction. Existing clients already
perform the equivalent echoed-item and local signature checks
(`scripts/network/person_custody.py:614-648`; `release/broker_signer.py:138-159,204-220`).

The broker stores the canonical exact-content digest at enqueue and consumes one approved request
at the sign endpoint. Its existing result path already distinguishes approved, declined, expired,
and consumed states and makes the result receipt one-shot
(`/home/prithvi/trisul@forgejo/main:infra/pillar-agent/pillar_signer_broker.py:1757-1783`).
For the wallet endpoint, the consumed record additionally stores `message_sha256`. Ed25519 signs the
raw message, not its text rendering; the digest is the approval/audit binding. TTL is 60 seconds and
is never extended. A retry may recover the result for the same request id, but a consumed request can
never sign a second message.

## 4. Decoder and policy

### 4.1 Decode exactly the bytes you sign

The pure decoder accepts `(raw_message, wallet_policy, cluster_view)` and returns a typed approval
plan or one refusal. It implements the legacy Solana message format directly, including canonical
short-vector encodings, header counts, ordered account keys, recent blockhash, compiled instruction
indexes, and instruction byte lengths. Versioned messages and address lookup tables are refused.
It rejects trailing bytes, non-canonical encodings, duplicate account keys, index overflow, privilege
escalation, and any mismatch between the header and reconstructed signer/writable sets.

There is one parse. The approval renderer consumes the typed plan; the signer receives the original
immutable byte string held by that plan. It is forbidden to rebuild a message from decoded fields.
Immediately before signing, the endpoint re-hashes and constant-time compares those held bytes with
the approved `message_sha256` and re-runs the decoder against current policy and cluster view.

The current builder compiles the fee payer first, records the required signer order in the message,
and signs `message.data` directly (`scripts/network/pay_chain.py:410-454,481-499`). The broker accepts
exactly one required signer, equal to both `wallet_id` and fee payer. Any additional or different
signer is refused.

### 4.2 Pinned program and exact allowlist

The deployed program id is `GLWqQsSqUsDkPvkc8rxcoQUyro7ZQd5denht21NtdTTo`
(`/home/prithvi/neuron/programs/nakshatra_pay/programs/nakshatra_pay/src/lib.rs:13`). Anchor instruction
data is discriminator plus the exact arguments; the program itself refuses trailing argument bytes
(`/home/prithvi/neuron/programs/nakshatra_pay/programs/nakshatra_pay/src/lib.rs:18-85`). All integer
fields below are little-endian.

| Allowed instruction | Exact data | Required ordered accounts and signers | Approval fields | Ceiling |
|---|---|---|---|---|
| `nakshatra_pay.delegate_set` (self ensure) | `8b10e667fea614a5` + delegate key (32) + `cap_total:u64` + `expires_at:i64` + `flags:u8` | wallet **signer,writable**; canonical config; canonical wallet escrow; canonical delegate PDA **writable**; System Program; canonical event authority; program id | wallet, own node delegate, cap in micro-USDC and USDC, expiry, flags, escrow, optional SOL top-up, message digest | `NAK_WALLET_MAX_DELEGATE_MICRO_USDC`; expiry horizon `NAK_WALLET_MAX_DELEGATE_TTL_S`; optional transfer ceiling below |
| `nakshatra_pay.delegate_set` (starter grant) | same discriminator/layout; `flags` must equal `OPEN_CHANNEL=1` | same account layout; delegate PDA must be derived from escrow and the decoded recipient node key | inviter wallet, recipient node, newly granted remaining capacity, resulting total cap, expiry, optional SOL top-up, escrow, digest | grant delta `NAK_WALLET_MAX_STARTER_MICRO_USDC`; expiry horizon `NAK_WALLET_MAX_STARTER_TTL_S`; transfer ceiling below |
| `nakshatra_pay.escrow_withdraw_request` | `f6f379c6f277d27a` + `amount:u64` | wallet **signer**; canonical config; canonical wallet escrow **writable**; canonical vault; event authority; program id | wallet, amount in micro-USDC and USDC, escrow, vault, earliest execute time from trusted chain state, digest | `NAK_WALLET_MAX_WITHDRAW_MICRO_USDC` |
| `nakshatra_pay.escrow_withdraw_execute` | `ac5dcf14503aa8f1` only | wallet **signer**; canonical config; pinned mint; canonical escrow, vault and wallet ATA **writable**; classic Token Program; event authority; program id | wallet, pending amount and unlock time from trusted devnet state, escrow, vault, destination ATA, digest | pending amount must be at or below `NAK_WALLET_MAX_WITHDRAW_MICRO_USDC` |
| System Program transfer | program `11111111111111111111111111111111`; data `2:u32` + `lamports:u64` | wallet **signer,writable**; decoded delegate node **writable** | destination node, lamports and SOL, whether self top-up or starter top-up | `NAK_WALLET_MAX_SELF_SOL_LAMPORTS` or `NAK_WALLET_MAX_STARTER_SOL_LAMPORTS` |
| Associated Token Account create-idempotent | program `ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL`; data exactly `01` | wallet **signer,writable** payer; canonical wallet ATA **writable**; wallet owner; pinned mint; System Program; classic Token Program | wallet, mint, canonical ATA, possible rent charge | no token amount; allowed once only in the execute template |

The account order and argument layouts above match the current Python builders
(`scripts/network/pay_chain.py:219-231,257-290`) and the program account constraints
(`/home/prithvi/neuron/programs/nakshatra_pay/programs/nakshatra_pay/src/instructions.rs:418-537,619-698`).
The decoder independently derives every PDA and ATA; it never trusts an account label from the
caller.

“Starter grant” is not a fourth on-chain discriminator. It is the closed template consisting of a
starter-form `delegate_set`, optionally followed by one System transfer to that same decoded
delegate key. The current starter builder uses exactly that order
(`scripts/network/nakd.py:1408-1443`). Self ensure allows an optional System transfer to the
principal's own node followed by a self-form `delegate_set`; a transfer alone is allowed when the
delegate needs no renewal. Current ensure can create the escrow too
(`scripts/network/nak.py:1036-1102`), but `escrow_init` is intentionally not in this allowlist; import
therefore requires the escrow to exist until question 1 below is resolved.

Withdraw request is exactly one instruction. Withdraw execute is exactly ATA create-idempotent then
execute, matching the current builder (`scripts/network/pay_chain.py:266-276`). No current payment
builder emits Compute Budget instructions; there is no Compute Budget instruction constructor or
program id in `pay_chain.py` (`scripts/network/pay_chain.py:27-35,219-388`). Therefore Compute Budget
is refused in this revision. If a builder later emits it, its exact opcode, byte length, unit limit,
unit-price ceiling, position, and multiplicity must be reviewed and added before deployment.

The broker environment must provide every ceiling explicitly; zero or missing values prevent the
wallet principal from loading. The renderer shows the decoded amount and its ceiling. A starter's
grant delta and a withdrawal execute's pending amount are state-derived, so the outer validator must
read the canonical accounts from the trusted devnet view before showing or signing them. Caller
claims are never used.

### 4.3 Cluster binding and refusals

A legacy Solana message contains a recent blockhash but no genesis hash. To keep the wire request to
raw message plus wallet id without trusting caller metadata, the broker uses its root-owned devnet
RPC configuration to obtain `getGenesisHash`, validate that the decoded recent blockhash is still
valid on that cluster, read any state needed by the plan, and pin the program account to the reviewed
executable. The pure module receives this as an injected immutable `cluster_view`; unit tests never
make RPC calls. The only accepted genesis is devnet
`EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG`. Current client code already refuses mainnet and
accepts only devnet or a loopback test validator (`scripts/network/pay_voucher.py:76-77,433-450`);
the production broker is stricter and never accepts the loopback exception.

Refuse before approval and again before signing on any of the following:

- a wallet id, node id, public key, program, mint, genesis, executable owner, or recent blockhash
  outside the pinned policy;
- an unknown program, unknown instruction, extra instruction, missing instruction, wrong order,
  duplicate instruction, trailing bytes, versioned message, or Compute Budget instruction;
- a fee payer other than the wallet; required signers other than exactly the wallet; signer or
  writable flags different from the template;
- a non-canonical config, escrow, vault, delegate, event authority, or ATA; a System transfer whose
  source is not the wallet or destination is not the decoded self/starter node;
- delegate flags, expiry, cap, starter grant delta, withdrawal amount, pending execute amount, or SOL
  transfer above its broker-environment ceiling;
- missing, malformed, stale, wrong-owner, wrong-generation, or internally inconsistent devnet state;
- a caller-supplied description or decoded field, approval TTL outside the fixed bound, digest
  mismatch, consumed request, expired approval, or custody/principal change during the act.

## 5. Client call sites and rv5#3

### 5.1 Every current `load_wallet()` caller

| Caller | What it actually needs | Change |
|---|---|---|
| `scripts/paid_admission.py:PaidAdmission.__init__` | Wallet address only, to identify the worker's payee wallet in chain admission (`scripts/paid_admission.py:633-675`). The worker node key signs its receipts and redeemer transactions. | Replace `load_wallet` with `load_wallet_identity`. This is rv5#3 and must work with only `wallet.pub`. |
| `scripts/client.py:main` chain-voucher setup | Wallet address only, passed as escrow owner to `ensure_channel`; the node key is loaded separately and opens the channel/signs vouchers (`scripts/client.py:1974-2012`; `scripts/network/pay_autochannel.py:252-267,377-396`). | Replace `load_wallet` with `load_wallet_identity`. This is the other rv5#3 call site. No wallet approval occurs on the normal paid-session path. |
| `scripts/network/nakd.py:NakDaemon._chain_starter` | Wallet public identity and a real wallet signature over `delegate_set` plus optional SOL transfer (`scripts/network/nakd.py:1283-1319,1408-1443`). | Load public identity, build/compile as today, and route the raw message through `wallet_sign`; never request or receive a seed. |
| `scripts/network/nak.py:_pay_command` `backup` | The seed, to derive and print the mnemonic (`scripts/network/nak.py:1230-1244`). | Keep `load_wallet`; broker custody refuses because export of broker key material is not supported. |
| `scripts/network/nak.py:_pay_command` `export` | The seed and public key, to print the keypair (`scripts/network/nak.py:1246-1253`). | Keep `load_wallet`; broker custody refuses. |
| `scripts/network/pay_wallet.py:load_wallet_identity` | In file custody, full validation and the seed for existing file-signing callers (`scripts/network/pay_wallet.py:289-306`). | Keep this internal delegation. Broker custody continues on the public-only branch. |
| `tests/test_pay_wallet.py:test_established_identity_adds_wallet_without_touching_identity_files` | Test assertion that a file wallet reloads (`tests/test_pay_wallet.py:126-135`). | Keep; add separate public-only and no-seed broker-custody tests. |

This is the complete repository call set found by `rg '\bload_wallet\s*\('`. The comment in
`tests/test_pay_redeemer.py:903` is not a call.

### 5.2 Other signing boundaries that change or stay closed

- Add `scripts/network/pay_wallet_custody.py` as the single file-or-broker wallet-signing boundary,
  analogous to `person_custody`. It accepts only `(raw_message, wallet_id)`, authenticates the exact
  dynamic principal, verifies the echoed bytes/digest/signature, and returns one 64-byte signature.
- Split transaction construction from signing in `scripts/network/pay_chain.py`: compile once, pass
  `CompiledMessage.data` to the custody boundary, then assemble the one-signature transaction without
  recompiling. Existing `sign_transaction` remains the file-custody implementation
  (`scripts/network/pay_chain.py:403-454,481-499`).
- Change `scripts/network/nak.py:_pay_wallet_and_node` so broker custody no longer fails at the
  blanket guard (`scripts/network/nak.py:974-982`). Change `_pay_send` to use the custody boundary
  for only the approved ensure/withdraw templates; it currently reads `wallet["seed"]` directly
  (`scripts/network/nak.py:996-1012`). Deposit and revoke stay explicitly unsupported in broker mode.
- Change `scripts/network/nakd.py:NakDaemon._chain_starter` as above. Preserve its durable attempt
  write before broadcast and blockhash-expiry reconciliation
  (`scripts/network/nakd.py:1427-1469`); the broker signature is not permission to silently rebuild.
- `scripts/network/pay_autochannel.py:ensure_channel`, voucher signing in
  `scripts/network/pay_voucher.py`, and the normal worker redeemer remain node-key paths. They do not
  call the wallet broker. The channel builder makes the node the signer and fee payer
  (`scripts/network/pay_autochannel.py:373-396`).
- `scripts/network/pay_redeemer.py:request_close` normally uses the node voucher signer. Its fallback
  for an owner-opened legacy channel genuinely needs the wallet seed and already refuses broker
  custody (`scripts/network/pay_redeemer.py:526-542`). `channel_close_request` is outside this
  allowlist, so that exceptional fallback remains fail-closed pending a separate template decision.

## 6. Failure modes

- **Broker down or wrong principal:** refuse before enqueue or broadcast. Never fall back to a file,
  cached signature, person key, or node key.
- **Approval denied or expired:** return “not approved; nothing was signed.” Do not enqueue again in
  the background.
- **Approval result uncertain:** retry only the same request id and exact content within its original
  deadline. If the broker says consumed but the signature was lost, require a new human approval.
- **TTL expires before signing:** discard the request. Rebuild with a fresh blockhash and ask again.
- **Transaction expires after approval/signing:** first use finalized status and block height to prove
  the old signature cannot still land. Then rebuild a new message and re-approve. Never re-sign or
  re-broadcast a replacement silently. Current redeemer and starter paths already retain signed
  attempts until finalized failure or blockhash expiry
  (`scripts/network/pay_redeemer.py:136-165,194-250`; `scripts/network/nakd.py:1219-1258`).
- **Broadcast outcome unknown:** keep the exact signed attempt unresolved and do not approve a
  competing replacement while it can land.
- **Decoder, state, ceiling, cluster, digest, echo, or signature mismatch:** fail closed before local
  state says “sent.” Log only request id, wallet id, operation class, digest, and closed error code;
  never raw key material or authorization credentials.
- **Custody changes during an act:** refuse. File and broker backends are pinned at preflight; there is
  no mid-act fallback.

## 7. Test plan

The first implementation branch contains a pure decoder/policy module plus fixtures only. It does
not call the approval broker, Buzz, or a network service.

Golden fixtures are generated by the real builders, then frozen as raw message bytes plus an
expected typed plan:

- self `delegate_set`; self transfer plus `delegate_set`; transfer-only self top-up;
- starter `delegate_set`; starter plus SOL top-up;
- withdraw request; ATA create-idempotent plus withdraw execute.

Each golden asserts canonical parsing, exact account order and privileges, PDA/ATA derivation,
operation classification, rendered fields, applicable ceilings, and SHA-256. A generation test may
call the builder; decoder tests consume frozen bytes so a builder and decoder cannot drift together
unnoticed.

Hostile fixtures mutate one property at a time: extra instruction, unknown program, known program
with unknown opcode, swapped account, writable/signer-bit change, wrong fee payer, extra signer,
wrong PDA/ATA/mint/program, System transfer to a different destination, non-canonical short-vector,
trailing byte, versioned message, mainnet genesis, invalid/stale devnet blockhash, over-ceiling cap,
expiry, starter delta, withdrawal, pending execute, and SOL amount. Include duplicate-signature replay:
the first approved request signs once; a second sign call for that request is refused, and the same
message under a new request still needs a new approval.

Client tests use a local fake transport, never the real endpoint. They cover wrong principal,
decline, expiry, consumed-with-lost-response, changed echoed message, changed digest, wrong wallet
signature, custody change, rv5#3 with `wallet.pub` and no seed, and expired transaction rebuild
requiring a second approval. Existing person/release tests demonstrate the same fail-closed cases
(`tests/test_person_custody.py:421-480,531-577`; `tests/test_release_broker.py:180-286`).

Finally run the sealed, network-disabled rehearsal:

```text
rehearsal/run.sh <reviewed-package> --mode import-nak-wallet
```

It proves real uid/mode boundaries, import, public-key match, principal registration, source
retirement, idempotent rerun, failure rollback, and decoder self-test. It does not make a Solana
transaction and does not touch the live broker.

## 8. Rollout and ownership

1. **This lane:** land this document only.
2. **Unification lane:** review the closed templates, cluster/state boundary, ceilings, installer
   transaction, and the three open questions below. No operational change yet.
3. **Implementation lane:** add the pure decoder and golden/hostile fixtures first. Review that diff
   independently before adding the broker endpoint or clients.
4. **Broker/unification lane:** add the dynamic wallet principal, root-owned wallet policy, one-shot
   endpoint, import transaction, environment ceilings, and rollback tests. Keep the service unchanged
   until the sealed rehearsal passes.
5. **Nakshatra lane:** add the custody client, compile-once/assemble seam, rv5#3 loader fixes, CLI and
   starter wiring, and fake-transport tests. No test contacts the live approval service or Buzz.
6. **Rehearsal owner:** run `rehearsal/run.sh --mode import-nak-wallet` with throwaway material in the
   sealed container; archive the PASS and reviewed package manifest.
7. **Biswa:** on his Mac, independently compute the reviewed package's manifest digest from the
   pinned commit. On the hub, compare that digest and perform only the documented `sudo` install/import
   invocation. He does not edit policy, move key files by hand, run tests against the live service, or
   approve a rehearsal transaction.
8. **Post-import verification by the unification lane:** as the unprivileged node user, confirm
   `wallet.pub`, `custody=broker`, absence of the seed, correct public projection, and a healthy
   authenticated principal. A live devnet approval is a separately scheduled acceptance drill, not
   part of Biswa's install work. Rollout stops before any withdrawal or mainnet action.

Order is fixed: design note, unification review, pure decoder plus fixtures, sealed rehearsal, then
Biswa's independent digest and install. No mainnet enablement is bundled with this work.

## 9. Open questions for the unification lane

1. **First escrow and excluded owner actions.** Current `nak pay ensure` can place `escrow_init` in
   the same message before `delegate_set`, while the agreed allowlist does not include it
   (`scripts/network/nak.py:1051-1098`). This note gates import on an existing escrow and leaves
   deposit, revoke, escrow close, and legacy wallet-authority channel close unsupported. Confirm that
   prerequisite, or commission separate exact templates; do not widen the generic program allowlist.
2. **Trusted devnet state provider.** Genesis/blockhash binding, starter grant delta, and withdrawal
   execute amount cannot be proven from legacy message bytes alone. Confirm that the broker may use a
   separately configured TLS devnet RPC with pinned genesis/program checks, or name an equivalent
   broker-trusted state feed. Caller-supplied cluster or account state is not acceptable.
3. **Reviewed ceiling values.** Agree the explicit broker-environment values for self delegate cap and
   lifetime, starter grant and lifetime, withdrawal amount, and self/starter SOL top-ups. Current code
   defaults self delegation to 5 USDC and 30 days
   (`scripts/network/pay_autochannel.py:18-22`), targets 0.03 SOL for self top-up
   (`scripts/network/nak.py:820-821,1053-1060`), and targets 0.01 SOL plus 30 days for a starter
   (`scripts/network/nakd.py:111-113,1408-1425`); these are evidence for defaults, not permission to
   omit hard ceilings.
