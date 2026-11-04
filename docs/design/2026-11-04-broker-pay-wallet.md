# Broker-held Solana pay wallet (M4c part 2c)

Status: unification-lane reviewed; send to build for rollout step 3 only. Devnet only.

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
- the devnet genesis hash, configured `nakshatra_pay` program id, its upgradeable-loader ProgramData
  address, reviewed program-byte SHA-256, expected multisig upgrade-authority address, the classic SPL
  mint, and all ceiling values from the root-owned broker environment.

The dynamic principal must authenticate as itself before enqueue, as the release client currently
checks `nakshatra-release` before asking for an approval (`release/broker_signer.py:162-183`). It must
not share the person or release principal credential.

The staged broker installer gains `--import-nak-wallet`. It follows the existing transactional
person-key import, which prepares one principal addition before touching authority state and keeps a
rollback record (`/home/prithvi/trisul@forgejo/main:infra/pillar-agent/deploy/prithvi-broker/install.sh:1352-1405`).
The wallet import does the following:

1. Inventory `wallet.ed25519` copies by SHA-256 only (never file content) across the known Syncthing
   roots, `backup-sources.tsv` targets, memory-sync roots, and git history, and report the count to
   Biswa in his install step.
2. Refuse unless the staged package digest, broker pre-state, source ownership/mode, `wallet.pub`,
   `custody=file`, node id, and broker policy are all valid. Derive the public key inside the trusted
   import helper and require a byte-for-byte match with `wallet.pub`; never print seed bytes.
3. Refuse a different pre-existing broker wallet key or registry row. An identical already-imported
   key is an idempotent resume, not a rotation.
4. Copy the existing raw 32-byte `wallet.ed25519` into broker-owned key storage, install the
   `nakshatra-wallet:<node>` principal credential and wallet policy, and self-test a fixed harmless
   message entirely inside the sealed import transaction.
5. Start the candidate broker and verify the public key, principal, decoder, and rollback path before
   retiring the source. No live Solana transaction is signed or sent during import.
6. Only after broker verification, atomically replace the custody marker with `broker`, move the node
   user's seed to a root-owned retired path, set it to mode `000`, and verify it is no longer readable
   through any installer-declared active source. Do not delete it. This follows the existing
   broker-key retirement pattern
   (`/home/prithvi/trisul@forgejo/main:infra/pillar-agent/deploy/prithvi-broker/README.md:292-305`). A
   failure before that boundary restores `custody=file` and the original source. A failure after it
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
  "idempotency_key": "<stable per-human-act key>",
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

The client creates and durably records `idempotency_key` with the exact approval attempt before its
first enqueue, then reuses it for every uncertain enqueue retry. A changed message or deliberately
new approval gets a new key; a replacement for the same logical payment act remains linked to that
act's local attempt journal and is allowed only after its prior signed transaction is reconciled. The
broker requires the field and returns the existing request only when the same principal/key names the
same request fingerprint
(`/home/prithvi/trisul@forgejo/main:infra/pillar-agent/pillar_signer_broker.py:1167-1174,1228-1261`).
Thus a lost enqueue response is recovered without creating a second approval.

The broker permits at most one outstanding wallet approval per principal. It refuses a second
enqueue while the first is pending, approved but unconsumed, or linked to an unreconciled signed
attempt, and rate-limits enqueue attempts per principal. Idempotent recovery with the same key and
fingerprint returns the existing request and does not count as a second approval.

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

The configured program id is `GLWqQsSqUsDkPvkc8rxcoQUyro7ZQd5denht21NtdTTo`
(`/home/prithvi/neuron/programs/nakshatra_pay/programs/nakshatra_pay/src/lib.rs:13`). Anchor instruction
data is discriminator plus the exact arguments; the program itself refuses trailing argument bytes
(`/home/prithvi/neuron/programs/nakshatra_pay/programs/nakshatra_pay/src/lib.rs:18-85`). All integer
fields below are little-endian.

| Allowed instruction | Exact data | Required ordered accounts and signers | Approval fields | Ceiling |
|---|---|---|---|---|
| `nakshatra_pay.delegate_set` (self ensure) | `8b10e667fea614a5` + delegate key (32) + `cap_total:u64` + `expires_at:i64` + `flags:u8` | wallet **signer,writable**; canonical config; canonical wallet escrow; canonical delegate PDA **writable**; System Program; canonical event authority; program id | wallet, own node delegate, real pre-state spent/cap, explicit older-generation reset disclosure, cap in micro-USDC and USDC, cap ceiling, expiry, flags, escrow, actual and worst-case delegate rent, optional SOL top-up, message digest | absolute encoded `cap_total` `NAK_WALLET_MAX_DELEGATE_MICRO_USDC`; expiry horizon `NAK_WALLET_MAX_DELEGATE_TTL_S`; rent and optional transfer ceilings below |
| `nakshatra_pay.delegate_set` (starter grant) | same discriminator/layout; `flags` must equal `OPEN_CHANNEL=1` | same account layout; delegate PDA must be derived from escrow and the decoded recipient node key | inviter wallet, recipient node, real pre-state spent/cap, explicit older-generation reset disclosure, resulting remaining capacity and total cap, expiry, actual and worst-case delegate rent, optional SOL top-up, escrow, digest | absolute encoded `cap_total` must be no greater than either `NAK_WALLET_MAX_STARTER_CAP_TOTAL_MICRO_USDC` or `NAK_WALLET_MAX_STARTER_MICRO_USDC`; expiry horizon `NAK_WALLET_MAX_STARTER_TTL_S`; rent and transfer ceilings below |
| `nakshatra_pay.escrow_withdraw_request` | `f6f379c6f277d27a` + `amount:u64` | wallet **signer**; canonical config; canonical wallet escrow **writable**; canonical vault; event authority; program id | wallet, amount in micro-USDC and USDC, escrow, vault, earliest execute time from trusted chain state, digest | `NAK_WALLET_MAX_WITHDRAW_MICRO_USDC` |
| `nakshatra_pay.escrow_withdraw_execute` | `ac5dcf14503aa8f1` only | wallet **signer**; canonical config; pinned mint; canonical escrow, vault and wallet ATA **writable**; classic Token Program; event authority; program id | wallet, pending amount, withdrawal ceiling and unlock time from trusted devnet state, escrow, vault, destination ATA, digest | pending amount must be at or below `NAK_WALLET_MAX_WITHDRAW_MICRO_USDC` |
| System Program transfer | program `11111111111111111111111111111111`; data `2:u32` + `lamports:u64` | wallet **signer,writable**; decoded delegate node **writable** | destination node, lamports and SOL, whether self top-up or starter top-up | `NAK_WALLET_MAX_SELF_SOL_LAMPORTS` or `NAK_WALLET_MAX_STARTER_SOL_LAMPORTS` |
| Associated Token Account create-idempotent | program `ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL`; data exactly `01` | wallet **signer,writable** payer; canonical wallet ATA **writable**; wallet owner; pinned mint; System Program; classic Token Program | wallet, mint, canonical ATA, actual zero-or-create shortfall and worst-case rent | `NAK_WALLET_MAX_ATA_RENT_LAMPORTS`; allowed once only in the execute template |

The account order and argument layouts above match the current Python builders
(`scripts/network/pay_chain.py:219-231,257-290`) and the program account constraints
(`/home/prithvi/neuron/programs/nakshatra_pay/programs/nakshatra_pay/src/instructions.rs:418-537,619-698`).
The decoder independently derives every PDA and ATA; it never trusts an account label from the
caller.

Every template also renders the exact network fee in lamports and SOL. The broker calls
`getFeeForMessage` for the exact message through its own pinned devnet RPC and refuses a quote above
`NAK_WALLET_MAX_NETWORK_FEE_LAMPORTS` from root-owned `broker.env`; the wallet is the fee payer, so
this debit exists even when an instruction fails. It stores the displayed quote and, immediately
before signing, repeats the quote and requires it to be unchanged and still below the ceiling. This
matches the existing client quote path (`scripts/network/pay_chain.py:662-670`;
`scripts/network/nak.py:996-1012`).

`delegate_set` is `init_if_needed, payer = owner`, so an absent 115-byte delegate PDA charges the
wallet rent (`/home/prithvi/neuron/programs/nakshatra_pay/programs/nakshatra_pay/src/instructions.rs:619-632`;
`scripts/network/pay_chain.py:724`). For both the delegate PDA and wallet ATA, the broker recognizes
three states: absent; correctly initialized under the expected program; or system-owned with zero
data and any pre-funded lamports (a dusted, still-uninitialized account). It renders the actual
shortfall as zero for an initialized account and otherwise as
`max(0, getMinimumBalanceForRentExemption(size) - current_lamports)`, using size 115 for the delegate
and 165 for the ATA; dust is therefore credited rather than refused or treated as zero. Separately,
it always renders and ceilings the full rent-exempt minimum as worst-case rent because current state
or dust can disappear before signing. Any other
owner/data shape is refused. This matches Anchor's fund-then-allocate/assign path and the associated
token program's equivalent PDA creation path
(`/home/prithvi/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/anchor-syn-1.2.0/src/codegen/accounts/constraints.rs:1778-1815`;
`/home/prithvi/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/spl-associated-token-account-1.1.3/src/tools/account.rs:27-53`).
Immediately before signing the broker repeats the account, lamport, and rent-quote reads and requires
the rendered shortfall to be unchanged and no higher than
`NAK_WALLET_MAX_DELEGATE_RENT_LAMPORTS` or `NAK_WALLET_MAX_ATA_RENT_LAMPORTS`. Current clients already
include these debits in affordability checks
(`scripts/network/nak.py:1091-1102,1371-1379`; `scripts/network/nakd.py:1427-1439`).

“Starter grant” is not a fourth on-chain discriminator. It is the closed template consisting of a
starter-form `delegate_set`, optionally followed by one System transfer to that same decoded
delegate key. The current starter builder uses exactly that order
(`scripts/network/nakd.py:1408-1443`). Self ensure allows an optional System transfer to the
principal's own node followed by a self-form `delegate_set`; a transfer alone is allowed when the
delegate needs no renewal. Current ensure can create the escrow too
(`scripts/network/nak.py:1036-1102`), but `escrow_init` is intentionally not in this allowlist; import
requires the escrow to exist. `escrow_init`, deposit, revoke, escrow close, and legacy
wallet-authority channel close remain outside the allowlist and fail closed; this review does not
widen it.

Withdraw request is exactly one instruction. Withdraw execute is exactly ATA create-idempotent then
execute, matching the current builder (`scripts/network/pay_chain.py:266-276`). No current payment
builder emits Compute Budget instructions; there is no Compute Budget instruction constructor or
program id in `pay_chain.py` (`scripts/network/pay_chain.py:27-35,219-388`). Therefore Compute Budget
is refused in this revision. If a builder later emits it, its exact opcode, byte length, unit limit,
unit-price ceiling, position, and multiplicity must be reviewed and added before deployment.

Root-owned `broker.env` must provide every ceiling explicitly; zero or missing values prevent the
wallet principal from loading. These are the reviewed devnet defaults; Biswa may tighten them at
install but may not raise them without another review.

| `broker.env` ceiling | Reviewed devnet default |
|---|---:|
| `NAK_WALLET_MAX_DELEGATE_MICRO_USDC` | `5_000_000` micro-USDC absolute `cap_total` |
| `NAK_WALLET_MAX_DELEGATE_TTL_S` | `2_592_000` (30 days) |
| `NAK_WALLET_MAX_STARTER_CAP_TOTAL_MICRO_USDC` | `1_000_000` micro-USDC absolute `cap_total` |
| `NAK_WALLET_MAX_STARTER_MICRO_USDC` | `1_000_000` micro-USDC resulting remaining capacity |
| `NAK_WALLET_MAX_STARTER_TTL_S` | `2_592_000` (30 days) |
| `NAK_WALLET_MAX_WITHDRAW_MICRO_USDC` | `5_000_000` micro-USDC |
| `NAK_WALLET_MAX_NETWORK_FEE_LAMPORTS` | `10_000` lamports |
| `NAK_WALLET_MAX_DELEGATE_RENT_LAMPORTS` | `1_700_000` lamports (115-byte minimum currently `1_691_280`) |
| `NAK_WALLET_MAX_ATA_RENT_LAMPORTS` | `2_040_000` lamports (165-byte minimum currently `2_039_280`) |
| `NAK_WALLET_MAX_SELF_SOL_LAMPORTS` | `30_000_000` lamports |
| `NAK_WALLET_MAX_STARTER_SOL_LAMPORTS` | `10_000_000` lamports |

The renderer shows each decoded or quoted debit and its ceiling. Every ceiling for an amount encoded
in the message is enforced from those bytes alone: self and starter `cap_total`, withdrawal-request
amount, and SOL transfers cannot be made acceptable by RPC state. RPC-derived state may only tighten
a decision, never loosen it. A withdrawal execute carries no amount, so its displayed pending amount
is necessarily state-derived; it is acceptable only because the originating withdrawal request was
absolutely ceiling-checked from its message bytes. The execute independently refuses if the current
pending amount exceeds the same ceiling. All starter pre-state is likewise read from the trusted
devnet view before showing and again before signing. Caller claims are never used.

Before any amount is labelled USDC, the broker decodes the canonical 208-byte Anchor `Config`
account at the derived Config PDA, requires its program owner and both stored PDA bumps, and requires
`Config.mint` to equal the pinned devnet USDC mint. The trusted withdrawal-delay quote must equal the
Config's signed `i64` delay. `chain_time` and the delay are each validated as `i64`, and their sum is
checked for `i64` overflow exactly as the program does.

For a delegate update, its generation must equal the escrow generation or be strictly older, never
newer. A same-generation delegate must not be disabled and the encoded cap must be at least its
spent amount. An older-generation delegate is eligible for the program's spent/disabled reset only
when `open_channels == 0` and `open_tasks == 0`; the approval shows the real pre-state and discloses
that reset separately. Any stored `spent > cap_total` or unknown flag bits are malformed state and
refused. These checks mirror the program transition
(`/home/prithvi/neuron/programs/nakshatra_pay/programs/nakshatra_pay/src/instructions.rs:641-688`).

For a starter grant, F1 is enforced from message bytes alone: encoded `cap_total` must be at or below
`min(NAK_WALLET_MAX_STARTER_CAP_TOTAL_MICRO_USDC, NAK_WALLET_MAX_STARTER_MICRO_USDC)` regardless of
RPC `spent`. Trusted state may only add refusals. The renderer still shows the post-transition
remaining capacity (`cap_total - spent` for the same generation, `cap_total` after an older-generation
reset), but state subtraction can never make over-ceiling bytes acceptable. This replaces the former
state-adjusted construction rule and prevents a false high `spent` from widening actual
authority. The current starter construction lives in `scripts/network/nakd.py:1362-1418`; channel
finalize does reduce `spent`
(`/home/prithvi/neuron/programs/nakshatra_pay/programs/nakshatra_pay/src/instructions.rs:1166-1173`).
Self `delegate_set` similarly applies `NAK_WALLET_MAX_DELEGATE_MICRO_USDC` to the encoded absolute
`cap_total`, not merely an increase.

### 4.3 Cluster binding and refusals

A read-only lead check on 2026-11-04 found no account for
`GLWqQsSqUsDkPvkc8rxcoQUyro7ZQd5denht21NtdTTo` on devnet: the program is not deployed, so it has no
current ProgramData or upgrade authority to report. Deployment under the required 48-hour
time-locked multisig, including the exact authority address, remains Biswa's decision and is a
prerequisite for rollout step 4 and the M6 live drill
(`docs/design/2026-10-26-noncustodial-payments.md:307`).

A legacy Solana message contains a recent blockhash but no genesis hash. To keep the wire request to
raw message plus wallet id without trusting caller metadata, the broker uses its root-owned devnet
RPC configuration to obtain `getGenesisHash`, validate that the decoded recent blockhash is still
valid on that cluster, and read any state needed by the plan. At approval and immediately before
signing it also requires the program account to be executable and owned by the BPF Upgradeable Loader,
decodes its ProgramData address, requires it to equal both the loader-derived address and broker
policy, and requires the ProgramData account to have the same loader owner. The SHA-256 of the
deployed program bytes (excluding ProgramData metadata) and the decoded ProgramData upgrade authority
must also equal the reviewed hash and expected multisig address pinned in root-owned `broker.env`.
These facts are read from the broker's trusted devnet RPC at signing time; caller-supplied program
state is never accepted. This preserves the controlling design's upgradeable, 48-hour time-locked
multisig authority rather than requiring immutability
(`docs/design/2026-10-26-noncustodial-payments.md:465-471`).
The pure module receives these facts as an injected immutable `cluster_view`; unit tests never make
RPC calls. The only accepted genesis is devnet
`EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG`. Current client code already refuses mainnet and
accepts only devnet or a loopback test validator (`scripts/network/pay_voucher.py:76-77,433-450`);
the production broker is stricter and never accepts the loopback exception.

The broker may use a separately configured TLS devnet RPC only with the genesis, ProgramData
address, deployed code hash, and upgrade authority pinned in root-owned `broker.env`. RPC responses
have strict byte-size and deadline limits. The network-facing component fetches the bounded state,
closes the connection, and only then hands immutable bytes to the decoder; the key-holding signing
path has no live RPC client. RPC failures log only closed error codes, never response bodies or
unbounded remote text.

An authorized upgrade can still occur after the final broker read while a signed transaction remains
valid. That residual window is only the recent-blockhash lifetime, approximately 60--90 seconds,
versus the required 48-hour announced multisig timelock. The notice period gives operators time to
quiesce signing and review the new binary; after activation, the old hash fails closed. Enabling a
reviewed upgrade for subsequent transactions is never automatic: Biswa updates the pinned hash in
root-owned `broker.env` as an explicit install step. A transaction signed immediately before an
activation can still execute during its remaining validity window, so upgrade procedure must quiesce
wallet signing across that short boundary rather than pretending the pin removes all TOCTOU risk.

Refuse before approval and again before signing on any of the following:

- a wallet id, node id, public key, program, mint, genesis, loader owner, ProgramData address,
  program-byte hash, upgrade authority, or recent blockhash outside the pinned policy;
- an unknown program, unknown instruction, extra instruction, missing instruction, wrong order,
  duplicate instruction, trailing bytes, versioned message, or Compute Budget instruction;
- a fee payer other than the wallet; required signers other than exactly the wallet; signer or
  writable flags different from the template;
- a non-canonical config, escrow, vault, delegate, event authority, or ATA; a System transfer whose
  source is not the wallet or destination is not the decoded self/starter node;
- any semantic role account that aliases another role or a protected runtime id (payment program,
  System, classic Token, ATA, ed25519, loaders, Compute Budget, or sysvars), checked uniformly before
  privilege aggregation for every template across the wallet, node or recipient, config, escrow,
  vault, ATA, mint, fee vault, delegate PDA, and event-authority roles the plan derives or displays;
- delegate flags, expiry, cap, starter resulting remaining capacity, withdrawal amount, pending
  execute amount, SOL transfer, network fee, delegate rent, or ATA rent above its broker-environment
  ceiling;
- missing, malformed, stale, wrong-owner, wrong-generation, internally inconsistent, oversized, or
  late devnet state;
- a caller-supplied description or decoded field, approval TTL outside the fixed bound, digest
  mismatch, consumed request, expired approval, or custody/principal change during the act.

## 5. Client call sites and rv5#3

### 5.1 Wallet load and setup paths

| Caller | What it actually needs | Change |
|---|---|---|
| `scripts/paid_admission.py:PaidAdmission.__init__` | Wallet address only, to identify the worker's payee wallet in chain admission (`scripts/paid_admission.py:633-675`). The worker node key signs receipts; redeemer fee-payer custody is separate. | Replace `load_wallet` with `load_wallet_identity`. This is rv5#3 and must work with only `wallet.pub`. |
| `scripts/client.py:main` chain-voucher setup | Wallet address only, passed as escrow owner to `ensure_channel`; the node key is loaded separately and opens the channel/signs vouchers (`scripts/client.py:1974-2012`; `scripts/network/pay_autochannel.py:252-267,377-396`). | Replace `load_wallet` with `load_wallet_identity`. This is the other rv5#3 call site. No wallet approval occurs on the normal paid-session path. |
| `scripts/network/nakd.py:NakDaemon._chain_starter` | Wallet public identity and a real wallet signature over `delegate_set` plus optional SOL transfer (`scripts/network/nakd.py:1283-1319,1408-1443`). | Load public identity, build/compile as today, and route the raw message through `wallet_sign`; never request or receive a seed. |
| `scripts/network/nak.py:_pay_command` `backup` | The seed, to derive and print the mnemonic (`scripts/network/nak.py:1230-1244`). | Keep `load_wallet`; broker custody refuses because export of broker key material is not supported. |
| `scripts/network/nak.py:_pay_command` `export` | The seed and public key, to print the keypair (`scripts/network/nak.py:1246-1253`). | Keep `load_wallet`; broker custody refuses. |
| `scripts/network/pay_wallet.py:load_wallet_identity` | In file custody, full validation and the seed for existing file-signing callers (`scripts/network/pay_wallet.py:289-306`). | Keep this internal delegation. Broker custody continues on the public-only branch. |
| `scripts/network/pay_wallet.py:preflight_wallet` | Setup preflight currently treats `wallet.pub`/`custody` without a seed as incomplete (`scripts/network/pay_wallet.py:180-193`). | Treat a valid `wallet.pub` plus `custody=broker` and no seed as complete; keep refusing partial, mismatched, or file-custody state without a seed. |
| `scripts/network/pay_wallet.py:ensure_wallet` | Every setup rerun calls preflight and currently creates a new seed whenever none exists (`scripts/network/pay_wallet.py:220-268`). | After broker-state preflight, return the public identity without creating or rewriting any wallet file. Add setup-rerun tests for this exact shape. |
| `tests/test_pay_wallet.py:test_established_identity_adds_wallet_without_touching_identity_files` | Test assertion that a file wallet reloads (`tests/test_pay_wallet.py:126-135`). | Keep; add separate public-only and no-seed broker-custody tests. |

The `load_wallet` rows are the complete repository call set found by
`rg '\bload_wallet\s*\('`; the two setup functions are included because they must also accept the
post-import shape. The comment in `tests/test_pay_redeemer.py:903` is not a call.

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
  (`scripts/network/nak.py:996-1012`). Every broker-backed `_pay_send`, including `ensure` and both
  withdrawal phases, must durably record the signed transaction's signature, message SHA-256,
  blockhash, and `lastValidBlockHeight` before broadcast. Deposit and revoke stay explicitly
  unsupported in broker mode. This record is required because `send_and_confirm` can report an
  unknown final state after broadcast (`scripts/network/pay_chain.py:695-719`).
- Every approved-withdraw journal record must carry the originating message digest, transaction signature,
  and escrow generation. Such records come only from reconciled, byte-ceiling-checked
  withdrawal requests; competing signed attempts keep the logical act unresolved and block another
  approval. Wallet import refuses while the escrow has a pending withdrawal. Current closed templates
  do not recreate escrows; if recreation is ever added, the execute lookup must bind escrow generation
  explicitly rather than relying on `(escrow, amount, withdraw_at)` alone.
- Change `scripts/network/nakd.py:NakDaemon._chain_starter` as above and retain its pre-broadcast
  durable-attempt boundary (`scripts/network/nakd.py:1427-1469`), extending the record with the message
  digest. Before **any** new approval request for the same logical starter, ensure, or withdrawal act,
  reconcile all prior signed attempts with finalized status and finalized block height. If height is
  past `lastValidBlockHeight`, re-read signature status and declare expiry only if that second read is
  still absent; `processed`/`confirmed` remains unresolved regardless of height. The existing redeemer
  and starter resolvers do not yet close that status/height race
  (`scripts/network/pay_redeemer.py:136-165`; `scripts/network/nakd.py:1219-1258`); the fix in
  `pay/starter-attempt-fence` must land before broker signing. The broker signature is not permission
  to silently rebuild or request a competing approval.
- `scripts/network/pay_autochannel.py:ensure_channel`, voucher signing in
  `scripts/network/pay_voucher.py`, and broker-custody worker redemption remain node-key paths. The
  redeemer chooses the node key as fee payer/signer only under broker custody; under file custody it
  uses the wallet as fee payer/signer (`scripts/network/pay_redeemer.py:353-368`). Neither case calls
  the wallet broker. The channel builder makes the node the signer and fee payer
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
- **Transaction expires after approval/signing:** reconcile the durable attempt before requesting a
  replacement approval. Any observed `processed`/`confirmed` status remains unresolved until
  finalized, regardless of block height. After finalized height exceeds `lastValidBlockHeight`, read
  signature status again and expire only if it is still absent; this closes the status/height race.
  Then rebuild a new message and re-approve. `pay/starter-attempt-fence` is fixing this rule in the
  existing redeemer and starter resolvers before broker signing is connected.
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
expiry, starter remaining capacity, withdrawal, pending execute, SOL amount, network fee, delegate/ATA
rent shortfall, ProgramData address/hash, and upgrade authority. Rent cases cover absent, initialized,
and system-owned zero-data accounts with zero, partial, exact, and over-ceiling prefunding; other
owners or nonzero data refuse. Starter cases cover both generations, disabled state, and nonzero open
channels/tasks, plus a hostile RPC view reporting `spent = cap_total - 1` while encoded `cap_total`
exceeds the absolute ceiling. Include duplicate-signature replay: the first approved request signs
once; a second sign call for that request is refused, and the same message under a new request still
needs a new approval.

Client tests use a local fake transport, never the real endpoint. They cover wrong principal,
stable-key recovery after a lost enqueue response, same-key/different-content conflict, decline,
expiry, consumed-with-lost-response, changed echoed message, changed digest, wrong wallet signature,
custody change, setup rerun plus rv5#3 with `wallet.pub` and no seed, and expired transaction rebuild
requiring a second approval. Broker tests refuse concurrent outstanding requests, enforce the
per-principal enqueue rate limit, reject oversized and late RPC responses before the decoder or
signer, and assert that logs contain only closed error codes. CLI and starter tests kill the process
after the pre-broadcast journal, prove all four attempt fields survive, and refuse a new approval
until reconciliation; the expiry race fixture makes status appear between the first read and the
over-height re-read. Existing person/release tests demonstrate the same fail-closed cases
(`tests/test_person_custody.py:421-480,531-577`; `tests/test_release_broker.py:180-286`).

Finally run the sealed, network-disabled rehearsal:

```text
rehearsal/run.sh <reviewed-package> --mode import-nak-wallet
```

It proves real uid/mode boundaries, import, public-key match, principal registration, source
retirement to a root-owned mode-`000` path, SHA-256-only copy inventory and reported count,
idempotent rerun, failure rollback, and decoder self-test. It does not make a Solana transaction and
does not touch the live broker.

## 8. Rollout and ownership

1. **This lane:** land this document only.
2. **Unification lane:** review complete. Round 3 sends only step 3 to build; no operational change.
3. **Implementation lane:** add the pure decoder and golden/hostile fixtures first. Review that diff
   independently before adding the broker endpoint or clients.
4. **Broker/unification lane:** add the dynamic wallet principal, root-owned wallet policy, one-shot
   endpoint, import transaction, environment ceilings, and rollback tests. This step is gated on
   Biswa choosing deployment under the 48-hour time-locked multisig and the lane reporting the
   resulting ProgramData upgrade authority, code hash, and exact multisig address. Keep the service
   unchanged until the sealed rehearsal passes.
5. **Nakshatra lane:** first land `pay/starter-attempt-fence` on main; then add the custody client,
   compile-once/assemble seam, rv5#3 plus setup-rerun wallet fixes, the shared durable attempt journal
   and reconciler for every broker-wallet send, CLI and starter wiring, and fake-transport tests. No
   test contacts the live approval service or Buzz.
6. **Rehearsal owner:** run `rehearsal/run.sh --mode import-nak-wallet` with throwaway material in the
   sealed container; archive the PASS and reviewed package manifest. The M6 live drill remains
   blocked until the deployment gate in step 4 is satisfied.
7. **Biswa:** decide whether and when to deploy the program under the 48-hour time-locked multisig.
   On his Mac, independently compute the reviewed package's manifest digest from the pinned commit.
   On the hub, compare that digest, receive the SHA-256-only wallet-copy count, and perform only the
   documented `sudo` install/import invocation. A later reviewed program upgrade requires a separate
   documented Biswa step to update the pinned program hash in root-owned `broker.env`; no daemon or
   caller advances that trust pin. He does not otherwise edit policy, move key files by hand, run
   tests against the live service, or approve a rehearsal transaction.
8. **Post-import verification by the unification lane:** as the unprivileged node user, confirm
   `wallet.pub`, `custody=broker`, absence of the seed from active user-readable paths, correct public
   projection, and a healthy authenticated principal. A live devnet approval is a separately
   scheduled acceptance drill, not part of Biswa's install work. Rollout stops before any withdrawal
   or mainnet action.

Order is fixed: design note, unification review, pure decoder plus fixtures, Biswa's deployment
decision and the authority report, broker/client build, sealed rehearsal, then Biswa's independent
digest and install. No mainnet enablement is bundled with this work.

## 9. Resolved answers and remaining decision

1. **Existing escrow required.** Import requires an existing escrow. `escrow_init`, deposit, revoke,
   escrow close, and legacy wallet-authority channel close stay outside this allowlist and fail
   closed. The current ensure path can emit `escrow_init` before `delegate_set`
   (`scripts/network/nak.py:1051-1098`), so that path must not use broker custody unless the escrow
   already exists.
2. **Trusted devnet state approved with strict pins.** The broker may use its separately configured
   TLS devnet RPC with genesis, ProgramData, code hash, and upgrade authority pinned in root-owned
   `broker.env`; RPC state may tighten but never loosen a message-byte check. The 2026-11-04 read-only
   check found that the configured program has no devnet account, deployment, ProgramData, or upgrade
   authority. Before step 4, the lane must report the authority created by deployment.
3. **Ceilings resolved.** The reviewed values are the `broker.env` table in section 4.2. They align
   with the existing 5-USDC/30-day self default, 0.03-SOL self target, and 0.01-SOL/30-day starter
   target (`scripts/network/pay_autochannel.py:18-22`; `scripts/network/nak.py:820-821,1053-1060`;
   `scripts/network/nakd.py:111-113,1408-1425`), while adding hard absolute ceilings.

The only remaining decision is Biswa's: whether and when to deploy the program, and which exact
48-hour time-locked multisig will hold upgrade authority. That deployment is required before rollout
step 4 or the M6 live drill; until then, only step 3 is authorized.

## 10. Review log

1. Added the broker-required stable per-act `idempotency_key` and uncertain-enqueue recovery rule.
2. Added broker-trusted `getFeeForMessage` quoting, rendering, recheck, and a hard environment ceiling.
3. Added exact 115-byte delegate and 165-byte ATA rent quotes plus explicit rent ceilings.
4. Pinned ProgramData and program-byte hash and required an immutable deployment to fence upgrades.
5. Added starter generation/disabled/open-work predicates and bounded resulting remaining capacity.
6. Made `pay/starter-attempt-fence` on main a prerequisite instead of claiming current retry safety.
7. Added `preflight_wallet` and `ensure_wallet` to the broker-custody setup-rerun changes.
8. Corrected redeemer custody: node-key payer only in broker custody; wallet payer in file custody.
9. **Round 2:** superseded item 4: kept the program upgradeable; pinned ProgramData, reviewed code
   hash, and the expected time-locked multisig authority, with explicit Biswa-controlled hash
   advancement after review.
10. **Round 2:** added system-owned zero-data dusted PDA/ATA state and ceilings on actual rent shortfall.
11. **Round 2:** required a pre-broadcast attempt record and race-safe reconciliation before any
    replacement approval for starter, ensure, or withdrawal sends.
12. **Round 3 (unification lane):** added message-byte absolute caps and RPC-only-tightens policy,
    linked withdrawal execute to its ceiling-checked request, made seed retirement mode-`000` with a
    SHA-256-only copy inventory, serialized and rate-limited approvals, isolated bounded RPC from the
    signer, resolved Q1/Q3, and recorded the Q2 fact and deployment/multisig gate. Verdict:
    send-to-build for rollout step 3 only.
