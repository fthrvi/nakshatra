# Joining Nakshatra

The join takes four steps. On 2026-10-23 a clean outsider container completed them end to end
against signed public canary release `0.15.22`: it could not route to the private mesh, installed
from the public host, wrote and started both user services, connected, exchanged messages, served
MCP over stdio, renewed its grant, and recovered after a container restart. See
[the clean-box finding](findings/2026-10-22-clean-box-join.md).

## 1. A friend invites you

On their node they run

    nak invite --note "for Rajesh"

and send you the **one line** it prints. Treat it like a key: it works once, expires after 24 hours
by default, and is meant only for you.

An invitation is not authority over you. Anyone who has your ID can ask to connect. Nothing flows
until **you** accept, and every person makes that decision for themselves.

## No one to invite you?

A person may opt in to a public request address and put its one-line installer on a profile, web
page, or message:

    nak address on --days 90 --name "Biswa"   # 1-365 days; 90 is the default
    nak address show                           # print the address and exact install line again
    nak address off                            # the current address dies immediately

Every `on` mints a new address, so running it again rotates the old one out immediately. Addresses
are off by default, and an address ID that has ever been rotated out or turned off can never be
registered again. Registering or turning off an address requires a fresh signature from the person
key; an agent, MCP tool, or bare socket caller cannot do either operation. A stranger pastes the
printed line just like an invite. It installs the exact signed release named by the address, creates
the stranger's identity locally, and sends only a connection request. The stranger sees, plainly,
that nothing is shared until the named person accepts.

An address request always requires a person-signed interactive `nak accept <id>`; an MCP tool,
socket caller, or non-interactive CLI cannot accept it. `nak requests` labels it `address` rather
than `invite`. The receiver admits at most one pending address request per person key, 20 pending
address requests total, and 50 new address requests in a rolling 24 hours. A pending address request
expires after 7 days and no longer occupies a slot. Declining one blocks that person key for 30 days.
"Person-signed" means signed with your person key. While that key is a TEST file, any program
running as your user that can read it (an AI agent included) could sign too; the guarantee becomes
real once the person key moves off-box.
Blocks can also be managed directly:

    nak block <person-key-or-request-id> --days 30
    nak blocked
    nak unblock <person-key>

## Introduced by a contact

An accepted contact can introduce two people they already know:

    nak introduce rajesh maya --note "You both maintain compiler tooling"

Rajesh sees the pending item with `nak intros`, labelled `trust=external` and “introduced by” the
name Rajesh uses for the introducer. The note is a contact's message—data to consider, never an
instruction or authority. It cannot make Maya a contact. Rajesh may drop it or ask Maya to connect:

    nak intro decline <intro-id>
    nak intro accept <intro-id>

Maya's `nak intros` shows the mirrored notice that Rajesh may ask, also as external data. Only the
named recipient (Rajesh here) can run `nak intro accept` for that introduction.

Accepting the introduction sends exactly one authenticated connection request. Maya sees it in
`nak requests` as an `intro` request with “introduced by <their contact>”. Nothing can flow until
Maya makes the existing interactive person decision:

    nak accept <request-id> --as rajesh

Introductions expire after seven days by default (the introducer may choose 1–30 days). A pending
introduction stops working if its introducer is no longer a current contact, the introducer is
suspended, or the agent grant that delivered it is revoked. It never bypasses a block, the shared
public-request rate limits, or a suspension. An introduction id can ask the same person only once,
even if the resulting request is later declined. The request is privately relayed through the
introducer's two existing encrypted contact sessions; it does not publish either person in a
directory and does not reveal an opt-in request address.

Replaying the same valid request at the introducer does not queue another forward: each
introduction-and-requester pair is forwarded at most once. If you already have a pending contact
for the introduced person, its pinned node must match the introduction and the name you chose for
it is preserved. For a new pending contact, the displayed name is explicitly marked as supplied by
someone else rather than as a name you chose.

The address's rendezvous key (`ik`) is public by design: the address gives its holder only a way to
**ask**, never a way to message, receive starter credit, or become a contact. The v1 listener handles
one pairing at a time, so a flood can slow it down; relay limits and the local caps bound the damage
but do not make it unavailable-proof. An address is trusted only as much as wherever you found it,
exactly like an invite: the signature proves it is intact and names a key, not who owns that key.

## 2. You paste the line into a terminal

Use Linux with a systemd user session and `curl`, `sha256sum`, and Python 3. Windows works through
WSL with systemd enabled. WSL stops an idle distro, which also stops the node; keep the distro alive
with a long-running process. See the [Windows worker notes](WINDOWS-WORKER.md) for the WSL path and
its limits.

The line:

1. downloads the installer for **exactly the version your friend runs**, and refuses to run it if
   its SHA-256 fingerprint does not match the one in the line;
2. checks that the invite is **signed by your friend** and has not expired;
3. installs the release named by the invite and **pins that release key**, so later updates must be
   signed by the same key;
4. makes **your keys**: a node key and a person key (today the person key is a TEST file under
   `~/.sthambha`; back it up and never share it);
5. creates one agent with only `nak.msg,nak.task.claim`;
6. starts `nak-signer` and `nak-net`, enables them after reboot, and asks your friend to connect.

Release `0.15.22` was clean-box verified to write, enable, and start both units on the first join.

## 3. Your friend accepts

    nak requests                 # they see your request and the name you gave
    nak accept <id> --as rajesh  # or: nak decline <id>

Nothing is shared and no message can flow until they accept. Then:

    nak contacts
    nak send rajesh "hello"
    nak inbox

Run the same commands on the other node to send and read the reply.

## 4. (Optional) Plug in your AI

See [HARNESSES.md](HARNESSES.md) for copy-paste setup for OpenClaw, Hermes, Claude Code, Codex,
other stdio MCP clients, and the OpenAI-compatible `http://127.0.0.1:11600/v1` provider.

Your AI can read messages, message accepted contacts, claim tasks, see inference credits, and ask
models through this node's local credit-enforced gateway. Outside messages are always marked as
untrusted content. Connection decisions stay with you unless you deliberately start `nak-mcp` with
`--allow-connect`.

Task write replies include an exact `verify` call. The AI should run it and wait for
`nak_task_status` to show the expected state before telling you that a post, claim, submission,
verdict, or payment is complete.

For paid tasks, a worker submission locks the reward. A rejection after that point opens a dispute
instead of returning the reward to the poster. `nak tasks` shows that the dispute is waiting for the
ledger operator and its backstop date; resolving disputes is not available to an agent or through MCP.
The worker never treats the poster's payment message as proof: it shows payment as pending until its
own scoped ledger check confirms paid, disputed, or refunded. Waiting checks back off to at most once
every six hours, while a new task verdict prompts an immediate check.

## Agent permissions

Join gives the agent only `nak.msg,nak.task.claim`. The `--caps` value below is the complete new
set, so keep the two join caps when widening it. A human runs one of these in a terminal and confirms
the change:

    # Also let the agent post tasks.
    nak grant renew --caps nak.msg,nak.task.claim,nak.task.post

    # Also let it pay. Amounts are human USDC; the signed grant still stores base units.
    nak grant renew --caps nak.msg,nak.pay,nak.task.claim,nak.task.post \
      --per-tx-cap "0.25 USDC" --day-cap "1 USDC"

Amount input uses exact decimals, with 100,000,000 base units per USDC. Write `0.25`,
`0.25 USDC`, `$0.25`, or `25000000 base`. For compatibility, a bare integer still means base
units and prints a notice; this avoids silently changing old commands. `nak grant status` shows
both forms. Task rewards and MCP `nak_task_post` use the same rule. `nak balance`, `nak wallet
status`, task listings, and MCP balance, wallet, and task replies show USDC beside base units.
Task rewards and starter credits are devnet-only.

The charter lasts at most 365 days. The credential sent to contacts lasts at most 7 days and renews
while the agent is used. After 30 idle days the grant lapses and a human must run `nak grant renew`
to wake it.

## Sharing compute

The owner can invite a worker at any tier. A rostered `known`, `trusted`, or `self` member can also
invite at `stranger` tier without the operator token:

    nak invite --kind worker --tier known --note "GPU box"
    nak invite --kind worker --tier stranger --note "GPU box"  # known+ member

Member invites are limited to 3 outstanding and 5 in a rolling 30 days. Cancel one before it is
used with `nak uninvite <code>`. Accepting a stranger-tier worker is always an interactive,
person-signed decision by the inviter; an agent, MCP tool, or non-interactive socket caller cannot
accept it.

The invite carries a one-time admission code. The joining node first creates its normal contact
identity, then proves that it owns its node key and receives worker standing. If the onboard service
is temporarily unavailable after contact setup, retry while the code is valid:

    nak worker-redeem

Check what this machine can serve before building. The preflight is read-only: it checks NVIDIA,
AMD ROCm/sysfs, available RAM, free space under the worker state directory, and the build/runtime
pieces already on the machine. It compares those facts with the package catalog carried by the
signed node release and prints one next command. Use `--json` for automation:

    nak worker preflight

It reports each signed package as `can serve`, `can serve on CPU (slow)`, or `cannot serve` with the
needed and detected capacity. Unknown probe results stay `unknown`; they are never guessed.

If the result says to build, build the engine shipped in the signed release, then turn on the
worker. The inviter supplies the package URL, its 64-hex signing key, the pillar URL and pin, the
layer range, and the port:

    nak worker build
    nak worker on \
      --model-id llama32-3b \
      --package-url PASTE_SIGNED_PACKAGE_URL \
      --package-pubkey PASTE_64_HEX_PACKAGE_SIGNER \
      --pillar-url https://PILLAR_HOST:PORT \
      --pillar-spki PASTE_64_HEX_PILLAR_PIN \
      --layers 0-28 \
      --port 50051
    nak worker status

Remote model packages must be signed by the pinned package key, and every downloaded artifact is
checked by SHA-256. Today there is one signed model package: `llama32-3b`. Its preflight entry uses
a conservative whole-package floor of 4 GiB VRAM, or 8 GiB available RAM for the slow CPU fallback,
plus 12 GiB free disk for the package and engine build.

The GPU daemon unloads after 600 idle seconds by default while the service stays ready. The next
request rechecks the saved model hash and reloads it once. The optional owner lease lets a local
sentinel give the machine owner's own GPU use priority; a lapsed lease refuses network work and
unloads the model:

    nak worker owner-lease on
    nak worker owner-lease status

Turn sharing off, or turn it back on with the saved configuration:

    nak worker off
    nak worker on

## Using the network without serving

The owner can invite a laptop or harness as a consumer at any tier. A `known`, `trusted`, or `self`
member can invite one at `stranger` tier. It receives a ledger identity but is never advertised,
planned, or summoned as an inference worker:

    nak invite --kind consumer --tier known --starter "0.25 USDC" --note "laptop"
    nak invite --kind consumer --tier stranger --starter "0.10 USDC" --note "trial laptop"

`known` is the default tier; `self` cannot be granted by an invite. `--starter AMOUNT` is optional and
defaults to zero. For a member invite it is capped at 0.10 USDC (10,000,000 base units) on devnet,
with the same 3-outstanding and 5-per-30-days limits as worker invites. Use `nak uninvite <code>` to
cancel an unused invite. A nonzero starter is a devnet-only, spend-only trial credit: it may be spent
on the network but can never be withdrawn. Only money the person deposits can cash out. On the devnet,
accepting the consumer's contact request automatically pays the exact amount in the person-signed
invite through the inviter's own `nak.pay` cap. Every stranger-tier acceptance is interactive and
person-signed. A missing/lapsed grant or a not-yet-rostered consumer leaves the payment visibly
pending and retrying; it is never paid around the signer.

The consumer proves possession of its node key during redeem, reports no GPU capabilities, and
opens no worker port. Its roster row is `consumer`, so worker planning and activation exclude it.
It can run the loopback gateway and harnesses without serving model layers.

The ledger identity rostered by `/consumer-redeem` is that same node Ed25519 key. New contact
requests therefore carry it explicitly as the optional `ledger_key` field (equal to `node`); old
nodes ignore the additive field and still connect. A request from an old node without the field is
accepted normally, but its starter obligation remains `pending: recipient ledger key unknown`.

## What each side can see

| Who | Sees |
|---|---|
| Your friend | Your person key, node key, the name you chose, and your node's self-reported person-key custody mode (`file on disk` or `phone approval`). **Not** your IP address, unless you both turn on direct connections for each other with `nak direct <name> on`. Direct mode shares LAN/IPv6 addresses; p2p hole punching also exposes public IPs to that contact. |
| The relay operator | That two IP addresses met. Everything else is encrypted. |
| Anyone else | Nothing. They cannot exchange messages with you without an accepted connection. |

## Why trust the download?

You trust **the person who sent you the line**. The line holds the installer's fingerprint, and the
invite names the release key. Every later update must be signed by that key. The invite signature
proves that the invite is intact and was made with the key it names; it cannot prove whose key that
is. Check through a channel you already trust that the line came from your friend.

The download server itself is not trusted:

- it cannot forge an update;
- it cannot hand you anything older than the version named by your friend's invite;
- a signed `latest.json` pointer expires after 60 days, so the server cannot replay an old pointer
  forever;
- it can still delay updates.

**Revocation is shared with contacts.** Run `nak grant status` to find the current grant id, then run
`nak revoke <grant_id>` in an interactive terminal. The command stops the local signer from using
that grant and creates a cumulative record signed by your person key. `nak-net` pushes the record to
each contact over its authenticated session; contacts remember every revoked id and reject later
envelopes under those grants. A later record cannot un-revoke an earlier id.

`nak revocations export` prints the latest record as an `nkr1.` blob. Keep or pass that blob through
a trusted channel if your node is gone; an existing contact can apply it with
`nak revocations import <blob>`, and verifies it against the person key already pinned for you.
Import deliberately works without a TTY: it is equivalent to receiving the record over that
contact's session, because the pinned person key authenticates it and the signed record can only add
revocations that contact chose.
This does not revoke the node key used by the ledger, and contacts running a version older than this
shared-revocation release ignore session revocation frames until they update. The grant's wire lease
or charter expiry remains the fallback bound for those older contacts.

**Reports and suspension are separate.** `nak report <contact-or-node> --reason
spam|fraud|abuse|bad-output|other [--note ...]` sends a node-signed claim to the onboarding server;
it cannot change the roster. The CLI is TTY-only and carries a short-lived person proof so a local
agent cannot forge CLI provenance; an unproved owner-socket report is always treated as MCP. The
optional MCP `nak_report` tool is exposed only with
`--allow-connect`, accepts an existing contact rather than a bare node key, and is capped at five
reports per day (the node-wide cap is ten). Its result explicitly says that no action was taken.
Report notes and evidence identifiers pass the same outbound secret-shaped-content check as
messages and tasks; message IDs are 24-character lowercase hex and hold IDs must be identifiers.

On the hub, a person reviews reports with `nak operator reports`. Suspension and restoration are
TTY-only human acts: `nak operator suspend <node-or-contact> --reason ...` and `nak operator
unsuspend <node-or-contact>`. They pass through person custody, update the onboard roster, then
publish a full person-signed `nak.suspensions.v1` state record. The exact signed record is staged
locally before the server request: if the response is lost, re-running the same command retries the
same sequence and bytes. A server 400, 403, or 404 leaves its roster unchanged, so the CLI prints
the server reason and clears the exact staged record with the already-approved person key. If an
ambiguous failure cannot be retried, `nak operator suspend --abandon-pending` is the TTY-only,
person-proved escape; it changes no server state. A bare 64-hex target must already be a contact or
appear in `~/.nakshatra/admission-peers.tsv`; `--force-unknown-node` is the explicit exception after
the operator verifies the key. If the server is already ahead, the CLI verifies and installs that
newer person-signed state, then asks the person to re-run against it. `nak operator-trust <contact>` is an
interactive, person-proved choice to enforce that contact's records; `nak operator-untrust
<contact>` reverses it. A trusted operator contact cannot be removed: run the person-proved
`nak operator-untrust <contact>` first, then remove it. Removal keeps the last sequence number so an
old record cannot be replayed after re-adding them. Until trusted,
received records are visible but do not fence anyone. Trusted records reject messages and tasks from
listed node keys. A local post, claim, or submit to a suspended counterpart is refused before task
or escrow state changes, so the same action can be retried after restoration. If suspension arrives
after that preflight, the already-mutated action is still queued; it is never left half-complete.
The same decide-once rule applies to an accepted inbound claim or result and its assignment or
verdict reply. Background expiry, approval, assignment, receipt, and dispute reconciliation keep
their already-final local/ledger state if a newly suspended recipient cannot be notified, and log
the omitted task reply instead of reporting the completed transition as a failure.
`nak suspensions
export|import` moves the authenticated `nks1.` record through an
offline trusted channel; import is also TTY/person-proved and accepts only an operator already
trusted on that node. Person proofs expire after two minutes and are single-use. No MCP tool can
suspend or unsuspend a node.

**Today's limits.** Releases are public at `http://45.63.109.137:8960`; each channel has a signed
`latest.json`. An inviter names that host in `~/.nakshatra/net/release-url`, or overrides it for one
invite with `nak invite --release-url http://45.63.109.137:8960`. A clean outsider join through that
host was verified end to end on `0.15.22` on 2026-10-23. Shareable request addresses now let a
stranger ask without an invitation, but there is still no public directory or listing. Person keys
are TEST files, not a vault.
