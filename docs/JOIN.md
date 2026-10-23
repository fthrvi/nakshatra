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
are off by default. A stranger pastes the printed line just like an invite. It installs the exact
signed release named by the address, creates the stranger's identity locally, and sends only a
connection request. The stranger sees, plainly, that nothing is shared until the named person
accepts.

An address request always requires a person-signed interactive `nak accept <id>`; an MCP tool,
socket caller, or non-interactive CLI cannot accept it. `nak requests` labels it `address` rather
than `invite`. The receiver admits at most one pending address request per person key, 20 pending
address requests total, and 50 new address requests in a rolling 24 hours. Declining one blocks
that person key for 30 days. Blocks can also be managed directly:

    nak block <person-key-or-request-id> --days 30
    nak blocked
    nak unblock <person-key>

The address's rendezvous key (`ik`) is public by design: the address gives its holder only a way to
**ask**, never a way to message, receive starter credit, or become a contact. The v1 listener handles
one pairing at a time, so a flood can slow it down; relay limits and the local caps bound the damage
but do not make it unavailable-proof. Trust the address only as much as the place where you found
it—the signature proves it is intact and names a key, not who owns that key, exactly like an invite.

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

## Agent permissions

Join gives the agent only `nak.msg,nak.task.claim`. The `--caps` value below is the complete new
set, so keep the two join caps when widening it. A human runs one of these in a terminal and confirms
the change:

    # Also let the agent post tasks.
    nak grant renew --caps nak.msg,nak.task.claim,nak.task.post

    # Also let it pay. Spend limits are base credit units; choose your own limits.
    nak grant renew --caps nak.msg,nak.pay,nak.task.claim,nak.task.post --per-tx-cap 500 --day-cap 2000

The charter lasts at most 365 days. The credential sent to contacts lasts at most 7 days and renews
while the agent is used. After 30 idle days the grant lapses and a human must run `nak grant renew`
to wake it.

## Sharing compute

An operator makes a signed worker invite:

    nak invite --kind worker --tier known --note "GPU box"

The invite carries a one-time admission code. The joining node first creates its normal contact
identity, then proves that it owns its node key and receives worker standing. If the onboard service
is temporarily unavailable after contact setup, retry while the code is valid:

    nak worker-redeem

Build the engine shipped in the signed release, then turn on the worker. The inviter supplies the
package URL, its 64-hex signing key, the pillar URL and pin, the layer range, and the port:

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
checked by SHA-256. Today there is one signed model package: `llama32-3b`.

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

An operator can invite a laptop or harness as a consumer. It receives a ledger identity but is
never advertised, planned, or summoned as an inference worker:

    nak invite --kind consumer --tier known --starter 0 --note "laptop"

`known` is the default tier; `self` cannot be granted by an invite. `--starter N` is optional and
defaults to zero. A nonzero starter is a spend-only trial credit: it may be spent on the network but
can never be withdrawn. Only money the person deposits can cash out. On the devnet, accepting the
consumer's contact request automatically pays the exact amount in the person-signed invite through
the inviter's own `nak.pay` cap. A missing/lapsed grant or a not-yet-rostered consumer leaves the
payment visibly pending and retrying; it is never paid around the signer.

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
| Your friend | Your person key, node key, and the name you chose. **Not** your IP address, unless you both turn on direct connections for each other with `nak direct <name> on`. Direct mode shares LAN/IPv6 addresses; p2p hole punching also exposes public IPs to that contact. |
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

**Revoking an agent is local today.** Your signer stops signing, but contacts do not receive that
revocation. Someone with a copied agent key and its current wire credential can keep using it until
that credential expires, at most 7 days. Shared revocation is still on the list.

**Today's limits.** Releases are public at `http://45.63.109.137:8960`; each channel has a signed
`latest.json`. An inviter names that host in `~/.nakshatra/net/release-url`, or overrides it for one
invite with `nak invite --release-url http://45.63.109.137:8960`. A clean outsider join through that
host was verified end to end on `0.15.22` on 2026-10-23. Shareable request addresses now let a
stranger ask without an invitation, but there is still no public directory or listing. Person keys
are TEST files, not a vault.
