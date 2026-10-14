# Joining Nakshatra

Four steps. You don't need an account, a website, or anyone's permission except the friend who
invites you.

## 1. A friend invites you

On their node they run

    nak invite --note "for Rajesh"

and send you the **one line** it prints. Treat it like a key: it works once, expires (24 h by
default), and is meant only for you.

## 2. You paste the line into a terminal

Linux, or Windows with WSL (systemd on). The line:

1. downloads the installer of **exactly the version your friend runs**, and refuses to run it if
   its fingerprint (sha256) doesn't match the one in the line;
2. checks the invite is **signed by your friend** and has not expired;
3. installs the node from the release your friend's invite names, **pinning that release key**:
   every later update must be signed by it;
4. makes **your keys**: a node key, and a person key (TEST: a file in `~/.sthambha`; back it up,
   never share it);
5. creates one **agent** that may message and claim tasks, nothing else. It cannot post tasks
   or pay; you grant those separately;
6. starts the services, which come back after a reboot, and **asks your friend to connect**.

## 3. Your friend accepts

    nak requests                 # they see your request and the name you gave
    nak accept <id> --as rajesh  # or: nak decline <id>

Nothing is shared and no message can flow until they accept. Then:

    nak contacts                 # each of you sees the other as active
    nak send rajesh "hello"      ·   nak inbox

## 4. (Optional) Plug in your AI

    # OpenClaw
    openclaw mcp add nakshatra --command ~/.local/bin/nak-mcp
    openclaw config set tools.toolSearch false   # small local models can't use tool search

    # Hermes: an mcp_servers entry whose command is ~/.local/bin/nak-mcp

Your AI can then read messages (always marked as untrusted outside content), message your
contacts, take on tasks your contacts post, see inference credits, and ask models through this
node's local, credit-enforced gateway. It can't accept connections; that stays with you. See
[HARNESSES.md](HARNESSES.md) for OpenClaw, Hermes, Claude Code, Codex, every MCP tool, and the
OpenAI-compatible `http://127.0.0.1:11600/v1` provider setup.

## Sharing compute

An operator can make a signed worker invite with the same installer and identity path:

    nak invite --kind worker --tier known --note "GPU box"

The command stages a one-time admission code with the onboard service (using
`NAK_ONBOARD_TOKEN`, or `~/.nakshatra/onboard.token` if that file is mode `0600`) and seals the
code, tier, and redeem URL inside the signed invite. The joining box creates the normal node and
contact identity first, then proves possession of that node key to redeem its worker standing. It
prints the admitted tier and junction. If the onboard service is temporarily unavailable, contact
setup remains complete and the box can retry while the code is valid:

    nak worker-redeem

Worker execution itself still uses the existing `worker.sh` runtime until phase B; phase A unifies
the signed invite, installer, admission proof, and node key.

## What each side can see

| Who | Sees |
|---|---|
| Your friend | your person key, node key, the name you chose. **Not** your IP address, unless you BOTH turn on direct connections for each other (`nak direct <name> on`): then they learn your LAN/IPv6 addresses. If p2p is also on, the hole-punch connection exposes your public IP to that contact. |
| The relay operator | that two IP addresses met. Everything else is encrypted. |
| Anyone else | nothing. They can't reach you without an accepted connection. |

## Why trust the download?

You trust **the person who sent you the line**. The line holds the installer's fingerprint, and
the invite names the release key. Every later update must be signed by that key. The invite's
signature proves the invite is intact and was made with the key it names; it can't prove *whose* key
that is. So check with your friend, through a channel you already trust, that the line came from them.

The download server itself is not trusted:
- it can't forge an update;
- it can't hand you anything older than the version your friend's invite names;
- a signed "latest" pointer expires after 60 days, so the server can't replay an old one forever;
- it *can* still delay updates.

**Revoking an agent is local today.** If you revoke one of your agents, your own signer stops signing for
it, but your contacts don't hear about it: a copied agent key keeps working toward them until its grant
expires (90 days for the agent a join creates). Shared revocation is on the list.

**Today's limits.** Releases are served from the mesh (`10.42.0.1:8960`), so only machines on the
mesh can join until a public release host is set up. Person keys are TEST files, not a vault.
