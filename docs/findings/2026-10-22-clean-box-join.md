# Clean-box outsider join, 2026-10-22

## Result

The join did **not** succeed end to end. The public release was reachable and cryptographically
accepted, and release `0.15.20` installed and made the new identity. The run then hit a code bug:
the first install had not written `nak-signer.service` or `nak-net.service`. The 90-second service
health wait expired before the node could redeem the invite. Per the task's stop rule, the test
stopped there. No workaround or code change was made.

The test ran on the hub in a container named `nak-cleanbox-20261022`. Times below are wall times
observed during the run. The local date was 2026-10-22 (`America/Denver`); the container displayed
UTC.

## Clean box and network cut

I built a disposable image from `ubuntu:24.04`. It added only the prerequisites needed for this
test: systemd, a user D-Bus session, `curl`, CA certificates, Python 3, `iproute2`, and `sudo`. A
non-root `joiner` account owned the node. The image entrypoint installed the three unreachable
routes before executing systemd as PID 1:

```sh
ip route add unreachable 10.0.0.0/8
ip route add unreachable 172.16.0.0/12
ip route add unreachable 192.168.0.0/16
exec /sbin/init
```

The connected Docker `172.17.0.0/16` route is more specific than the `172.16.0.0/12` rejection, so
the Docker gateway remained reachable. The effective routes were:

```text
default via 172.17.0.1 dev eth0
unreachable 10.0.0.0/8
unreachable 172.16.0.0/12
172.17.0.0/16 dev eth0 proto kernel scope link src 172.17.0.6
unreachable 192.168.0.0/16
```

The first systemd launch without full container privileges exited before the test began. It minted
no invite and changed no Nakshatra state. I recreated the same named container with systemd's
required container privileges, `--cap-add NET_ADMIN`, and a cgroup mount. PID 1 then reported
`systemd` and the system state reported `running`.

The required isolation proof passed:

```sh
curl -m5 -fsS -o /dev/null -w 'HTTP=%{http_code}\n' http://10.42.0.1:8960/
# curl: (7) Failed to connect ...
# HTTP=000
# rc=7 elapsed_ms=4

curl -m5 -fsS -o /dev/null -w 'HTTP=%{http_code}\n' \
  http://45.63.109.137:8960/canary/latest.json
# HTTP=200
# rc=0 elapsed_ms=127
```

## Invite on the hub

The hub's configured release source was already public:

```sh
cat ~/.nakshatra/net/release-url
# http://45.63.109.137:8960
```

The baseline hub contacts were `bro` and `testbox`; there was no `cleanbox` contact. I ran the
documented command exactly:

```sh
nak invite --note "clean-box test 2026-10-22"
```

It returned in 52 ms. The one-line output named the public release host and the running canary
version. The secret invite and installer fingerprints are redacted here:

```sh
mkdir -p ~/nak-join && cd ~/nak-join && \
curl -fsSO http://45.63.109.137:8960/canary/0.15.20/install.py \
  -O http://45.63.109.137:8960/canary/0.15.20/releasekit.py && \
printf '%s  install.py\n%s  releasekit.py\n' <sha256> <sha256> | \
sha256sum -c --quiet && python3 install.py join 'nki1.<redacted>'
```

The real line was kept in a mode-`0600` temporary file, passed to the container on standard input,
and unlinked during cleanup. No invite, token, signature, or private key is in this file.

## Paste and install

I pasted that line as the unprivileged `joiner` user with its live systemd user session. The run
took 95.932 seconds and exited 1. Trimmed output:

```text
invite is intact and signed by key <redacted>…; installing Nakshatra from
http://45.63.109.137:8960 (canary)
installed 0.15.20 (canary)
setting up this node:
  · node key: created
  · person key: created (TEST custody: a file on this machine)
  · signer: set up
  · agent: 'agent' may nak.msg, nak.task.claim under a 365-day charter;
    active while used and lapses after 30 idle days
Failed to enable unit: Unit file nak-signer.service does not exist.
Failed to restart nak-net.service: Unit nak-net.service not found.
the messaging service did not come up; see: journalctl --user -u nak-net -n 30
```

The successful signed install also proves that the public host's release metadata and package
passed the pinned signature and SHA-256 checks. The failure was after installation and identity
creation.

## Code bug

This is an ordering bug between the installer and join setup:

- `release/install.py:749-765` writes an opt-in unit only if its signed `requires` path already
  exists. On first install, `~/.sthambha/signer/person.pub` and `~/.nakshatra/net/agent` do not yet
  exist, so `nak-signer.service` and `nak-net.service` are deliberately skipped.
- `scripts/network/join_setup.py:450-502` creates those identity files after the release install.
- `scripts/network/join_setup.py:505-517` then runs `systemctl --user daemon-reload`, enables the
  two unit names, and waits for `nakd`, but it never asks the signed release installer to write the
  newly eligible units.

The container confirmed that `~/.config/systemd/user/` contained only `nak-update.service` and
`nak-update.timer`, while the signed manifest did contain both `nak-signer` and `nak-net` service
definitions. This blocks every fresh systemd join on release `0.15.20`.

I did not take the one-line-fix exception. A correct fix needs a regression test, a new signed
release, and another clean-box run; working around the public release inside the test would not
prove the documented path.

## Steps not reached

| Step | Result |
|---|---|
| Public release reachable outside mesh | Passed |
| Invite uses `~/.nakshatra/net/release-url` | Passed |
| Invite signature and expiry check | Passed |
| Signed release install and hash checks | Passed |
| Node, person, signer, and agent creation | Passed |
| Agent caps and printed charter/idle policy | Passed: `nak.msg,nak.task.claim`, 365-day charter, 30 idle days |
| `nak-signer` and `nak-net` start | Failed: units absent |
| Connection request reaches inviter | Not reached |
| Inviter accepts as `cleanbox` | Not reached |
| Contacts and two-way messages | Not reached |
| `nak-mcp` initialize and `tools/list` | Not reached |
| Live `nak grant renew --caps …` on joined node | Not reached; the same commands passed later against a temporary identity under the installed signed release |
| Restart and service persistence | Not reached |

The hub already had one unrelated old pending request named `friend`. The failed clean box created
no new request, and I did not touch the existing one.

## Documentation divergences and corrections

1. `JOIN.md` said the four-step path worked. It stops before services and invite redemption on a
   fresh `0.15.20` install. The guide now opens with the blocker and marks the exact failed step.
2. `JOIN.md` said releases were mesh-only. The public host returned 200, the invite named it, and
   the signed install succeeded. The guide now documents the public URL, `release-url`, and
   `--release-url` override.
3. The guide did not name its executable prerequisites. The real one-liner uses `curl`,
   `sha256sum`, and `python3`; these are now named.
4. The installed OpenClaw `2026.4.5` has `openclaw mcp set`, not the documented `mcp add` form.
   `HARNESSES.md` now uses the installed syntax.
5. The worker section still said `worker.sh`. The current CLI is
   `nak worker build/on/off/status`, with signed packages, idle unload, and an optional owner lease. The guide
   now shows that flow and names today's single signed package, `llama32-3b`.
6. The old revocation paragraph said a joined grant lasted 90 days. Join actually issues a
   365-day charter with a sliding wire credential capped at 7 days and a 30-day idle lapse. The
   revocation exposure is now stated as at most 7 days for a copied agent key plus current lease.
7. The guide did not show the human grant-widening commands. It now preserves the join caps while
   adding `nak.task.post` or `nak.pay`; the payment example also sets the required spend limits.
8. Starter credits were described only as a pending transfer. The guide now says the transfer is
   not wired and records the policy: starter value is spend-only trial credit, never cashable; only
   deposited money can cash out.
9. The guide did not warn that an idle WSL distro stops the node. It now does and links the Windows
   worker notes without editing that separately owned file.
10. Harness docs claimed 22 MCP tools and omitted four typed-agent tools. The server exposes 18
    default tools plus 3 opt-in connection tools, 21 total; the table now lists all of them.

## Harness and grant checks after cleanup

These checks did not stand in for the blocked clean-box steps. They only verified that the refreshed
examples match the software installed on the hub:

- `which` found OpenClaw `2026.4.5`, Claude Code, and Codex. Hermes was absent. Their local help
  confirmed `openclaw mcp set`, `claude mcp add --scope user ... -- <command>`, and Codex's
  `[mcp_servers.<name>]` stdio table. The Codex form also matches its official MCP documentation.
- Sending JSON-RPC `initialize` and `tools/list` by hand to the hub's installed `nak-mcp` returned
  server `nakshatra` version `0.3` and 18 default tools. Source and help confirmed the 3 additional
  `--allow-connect` tools.
- Under a new temporary HOME, the installed signed release created a fresh identity and accepted
  both documented grant forms (automation added `--yes` because it had no terminal). The final
  grant had `nak.msg,nak.pay,nak.task.claim,nak.task.post`, spend limits 500/2000, a 7-day wire
  lease, 30-day idle lapse, and 365-day charter. The temporary identity was deleted afterward.

## Test results

The requested command was run exactly, without extra environment:

```sh
pytest -q tests -k "doc or join"
```

It failed during repository-wide collection before running the selection: 37 collection errors
from the host test environment (`hivemind`, `petals`, `grpc`, and `pytest_asyncio` missing; an old
protobuf; `scripts/packaging` shadowing the installed package; and `INITIAL_PEERS` unset).

Running the directly named join files that do not depend on that environment passed:

```text
136 passed in 0.87s
```

An additional combined join/grant run produced 150 passes and 93 failures. Every reported failure
was caused by `tests/test_nakd.py` loading the older `~/sthambha-signer`, which lacks
`validate_agent_name` and the sliding-lease argument. The isolated installed-release grant check
above covered the two exact doc commands despite that test-path skew. `git diff --check` passed.

## Cleanup proof

Because the failure happened before invite redemption, there was no `cleanbox` contact to remove.
After stopping and removing the disposable container:

```text
HUB_CONTACTS_AFTER_CLEANUP
bro              active    online via direct ...
testbox          active    offline ...

CONTAINER_LOOKUP
<no rows>

SECRET_FILES
absent /tmp/nak-cleanbox-20261022.invite
absent /tmp/nak-cleanbox-20261022.invite.stderr
```

No hub configuration, service, grant, or existing contact was changed or restarted.
The disposable `nak-cleanbox-systemd:ubuntu24.04` image was also removed.
