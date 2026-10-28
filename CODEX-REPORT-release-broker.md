# CODEX report — release approval-broker client

Date: 2026-10-28
Branch: `release/broker-signer`

## What changed

- Added `release/broker_signer.py` for the `nakshatra-release` principal. It loads the closed
  `release-broker.json` config, a user-owned `0600` token, and `release.pub`; refuses redirects;
  preserves uncertain result/sign failures; and locally verifies every returned item and Ed25519
  signature.
- Added the one-enqueue `release` batch. The client sends canonical manifest bytes plus a zero-hash
  latest template with `record_ref=0`, reconstructs the signed manifest locally, recomputes the
  exact pretty-JSON manifest hash, and refuses substituted bytes, a different filled hash, swapped
  output, or invalid signatures before anything is published.
- Added broker `refresh` and `keyroll` acts. Refresh verifies the existing immutable manifest before
  atomically replacing `latest.json`; key-roll refuses a statement whose `new_pubkey` differs from
  the broker's `release.pub`.
- Added `--release-broker CONFIG_DIR` to `release/build.py`, mutually exclusive with the existing
  file and external signers; added `--new-broker CONFIG_DIR` to the new side of `release/keyroll.py`;
  and added `release/refresh_latest.py`.
- Shared secure config/token reads, URL validation, the no-redirect opener, bounded JSON transport,
  and uncertain-failure retry behavior in `scripts/network/broker_transport.py`. Person custody and
  release signing both use this node-shipped module.
- Updated `docs/RELEASE-KEY.md` with configuration, one-approval release/refresh use, and the
  TEST-key-to-broker-key ceremony.
- Added in-process loopback fake-broker tests for release/key-roll/refresh happy paths, single
  enqueue behavior, wrong signature, wrong `items_b64`, incorrect filled manifest hash, swapped
  order, redirect refusal, non-`0600` token refusal, no-publication guarantees, and wrong key-roll
  public key refusal.

## Fix round 1

- Moved the shared broker transport from `release/` into `scripts/network/`. Installed nodes have
  only the release's `scripts` tree and Sthambha on `PYTHONPATH`, so `network.person_custody`,
  `network.nak`, and `network.nakd` no longer depend on an unshipped `release` package. Release-side
  code resolves the repository `scripts` directory from `__file__` and imports the same module.
- Added an isolated subprocess regression that starts from a temporary working directory with
  `python -I`, prepends exactly the release `scripts` and Sthambha roots, and imports all three node
  modules. It passes. Inspection of `release/install.py` and `release/spec.json` confirmed that the
  installed health check and command wrappers import only shipped scripts/Sthambha modules.
- Hardened `refresh_latest_pointer` to require `schema == releasekit.MANIFEST_SCHEMA`, matching
  version/channel, and a valid signature under `release.pub`. A signed `latest.json` copied into a
  version's `manifest.json` is now refused before enqueue and leaves `latest.json` absent.
- Restored tracked `CODEX-REPORT.md` byte-for-byte from `forgejo/main`; this lane's report is this
  separate file. `CODEX-TASK.md` and `CODEX-FIX1.md` are not included in the commit.

No real broker, `10.42.0.1:8902`, approval UI, SSH, publish script, `systemctl`, `sudo`,
`journalctl`, or polkit path was contacted or invoked. Every test used a fresh temporary `HOME`;
all HTTP brokers and network relays were in-process and loopback-only.

## Gates

- Prescribed release/person/key-roll gate: **208 passed**, 0 failed, 2 pre-existing
  `path_probe.py` `SyntaxWarning` warnings.
- Prescribed network matrix: **1185 passed**, **1 deselected**, **9 failed**, 10 warnings. All nine
  failures are the documented main-branch `tests/test_nak_direct.py` p2p-toggle baseline
  (`_unit_enabled_or_active` receives the tests' `None` subprocess shim); there were no other
  failures.
- New broker/import tests alone: **12 passed**, 0 failed.
- `py_compile` for all changed Python modules/tests and `git diff --check` passed.
