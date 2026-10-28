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
- Factored secure config/token reads, URL validation, the no-redirect opener, bounded JSON transport,
  and uncertain-failure retry behavior into `release/broker_transport.py`. Person custody now uses
  that shared implementation; its behavior and tests remain unchanged.
- Updated `docs/RELEASE-KEY.md` with configuration, one-approval release/refresh use, and the
  TEST-key-to-broker-key ceremony.
- Added in-process loopback fake-broker tests for release/key-roll/refresh happy paths, single
  enqueue behavior, wrong signature, wrong `items_b64`, incorrect filled manifest hash, swapped
  order, redirect refusal, non-`0600` token refusal, no-publication guarantees, and wrong key-roll
  public key refusal.

No real broker, `10.42.0.1:8902`, approval UI, SSH, publish script, `systemctl`, `sudo`,
`journalctl`, or polkit path was contacted or invoked.

## Gates

Release/person/key-roll gate (isolated temporary `HOME`, required `PYTHONPATH`):

- **206 passed**, 0 failed, 2 pre-existing `SyntaxWarning` warnings.

Network matrix with `schema_migration_is_idempotent_on_a_forgejo_main_store` deselected:

- **1185 passed**, **1 deselected**, **9 failed**, 10 warnings.
- All nine failures are the documented main-branch `tests/test_nak_direct.py` p2p-toggle baseline
  (`_unit_enabled_or_active` receives the tests' `None` subprocess shim). There were **no other
  failures**.

Additional focused release/key-roll regression run before the prescribed gates:

- **123 passed**, 0 failed.
