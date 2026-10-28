# Release signing and key rolls

Each node pins one Ed25519 release public key in `~/.nakshatra-node/config.json`. A release signed by
another key is accepted only when the channel publishes `keyroll.json`: an oldest-first chain of at
most eight `nak-release-keyroll/1` statements. Every link is signed by the old key (authorization)
and the new key (proof of possession).

## Approval-broker custody

The production path uses the broker principal `nakshatra-release`, action
`nakshatra.release.v1`, high tier, and biometric approval. The broker owns a dedicated Ed25519 key;
it is not the person key and the private half is never exported. Put its public half and client
configuration in a private builder-side directory (normally `~/.nakshatra-release`):

```text
release.pub              # exactly 64 lowercase hex characters
release-broker.json      # {"url":"http://HOST:PORT","token_file":"/absolute/token/path"}
```

The token file must be owned by the effective user, be a regular non-symlink file, and have mode
`0600`. The URL must be exactly `http://host:port` with no credentials, path, query, or fragment.
Redirects are refused. The client authenticates the `nakshatra-release` principal before enqueueing
and verifies every returned byte string and Ed25519 signature against `release.pub`.

A release build is one approval, not two: the client enqueues the unsigned manifest and a zero-hash
`latest.json` template together. It reconstructs the signed manifest locally, hashes its exact
pretty-printed bytes, requires the broker's filled pointer to contain that hash, and publishes
nothing until both signatures pass:

```bash
python3 release/build.py --version X.Y.Z --channel canary \
  --ref nakshatra=COMMIT --ref sthambha=COMMIT \
  --release-broker ~/.nakshatra-release \
  --out /path/to/nakshatra-dist
```

To extend the 60-day lifetime of the pointer for an already-built immutable version, use a separate
`refresh` approval. This first verifies that version's manifest with `release.pub`, hashes its exact
on-disk bytes, and only then atomically replaces `latest.json`:

```bash
python3 release/refresh_latest.py --version X.Y.Z --channel canary \
  --release-broker ~/.nakshatra-release \
  --out /path/to/nakshatra-dist
```

## Key-roll ceremony

Roll only after the release containing key-roll support is installed across the fleet. For the
initial move from the test file key, the old side remains local file custody while the new side uses
the broker. The broker refuses a statement whose `new_pubkey` is not its own `release.pub`:

```bash
python3 release/keyroll.py make \
  --old-key ~/.nakshatra-release/release-TEST.key \
  --new-broker ~/.nakshatra-release \
  --reason 'move release authority to the approval broker' \
  --out /path/to/nakshatra-dist/canary/keyroll.json
```

The complete ceremony is:

1. Import the `nakshatra-release` principal on the broker from its Mac-verified manifest. Import
   generates the dedicated release key in broker custody and exports only `release.pub`.
2. Approve one `keyroll` act to create the TEST-key-to-broker-key link; publish `keyroll.json`.
3. Approve one `release` act for the first broker-signed release. Publish it and let the fleet verify
   the roll chain and re-pin to the broker key.
4. Move `release-TEST.key` into root custody. In the same operational step, update the onboarding
   server's pinned release public key so newly issued invites pin the broker key.

Never remove older roll links while nodes may still pin an older key. `release/publish.sh` uploads
`keyroll.json` atomically before `latest.json`; this document does not change that publish boundary.

## Legacy and external signers

Build from explicit component refs. The Nakshatra ref supplies `release/spec.json`, so a dirty or
stale working-tree spec cannot enter the release:

```bash
python3 release/build.py --version X.Y.Z --channel canary \
  --ref nakshatra=COMMIT --ref sthambha=COMMIT \
  --release-signer '/path/to/new-approval-broker sign' \
  --release-pubkey NEW_HEX --out /path/to/nakshatra-dist
```

The signer command receives canonical JSON bytes on stdin and must print one hex Ed25519 signature.
The builder verifies every returned signature locally before publishing either the immutable version
or `latest.json`. File-backed `--release-key` remains available for test and migration work. These
three build signer options are mutually exclusive.

During update, a node walks forward only from its current pin and verifies both signatures on every
link. It switches and health-checks the complete release while the old pin remains committed. Only
after the install commits does it re-pin to the chain tip. A crash before commit restores the old
release and pin; a completed commit keeps the new pin. Previously trusted keys are recorded and
cannot be reintroduced by a later chain.
