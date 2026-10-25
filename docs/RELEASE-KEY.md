# Release signing and key rolls

Each node pins one Ed25519 release public key in `~/.nakshatra-node/config.json`. A release signed by
another key is accepted only when the channel publishes `keyroll.json`: an oldest-first chain of at
most eight `nak-release-keyroll/1` statements. Every link is signed by the old key (authorization)
and the new key (proof of possession).

Roll only after the release containing key-roll support is installed across the fleet. Prepare the
new signer, then create the first link without putting production private keys in command lines:

```bash
python3 release/keyroll.py make \
  --old-signer '/path/to/old-approval-broker sign' --old-pubkey OLD_HEX \
  --new-signer '/path/to/new-approval-broker sign' --new-pubkey NEW_HEX \
  --reason 'move release authority to the approval broker' \
  --out /path/to/nakshatra-dist/canary/keyroll.json
```

For file-backed keys, use `--old-key FILE` or `--new-key FILE` instead of the corresponding signer
and public-key arguments. To make a later roll, use `append --chain ...` with the same signer options;
the old key must equal the existing chain tip. Never remove older links while nodes may still pin an
older key.

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
or `latest.json`. `release/publish.sh` uploads `keyroll.json` atomically before `latest.json`.

During update, a node walks forward only from its current pin and verifies both signatures on every
link. It switches and health-checks the complete release while the old pin remains committed. Only
after the install commits does it re-pin to the chain tip. A crash before commit restores the old
release and pin; a completed commit keeps the new pin. Previously trusted keys are recorded and
cannot be reintroduced by a later chain.
