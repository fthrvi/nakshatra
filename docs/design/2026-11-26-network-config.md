# Payment network configuration

Nakshatra payment code reads one node-local file:
`~/.nakshatra/pay/network.json`. The file must be a user-owned, regular (not
symlinked) mode `0600` file. Its `active` field selects one entry from
`networks`; changing that one value and restarting the installed payment
services is the whole network switch. No code change or release is involved.

```json
{
  "active": "devnet",
  "networks": {
    "devnet": {
      "rpc": "https://api.devnet.solana.com",
      "genesis": "EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG",
      "program": "GLWqQsSqUsDkPvkc8rxcoQUyro7ZQd5denht21NtdTTo",
      "mint": "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU",
      "explorer_cluster": "devnet"
    },
    "mainnet": {
      "rpc": "",
      "genesis": "5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d",
      "program": "",
      "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
      "explorer_cluster": ""
    }
  }
}
```

Remote RPCs must use HTTPS; HTTP is accepted only for loopback validators. Every
payment path asks the selected RPC for `getGenesisHash` and refuses it unless it
equals the selected entry. When the file exists it is authoritative: stale
`NAK_PAY_RPC`, `NAK_PAY_PROGRAM`, or `NAK_PAY_MINT` values that disagree are an
error, never a mixed configuration. If the file is absent, the old environment
configuration and its devnet-only mainnet refusal remain unchanged.

Commands:

- `nak pay network init` migrates the current devnet `NAK_PAY_*` values once.
  Remove those legacy values from service environments after migration; leaving
  equal values is tolerated, but a future network switch correctly refuses stale
  values that disagree.
- `nak pay network status` shows the active name, RPC host only, public program
  and mint, plus every mainnet gate.
- `nak pay network use NAME` changes only `active`, keeps a timestamped backup,
  and names the installed user services that the operator must restart.
- `nak pay network arm-request-info` prints this node's public key and the exact
  mainnet RPC, program, limits file, and limits SHA-256 for the release hub.
- `nak pay network arm mainnet --file FILE` verifies and installs a release-key
  signature made after the owner approves the request with Face ID. The signed
  statement binds the node key, genesis, RPC, program, mint, and exact
  `limits.mainnet.json` bytes. Any later edit disarms it.
- `nak pay network disarm` removes that arm record.

## Mainnet switch

Mainnet is the owner's decision. Nothing in this change deploys a program,
funds a wallet, restarts a service, or enables mainnet by itself.

1. Deploy and audit the escrow program on Solana mainnet.
2. On each node, put the mainnet HTTPS RPC and audited program ID in
   `network.json`; create explicit, mode-`0600`
   `~/.nakshatra/pay/limits.mainnet.json` spend limits. Devnet continues to use
   `limits.json` and may create its defaults automatically; mainnet never does.
3. On the node, run `nak pay network arm-request-info`.
4. On the release hub, use those exact values:
   `python3 release/arm_mainnet.py --node-pubkey HEX --rpc HTTPS_URL
   --program PUBKEY --limits /copied/limits.mainnet.json --out NODE.arm`.
   Approve the broker request with Face ID, then copy `NODE.arm` to the node.
5. On the node, run `nak pay network arm mainnet --file NODE.arm`, then
   `nak pay network use mainnet`, and restart the user services it prints.
6. Fund the node with real USDC and enough SOL for fees and rent.

A process running as the same Unix user can still edit `network.json` or either
limits file. It cannot produce a valid replacement arm statement: that requires
the release key held behind the hub's Face ID approval broker. Editing a bound
value therefore makes the mainnet gate fail closed until the owner signs a new
statement. The spend path parses only the limits bytes the signed statement
names, so swapping the limits file after a check cannot take effect.

What this does NOT protect against: code that is hostile and running as that
same user. Such code can rewrite the pinned release key in
`~/.nakshatra-node/config.json` or the installed Nakshatra code itself, and on a
file-custody node it can simply read `~/.nakshatra/pay/wallet.ed25519` and sign
mainnet transactions without Nakshatra at all. The Face ID arm stops mistakes,
misconfiguration and agents acting through `nak` from moving real money; it is
not a sandbox. A node that must resist hostile local code needs broker custody
(the wallet key outside the user's reach, like the hub).

For a broker-custody wallet such as the hub's, the copied broker policy remains
devnet-only. It needs its own reviewed policy update before that wallet can be
used on mainnet. That wallet-policy gate is separate from the release-key arm.
