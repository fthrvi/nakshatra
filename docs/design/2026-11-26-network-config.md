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
- `nak pay network arm mainnet` is terminal-only and binds the arm record to the
  exact mainnet genesis, program, and mint. Any later edit disarms it.
- `nak pay network disarm` removes that arm record.

## Mainnet switch

Mainnet is the owner's decision. Nothing in this change deploys a program,
funds a wallet, restarts a service, or enables mainnet by itself.

1. Deploy and audit the escrow program on Solana mainnet.
2. On each node, put the mainnet HTTPS RPC and audited program ID in
   `network.json`; create explicit `~/.nakshatra/pay/limits.json` spend limits.
3. Run `nak pay network arm mainnet`.
4. Run `nak pay network use mainnet`, then restart the user services it prints.
5. Fund the node with real USDC and enough SOL for fees and rent.

For a broker-custody wallet such as the hub's, the copied broker policy remains
devnet-only. It needs its own reviewed policy update before that wallet can be
used on mainnet.
