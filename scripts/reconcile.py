from identity_binding import verify_participation, account_id
from cryptography.hazmat.primitives.asymmetric import ed25519


def reconcile(balances: dict[str, int], receipts: list[dict],
              pinned: dict[str, str]) -> dict:
    computed = {}
    skipped = 0
    accounts_with_receipts = set()

    for receipt in receipts:
        try:
            # Check required fields exist
            if not isinstance(receipt, dict):
                skipped += 1
                continue

            # Get worker signatures from the receipt
            worker_sigs = receipt.get("worker_signatures", [])
            if not isinstance(worker_sigs, list):
                skipped += 1
                continue

            # Extract run info for verification
            run_id = receipt.get("run_id", "")
            output_sha256 = receipt.get("output_sha256", "")

            # Skip if missing required run info
            if not run_id or not output_sha256:
                skipped += 1
                continue

            for sig_entry in worker_sigs:
                try:
                    # Verify participation signature
                    ok, _ = verify_participation(sig_entry, run_id=run_id, 
                                                output_sha256=output_sha256, 
                                                pinned=pinned)
                    
                    if ok:
                        # Extract account and compute units
                        pub = sig_entry.get("pubkey", "")
                        acc = account_id(pub)
                        accounts_with_receipts.add(acc)

                        # Extract layer info
                        layer_start = sig_entry.get("layer_start")
                        layer_end = sig_entry.get("layer_end")
                        n_generated = receipt.get("n_generated", 0)

                        # Validate required fields
                        if layer_start is None or layer_end is None:
                            continue

                        units = n_generated * (layer_end - layer_start)

                        if acc not in computed:
                            computed[acc] = 0
                        computed[acc] += units

                except Exception:
                    continue

        except Exception:
            skipped += 1
            continue

    # Build result
    agree = {}
    over = {}
    under = {}
    unknown_accounts = []

    for account, ledger_balance in balances.items():
        computed_balance = computed.get(account, 0)

        if account not in accounts_with_receipts:
            unknown_accounts.append(account)
        elif ledger_balance == computed_balance:
            agree[account] = ledger_balance
        elif ledger_balance > computed_balance:
            over[account] = (ledger_balance, computed_balance)
        else:
            under[account] = (ledger_balance, computed_balance)

    clean = len(over) == 0 and len(under) == 0 and len(unknown_accounts) == 0

    return {
        "agree": agree,
        "over": over,
        "under": under,
        "unknown_accounts": unknown_accounts,
        "clean": clean,
        "skipped": skipped
    }