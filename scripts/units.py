from identity_binding import verify_participation, pub_of, sign_participation, account_id


def credit_units(receipt: dict, pinned: dict[str, str]) -> dict[str, int]:
    # Handle non-dict receipt
    if not isinstance(receipt, dict):
        return {}

    # Handle missing or non-list worker_signatures
    if 'worker_signatures' not in receipt or not isinstance(receipt['worker_signatures'], list):
        return {}

    # Handle missing or invalid n_generated
    n_generated = receipt.get('n_generated')
    if not isinstance(n_generated, int) or n_generated < 0:
        return {}

    # Extract run_id and output_sha256 from receipt
    run_id = receipt.get('run_id')
    output_sha256 = receipt.get('output_sha256')

    # If run_id or output_sha256 are missing, return empty dict
    if run_id is None or output_sha256 is None:
        return {}

    # Dictionary to collect spans per account: {account_id: [(layer_start, layer_end), ...]}
    account_spans: dict[str, list[tuple[int, int]]] = {}

    for entry in receipt['worker_signatures']:
        # Skip non-dict entries
        if not isinstance(entry, dict):
            continue

        # Verify the signature using verify_participation
        ok, _ = verify_participation(entry, run_id=run_id, output_sha256=output_sha256, pinned=pinned)
        if not ok:
            continue

        # Get the account_id from the entry's pubkey
        try:
            pub = entry.get('pubkey')
            if pub is None:
                continue
            acc = account_id(pub)
        except Exception:
            continue

        # Extract layer span
        try:
            layer_start = int(entry['layer_start'])
            layer_end = int(entry['layer_end'])
        except (KeyError, ValueError, TypeError):
            continue

        # Record the span for this account
        if acc not in account_spans:
            account_spans[acc] = []
        account_spans[acc].append((layer_start, layer_end))

    # Now compute units per account, checking for overlaps
    result: dict[str, int] = {}

    for acc, spans in account_spans.items():
        # Sort spans by layer_start
        spans_sorted = sorted(spans, key=lambda x: x[0])

        # Check for overlaps between consecutive spans
        for i in range(1, len(spans_sorted)):
            prev_start, prev_end = spans_sorted[i - 1]
            curr_start, curr_end = spans_sorted[i]
            # Overlap if curr_start < prev_end (since spans are [start, end))
            if curr_start < prev_end:
                raise ValueError(f"Overlapping spans detected for account {acc}")

        # Compute total units for this account
        total_units = 0
        for start, end in spans_sorted:
            total_units += (end - start)

        units = n_generated * total_units
        if units > 0:
            result[acc] = units

    return result