from identity_binding import sign_participation


def build_worker_signatures(stages, keys, *, run_id: str, output_sha256: str) -> list[dict]:
    """
    Build worker signatures from a run's stage assignments.

    Args:
        stages: List of stage dicts with keys 'node_id', 'layer_start', 'layer_end'
        keys: Dict mapping node_id -> priv_hex (raw 32-byte hex string)
        run_id: The run identifier
        output_sha256: SHA256 hash of the output

    Returns:
        List of signed entry dicts in the same order as stages (skipping invalid ones)

    Raises:
        ValueError: If two stages for the same node_id have overlapping layer ranges
    """
    # Track layer ranges per node_id to detect overlaps
    node_layer_ranges = {}

    signed_entries = []

    for stage in stages:
        node_id = stage["node_id"]
        layer_start = stage["layer_start"]
        layer_end = stage["layer_end"]

        # Skip stages where node_id is not in keys
        if node_id not in keys:
            continue

        # Skip stages where layer_end <= layer_start
        if layer_end <= layer_start:
            continue

        # Check for overlapping layer ranges for the same node_id
        if node_id in node_layer_ranges:
            for existing_start, existing_end in node_layer_ranges[node_id]:
                # Overlap check: two intervals [a,b) and [c,d) overlap if a < d and c < b
                if layer_start < existing_end and existing_start < layer_end:
                    raise ValueError(
                        f"Overlapping layer ranges for node {node_id}: "
                        f"[{layer_start}, {layer_end}) overlaps with "
                        f"[{existing_start}, {existing_end})"
                    )
            node_layer_ranges[node_id].append((layer_start, layer_end))
        else:
            node_layer_ranges[node_id] = [(layer_start, layer_end)]

        # Sign the participation
        priv_hex = keys[node_id]
        entry = sign_participation(
            node_id=node_id,
            layer_start=layer_start,
            layer_end=layer_end,
            run_id=run_id,
            output_sha256=output_sha256,
            priv_hex=priv_hex,
        )
        signed_entries.append(entry)

    return signed_entries