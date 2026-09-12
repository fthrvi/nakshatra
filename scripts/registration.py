def registration_payload(facts: dict) -> dict:
    # Extract and validate node_id
    node_id = facts.get("node_id")
    if not node_id:
        raise ValueError("node_id must not be empty or missing")
    
    # Extract and validate pubkey (must be 64 lowercase hex characters)
    pubkey = facts.get("pubkey")
    if not isinstance(pubkey, str) or len(pubkey) != 64 or not all(c in "0123456789abcdef" for c in pubkey):
        raise ValueError("pubkey must be 64 lowercase hex characters")
    
    # Extract max_layers, coerce to int, floor at 0
    max_layers_raw = facts.get("max_layers", 0)
    try:
        max_layers = int(max_layers_raw)
    except (ValueError, TypeError):
        max_layers = 0
    max_layers = max(0, max_layers)
    
    # Extract ctx_tokens, coerce to int, floor at 0
    ctx_tokens_raw = facts.get("ctx_tokens", 0)
    try:
        ctx_tokens = int(ctx_tokens_raw)
    except (ValueError, TypeError):
        ctx_tokens = 0
    ctx_tokens = max(0, ctx_tokens)
    
    # Extract addrs, keep only strings; non-list becomes []
    addrs_raw = facts.get("addrs")
    if not isinstance(addrs_raw, list):
        addrs = []
    else:
        addrs = [addr for addr in addrs_raw if isinstance(addr, str)]
    
    # Extract version, default to "" if missing
    version = facts.get("version", "")
    
    # Build the payload with exactly the required keys
    return {
        "node_id": node_id,
        "pubkey": pubkey,
        "accel": facts.get("accel"),
        "max_layers": max_layers,
        "ctx_tokens": ctx_tokens,
        "addrs": addrs,
        "version": version,
    }