def identity_facts(key_existed: bool, pubkey: str, roster: dict, node_id: str) -> dict:
    # identity_is_new is not key_existed
    identity_is_new = not key_existed
    
    # roster_pubkey is roster.get(node_id) when roster is a dict, else None
    if isinstance(roster, dict):
        roster_pubkey = roster.get(node_id)
    else:
        roster_pubkey = None
    
    # registered is True only when roster_pubkey is a non-empty string AND equals pubkey
    # Also, pubkey must be valid (64 lowercase hex) for registered to be True
    if isinstance(roster_pubkey, str) and roster_pubkey != "" and isinstance(pubkey, str):
        # Check if pubkey is a 64-lowercase-hex string
        if len(pubkey) == 64 and all(c in '0123456789abcdef' for c in pubkey):
            registered = roster_pubkey == pubkey
        else:
            registered = False
    else:
        registered = False
    
    # identity_pubkey is pubkey when it is a 64-lowercase-hex string, else ""
    identity_pubkey = ""
    if isinstance(pubkey, str) and len(pubkey) == 64:
        # Check if all characters are lowercase hex digits
        if all(c in '0123456789abcdef' for c in pubkey):
            identity_pubkey = pubkey
    
    return {
        "identity_pubkey": identity_pubkey,
        "identity_is_new": identity_is_new,
        "registered": registered,
        "roster_pubkey": roster_pubkey
    }