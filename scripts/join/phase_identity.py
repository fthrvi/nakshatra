def identity(facts: dict) -> tuple[bool, list[str], dict]:
    problems = []
    updates = {}

    # Extract facts with defaults
    identity_pubkey = facts.get("identity_pubkey")
    identity_is_new = facts.get("identity_is_new", False)
    registered = facts.get("registered", False)
    roster_pubkey = facts.get("roster_pubkey")

    # Validate identity_pubkey is 64 lowercase hex
    if not isinstance(identity_pubkey, str):
        problems.append("identity_pubkey must be a string")
    elif len(identity_pubkey) != 64:
        problems.append("identity_pubkey must be 64 characters")
    elif not all(c in "0123456789abcdef" for c in identity_pubkey):
        problems.append("identity_pubkey must be 64 lowercase hex characters")

    # Check if registered
    if not isinstance(registered, bool):
        problems.append("registered must be a boolean")
    elif not registered:
        problems.append("node is not registered")

    # Check roster_pubkey consistency - only one of these two checks should apply
    if roster_pubkey is not None and identity_pubkey is not None and isinstance(identity_pubkey, str):
        if roster_pubkey != identity_pubkey:
            # Only report re-key if identity_is_new is False
            if not identity_is_new:
                problems.append("node appears to have re-keyed: everything earned under the old key is now unreachable")
        # If they match, no problem

    # Check for new key with existing roster_pubkey
    if identity_is_new is True and roster_pubkey is not None:
        if isinstance(roster_pubkey, str):
            # Only add this problem if we didn't already add the re-key problem
            if not (roster_pubkey != identity_pubkey and not identity_is_new):
                problems.append("fresh key minted for node that already had an identity on file")

    # If no problems, set updates
    if not problems and identity_pubkey is not None and isinstance(identity_pubkey, str):
        updates = {
            "identity_ok": True,
            "account": "nak:" + identity_pubkey
        }

    return (True, problems, updates) if not problems else (False, problems, updates)