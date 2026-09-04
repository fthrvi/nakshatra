def should_dispute(checks: list[dict], *, min_checks: int = 3,
                   fail_ratio: float = 0.5) -> tuple[bool, str]:
    # Handle non-list cases
    if not isinstance(checks, list):
        return (False, "invalid checks: not a list")
    
    # Handle empty list
    if len(checks) == 0:
        return (False, "invalid checks: empty list")
    
    # Deduplicate based on (checked_at, reason) pairs
    seen = set()
    unique_checks = []
    for check in checks:
        if not isinstance(check, dict):
            return (False, "invalid checks: contains non-dict entry")
        
        # Extract checked_at and reason, defaulting to None if missing
        checked_at = check.get("checked_at")
        reason = check.get("reason")
        key = (checked_at, reason)
        
        if key not in seen:
            seen.add(key)
            unique_checks.append(check)
    
    # Count failures and total unique checks
    total_checks = len(unique_checks)
    failed_checks = sum(1 for check in unique_checks if check.get("ok") is False)
    
    # Check if we have sufficient evidence
    if total_checks < min_checks:
        return (False, f"insufficient evidence: {total_checks} of {min_checks} checks")
    
    # Calculate failure fraction
    failure_fraction = failed_checks / total_checks
    
    # Dispute only if failure fraction is strictly greater than fail_ratio
    if failure_fraction > fail_ratio:
        return (True, f"{failed_checks} of {total_checks} checks failed ({failure_fraction:.2f} > {fail_ratio:.2f})")
    else:
        return (False, f"failure rate ({failure_fraction:.2f}) not above threshold ({fail_ratio:.2f})")