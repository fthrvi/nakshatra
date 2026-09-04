def supply_problems(balances: dict[str, int]) -> list[str]:
    """Return list of problems with the given balances ledger."""
    problems = []
    
    # Check for empty-string account id
    for account, balance in balances.items():
        if account == "":
            problems.append("empty account id")
    
    # Check for non-int balances
    for account, balance in balances.items():
        if not isinstance(balance, int):
            problems.append(f"non-int balance for account '{account}'")
    
    # Check total sum
    total = 0
    for account, balance in balances.items():
        if isinstance(balance, int):
            total += balance
    
    if not isinstance(total, int):
        problems.append("non-int total")
    elif total != 0:
        problems.append(f"drift {total}")
    
    return problems


def would_break_invariant(balances: dict[str, int], credits: dict[str, int],
                          requester: str) -> tuple[bool, str]:
    """Check if applying the settlement would break the invariant.
    
    Returns (True, why) if it would break, (False, "") otherwise.
    """
    # Check requester validity
    if not isinstance(requester, str) or requester == "":
        return (True, "invalid requester")
    
    # Check credits is a dict
    if not isinstance(credits, dict):
        return (True, "credits must be a dict")
    
    # Check credits for non-int or negative amounts
    for account, amount in credits.items():
        if not isinstance(amount, int):
            return (True, f"non-int credit for account '{account}'")
        if amount < 0:
            return (True, f"negative credit for account '{account}'")
    
    # Calculate current total (only for int balances)
    current_total = 0
    if isinstance(balances, dict):
        for account, balance in balances.items():
            if isinstance(balance, int):
                current_total += balance
    
    # Calculate the total credits being issued
    total_credits = sum(credits.values())
    
    # The settlement credits the accounts in credits and debits the requester
    # So the new total would be: current_total + total_credits - total_credits = current_total
    # But we need to check if any of the operations would break the invariant
    
    # Since we're adding credits and debiting the requester by the same amount,
    # the total should remain unchanged IF all values are valid integers.
    # The only way this breaks is if we have invalid inputs (already checked above)
    
    return (False, "")