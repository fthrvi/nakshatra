def node_state(last_seen: float, now: float, *, interval: float = 30.0,
               miss_stale: int = 2, miss_dead: int = 10) -> tuple[str, str]:
    # Check for non-numeric or boolean inputs
    for val, name in [(last_seen, "last_seen"), (now, "now")]:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return ("unknown", f"non-numeric {name}")
    
    # Check for clock skew (future timestamp)
    if last_seen > now:
        if last_seen - now > interval:
            return ("unknown", "clock skew")
        # Small skew tolerated, treat as live
        return ("live", "small clock skew")
    
    age = now - last_seen
    
    # Handle miss_stale >= miss_dead case
    if miss_stale >= miss_dead:
        # Clamp miss_dead to be at least miss_stale + 1
        clamped_miss_dead = miss_stale + 1
        # Update the logic with clamped value
        if age <= interval * miss_stale:
            return ("live", f"miss_stale >= miss_dead, clamped miss_dead to {clamped_miss_dead}")
        elif age <= interval * clamped_miss_dead:
            return ("stale", f"miss_stale >= miss_dead, clamped miss_dead to {clamped_miss_dead}")
        else:
            return ("dead", f"miss_stale >= miss_dead, clamped miss_dead to {clamped_miss_dead}")
    
    # Normal case with valid miss_stale and miss_dead
    if age <= interval * miss_stale:
        return ("live", "within miss_stale threshold")
    elif age <= interval * miss_dead:
        return ("stale", "within miss_dead threshold")
    else:
        return ("dead", "exceeded miss_dead threshold")