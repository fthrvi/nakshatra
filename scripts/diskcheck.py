def room_for(free_bytes: int, need_bytes: int, *, reserve: float = 0.05) -> tuple[bool, str]:
    # Validate free_bytes
    if not isinstance(free_bytes, int) or free_bytes < 0:
        return (False, f"invalid free_bytes: {free_bytes}")
    
    # Validate need_bytes
    if not isinstance(need_bytes, int) or need_bytes < 0:
        return (False, f"invalid need_bytes: {need_bytes}")
    
    # Clamp reserve to [0.0, 0.5]
    if reserve < 0.0:
        reserve = 0.0
    elif reserve > 0.5:
        reserve = 0.5
    
    # Handle zero need case
    if need_bytes == 0:
        usable = free_bytes * (1 - reserve)
        headroom_mib = int(usable / (1024 * 1024))
        return (True, f"{headroom_mib} MiB headroom")
    
    # Calculate usable space
    usable = free_bytes * (1 - reserve)
    
    # Check if we have enough space
    if usable >= need_bytes:
        headroom_mib = int((usable - need_bytes) / (1024 * 1024))
        return (True, f"{headroom_mib} MiB headroom")
    else:
        shortfall_bytes = need_bytes - usable
        shortfall_mib = int(shortfall_bytes / (1024 * 1024))
        return (False, f"need {shortfall_mib} MiB more")