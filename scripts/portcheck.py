def port_conflicts(want: int, listeners: list[dict]) -> tuple[bool, str]:
    # Check if port is out of valid range
    if not isinstance(want, int) or want < 1 or want > 65535:
        return (False, f"Port {want} out of range (1-65535)")
    
    # Check if port is privileged (below 1024)
    if want < 1024:
        return (False, f"Port {want} requires elevated privileges (privileged port)")
    
    # Handle malformed listeners input
    if not isinstance(listeners, list):
        return (True, "")
    
    # Process listeners, ignoring malformed entries
    for listener in listeners:
        if not isinstance(listener, dict):
            continue
        
        # Extract fields safely
        port = listener.get("port")
        addr = listener.get("addr")
        
        # Skip if port or addr is missing or not the right type
        if not isinstance(port, int) or not isinstance(addr, str):
            continue
        
        # Check if port matches
        if port == want:
            # Determine conflict type
            name = listener.get("name", "unknown")
            pid = listener.get("pid", "unknown")
            return (False, f"Port {want} held by {name} (pid {pid})")
    
    # No conflicts found
    return (True, "")