def shape_observations(replies: list) -> tuple[list[dict], list[str]]:
    """
    Turn raw STUN replies into observations natclass reads.
    
    Args:
        replies: List of reply dicts with keys:
            - server: str
            - local_port: int
            - local_ip: str
            - mapped: str | None (format "ip:port" or "[ipv6]:port")
            - error: str | None
    
    Returns:
        (observations, problems) where observations is a list of dicts with keys:
            - server, local_port, local_ip, mapped_ip, mapped_port
        and problems is a list of error strings.
    """
    observations = []
    problems = []
    seen_servers = set()
    
    # Handle non-list input
    if not isinstance(replies, list):
        return observations, problems
    
    for reply in replies:
        # Skip None entries
        if reply is None:
            continue
        
        # Must be a dict
        if not isinstance(reply, dict):
            continue
        
        server = reply.get("server")
        local_port = reply.get("local_port")
        local_ip = reply.get("local_ip")
        mapped = reply.get("mapped")
        error = reply.get("error")
        
        # Check for error or mapped=None
        if error is not None:
            if server is not None:
                problems.append(f"error from {server}")
            continue
        
        if mapped is None:
            if server is not None:
                problems.append(f"no mapped address from {server}")
            continue
        
        # Check for duplicate server
        if server in seen_servers:
            if server is not None:
                problems.append(f"duplicate server {server}")
            continue
        
        # Parse mapped address
        # Handle IPv6 format: [2001:db8::1]:40000
        if mapped.startswith("["):
            # IPv6 format
            bracket_end = mapped.find("]")
            if bracket_end == -1:
                if server is not None:
                    problems.append(f"malformed mapped address from {server}")
                continue
            
            host = mapped[1:bracket_end]
            # After the closing bracket, there should be a colon and port
            if len(mapped) <= bracket_end + 1 or mapped[bracket_end + 1] != ":":
                if server is not None:
                    problems.append(f"malformed mapped address from {server}")
                continue
            
            port_str = mapped[bracket_end + 2:]
        else:
            # IPv4 format - split on last colon
            colon_idx = mapped.rfind(":")
            if colon_idx == -1:
                if server is not None:
                    problems.append(f"malformed mapped address from {server}")
                continue
            
            host = mapped[:colon_idx]
            port_str = mapped[colon_idx + 1:]
        
        # Validate port
        try:
            port = int(port_str)
        except ValueError:
            if server is not None:
                problems.append(f"malformed mapped address from {server}")
            continue
        
        # Check port range (valid ports are 1-65535)
        if port < 1 or port > 65535:
            if server is not None:
                problems.append(f"malformed mapped address from {server}")
            continue
        
        # Valid observation
        seen_servers.add(server)
        observations.append({
            "server": server,
            "local_port": local_port,
            "local_ip": local_ip,
            "mapped_ip": host,
            "mapped_port": port
        })
    
    return observations, problems