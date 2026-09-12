import ipaddress


def usable_endpoints(endpoints: list, *, allow_lan: bool = True) -> tuple[list, list[str]]:
    """
    Filter endpoints to only those that are usable for direct connection.
    
    Returns (usable_endpoints, problems).
    """
    if not isinstance(endpoints, list):
        return [], ["non-list input"]
    
    usable = []
    problems = []
    seen = set()
    
    for endpoint in endpoints:
        # Handle None entries and non-tuple/list entries
        if endpoint is None:
            problems.append("None entry")
            continue
        
        if not isinstance(endpoint, (list, tuple)):
            problems.append(f"malformed entry: {endpoint}")
            continue
        
        if len(endpoint) != 2:
            problems.append(f"malformed entry: {endpoint}")
            continue
        
        host, port = endpoint
        
        # Validate types
        if not isinstance(host, str) or not isinstance(port, int):
            problems.append(f"malformed entry: {endpoint}")
            continue
        
        # Validate port range
        if port < 1 or port > 65535:
            problems.append(f"invalid port {port} for endpoint {endpoint}")
            continue
        
        # Check if host is a DNS name (not an IP literal)
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            problems.append(f"DNS name not allowed: {host}")
            continue
        
        # Check for loopback addresses
        if ip.is_loopback:
            problems.append(f"loopback address: {host}")
            continue
        
        # Check for unspecified addresses
        if ip.is_unspecified:
            problems.append(f"unspecified address: {host}")
            continue
        
        # Check for multicast
        if ip.is_multicast:
            problems.append(f"multicast address: {host}")
            continue
        
        # Check for link-local addresses
        if isinstance(ip, ipaddress.IPv4Address):
            if ip.is_link_local or str(ip).startswith("169.254."):
                problems.append(f"link-local address: {host}")
                continue
        elif isinstance(ip, ipaddress.IPv6Address):
            if ip.is_link_local:
                problems.append(f"link-local address: {host}")
                continue
        
        # Check for private addresses when allow_lan is False
        if not allow_lan and ip.is_private:
            problems.append(f"private address: {host}")
            continue
        
        # Deduplicate, keeping first occurrence
        endpoint_tuple = (host, port)
        if endpoint_tuple in seen:
            problems.append(f"duplicate endpoint: {host}:{port}")
            continue
        
        seen.add(endpoint_tuple)
        usable.append(endpoint_tuple)
    
    return usable, problems