def classify_nat(observations: list[dict]) -> tuple[str, str]:
    # Handle non-list input
    if not isinstance(observations, list):
        return ("unknown", "non-list input")
    
    # Handle empty list
    if len(observations) == 0:
        return ("unknown", "no observations")
    
    # Validate entries and extract data
    valid_obs = []
    for obs in observations:
        try:
            if not isinstance(obs, dict):
                return ("unknown", "non-dict entry")
            
            required_keys = ["local_port", "server", "mapped_ip", "mapped_port"]
            if not all(key in obs for key in required_keys):
                return ("unknown", "missing required keys")
            
            # Check if local_ip is present and use it for "none" detection
            local_ip = obs.get("local_ip")
            
            # Validate types
            if not isinstance(obs["local_port"], int) or not isinstance(obs["server"], str) or \
               not isinstance(obs["mapped_ip"], str) or not isinstance(obs["mapped_port"], int):
                return ("unknown", "non-int ports or non-string fields")
            
            valid_obs.append(obs)
        except Exception:
            return ("unknown", "exception during validation")
    
    # Check for "none" case: mapped_ip equals local_ip (when present)
    for obs in valid_obs:
        if "local_ip" in obs and obs["mapped_ip"] == obs["local_ip"]:
            return ("none", "mapped_ip equals local_ip")
    
    # Group observations by local_port
    port_groups = {}
    for obs in valid_obs:
        port = obs["local_port"]
        if port not in port_groups:
            port_groups[port] = []
        port_groups[port].append(obs)
    
    # Check for cases with fewer than 2 observations from different servers
    for port, group in port_groups.items():
        servers = set(obs["server"] for obs in group)
        if len(servers) >= 2:
            # We have at least two different servers for this port
            # Check if all mapped_ports are the same
            mapped_ports = set(obs["mapped_port"] for obs in group)
            if len(mapped_ports) == 1:
                return ("port-restricted", "same local_port, two servers, same mapped_port")
            else:
                return ("symmetric", "same local_port, two servers, different mapped_ports")
    
    # If we didn't find two different servers for any local_port
    return ("unknown", "fewer than two observations from different servers")