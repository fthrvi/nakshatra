def download_argv(url: str, dest: str, *, resume: bool = True, timeout_s: int = 1800) -> list[str]:
    # Validate inputs
    if not dest:
        raise ValueError("dest cannot be empty")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    
    # Check URL scheme
    if not url.startswith("https://"):
        raise ValueError("url must use https:// scheme")
    
    # Check for embedded credentials
    # Find the first '/' after the scheme
    after_scheme = url[len("https://"):]
    slash_pos = after_scheme.find('/')
    if slash_pos == -1:
        # No path, check entire remainder for '@'
        if '@' in after_scheme:
            raise ValueError("url contains embedded credentials")
        # Also raise if there's no path - URL must have at least a path separator
        raise ValueError("url must have a path")
    else:
        # Check only the part before the first '/'
        host_part = after_scheme[:slash_pos]
        if '@' in host_part:
            raise ValueError("url contains embedded credentials")
    
    # Build the argv
    argv = ["curl", "-fSL", "--proto", "=https", "-o", dest]
    
    # Add timeouts
    argv.extend(["--max-time", str(timeout_s), "--connect-timeout", "30"])
    
    # Add resume flag if requested
    if resume:
        argv.append("-C")
        argv.append("-")
    
    # Add the URL
    argv.append(url)
    
    return argv