import ipaddress
from urllib.parse import urlparse


def check_package_url(url: str, *, allow_hosts: list[str] | None = None) -> tuple[bool, str]:
    """
    Validate a package URL for safe fetching.
    
    Returns (ok, why) where ok is True if the URL is safe, False otherwise.
    """
    try:
        # Must be a string
        if not isinstance(url, str):
            return (False, "URL must be a string")
        
        # Parse the URL
        parsed = urlparse(url)
        
        # Check scheme - only https allowed
        if parsed.scheme != "https":
            return (False, f"Scheme must be https, got {parsed.scheme!r}")
        
        # Check for embedded credentials
        if parsed.username or parsed.password:
            return (False, "URL must not contain embedded credentials")
        
        # Get the hostname
        hostname = parsed.hostname
        if not hostname:
            return (False, "URL must have a valid hostname")
        
        # Check allow_hosts if provided and non-empty
        if allow_hosts is not None and len(allow_hosts) > 0:
            # Case-insensitive exact match
            if hostname.lower() not in [h.lower() for h in allow_hosts]:
                return (False, f"Host {hostname!r} not in allowed hosts")
        
        # Check for IP literals in private ranges
        try:
            ip = ipaddress.ip_address(hostname)
            
            # Check if it's a private or reserved IP
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_unspecified:
                return (False, "IP address is in a private or reserved range")
            
            # Additional checks for specific ranges
            # IPv4 private ranges: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
            if isinstance(ip, ipaddress.IPv4Address):
                if ip in ipaddress.ip_network("10.0.0.0/8"):
                    return (False, "IP address is in private range 10.0.0.0/8")
                if ip in ipaddress.ip_network("172.16.0.0/12"):
                    return (False, "IP address is in private range 172.16.0.0/12")
                if ip in ipaddress.ip_network("192.168.0.0/16"):
                    return (False, "IP address is in private range 192.168.0.0/16")
                if ip in ipaddress.ip_network("127.0.0.0/8"):
                    return (False, "IP address is in loopback range 127.0.0.0/8")
                if ip in ipaddress.ip_network("169.254.0.0/16"):
                    return (False, "IP address is in link-local range 169.254.0.0/16")
            
            # IPv6 checks
            if isinstance(ip, ipaddress.IPv6Address):
                if ip == ipaddress.IPv6Address("::1"):
                    return (False, "IP address is IPv6 loopback")
                # fc00::/7 - unique local addresses
                if ip in ipaddress.ip_network("fc00::/7"):
                    return (False, "IP address is in IPv6 unique local range fc00::/7")
        except ValueError:
            # Not an IP address, that's fine - it's a hostname
            pass
        
        # Check path - must not be empty or "/"
        if not parsed.path or parsed.path == "/":
            return (False, "Path must not be empty or '/'")
        
        return (True, "OK")
    
    except Exception:
        # Never raise, whatever the input
        return (False, "Invalid URL format")