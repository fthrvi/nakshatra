import ipaddress
from urllib.parse import urlparse


def admit(facts: dict) -> tuple[bool, list[str], dict]:
    problems = []
    updates = {}

    # Check for missing or incomplete join
    if "join" not in facts:
        problems.append("missing join")
        return (False, problems, updates)

    join = facts.get("join", {})
    if not isinstance(join, dict):
        problems.append("invalid join")
        return (False, problems, updates)

    coordinator = join.get("coordinator")
    token = join.get("token")

    if not coordinator:
        problems.append("missing coordinator")
        return (False, problems, updates)

    if not token:
        problems.append("missing token")
        return (False, problems, updates)

    # Check expiration
    now = facts.get("now", 0)
    expires_at = join.get("expires_at", 0)
    if expires_at != 0 and now >= expires_at:
        problems.append("join code expired")
        return (False, problems, updates)

    # Check version compatibility (major version must match)
    node_version = facts.get("node_version", "")
    coordinator_version = facts.get("coordinator_version", "")

    if node_version and coordinator_version:
        try:
            node_major = int(node_version.split(".")[0])
            coord_major = int(coordinator_version.split(".")[0])
            if node_major != coord_major:
                problems.append("major version mismatch")
                return (False, problems, updates)
        except (ValueError, IndexError):
            # If version parsing fails, treat as mismatch
            problems.append("invalid version format")
            return (False, problems, updates)

    # Check package_url
    package_url = facts.get("package_url", "")
    if not package_url:
        problems.append("missing package_url")
        return (False, problems, updates)

    try:
        parsed = urlparse(package_url)
    except Exception:
        problems.append("invalid package_url")
        return (False, problems, updates)

    if parsed.scheme != "https":
        problems.append("package_url must use https")
        return (False, problems, updates)

    host = parsed.hostname
    if not host:
        problems.append("invalid package_url host")
        return (False, problems, updates)

    # Check for SSRF: private, loopback, or link-local IPs
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_unspecified:
            problems.append("package_url host is a private or reserved IP")
            return (False, problems, updates)
    except ValueError:
        # Not an IP address, so it's a hostname - that's acceptable
        pass

    # Success
    updates = {
        "coordinator": coordinator,
        "admitted": True
    }
    return (True, problems, updates)