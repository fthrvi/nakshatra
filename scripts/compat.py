import re


def compatible(node_version: str, coordinator_version: str) -> tuple[bool, str]:
    """
    Check if a node's version is compatible with a coordinator's version.
    
    Returns:
        tuple[bool, str]: (ok, why)
            - ok: True if compatible, False otherwise
            - why: Explanation of compatibility or incompatibility
    """
    try:
        # Parse node version
        node_parsed = _parse_version(node_version, "node")
        if node_parsed is None:
            return (False, "node version is unparseable")
        
        # Parse coordinator version
        coord_parsed = _parse_version(coordinator_version, "coordinator")
        if coord_parsed is None:
            return (False, "coordinator version is unparseable")
        
        node_major, node_minor, node_patch, node_extra = node_parsed
        coord_major, coord_minor, coord_patch, coord_extra = coord_parsed
        
        # Check major version compatibility
        if node_major != coord_major:
            return (False, f"major version mismatch: node {node_major}, coordinator {coord_major}")
        
        # Build the explanation
        reasons = []
        
        # Check for pre-release notes
        if node_extra or coord_extra:
            if node_extra and coord_extra:
                reasons.append(f"node has pre-release '{node_extra}', coordinator has pre-release '{coord_extra}'")
            elif node_extra:
                reasons.append(f"node has pre-release '{node_extra}'")
            elif coord_extra:
                reasons.append(f"coordinator has pre-release '{coord_extra}'")
        
        # Check minor version differences
        if node_minor > coord_minor:
            reasons.append(f"node minor ({node_minor}) is ahead of coordinator minor ({coord_minor})")
        elif node_minor < coord_minor:
            reasons.append(f"coordinator minor ({coord_minor}) is ahead of node minor ({node_minor})")
        
        # If no special reasons, just say they're compatible
        if not reasons:
            why = "compatible"
        else:
            why = "; ".join(reasons)
        
        return (True, why)
    
    except Exception:
        # Never raise, just return failure
        return (False, "version parsing error")


def _parse_version(version: str, side: str) -> tuple[int, int, int, str] | None:
    """
    Parse a version string into (major, minor, patch, extra).
    
    Returns None if unparseable.
    """
    if not isinstance(version, str):
        return None
    
    # Handle empty strings
    if not version:
        return None
    
    # Remove leading/trailing whitespace
    version = version.strip()
    
    # Match version pattern: MAJOR.MINOR.PATCH[-extra]
    # Allow missing components (treat as 0)
    # Only take up to 3 numeric components, ignore the rest
    match = re.match(r'^(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\.\d+)*?(?:-(.+))?$', version)
    if not match:
        return None
    
    major_str, minor_str, patch_str, extra = match.groups()
    
    try:
        major = int(major_str)
        minor = int(minor_str) if minor_str else 0
        patch = int(patch_str) if patch_str else 0
    except ValueError:
        return None
    
    # If there were missing components, note it in the extra field
    if minor_str is None or patch_str is None:
        if extra:
            extra = f"missing components; {extra}"
        else:
            extra = "missing components"
    
    return (major, minor, patch, extra)