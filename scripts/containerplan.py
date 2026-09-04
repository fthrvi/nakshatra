def container_argv(image: str, script_path: str, *, network: str = "bridge",
                   memory_gb: int = 8, timeout_s: int = 3600) -> list[str]:
    # Validate image
    if not image or ' ' in image:
        raise ValueError("image must be non-empty and contain no spaces")
    
    # Validate script_path
    if not script_path:
        raise ValueError("script_path must be non-empty")
    if not script_path.startswith('/'):
        raise ValueError("script_path must be absolute")
    
    # Validate memory_gb
    if memory_gb <= 0:
        raise ValueError("memory_gb must be positive")
    
    # Validate timeout_s
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    
    return [
        "docker", "run",
        "--rm",
        "--network", network,
        "--memory", f"{memory_gb}g",
        "-v", f"{script_path}:/provision.sh:ro",
        "--stop-timeout", str(timeout_s),
        image,
        "bash", "/provision.sh"
    ]