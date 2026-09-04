def flag_enabled(name: str, env: dict[str, str], *, default: bool = False) -> bool:
    # Validate inputs
    if not isinstance(name, str) or not isinstance(env, dict):
        return False
    
    # Empty name is invalid
    if not name:
        return False
    
    # Construct the environment variable name
    var_name = f"NAKSHATRA_{name.upper()}"
    
    # Get the value from env, return default if absent
    value = env.get(var_name)
    if value is None:
        return default
    
    # Normalize the value: strip whitespace and convert to lowercase
    normalized = value.strip().lower()
    
    # Truthy values
    if normalized in ("1", "true", "yes", "on"):
        return True
    
    # Falsy values (including empty string)
    if normalized in ("0", "false", "no", "off", ""):
        return False
    
    # Any other value is False (fail closed)
    return False