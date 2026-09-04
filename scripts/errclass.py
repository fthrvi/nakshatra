def classify(exit_code: int, stderr: str) -> tuple[str, str]:
    # Handle non-int exit_code or non-string stderr
    if not isinstance(exit_code, int):
        return ("unknown", "")
    if not isinstance(stderr, str):
        return ("unknown", "")
    
    # Exit code 0 means success, regardless of stderr content
    if exit_code == 0:
        return ("unknown", "")
    
    stderr_lower = stderr.lower()
    
    # Define categories and their keywords
    transient_keywords = [
        "timeout",
        "temporary failure",
        "connection reset",
        "503",
        "try again",
        "network is unreachable"
    ]
    
    needs_operator_keywords = [
        "no space left on device",
        "permission denied",
        "command not found",
        "unable to locate package",
        "disk quota exceeded"
    ]
    
    fatal_keywords = [
        "404",
        "no such file or directory",
        "unsupported architecture",
        "checksum mismatch"
    ]
    
    # Check for needs_operator first (highest precedence)
    for keyword in needs_operator_keywords:
        if keyword in stderr_lower:
            return ("needs_operator", keyword)
    
    # Check for fatal
    for keyword in fatal_keywords:
        if keyword in stderr_lower:
            return ("fatal", keyword)
    
    # Check for transient
    for keyword in transient_keywords:
        if keyword in stderr_lower:
            return ("transient", keyword)
    
    # None matched
    return ("unknown", "")