import re


def redact_keys(text: str) -> str:
    """Replace every run of exactly 64 hex characters with «redacted:key»."""
    if not isinstance(text, str):
        return ""
    
    # Pattern matches exactly 64 hex characters not surrounded by other hex chars
    pattern = r'(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])'
    
    # Replace with the marker
    result = re.sub(pattern, '«redacted:key»', text)
    
    return result