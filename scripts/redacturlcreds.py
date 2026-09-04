import re


def redact_url_creds(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    # Pattern explanation:
    # ([a-zA-Z][a-zA-Z0-9+.-]*://) - captures the scheme and ://
    # [^/\s@]+ - matches the credential part (anything except /, whitespace, or @)
    # @ - matches the @ that separates credentials from host
    # The replacement keeps the scheme and ://, then inserts the redacted marker
    pattern = r'([a-zA-Z][a-zA-Z0-9+.-]*://)[^/\s@]+@'
    replacement = r'\1«redacted:token»@'
    
    return re.sub(pattern, replacement, text)