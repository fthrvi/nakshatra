import re

def redact_paths(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    # Pattern to match home directory paths with username
    # Matches /home/username or /Users/username where username is non-empty
    # Uses negative lookahead to avoid matching already redacted paths
    pattern = r'(/(?:home|Users)/)(?!«redacted:user»/)([^/\n]+)(?=/)'
    
    def replace_match(match):
        prefix = match.group(1)  # /home/ or /Users/
        # Replace the username with the redacted marker
        return prefix + '«redacted:user»'
    
    result = re.sub(pattern, replace_match, text)
    return result