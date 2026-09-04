import json
import re


def validate_listing(event_content: str, *, max_bytes: int = 8192) -> tuple[dict | None, list[str]]:
    problems: list[str] = []

    # Check size BEFORE parsing
    if not isinstance(event_content, str):
        problems.append("invalid JSON")
        return None, problems

    if len(event_content) > max_bytes:
        problems.append("content exceeds maximum size")
        return None, problems

    # Try to parse JSON
    try:
        data = json.loads(event_content)
    except (json.JSONDecodeError, TypeError, ValueError):
        problems.append("invalid JSON")
        return None, problems

    # Must be a dict
    if not isinstance(data, dict):
        problems.append("top-level is not an object")
        return None, problems

    # Define allowed fields and their validators
    allowed_fields = {
        'node_id': (str, lambda s: 1 <= len(s) <= 64 and re.fullmatch(r'[A-Za-z0-9_.-]+', s)),
        'pubkey': (str, lambda s: len(s) == 64 and re.fullmatch(r'[0-9a-f]+', s)),
        'accel': (str, lambda s: s in {'cuda', 'rocm', 'vulkan', 'metal', 'cpu'}),
        'max_layers': (int, lambda i: 0 <= i <= 1000),
        'ctx_tokens': (int, lambda i: 0 <= i <= 10_000_000),
        'addrs': (list, lambda lst: len(lst) <= 8 and all(isinstance(s, str) and len(s) <= 128 for s in lst)),
        'relay': (str, lambda s: len(s) <= 256),
        'version': (str, lambda s: len(s) <= 32),
    }

    required_fields = set(allowed_fields.keys())

    # Check for unexpected fields
    for key in data:
        if key not in allowed_fields:
            problems.append(f"unexpected field: {key}")
            return None, problems

    # Check for missing required fields
    for field in required_fields:
        if field not in data:
            problems.append(f"missing required field: {field}")
            return None, problems

    # Validate each field
    for field, (expected_type, validator) in allowed_fields.items():
        value = data[field]

        # Type check
        if not isinstance(value, expected_type):
            problems.append(f"field '{field}' has wrong type")
            return None, problems

        # Check for control characters and newlines in strings BEFORE value validation
        if isinstance(value, str):
            if any(ord(c) < 32 or c == '\x7f' for c in value):
                problems.append(f"field '{field}' contains control characters")
                return None, problems

        # Value validation
        if not validator(value):
            problems.append(f"field '{field}' has invalid value")
            return None, problems

    # Special check for addrs list items
    for i, addr in enumerate(data['addrs']):
        if any(ord(c) < 32 or c == '\x7f' for c in addr):
            problems.append(f"field 'addrs' contains control characters in item {i}")
            return None, problems

    return data, []