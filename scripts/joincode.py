import base64
import json


def encode_join(coordinator: str, token: str, *, relay: str = "", expires_at: int = 0) -> str:
    obj = {
        "c": coordinator,
        "t": token,
    }
    if relay:
        obj["r"] = relay
    if expires_at:
        obj["x"] = expires_at
    payload = json.dumps(obj, separators=(",", ":"))
    # base64url encoding without padding
    encoded = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")
    return encoded


def decode_join(code: str, *, now: int) -> dict:
    # Validate base64url
    try:
        # Add padding back if needed
        padding = 4 - (len(code) % 4)
        if padding != 4:
            code += "=" * padding
        decoded = base64.urlsafe_b64decode(code)
        payload = decoded.decode("utf-8")
    except Exception:
        raise ValueError("invalid base64url encoding")

    # Parse JSON
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        raise ValueError("invalid JSON")

    # Must be a JSON object
    if not isinstance(data, dict):
        raise ValueError("payload is not a JSON object")

    # Check required keys
    if "c" not in data:
        raise ValueError("coordinator missing")
    if "t" not in data:
        raise ValueError("token missing")

    coordinator = data["c"]
    token = data["t"]

    # Validate coordinator is a non-empty string
    if not isinstance(coordinator, str) or not coordinator:
        raise ValueError("coordinator missing or invalid")

    # Validate token is a non-empty string
    if not isinstance(token, str) or not token:
        raise ValueError("token missing or invalid")

    # Coordinator must start with http:// or https://
    if not (coordinator.startswith("http://") or coordinator.startswith("https://")):
        raise ValueError("coordinator must start with http:// or https://")

    # Check for unknown keys
    allowed_keys = {"c", "t", "r", "x"}
    for key in data:
        if key not in allowed_keys:
            raise ValueError(f"unknown key in code: {key}")

    # Handle relay (optional)
    relay = data.get("r", "")

    # Handle expiry
    expires_at = data.get("x", 0)
    if expires_at and now >= expires_at:
        raise ValueError("code expired")

    return {
        "coordinator": coordinator,
        "token": token,
        "relay": relay,
        "expires_at": expires_at,
    }