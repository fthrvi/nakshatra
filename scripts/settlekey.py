import hashlib
import json


def settlement_key(receipt: dict) -> str:
    """Return a stable hex digest identifying this settlement.

    ⚠️⚠️ WHY THIS DOES NOT SWALLOW EXCEPTIONS. It used to: `except Exception: return
    hashlib.sha256(b"").hexdigest()` collapsed EVERY malformed input — `None`, a string, an
    int, a list, anything without `.get` — onto the SAME constant hash. That defeats
    settle-idempotency by construction: two unrelated malformed/None receipts collide and the
    second is silently treated as "already settled". A receipt broken enough to not even be a
    dict is a bug upstream of this function, not a hashing problem — it must surface loudly at
    the caller, not vanish into a fake key that looks like a real settlement.
    """
    if not isinstance(receipt, dict):
        raise TypeError(f"settlement_key requires a dict receipt, got {type(receipt).__name__}")

    run_id = receipt.get("run_id", "")
    output_sha256 = receipt.get("output_sha256", "")
    worker_sigs = receipt.get("worker_signatures", [])

    # Ensure worker_sigs is a list (even if malformed)
    if not isinstance(worker_sigs, list):
        worker_sigs = []

    # Sort worker signatures by (node_id, layer_start, layer_end)
    sorted_sigs = []
    for sig in worker_sigs:
        if isinstance(sig, dict):
            node_id = sig.get("node_id", "")
            layer_start = sig.get("layer_start", 0)
            layer_end = sig.get("layer_end", 0)
            sorted_sigs.append((node_id, layer_start, layer_end))

    # Sort the list of tuples
    sorted_sigs.sort(key=lambda x: (x[0], x[1], x[2]))

    # Build canonical representation
    data = {
        "run_id": run_id,
        "output_sha256": output_sha256,
        "worker_signatures": sorted_sigs
    }

    # Canonical JSON encoding
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))

    # SHA256 hash
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_duplicate(key: str, seen: dict[str, str]) -> tuple[bool, str]:
    """Return (True, when_first_seen) if key is in seen, else (False, "")."""
    if key in seen:
        return (True, seen[key])
    return (False, "")