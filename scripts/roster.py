import json
import re
from pathlib import Path


def load_roster(path: str) -> dict[str, str]:
    """
    Load the admission roster from a JSON file and return a dict of {node_id: pubkey}
    containing only admissible records.

    Rules:
    - Only records with status == "active" are included.
    - pubkey must be exactly 64 lowercase hex characters.
    - Duplicate node_ids with different pubkeys raise ValueError.
    - Missing file, unreadable file, or malformed JSON raises appropriate error.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Roster file not found: {path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except (OSError, UnicodeDecodeError) as e:
        raise FileNotFoundError(f"Cannot read roster file: {path}") from e

    try:
        records = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in roster file: {path}") from e

    if not isinstance(records, list):
        raise ValueError(f"Roster file must contain a JSON array: {path}")

    roster = {}
    pubkey_pattern = re.compile(r'^[0-9a-f]{64}$')

    for record in records:
        if not isinstance(record, dict):
            continue

        status = record.get('status')
        if status != 'active':
            continue

        node_id = record.get('node_id')
        pubkey = record.get('pubkey')

        # node_id must be a string
        if not isinstance(node_id, str):
            continue

        # pubkey must be a string
        if not isinstance(pubkey, str):
            continue

        # pubkey must be exactly 64 lowercase hex characters
        if not pubkey_pattern.match(pubkey):
            continue

        # Check for duplicate node_id with different pubkey
        if node_id in roster:
            if roster[node_id] != pubkey:
                raise ValueError(
                    f"Duplicate node_id '{node_id}' with different pubkeys in roster"
                )
            # If same pubkey, we can skip (collapse duplicates)
            continue

        roster[node_id] = pubkey

    return roster