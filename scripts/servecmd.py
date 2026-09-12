def serve_argv(python: str, script: str, facts: dict) -> list[str]:
    required_keys = ["port", "role", "layer_start", "layer_end", "slice_path", "backend", "ngl"]
    
    # Check for missing required facts
    for key in required_keys:
        if key not in facts:
            raise ValueError(f"Missing required fact: {key}")
    
    # Validate role
    role = facts["role"]
    if role not in ("first", "middle", "last"):
        raise ValueError(f"Invalid role: {role}")
    
    # Validate layer range
    layer_start = facts["layer_start"]
    layer_end = facts["layer_end"]
    if layer_end <= layer_start:
        raise ValueError("layer_end must be greater than layer_start")
    
    # Build the argument list
    args = [
        python,
        script,
        "--port", str(facts["port"]),
        "--role", role,
        "--layer-start", str(layer_start),
        "--layer-end", str(layer_end),
        "--sub-gguf", facts["slice_path"],
        "--gpu-backend", facts["backend"],
        "--n-gpu-layers", str(facts["ngl"]),
    ]
    
    # Add model_id if present and non-empty
    if "model_id" in facts and facts["model_id"]:
        args.extend(["--model-id", facts["model_id"]])

    # ⚠️⚠️ AUTH IS NOT OPTIONAL FOR A JOINED NODE. worker.py's resolve_auth_required() reads
    # (env unset, pillar_url "") as "Mode A legacy" — TLS on, but NO peer authentication, and
    # peer_resolver=None, which also disarms the push-address SSRF gate. Until 2026-09-04 this
    # function passed neither, so every node produced by `nakshatra join` answered Forward /
    # Inference to anyone who could reach the port, and would push to any address a peer
    # named. Passing the coordinator as --pillar-url flips the truth table to Mode B/C; the
    # explicit env switch in act.start_daemon is the second lock on the same door.
    # The coordinator is where observe.py put it: inside the decoded join code. A top-level
    # key is honoured too, for callers that pass a flat dict.
    join = facts.get("join")
    coordinator = (join.get("coordinator") if isinstance(join, dict) else None) or facts.get("coordinator")
    if isinstance(coordinator, str) and coordinator:
        args.extend(["--pillar-url", coordinator])

    return args