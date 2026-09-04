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
    
    return args