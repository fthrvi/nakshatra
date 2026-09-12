def serving_capacity(vram_bytes: int, *, model_bytes_per_layer: int,
                     n_layers: int, ctx_tokens: int, kv_bytes_per_token: int,
                     headroom: float = 0.10) -> dict:
    # Validate inputs
    if not isinstance(vram_bytes, int) or vram_bytes <= 0:
        return {"max_layers": 0, "holds_full_model": False,
                "usable_bytes": 0, "kv_bytes": 0,
                "reason": "vram_bytes must be a positive integer"}
    
    if not isinstance(model_bytes_per_layer, int) or model_bytes_per_layer <= 0:
        return {"max_layers": 0, "holds_full_model": False,
                "usable_bytes": 0, "kv_bytes": 0,
                "reason": "model_bytes_per_layer must be a positive integer"}
    
    if not isinstance(n_layers, int) or n_layers <= 0:
        return {"max_layers": 0, "holds_full_model": False,
                "usable_bytes": 0, "kv_bytes": 0,
                "reason": "n_layers must be a positive integer"}
    
    if not isinstance(ctx_tokens, int) or ctx_tokens <= 0:
        return {"max_layers": 0, "holds_full_model": False,
                "usable_bytes": 0, "kv_bytes": 0,
                "reason": "ctx_tokens must be a positive integer"}
    
    if not isinstance(kv_bytes_per_token, int) or kv_bytes_per_token <= 0:
        return {"max_layers": 0, "holds_full_model": False,
                "usable_bytes": 0, "kv_bytes": 0,
                "reason": "kv_bytes_per_token must be a positive integer"}
    
    # Clamp headroom to [0.0, 0.9]
    headroom = max(0.0, min(0.9, headroom))
    
    # Calculate usable bytes
    usable_bytes = int(vram_bytes * (1 - headroom))
    
    # Calculate KV bytes
    kv_bytes = ctx_tokens * kv_bytes_per_token
    
    # Check if KV cache exceeds usable memory
    if kv_bytes >= usable_bytes:
        return {"max_layers": 0, "holds_full_model": False,
                "usable_bytes": usable_bytes, "kv_bytes": kv_bytes,
                "reason": f"KV cache for {ctx_tokens} tokens needs more than this card has"}
    
    # Calculate remaining bytes after KV cache
    remaining = usable_bytes - kv_bytes
    
    # Calculate max layers
    max_layers = remaining // model_bytes_per_layer
    max_layers = max(0, min(max_layers, n_layers))
    
    # Determine if holds full model
    holds_full_model = max_layers >= n_layers
    
    return {"max_layers": max_layers, "holds_full_model": holds_full_model,
            "usable_bytes": usable_bytes, "kv_bytes": kv_bytes,
            "reason": "OK"}