def weighted_units(tokens: int, layer_start: int, layer_end: int,
                   slice_bytes: int, *, ref_bytes: int = 419430400) -> int:
    # Validate input types - all must be integers
    if not all(isinstance(x, int) for x in [tokens, layer_start, layer_end, slice_bytes]):
        return 0
    
    # Handle non-positive tokens or invalid layer range
    if tokens <= 0 or layer_end <= layer_start:
        return 0
    
    # Handle non-positive slice_bytes - fall back to unweighted calculation
    if slice_bytes <= 0:
        layers = layer_end - layer_start
        return tokens * layers
    
    # Handle non-positive ref_bytes - use default
    if ref_bytes <= 0:
        ref_bytes = 419430400
    
    # Calculate weighted units: tokens × layers × (slice_bytes / ref_bytes)
    layers = layer_end - layer_start
    result = tokens * layers * (slice_bytes / ref_bytes)
    
    # Round to nearest int and floor at 0
    return max(0, round(result))