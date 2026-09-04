def assign_layers(nodes: list[dict], n_layers: int) -> list[dict]:
    # Handle invalid inputs
    if not isinstance(nodes, list):
        return []
    if not isinstance(n_layers, int) or n_layers <= 0:
        return []
    
    # Validate each node and collect valid ones
    valid_nodes = []
    for node in nodes:
        if not isinstance(node, dict):
            return []
        if "node_id" not in node or "max_layers" not in node or "rtt_ms" not in node:
            return []
        if not isinstance(node["node_id"], str):
            return []
        if not isinstance(node["max_layers"], int):
            return []
        if not isinstance(node["rtt_ms"], (int, float)):
            return []
        
        # Skip nodes with max_layers <= 0
        if node["max_layers"] <= 0:
            continue
        
        valid_nodes.append(node)
    
    # Sort by rtt_ms ascending, then by node_id for deterministic tie-breaking
    try:
        valid_nodes.sort(key=lambda x: (x["rtt_ms"], x["node_id"]))
    except (TypeError, KeyError):
        return []
    
    # Check if total capacity is sufficient
    total_capacity = sum(node["max_layers"] for node in valid_nodes)
    if total_capacity < n_layers:
        return []
    
    # Assign layers
    result = []
    current_layer = 0
    
    for node in valid_nodes:
        if current_layer >= n_layers:
            break
        
        # Assign up to max_layers, but not more than needed
        layers_to_assign = min(node["max_layers"], n_layers - current_layer)
        
        if layers_to_assign > 0:
            result.append({
                "node_id": node["node_id"],
                "layer_start": current_layer,
                "layer_end": current_layer + layers_to_assign
            })
            current_layer += layers_to_assign
    
    # Ensure we covered exactly n_layers
    if current_layer != n_layers:
        return []
    
    return result