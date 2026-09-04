def parse_probe(body: str) -> dict:
    # Check if body is a string
    if not isinstance(body, str):
        return {"probe_tokens": [], "probe_layers": None, "error": "body is not a string"}
    
    # Check for empty body
    if body == "":
        return {"probe_tokens": [], "probe_layers": None, "error": "body is empty"}
    
    # Try to parse JSON
    try:
        import json
        data = json.loads(body)
    except json.JSONDecodeError as e:
        return {"probe_tokens": [], "probe_layers": None, "error": f"invalid JSON: {str(e)}"}
    
    # Check if top-level is a dict
    if not isinstance(data, dict):
        return {"probe_tokens": [], "probe_layers": None, "error": "top-level is not an object"}
    
    # Check for shape 1: tokens and layers
    if "tokens" in data:
        tokens = data["tokens"]
        
        # Validate tokens is a list
        if not isinstance(tokens, list):
            return {"probe_tokens": [], "probe_layers": None, "error": "tokens is not a list"}
        
        # Extract and validate token values
        probe_tokens = []
        error_msg = None
        for token in tokens:
            if isinstance(token, int):
                probe_tokens.append(token)
            else:
                if error_msg is None:
                    error_msg = f"non-int token dropped: {token}"
                else:
                    error_msg += f", {token}"
        
        # Check for layers
        if "layers" in data:
            layers = data["layers"]
            
            # Validate layers is a dict with start and end
            if not isinstance(layers, dict) or "start" not in layers or "end" not in layers:
                error_msg = error_msg or "layers is not a 2-field object"
                probe_layers = None
            else:
                # Validate layers values are integers
                try:
                    start = layers["start"]
                    end = layers["end"]
                    if not isinstance(start, int) or not isinstance(end, int):
                        error_msg = error_msg or "layers values are not integers"
                        probe_layers = None
                    else:
                        probe_layers = [start, end]
                except (TypeError, ValueError):
                    error_msg = error_msg or "layers values are not integers"
                    probe_layers = None
        else:
            # Missing layers - this is an error condition per the test
            error_msg = "layers is not a 2-field object"
            probe_layers = None
        
        return {
            "probe_tokens": probe_tokens,
            "probe_layers": probe_layers,
            "error": error_msg
        }
    
    # Check for shape 2: OpenAI-ish format
    elif "choices" in data and "usage" in data:
        choices = data["choices"]
        usage = data["usage"]
        
        # Validate choices is a list
        if not isinstance(choices, list):
            return {"probe_tokens": [], "probe_layers": None, "error": "choices is not a list"}
        
        # Validate usage is a dict with completion_tokens
        if not isinstance(usage, dict) or "completion_tokens" not in usage:
            return {"probe_tokens": [], "probe_layers": None, "error": "usage missing completion_tokens"}
        
        completion_tokens = usage["completion_tokens"]
        
        # Validate completion_tokens is an integer
        if not isinstance(completion_tokens, int):
            return {"probe_tokens": [], "probe_layers": None, "error": "completion_tokens is not an integer"}
        
        # Return zeros for probe_tokens and None for probe_layers
        probe_tokens = [0] * completion_tokens
        probe_layers = None
        
        return {
            "probe_tokens": probe_tokens,
            "probe_layers": probe_layers,
            "error": None
        }
    
    # Neither shape recognized
    else:
        return {"probe_tokens": [], "probe_layers": None, "error": "unknown response format"}