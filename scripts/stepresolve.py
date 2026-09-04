def next_steps(steps: list[dict], done: list[str]) -> list[dict]:
    # Validate input types
    if not isinstance(steps, list) or not isinstance(done, list):
        return []
    
    # Create a set of valid step IDs from steps
    valid_ids = set()
    for step in steps:
        if isinstance(step, dict) and "id" in step:
            valid_ids.add(step["id"])
    
    # Create a set of done IDs for quick lookup
    done_set = set()
    for item in done:
        if isinstance(item, str):
            done_set.add(item)
    
    result = []
    for step in steps:
        # Skip non-dict entries
        if not isinstance(step, dict):
            continue
        
        # Skip entries without 'id' or 'needs'
        if "id" not in step or "needs" not in step:
            continue
        
        step_id = step["id"]
        
        # Skip if step_id is not a string
        if not isinstance(step_id, str):
            continue
        
        # Skip if needs is not a list
        needs = step["needs"]
        if not isinstance(needs, list):
            continue
        
        # Skip if any need is not a string
        if not all(isinstance(n, str) for n in needs):
            continue
        
        # Skip if any need references a step that doesn't exist in steps
        # (this makes the step never runnable)
        if any(need not in valid_ids for need in needs):
            continue
        
        # Check if all needs are satisfied
        if all(need in done_set for need in needs):
            # Only add if step_id is not already in done_set
            # (a step that's done shouldn't appear again)
            if step_id not in done_set:
                result.append(step)
    
    return result