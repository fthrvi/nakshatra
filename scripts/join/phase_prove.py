def prove(facts: dict) -> tuple[bool, list[str], dict]:
    problems = []
    updates = {}

    # Check probe_tokens
    probe_tokens = facts.get("probe_tokens")
    if probe_tokens is None or not isinstance(probe_tokens, list) or len(probe_tokens) == 0 or not all(isinstance(t, int) for t in probe_tokens):
        problems.append("probe_tokens is empty or not a list of ints — it produced nothing")

    # Check probe_layers
    probe_layers = facts.get("probe_layers")
    if probe_layers is None or not isinstance(probe_layers, list) or len(probe_layers) != 2:
        problems.append("probe_layers is not a 2-list")
    else:
        start, end = probe_layers
        n_layers = facts.get("n_layers")
        if not isinstance(start, int) or not isinstance(end, int):
            problems.append("probe_layers start/end are not integers")
        elif not isinstance(n_layers, int):
            problems.append("probe_layers is not a 2-list with 0 <= start < end <= n_layers")
        elif not (0 <= start < end <= n_layers):
            problems.append("probe_layers is not a 2-list with 0 <= start < end <= n_layers")

    # Check serve_ngl and accel for GPU idle card issue
    serve_ngl = facts.get("serve_ngl")
    accel = facts.get("accel")
    if serve_ngl is not None and accel is not None:
        if isinstance(serve_ngl, (int, float)) and isinstance(accel, str):
            if serve_ngl <= 0 and accel.lower() in ("cuda", "gpu"):
                problems.append("a node can pass every earlier check, answer the probe, and still be running on CPU with an idle card")

    # Check answered_probe_ms
    # ⚠️ THIS IS THE ONLY PLACE THAT MAY REQUIRE `answered_probe_ms` — moved from phase_serve
    # 2026-09-07, which ran before the probe existed and so could never pass on a real run.
    # A missing/invalid value here means the probe never produced a real answer at all, which
    # is a distinct, worse problem than answering too slowly, and must not be silently ignored.
    answered_probe_ms = facts.get("answered_probe_ms")
    if not isinstance(answered_probe_ms, (int, float)) or isinstance(answered_probe_ms, bool) or answered_probe_ms <= 0:
        problems.append("probe did not answer — no real response time was observed")
    elif answered_probe_ms >= 60000:
        problems.append("it answered, eventually, in a way no requester will wait for")

    # If no problems, set success updates
    if not problems:
        updates = {"joined": True, "serving_layers": probe_layers}

    return (len(problems) == 0, problems, updates)