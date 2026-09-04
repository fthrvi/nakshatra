def serve(facts: dict) -> tuple[bool, list[str], dict]:
    problems = []
    updates = {}

    # Check daemon_pid
    daemon_pid = facts.get("daemon_pid")
    if not isinstance(daemon_pid, int) or daemon_pid <= 0:
        problems.append("daemon_pid is not a positive int")

    # Check running_argv
    running_argv = facts.get("running_argv")
    if not isinstance(running_argv, list):
        problems.append("running_argv is missing or not a list")
        return (False, problems, updates)

    # Parse --n-gpu-layers from running_argv
    ngl = None
    i = 0
    while i < len(running_argv):
        arg = running_argv[i]
        if arg == "--n-gpu-layers":
            # Next argument should be the value
            if i + 1 < len(running_argv):
                try:
                    ngl = int(running_argv[i + 1])
                except (ValueError, IndexError):
                    pass
            i += 1
        elif arg.startswith("--n-gpu-layers="):
            try:
                ngl = int(arg.split("=", 1)[1])
            except (ValueError, IndexError):
                pass
            i += 1
        else:
            i += 1

    # Check if --n-gpu-layers is present (exact match or with =)
    has_flag = False
    for arg in running_argv:
        if arg == "--n-gpu-layers" or arg.startswith("--n-gpu-layers="):
            has_flag = True
            break

    if not has_flag:
        problems.append("running_argv has no --n-gpu-layers")

    # Check accel
    accel = facts.get("accel")
    if accel not in ("cuda", "rocm", "vulkan", "cpu"):
        problems.append("accel is missing or invalid")
        return (False, problems, updates)

    # Check layer count against accel
    if accel in ("cuda", "rocm", "vulkan"):
        if ngl is not None and ngl <= 0:
            problems.append("the daemon is up and the card is idle")
    elif accel == "cpu":
        if ngl is not None and ngl != 0:
            problems.append("claiming offload with no GPU")

    # Check answered_probe_ms
    answered_probe_ms = facts.get("answered_probe_ms")
    if answered_probe_ms is None or answered_probe_ms <= 0 or answered_probe_ms >= 60000:
        problems.append("probe did not answer in time")

    # If no problems, set success updates
    if not problems:
        updates = {"serving": True, "serve_ngl": ngl if ngl is not None else 0}

    return (len(problems) == 0, problems, updates)