def onboard_succeeded(result: dict) -> tuple[bool, list[str]]:
    problems = []
    
    # Check exit_code
    try:
        exit_code = result.get("exit_code")
        if exit_code is None or exit_code != 0:
            problems.append("exit_code is not 0")
    except Exception:
        problems.append("exit_code is not 0")
    
    # Check daemon_pid
    try:
        daemon_pid = result.get("daemon_pid")
        if daemon_pid is None or not isinstance(daemon_pid, int) or daemon_pid <= 0:
            problems.append("daemon_pid is missing or not a positive integer")
    except Exception:
        problems.append("daemon_pid is missing or not a positive integer")
    
    # Check running_argv and --n-gpu-layers
    try:
        running_argv = result.get("running_argv")
        if running_argv is None or not isinstance(running_argv, list):
            problems.append("running_argv is missing")
        else:
            # Parse --n-gpu-layers from running_argv
            ngl_value = None
            i = 0
            while i < len(running_argv):
                arg = running_argv[i]
                if arg == "--n-gpu-layers":
                    if i + 1 < len(running_argv):
                        try:
                            ngl_value = int(running_argv[i + 1])
                        except (ValueError, TypeError):
                            pass
                    i += 2
                    continue
                elif arg.startswith("--n-gpu-layers="):
                    try:
                        ngl_value = int(arg.split("=", 1)[1])
                    except (ValueError, TypeError):
                        pass
                i += 1
            
            # Check if we need to validate ngl_value
            detected_accel = result.get("detected_accel")
            if detected_accel in ("cuda", "rocm", "vulkan"):
                if ngl_value is None:
                    problems.append("--n-gpu-layers not found in running_argv while detected_accel is " + str(detected_accel))
                elif ngl_value <= 0:
                    problems.append("--n-gpu-layers is " + str(ngl_value) + " while detected_accel is " + str(detected_accel))
    except Exception:
        # If anything goes wrong during parsing, just add a generic problem
        pass
    
    # Check http_status
    try:
        http_status = result.get("http_status")
        if http_status is None or http_status != 200:
            problems.append("http_status is not 200")
    except Exception:
        problems.append("http_status is not 200")
    
    ok = len(problems) == 0
    return (ok, problems)