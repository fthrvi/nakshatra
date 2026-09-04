def serve_args(pid: int, *, proc_root: str = "/proc") -> list[str]:
    """
    Return the command line arguments of a process as a list, read from
    {proc_root}/{pid}/cmdline. The file is NUL-separated with a trailing NUL,
    so we drop the trailing empty element.
    """
    try:
        cmdline_path = f"{proc_root}/{pid}/cmdline"
        with open(cmdline_path, "rb") as f:
            data = f.read()
        
        if not data:
            return []
        
        parts = data.decode("utf-8", errors="replace").split("\x00")
        # Drop trailing empty element due to trailing NUL
        if parts and parts[-1] == "":
            parts = parts[:-1]
        return parts
    except (OSError, IOError, UnicodeDecodeError):
        return []


def gpu_layers(pid: int, *, proc_root: str = "/proc") -> int | None:
    """
    Return the int value of --n-gpu-layers from the process's command line,
    or None if absent or invalid.
    """
    args = serve_args(pid, proc_root=proc_root)
    if not args:
        return None
    
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--n-gpu-layers":
            # Two-argument form: --n-gpu-layers <value>
            if i + 1 < len(args):
                try:
                    return int(args[i + 1])
                except ValueError:
                    return None
            return None
        elif arg.startswith("--n-gpu-layers="):
            # One-argument form: --n-gpu-layers=<value>
            try:
                return int(arg[len("--n-gpu-layers="):])
            except ValueError:
                return None
        i += 1
    
    return None