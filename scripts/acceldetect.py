def detect_accel(*, nvidia_smi: str = "", rocminfo: str = "", vulkaninfo: str = "",
                 sysfs_drm: list[str] | None = None) -> str:
    # Check CUDA first
    if nvidia_smi:
        lines = nvidia_smi.splitlines()
        has_driver_version = any("Driver Version:" in line for line in lines)
        has_memory_column = any("MiB /" in line for line in lines)
        if has_driver_version and has_memory_column:
            return "cuda"

    # Check ROCm
    if rocminfo:
        # Look for gfx followed by digits (e.g., gfx1201)
        import re
        if re.search(r'gfx\d+', rocminfo):
            return "rocm"

    # Check Vulkan
    if vulkaninfo:
        # Look for discrete or integrated GPU, but not CPU
        import re
        # Match deviceType lines with discrete or integrated GPU
        if re.search(r'PHYSICAL_DEVICE_TYPE_DISCRETE_GPU|PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU', vulkaninfo):
            return "vulkan"

    # sysfs_drm alone cannot promote above "cpu"
    # If nothing else matched, return cpu
    return "cpu"