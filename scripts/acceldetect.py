"""acceldetect — decide the accelerator from CAPTURED probe output. Pure; runs nothing.

⚠️⚠️ THIS OVERLAPS `fabric/worker_join.detect_capabilities()`, AND THAT IS KNOWN. That
function runs the probes and reads the VRAM number — it needs the machine. This one decides
the backend from the captured stdout, so the vendor rules are testable on a laptop with no
card:

  · an `nvidia-smi` header with NO GPU rows is not a GPU (driver present, card removed or
    claimed by another container)
  · `rocminfo` lists the CPU agent too — a `gfx` marker is what separates a real GPU agent
  · a `vulkaninfo` reporting PHYSICAL_DEVICE_TYPE_CPU is lavapipe, a SOFTWARE rasteriser that
    serves slower than CPU while looking like a GPU

The right shape is for `detect_capabilities()` to keep the probing and delegate the DECIDING
here — the same split the join phases use. That refactor is deliberately NOT done: worker_join
is live, its tests pass, and this is a testability improvement rather than a bug fix. The two
never disagree about anything load-bearing — unlike the trust tiers, which really were two
answers to one question and are now read from `admission.TIER_RANK`.

⚠️ IF YOU ADD A VENDOR HERE, ADD IT THERE TOO, or the fleet detects it on one path and not
the other — and the node that joins through the wrong one serves on its CPU.
"""
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