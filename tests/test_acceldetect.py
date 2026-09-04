import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
from acceldetect import detect_accel


class TestDetectAccel:
    # Real nvidia-smi block with GPU rows
    def test_cuda_real_nvidia_smi(self):
        nvidia_smi = """
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.03    Driver Version: 535.54.03    CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap| Memory-Usage     GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... Off  | 00000000:01:00.0 Off |                  N/A |
| 30%   45C    P8    15W / 250W |   1024MiB / 16384MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
"""
        assert detect_accel(nvidia_smi=nvidia_smi) == "cuda"

    # nvidia-smi header with no GPU rows (no "MiB /" lines)
    def test_cuda_header_only_no_gpu_rows(self):
        nvidia_smi = """
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.03    Driver Version: 535.54.03    CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap| Memory-Usage     GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
"""
        assert detect_accel(nvidia_smi=nvidia_smi) == "cpu"

    # rocminfo with real GPU agent (gfx1201)
    def test_rocm_with_gfx(self):
        rocminfo = """
Agent 0:
  Name:                    AMD Ryzen 9 7950X
  Marketing Name:          AMD Ryzen 9 7950X
  Agent Type:              CPU
  ... other fields ...
  Agent 1:
    Name:                    AMD Radeon Graphics
    Marketing Name:          AMD Radeon Graphics
    Agent Type:              GPU
    ... other fields ...
    Machine Models:          gfx1201
    ... other fields ...
"""
        assert detect_accel(rocminfo=rocminfo) == "rocm"

    # rocminfo with only CPU agent (no gfx)
    def test_rocm_cpu_only_no_gfx(self):
        rocminfo = """
Agent 0:
  Name:                    AMD Ryzen 9 7950X
  Marketing Name:          AMD Ryzen 9 7950X
  Agent Type:              CPU
  ... other fields ...
"""
        assert detect_accel(rocminfo=rocminfo) == "cpu"

    # vulkaninfo with discrete GPU
    def test_vulkan_discrete_gpu(self):
        vulkaninfo = """
Device Properties:
  deviceName: NVIDIA GeForce RTX 4090
  deviceType: PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
  ... other fields ...
"""
        assert detect_accel(vulkaninfo=vulkaninfo) == "vulkan"

    # vulkaninfo with integrated GPU
    def test_vulkan_integrated_gpu(self):
        vulkaninfo = """
Device Properties:
  deviceName: Intel(R) Iris(R) Xe Graphics
  deviceType: PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
  ... other fields ...
"""
        assert detect_accel(vulkaninfo=vulkaninfo) == "vulkan"

    # vulkaninfo with only CPU (lavapipe software rasterizer)
    def test_vulkan_cpu_only(self):
        vulkaninfo = """
Device Properties:
  deviceName: llvmpipe (LLVM 15.0.0, 256 bits)
  deviceType: PHYSICAL_DEVICE_TYPE_CPU
  ... other fields ...
"""
        assert detect_accel(vulkaninfo=vulkaninfo) == "cpu"

    # Precedence: cuda wins over rocm and vulkan
    def test_precedence_cuda_over_rocm_and_vulkan(self):
        nvidia_smi = """
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.03    Driver Version: 535.54.03    CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap| Memory-Usage     GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... Off  | 00000000:01:00.0 Off |                  N/A |
| 30%   45C    P8    15W / 250W |   1024MiB / 16384MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
"""
        rocminfo = """
Agent 0:
  Name:                    AMD Ryzen 9 7950X
  Marketing Name:          AMD Ryzen 9 7950X
  Agent Type:              CPU
  ... other fields ...
Agent 1:
  Name:                    AMD Radeon Graphics
  Marketing Name:          AMD Radeon Graphics
  Agent Type:              GPU
  ... other fields ...
  Machine Models:          gfx1201
  ... other fields ...
"""
        vulkaninfo = """
Device Properties:
  deviceName: NVIDIA GeForce RTX 4090
  deviceType: PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
  ... other fields ...
"""
        assert detect_accel(nvidia_smi=nvidia_smi, rocminfo=rocminfo, vulkaninfo=vulkaninfo) == "cuda"

    # All empty inputs
    def test_all_empty_returns_cpu(self):
        assert detect_accel() == "cpu"

    # sysfs_drm alone should never produce anything but cpu
    def test_sysfs_drm_alone_returns_cpu(self):
        assert detect_accel(sysfs_drm=["card0", "card1", "renderD128"]) == "cpu"

    # sysfs_drm with cuda should still return cuda
    def test_sysfs_drm_with_cuda(self):
        nvidia_smi = """
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.03    Driver Version: 535.54.03    CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap| Memory-Usage     GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... Off  | 00000000:01:00.0 Off |                  N/A |
| 30%   45C    P8    15W / 250W |   1024MiB / 16384MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
"""
        assert detect_accel(nvidia_smi=nvidia_smi, sysfs_drm=["card0"]) == "cuda"

    # sysfs_drm with vulkan should still return vulkan
    def test_sysfs_drm_with_vulkan(self):
        vulkaninfo = """
Device Properties:
  deviceName: NVIDIA GeForce RTX 4090
  deviceType: PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
  ... other fields ...
"""
        assert detect_accel(vulkaninfo=vulkaninfo, sysfs_drm=["card0"]) == "vulkan"

    # sysfs_drm with rocm should still return rocm
    def test_sysfs_drm_with_rocm(self):
        rocminfo = """
Agent 0:
  Name:                    AMD Ryzen 9 7950X
  Marketing Name:          AMD Ryzen 9 7950X
  Agent Type:              CPU
  ... other fields ...
Agent 1:
  Name:                    AMD Radeon Graphics
  Marketing Name:          AMD Radeon Graphics
  Agent Type:              GPU
  ... other fields ...
  Machine Models:          gfx1201
  ... other fields ...
"""
        assert detect_accel(rocminfo=rocminfo, sysfs_drm=["card0"]) == "rocm"