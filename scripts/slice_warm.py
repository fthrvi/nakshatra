"""slice_warm.py — keep a model's GGUF layer-slices hot in the OS page cache so
a worker re-summon (or first request) skips the multi-GB disk read.

Why this exists: our cold-start (~7.5s) is process-spawn + ROCm-init + DISK READ
of the slice + PCIe upload to VRAM. llama.cpp mmaps the GGUF and demand-pages it
on first touch, so the first load eats the full disk→page-cache fault. Reading
the slice into the page cache ahead of demand (and, best-effort, mlock'ing it)
removes the disk portion from the critical path — and, crucially, the page cache
**survives the worker process being reaped**, so re-summon stays fast.

This is the cheapest, exact-fit cold-start win for our llama.cpp-GGUF stack
(research 2026-06-21): no model-path code change, no GPU, privilege-free in its
default (plain sequential read) mode. mlock is opt-in and best-effort.

Portable substitute for NVIDIA-only tricks (GPUDirect Storage / CRIU snapshot)
that don't exist on our AMD/ROCm GPU.
"""
from __future__ import annotations
import os
from typing import Callable, Iterable, Optional

_CHUNK = 8 << 20  # 8 MiB sequential reads


def warm_path(path: str, mlock: bool = False,
              log: Optional[Callable[[str], None]] = None) -> dict:
    """Read one file fully into the page cache (sequential), optionally mlock it.

    Returns {path, bytes, ok, mlocked, error}. Never raises — a warm failure
    must never break the summon path (degrades to a normal cold load).
    """
    res = {"path": path, "bytes": 0, "ok": False, "mlocked": False, "error": None}
    try:
        size = os.path.getsize(path)
    except OSError as e:
        res["error"] = f"stat: {e}"
        return res
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as e:
        res["error"] = f"open: {e}"
        return res
    try:
        # Hint the kernel we'll read it all, then actually fault it in.
        try:
            os.posix_fadvise(fd, 0, size, os.POSIX_FADV_WILLNEED)
            os.posix_fadvise(fd, 0, size, os.POSIX_FADV_SEQUENTIAL)
        except (AttributeError, OSError):
            pass
        read = 0
        while True:
            b = os.read(fd, _CHUNK)
            if not b:
                break
            read += len(b)
        res["bytes"] = read
        res["ok"] = True
        if mlock:
            res["mlocked"] = _try_mlock(path, size, log)
    finally:
        os.close(fd)
    if log:
        tag = " +mlock" if res["mlocked"] else ""
        log(f"[slice-warm] {os.path.basename(path)} {res['bytes']/1e9:.2f}GB cached{tag}")
    return res


def _try_mlock(path: str, size: int,
               log: Optional[Callable[[str], None]]) -> bool:
    """Best-effort mlock of the file's pages so memory pressure can't evict the
    warmed slice. Requires holding the mmap alive, so we stash it on a module
    registry. Fails gracefully on RLIMIT_MEMLOCK (the common unprivileged case)."""
    import mmap
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        # ACCESS_COPY = private, writable view (needed so ctypes can take the
        # address); COW pages still reflect the file's page-cache residency.
        mm = mmap.mmap(fd, 0, access=mmap.ACCESS_COPY)
        if hasattr(mm, "madvise"):
            try:
                mm.madvise(mmap.MADV_WILLNEED)
            except OSError:
                pass
        try:
            _mlock_ctypes(mm, size)
        except (OSError, AttributeError, TypeError, ValueError) as e:
            mm.close()
            if log:
                log(f"[slice-warm] mlock skipped ({e}); page-cache warm only")
            return False
        _PINNED[path] = mm   # hold the mapping so the lock persists
        return True
    finally:
        os.close(fd)


def _mlock_ctypes(mm, size: int) -> None:
    import ctypes, ctypes.util
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
    if libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(size)) != 0:
        raise OSError(ctypes.get_errno(), "mlock failed")


# module-level registry of pinned mappings (keeps mlock alive for process life)
_PINNED: "dict[str, object]" = {}


def warm_paths(paths: Iterable[str], mlock: bool = False,
               log: Optional[Callable[[str], None]] = None) -> list[dict]:
    """Warm several slice files. Returns a result dict per path. Best-effort."""
    return [warm_path(p, mlock=mlock, log=log) for p in paths if p]


def resident_fraction(path: str) -> float:
    """Fraction of the file currently resident in the page cache (0..1), via
    mincore. Returns -1.0 if it can't be determined. For tests / observability."""
    import ctypes, ctypes.util, mmap, math
    try:
        size = os.path.getsize(path)
        if size == 0:
            return 1.0
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return -1.0
    try:
        # ACCESS_COPY so ctypes.from_buffer (needs writable) can take the address;
        # the COW view still reports the underlying file pages' residency.
        mm = mmap.mmap(fd, 0, access=mmap.ACCESS_COPY)
    except (OSError, ValueError):
        os.close(fd)
        return -1.0
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        page = os.sysconf("SC_PAGE_SIZE")
        n = math.ceil(size / page)
        vec = (ctypes.c_ubyte * n)()
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
            return -1.0
        resident = sum(1 for b in vec if b & 1)
        return resident / n
    except (TypeError, ValueError, OSError):
        return -1.0
    finally:
        mm.close()
        os.close(fd)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        before = resident_fraction(p)
        r = warm_path(p, mlock=("--mlock" in sys.argv), log=print)
        after = resident_fraction(p)
        print(f"  {p}: resident {before:.0%} -> {after:.0%}  ({r['bytes']/1e9:.2f}GB)")
