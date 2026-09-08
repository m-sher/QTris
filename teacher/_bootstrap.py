"""CUDA library bootstrap that must run before numba.cuda or cupy is imported.

`libcuda.so.1` is resolved from the WSL passthrough directory and
`libnvJitLink.so.12` must be 12.3 or newer.
"""

import ctypes
import glob
import os
import re

WSL_DRIVER_DIR = "/usr/lib/wsl/lib"
MIN_NVJITLINK = (12, 3)

_loaded: dict[str, str] = {}


def _load(path: str, tag: str) -> bool:
    try:
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return False
    _loaded[tag] = path
    return True


def _nvjitlink_candidates() -> list[tuple[tuple[int, int, int], str]]:
    roots = [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")]
    roots += sorted(glob.glob("/usr/local/cuda-12.*"), reverse=True)
    out = []
    for root in filter(None, roots):
        for path in glob.glob(f"{root}/lib64/libnvJitLink.so.12.*"):
            m = re.search(r"\.so\.(\d+)\.(\d+)\.(\d+)$", path)
            if m:
                out.append((tuple(int(g) for g in m.groups()), path))
    return sorted(out, reverse=True)


def bootstrap() -> dict[str, str]:
    """Preload the driver and linker libraries; returns what was loaded, by tag."""
    if _loaded:
        return dict(_loaded)
    _load(f"{WSL_DRIVER_DIR}/libcuda.so.1", "libcuda")
    for version, path in _nvjitlink_candidates():
        if version >= MIN_NVJITLINK and _load(path, "libnvJitLink"):
            break
    return dict(_loaded)
