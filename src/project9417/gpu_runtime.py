from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .paths import PROJECT_ROOT


def _torch_cuda_lib_dir() -> Path | None:
    try:
        import torch
    except ImportError:
        return None

    lib_dir = Path(torch.__file__).resolve().parent / "lib"
    if not lib_dir.exists():
        return None
    return lib_dir


def configure_cupy_runtime() -> None:
    """Point CuPy at a writable temp/cache area and a CUDA-style DLL layout.

    On this Windows environment PyTorch ships the CUDA DLLs in ``torch/lib`` instead of
    ``%CUDA_PATH%/bin``. CuPy expects the latter, so we mirror the DLLs into a local
    shim directory under the workspace and keep all generated files inside the project.
    """

    torch_lib_dir = _torch_cuda_lib_dir()
    if torch_lib_dir is None:
        return

    shim_root = PROJECT_ROOT / ".cuda-shim"
    shim_bin_dir = shim_root / "bin"
    cache_dir = PROJECT_ROOT / ".cache" / "cupy"
    temp_dir = PROJECT_ROOT / ".tmp" / "cupy-temp"
    shim_bin_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    dll_names = [path.name for path in torch_lib_dir.glob("*.dll")]
    for dll_name in dll_names:
        src = torch_lib_dir / dll_name
        dst = shim_bin_dir / dll_name
        if not dst.exists():
            shutil.copy2(src, dst)

    os.environ.setdefault("CUDA_PATH", str(shim_root))
    os.environ.setdefault("CUPY_CACHE_DIR", str(cache_dir))
    os.environ["TMP"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMPDIR"] = str(temp_dir)
    tempfile.tempdir = str(temp_dir)

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    shim_bin = str(shim_bin_dir)
    if shim_bin not in path_entries:
        os.environ["PATH"] = os.pathsep.join([shim_bin, *[entry for entry in path_entries if entry]])
