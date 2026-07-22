from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from importlib import import_module
from io import StringIO
from typing import Any, Tuple


def load_faiss() -> Tuple[Any, bool, bool]:
    """
    Attempt to import FAISS while suppressing noisy stderr/stdout from
    binary/ABI import failures (for example NumPy major-version mismatches).

    Returns
    -------
    (faiss_module_or_none, is_available, has_gpu_support)
    """
    sink = StringIO()
    try:
        with redirect_stdout(sink), redirect_stderr(sink):
            faiss = import_module("faiss")
    except Exception:
        return None, False, False

    try:
        has_gpu = bool(
            getattr(faiss, "get_num_gpus", lambda: 0)() > 0
            and hasattr(faiss, "StandardGpuResources")
        )
    except Exception:
        has_gpu = False
    return faiss, True, has_gpu

