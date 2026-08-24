import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent

if "ORT_DYLIB_PATH" not in os.environ:
    if sys.platform == "win32":
        _name = "onnxruntime.dll"
    elif sys.platform == "darwin":
        _name = "libonnxruntime.dylib"
    else:
        _name = "libonnxruntime.so"
    _candidate = _HERE / "libs" / _name
    if _candidate.exists():
        os.environ["ORT_DYLIB_PATH"] = str(_candidate)

from ._native import SearchResult, SwiftVec

__all__ = ["SwiftVec", "SearchResult"]
__version__ = "0.3.0"
