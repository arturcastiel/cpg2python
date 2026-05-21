"""Lightweight CPG → unstructured-mesh + graph."""
from __future__ import annotations

from cpg2unstructured._errors import (
    Cpg2UError,
    GrdeclConsistencyError,
    GrdeclDuplicatePropertyWarning,
    GrdeclError,
    GrdeclIncludeError,
    GrdeclSyntaxError,
    NonMatchingZError,
)
from cpg2unstructured.grdecl_reader import Grdecl, read_grdecl

__version__ = "0.1.0.dev0"

__all__ = [
    "__version__",
    "Grdecl",
    "read_grdecl",
    "Cpg2UError",
    "GrdeclError",
    "GrdeclSyntaxError",
    "GrdeclConsistencyError",
    "GrdeclIncludeError",
    "GrdeclDuplicatePropertyWarning",
    "NonMatchingZError",
]
