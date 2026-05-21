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
from cpg2unstructured.cpg import CornerCube, from_grdecl
from cpg2unstructured.grdecl_reader import Grdecl, read_grdecl
from cpg2unstructured.unstructured import (
    Faces,
    find_i_faces,
    find_ij_faces,
    find_j_faces,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "__version__",
    "Grdecl",
    "read_grdecl",
    "CornerCube",
    "from_grdecl",
    "Faces",
    "find_i_faces",
    "find_j_faces",
    "find_ij_faces",
    "Cpg2UError",
    "GrdeclError",
    "GrdeclSyntaxError",
    "GrdeclConsistencyError",
    "GrdeclIncludeError",
    "GrdeclDuplicatePropertyWarning",
    "NonMatchingZError",
]
