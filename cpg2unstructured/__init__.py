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
from cpg2unstructured.geometry import (
    CellGeometry,
    FaceGeometry,
    cell_geometry,
    face_geometry,
)
from cpg2unstructured.graph import (
    cell_adjacency,
    cell_adjacency_csgraph,
    shortest_path,
    shortest_path_fast,
)
from cpg2unstructured.grdecl_reader import Grdecl, read_grdecl
from cpg2unstructured.index_map import IndexMap
from cpg2unstructured.unstructured import (
    Faces,
    UnstructuredGrid,
    attach_props,
    build_topology,
    find_i_faces,
    find_ij_faces,
    find_j_faces,
    find_k_faces,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Grdecl",
    "read_grdecl",
    "CornerCube",
    "from_grdecl",
    "Faces",
    "find_i_faces",
    "find_j_faces",
    "find_k_faces",
    "find_ij_faces",
    "build_topology",
    "UnstructuredGrid",
    "attach_props",
    "FaceGeometry",
    "CellGeometry",
    "face_geometry",
    "cell_geometry",
    "IndexMap",
    "cell_adjacency",
    "cell_adjacency_csgraph",
    "shortest_path",
    "shortest_path_fast",
    "Cpg2UError",
    "GrdeclError",
    "GrdeclSyntaxError",
    "GrdeclConsistencyError",
    "GrdeclIncludeError",
    "GrdeclDuplicatePropertyWarning",
    "NonMatchingZError",
]
