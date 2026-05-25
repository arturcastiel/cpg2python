"""Shared pytest fixtures.

Centralised here so test files can share mesh fixtures without
re-defining them. Introduced by  (spec hygiene F-PR11-1) —
``mesh_2x2x2`` moved out of ``tests/test_graph.py``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cpg2unstructured import (
    build_topology,
    cell_geometry,
    face_geometry,
    from_grdecl,
    read_grdecl,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    """Path to the shared GRDECL fixture directory."""
    return FIXTURES


@pytest.fixture
def mesh_2x2x2():
    """Topology mesh for the all-active 2x2x2 cartesian deck."""
    cube = from_grdecl(read_grdecl(FIXTURES / "simple_2x2x2.grdecl"))
    return build_topology(cube)


@pytest.fixture
def mesh_3x3x2():
    """Topology mesh for the all-active 3x3x2 cartesian deck."""
    cube = from_grdecl(read_grdecl(FIXTURES / "simple_3x3x2.grdecl"))
    return build_topology(cube)


def _make_cartesian_grdecl_text(nx: int, ny: int, nz: int) -> str:
    """Synthesize an all-active cartesian matching-Z GRDECL deck.

    Used to build larger meshes on demand for the  numpy↔scipy
    agreement tests without committing extra fixture files.
    """
    lines: list[str] = [
        f"-- synthetic {nx}x{ny}x{nz} cartesian matching-Z deck",
        "SPECGRID",
        f"  {nx} {ny} {nz} 1 F /",
        "",
        "COORD",
    ]
    pillars: list[str] = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            pillars.append(f"  {i} {j} 0  {i} {j} {nz}")
    pillars[-1] = pillars[-1] + "  /"
    lines.extend(pillars)
    lines.append("")
    lines.append("ZCORN")
    per_plane = 4 * nx * ny
    zlines: list[str] = []
    for k in range(nz):
        zlines.append(f"  {per_plane}*{k}   {per_plane}*{k + 1}")
    zlines[-1] = zlines[-1] + " /"
    lines.extend(zlines)
    lines.append("")
    lines.append("ACTNUM")
    lines.append(f"  {nx * ny * nz}*1 /")
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="session")
def mesh_5x5x3_with_centroids(tmp_path_factory):
    """Synthetic 5x5x3 mesh + centroids for fast-path agreement tests.

    Scaled down from the  spec's 10x10x5 to keep the synthetic
    GRDECL string + parse cheap. 75 cells still exercises non-trivial
    paths and is plenty to surface a weight-mode disagreement between
    heapq-Python and scipy's C impl.
    """
    deck_dir = tmp_path_factory.mktemp("synthetic")
    path = deck_dir / "cartesian_5x5x3.grdecl"
    path.write_text(_make_cartesian_grdecl_text(5, 5, 3))
    mesh = build_topology(from_grdecl(read_grdecl(path)))
    fg = face_geometry(mesh.nodes, mesh.faces, mesh.cell_corners)
    cg = cell_geometry(mesh.nodes, mesh.faces, mesh.cell_faces, fg)
    return mesh, cg.centroids
