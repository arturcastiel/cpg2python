"""End-to-end tests for ``build_topology`` on faulted decks
(PR-8 of phase 2-faults).

The headline outcome of PR-8: ``build_topology`` no longer raises on
i- or j-direction faulted decks. The i/j fault dispatch from
PR-6/PR-7 + the removal of the top-level guard in ``build_topology``
mean a faulted deck flows end-to-end through the topology builder.

k-direction faults still raise — k-fault dispatch is deferred to a
follow-up because the 2-pillar sweep-line algorithm doesn't apply
directly to the 4-pillar geometry of k-faces.
"""
from __future__ import annotations

import numpy as np
import pytest

from cpg2unstructured import (
    UnstructuredGrid,
    build_topology,
    from_grdecl,
    read_grdecl,
)
from cpg2unstructured._errors import NonMatchingZError


# ── headline: build_topology on i-faulted deck ─────────────────────


def test_build_topology_on_simple_faulted_2x2x2_i(fixtures):
    g = read_grdecl(fixtures / "simple_faulted_2x2x2.grdecl")
    cube = from_grdecl(g)
    grid = build_topology(cube)
    assert isinstance(grid, UnstructuredGrid)
    assert grid.n_cells == 8     # all-active 2x2x2
    # Sanity on outputs:
    assert grid.cell_faces.shape == (8, 6)
    assert grid.faces.nodes.shape[1] == 4
    assert grid.faces.neighbours.shape[1] == 2
    # Direction labels: 0 = i, 1 = j, 2 = k
    assert set(np.unique(grid.faces.direction).tolist()) <= {0, 1, 2}


def test_build_topology_on_simple_faulted_2x2x2_j(fixtures):
    g = read_grdecl(fixtures / "simple_faulted_2x2x2_j.grdecl")
    cube = from_grdecl(g)
    grid = build_topology(cube)
    assert grid.n_cells == 8
    assert grid.cell_faces.shape == (8, 6)


# ── still raises on k-faults ───────────────────────────────────────


def test_build_topology_raises_on_k_fault(tmp_path, fixtures):
    """Synthetic k-fault: simple_2x2x2 deck with one ZCORN value
    perturbed at the k=0 ↔ k=1 boundary (z-slot 1 on cell (0,0,0)
    set to 1.5 instead of 1)."""
    src = (fixtures / "simple_2x2x2.grdecl").read_text()
    # Original ZCORN block:
    #   ZCORN
    #     16*0   16*1
    #     16*1   16*2 /
    # The k=0-bot row is the 16*1 in the first ZCORN line. Replace
    # that with one explicit value of 1.5 + 15*1 to break matching-Z
    # between k=0 and k=1 at slot (i_slot=0, j_slot=0, k_slot=1).
    src = src.replace(
        "16*0   16*1\n  16*1   16*2 /",
        "16*0   1.5 15*1\n  16*1   16*2 /",
    )
    p = tmp_path / "k_fault.grdecl"
    p.write_text(src)
    g = read_grdecl(p)
    cube = from_grdecl(g)
    with pytest.raises(NonMatchingZError, match="k-direction"):
        build_topology(cube)
