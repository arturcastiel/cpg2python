"""Property pass-through — ``attach_props`` + ``UnstructuredGrid``."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from cpg2unstructured import (
    UnstructuredGrid,
    attach_props,
    build_topology,
    from_grdecl,
    read_grdecl,
)

# ── basic round-trip ─────────────────────────────────────────────────


def test_poro_round_trip(fixtures):
    g = read_grdecl(fixtures / "with_include" / "main.grdecl")
    grid = attach_props(build_topology(from_grdecl(g)), g)
    assert "PORO" in grid
    assert grid["PORO"].shape == (grid.n_cells,)
    assert grid["PORO"].dtype == np.float64
    # All-active 2x2x2 with_include deck → identity reordering
    # (linear_to_cell is 0..7), so PORO values match grdecl.props directly.
    np.testing.assert_allclose(grid["PORO"],
                               g.props["PORO"][grid.index_map.linear_to_cell])


def test_permx_also_attached(fixtures):
    g = read_grdecl(fixtures / "with_include" / "main.grdecl")
    grid = attach_props(build_topology(from_grdecl(g)), g)
    assert "PERMX" in grid
    assert grid["PERMX"].shape == (grid.n_cells,)
    assert grid["PERMX"].dtype == np.float64


# ── active-cell filtering ─────────────────────────────────────────────


def test_active_cell_filtering(tmp_path, fixtures):
    src = (fixtures / "simple_2x2x2.grdecl").read_text().replace(
        "ACTNUM\n  8*1 /", "ACTNUM\n  1 1 1 1  1 1 1 0 /"
    ) + "\nPORO\n 0.10 0.11 0.12 0.13  0.20 0.21 0.22 0.99 /\n"
    p = tmp_path / "deck.grdecl"
    p.write_text(src)
    g = read_grdecl(p)
    grid = attach_props(build_topology(from_grdecl(g)), g)
    assert grid["PORO"].shape == (7,)
    # The 0.99 belongs to the deactivated cell (id 7) → must be absent.
    assert 0.99 not in grid["PORO"].tolist()


# ── error paths ───────────────────────────────────────────────────────


def test_missing_property_raises(fixtures):
    g = read_grdecl(fixtures / "simple_2x2x2.grdecl")
    grid = attach_props(build_topology(from_grdecl(g)), g)
    # simple_2x2x2 fixture has no PORO — attach_props produces an
    # empty props dict; lookup raises with the "available keys" hint.
    with pytest.raises(KeyError, match="not attached"):
        _ = grid["PORO"]


def test_missing_property_lists_available_keys(fixtures):
    g = read_grdecl(fixtures / "with_include" / "main.grdecl")
    grid = attach_props(build_topology(from_grdecl(g)), g)
    with pytest.raises(KeyError) as excinfo:
        _ = grid["DOES_NOT_EXIST"]
    # Helpful error names every attached property.
    assert "PORO" in str(excinfo.value)
    assert "PERMX" in str(excinfo.value)


def test_cart_dims_mismatch_raises(fixtures):
    g = read_grdecl(fixtures / "simple_2x2x2.grdecl")
    grid = build_topology(from_grdecl(g))
    bad = replace(g, cart_dims=(3, 3, 3))
    with pytest.raises(ValueError, match="cart_dims mismatch"):
        attach_props(grid, bad)


# ── dtype preservation ───────────────────────────────────────────────


def test_int_property_preserves_dtype(tmp_path, fixtures):
    src = (fixtures / "simple_2x2x2.grdecl").read_text() \
        + "\nSATNUM\n 1 2 1 2  3 1 2 3 /\n"
    p = tmp_path / "deck.grdecl"
    p.write_text(src)
    g = read_grdecl(p)
    grid = attach_props(build_topology(from_grdecl(g)), g)
    #  classifies SATNUM as int32; attach_props must preserve dtype.
    assert grid["SATNUM"].dtype.kind == "i"
    assert grid["SATNUM"].shape == (grid.n_cells,)


# ── purity / immutability ─────────────────────────────────────────────


def test_attach_is_pure(fixtures):
    """attach_props returns a NEW grid without mutating the input."""
    g = read_grdecl(fixtures / "with_include" / "main.grdecl")
    grid = build_topology(from_grdecl(g))
    grid2 = attach_props(grid, g)
    assert grid is not grid2
    assert grid.props == {}
    assert "PORO" in grid2.props


def test_grid_is_frozen(fixtures):
    g = read_grdecl(fixtures / "simple_2x2x2.grdecl")
    grid = build_topology(from_grdecl(g))
    with pytest.raises(FrozenInstanceError):
        grid.props = {"PORO": np.zeros(grid.n_cells)}  # type: ignore[misc]


# ── derived properties ───────────────────────────────────────────────


def test_n_cells_matches_cell_faces(fixtures):
    g = read_grdecl(fixtures / "simple_2x2x2.grdecl")
    grid = build_topology(from_grdecl(g))
    assert grid.n_cells == grid.cell_faces.shape[0]
    assert grid.n_cells == 8


def test_index_map_lazy_cached(fixtures):
    g = read_grdecl(fixtures / "simple_2x2x2.grdecl")
    grid = build_topology(from_grdecl(g))
    im1 = grid.index_map
    im2 = grid.index_map
    assert im1 is im2  # cached on first access


def test_unstructuredgrid_re_exported():
    """UnstructuredGrid is the public face of build_topology output."""
    g = read_grdecl(__import__("pathlib").Path(__file__).parent
                    / "fixtures" / "simple_2x2x2.grdecl")
    grid = build_topology(from_grdecl(g))
    assert isinstance(grid, UnstructuredGrid)


# ── interop with  /  graph (regression for the rename) ─────


def test_graph_still_works_on_unstructuredgrid(fixtures):
    """Renaming _TopoMesh → UnstructuredGrid must not break /
    callers (the backward-compat alias keeps the type hints valid)."""
    from cpg2unstructured import cell_adjacency, shortest_path
    g = read_grdecl(fixtures / "simple_2x2x2.grdecl")
    grid = build_topology(from_grdecl(g))
    adj = cell_adjacency(grid, weights="unit")
    dist, path = shortest_path(adj, 0, grid.n_cells - 1)
    assert dist == 3.0
    assert path[0] == 0
    assert path[-1] == grid.n_cells - 1
