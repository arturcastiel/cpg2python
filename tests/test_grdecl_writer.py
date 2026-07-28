"""write_grid_include — round-trip fidelity and the MAPAXES refusal."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cpg2unstructured.corner_point import CornerPointGrid
from cpg2unstructured.grdecl_reader import read_grdecl
from cpg2unstructured.grdecl_writer import (
    MapaxesNotInvertedError,
    format_keyword,
    write_grid_include,
)

FIXTURES = Path(__file__).parent / "fixtures"
GRDECLS = sorted(FIXTURES.glob("*.grdecl"))


def test_fixtures_present():
    assert GRDECLS, "no .grdecl fixtures found — the round-trip suite would vacuously pass"


@pytest.mark.parametrize("src", GRDECLS, ids=lambda p: p.stem)
def test_round_trip_is_array_identical(src, tmp_path):
    """read -> write -> read must reproduce every GRID array exactly."""
    a = read_grdecl(src)
    out = write_grid_include(a, tmp_path / "grid.grdecl")
    b = read_grdecl(out)

    assert b.cart_dims == a.cart_dims
    np.testing.assert_array_equal(
        np.asarray(b.coord, float).reshape(-1, 6),
        np.asarray(a.coord, float).reshape(-1, 6),
    )
    np.testing.assert_array_equal(
        np.asarray(b.zcorn, float).ravel(), np.asarray(a.zcorn, float).ravel()
    )
    np.testing.assert_array_equal(
        np.asarray(b.actnum).ravel(), np.asarray(a.actnum).ravel()
    )


@pytest.mark.parametrize("src", GRDECLS, ids=lambda p: p.stem)
def test_round_trip_is_idempotent(src, tmp_path):
    """Writing twice produces byte-identical files."""
    g = read_grdecl(src)
    p1 = write_grid_include(g, tmp_path / "a.grdecl")
    p2 = write_grid_include(read_grdecl(p1), tmp_path / "b.grdecl")
    assert p1.read_text() == p2.read_text()


def test_source_deck_is_never_modified(tmp_path):
    src = GRDECLS[0]
    before = src.read_bytes()
    write_grid_include(read_grdecl(src), tmp_path / "out.grdecl")
    assert src.read_bytes() == before, "the source deck must never be touched"


def test_mapaxes_applied_is_refused(tmp_path):
    """Map-frame coordinates must not be written silently — they would double-rotate."""
    g = read_grdecl(GRDECLS[0])
    rotated = CornerPointGrid(
        cart_dims=g.cart_dims,
        coord=g.coord,
        zcorn=g.zcorn,
        actnum=g.actnum,
        mapaxes=np.arange(6, dtype=np.float64),
        mapaxes_applied=True,
        source="egrid",
    )
    with pytest.raises(MapaxesNotInvertedError):
        write_grid_include(rotated, tmp_path / "x.grdecl")

    # explicit opt-in is honoured
    out = write_grid_include(rotated, tmp_path / "y.grdecl", allow_mapaxes_applied=True)
    assert out.exists()


def test_mapaxes_not_applied_is_allowed(tmp_path):
    g = read_grdecl(GRDECLS[0])
    raw = CornerPointGrid(
        cart_dims=g.cart_dims, coord=g.coord, zcorn=g.zcorn, actnum=g.actnum,
        mapaxes=np.arange(6, dtype=np.float64), mapaxes_applied=False, source="egrid",
    )
    assert write_grid_include(raw, tmp_path / "z.grdecl").exists()


def test_actnum_can_be_omitted(tmp_path):
    g = read_grdecl(GRDECLS[0])
    out = write_grid_include(g, tmp_path / "no_actnum.grdecl", include_actnum=False)
    assert "ACTNUM" not in out.read_text()


def test_header_is_commented(tmp_path):
    g = read_grdecl(GRDECLS[0])
    out = write_grid_include(g, tmp_path / "h.grdecl", header="fault-adapted\ntwo lines")
    text = out.read_text()
    assert "-- fault-adapted" in text and "-- two lines" in text


def test_dimension_mismatch_raises(tmp_path):
    g = read_grdecl(GRDECLS[0])
    nx, ny, nz = g.cart_dims
    bad = CornerPointGrid(
        cart_dims=(nx + 1, ny, nz), coord=g.coord, zcorn=g.zcorn, actnum=g.actnum,
    )
    with pytest.raises(ValueError, match="COORD has"):
        write_grid_include(bad, tmp_path / "bad.grdecl")


class TestRunLengthEncoding:
    def test_runs_are_collapsed(self):
        text = format_keyword("K", [1, 1, 1, 1], integer=True)
        assert "4*1" in text

    def test_single_values_are_bare(self):
        text = format_keyword("K", [1, 2, 3], integer=True)
        assert "*" not in text

    def test_terminator_and_name(self):
        text = format_keyword("K", [7], integer=True)
        assert text.startswith("K\n") and text.rstrip().endswith("/")

    def test_encoded_values_survive_a_read(self, tmp_path):
        """RLE is only worth having if the reader expands it back."""
        g = read_grdecl(GRDECLS[0])
        out = write_grid_include(g, tmp_path / "rle.grdecl")
        assert "*" in out.read_text(), "fixture should contain at least one run"
        np.testing.assert_array_equal(
            np.asarray(read_grdecl(out).zcorn, float).ravel(),
            np.asarray(g.zcorn, float).ravel(),
        )
