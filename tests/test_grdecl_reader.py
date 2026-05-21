import numpy as np
import pytest
from pathlib import Path

from cpg2unstructured import Grdecl, read_grdecl
from cpg2unstructured._errors import GrdeclConsistencyError, GrdeclSyntaxError

FIXTURES = Path(__file__).parent / "fixtures"


def test_simple_2x2x2_shapes():
    g = read_grdecl(FIXTURES / "simple_2x2x2.grdecl")
    assert isinstance(g, Grdecl)
    assert g.cart_dims == (2, 2, 2)
    assert g.coord.shape == (9, 6)
    assert g.zcorn.shape == (8 * 2 * 2 * 2,)
    assert g.actnum.shape == (8,)
    assert g.actnum.all()
    assert g.unrecognized == []


def test_simple_3x3x2_shapes():
    g = read_grdecl(FIXTURES / "simple_3x3x2.grdecl")
    assert g.cart_dims == (3, 3, 2)
    assert g.coord.shape == (16, 6)
    assert g.zcorn.size == 8 * 3 * 3 * 2
    assert g.actnum.size == 18


def test_zcorn_values_2x2x2():
    g = read_grdecl(FIXTURES / "simple_2x2x2.grdecl")
    assert np.unique(g.zcorn).tolist() == [0.0, 1.0, 2.0]


def test_unknown_keyword_collected_not_raised(tmp_path):
    p = tmp_path / "deck.grdecl"
    p.write_text(
        "SPECGRID\n 1 1 1 1 F /\n"
        "COORD\n  0 0 0  0 0 1   1 0 0  1 0 1   0 1 0  0 1 1   1 1 0  1 1 1 /\n"
        "ZCORN\n  4*0  4*1 /\n"
        "ACTNUM\n  1 /\n"
        "MULTPV\n  0.95 /\n"
    )
    g = read_grdecl(p)
    assert "MULTPV" in g.unrecognized
    assert g.cart_dims == (1, 1, 1)


def test_dimens_specgrid_mismatch_raises(tmp_path):
    p = tmp_path / "bad.grdecl"
    p.write_text(
        "DIMENS\n 2 2 2 /\n"
        "SPECGRID\n 3 2 2 1 F /\n"
        "COORD\n /\n"
        "ZCORN\n /\n"
    )
    with pytest.raises(GrdeclConsistencyError):
        read_grdecl(p)


def test_missing_terminator_raises(tmp_path):
    p = tmp_path / "trunc.grdecl"
    p.write_text("SPECGRID\n 2 2 2 1 F\n")  # no `/`
    with pytest.raises(GrdeclSyntaxError):
        read_grdecl(p)


def test_coord_size_mismatch_raises(tmp_path):
    p = tmp_path / "wrongcoord.grdecl"
    p.write_text(
        "SPECGRID\n 2 2 1 1 F /\n"
        "COORD\n  0 0 0  0 0 1 /\n"   # only 1 pillar's worth, need 9
        "ZCORN\n  16*0 /\n"
        "ACTNUM\n  4*1 /\n"
    )
    with pytest.raises(GrdeclConsistencyError):
        read_grdecl(p)


def test_dimens_only_works(tmp_path):
    # DIMENS without SPECGRID is also valid.
    p = tmp_path / "dimens_only.grdecl"
    p.write_text(
        "DIMENS\n 1 1 1 /\n"
        "COORD\n  0 0 0  0 0 1   1 0 0  1 0 1   0 1 0  0 1 1   1 1 0  1 1 1 /\n"
        "ZCORN\n  4*0  4*1 /\n"
        "ACTNUM\n  1 /\n"
    )
    g = read_grdecl(p)
    assert g.cart_dims == (1, 1, 1)


def test_actnum_optional_defaults_all_active(tmp_path):
    p = tmp_path / "no_actnum.grdecl"
    p.write_text(
        "SPECGRID\n 1 1 1 1 F /\n"
        "COORD\n  0 0 0  0 0 1   1 0 0  1 0 1   0 1 0  0 1 1   1 1 0  1 1 1 /\n"
        "ZCORN\n  4*0  4*1 /\n"
    )
    g = read_grdecl(p)
    assert g.actnum.shape == (1,)
    assert g.actnum.all()
