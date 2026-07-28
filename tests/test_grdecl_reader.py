from pathlib import Path

import numpy as np
import pytest

from cpg2unstructured import Grdecl, read_grdecl
from cpg2unstructured._errors import (
    GrdeclConsistencyError,
    GrdeclDuplicatePropertyWarning,
    GrdeclIncludeError,
    GrdeclSyntaxError,
)

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


# ────────────────────────────────────────────────────────────────────
#   — INCLUDE recursion + property pass-through
# ────────────────────────────────────────────────────────────────────


def test_include_resolves_relative_to_includer():
    g = read_grdecl(FIXTURES / "with_include" / "main.grdecl")
    assert g.cart_dims == (2, 2, 2)
    assert g.coord.shape == (9, 6)
    assert "PORO" in g.props
    assert g.props["PORO"].shape == (8,)
    assert g.props["PORO"].dtype == np.float64
    assert "PERMX" in g.props
    assert g.props["PERMX"].shape == (8,)
    assert "PORO" not in g.unrecognized
    assert "PERMX" not in g.unrecognized
    # Spot-check the property values came from props.inc, in order.
    np.testing.assert_allclose(
        g.props["PORO"],
        [0.10, 0.12, 0.13, 0.11, 0.18, 0.21, 0.20, 0.19],
    )


def test_include_cycle_raises():
    with pytest.raises(GrdeclIncludeError):
        read_grdecl(FIXTURES / "with_include_cycle" / "a.grdecl")


def test_property_array_wrong_length_raises(tmp_path):
    # NOTE: do NOT prepend an extra SPECGRID — splitting after "SPECGRID"
    # already returns the keyword's payload (`" 2 2 2 1 F /\n…"`). A naive
    # prepend produces two SPECGRID payloads in one file and the parser
    # raises GrdeclSyntaxError on the orphaned `2` before reaching PORO.
    base = (FIXTURES / "simple_2x2x2.grdecl").read_text()
    head = "SPECGRID" + base.split("SPECGRID", 1)[1].split("ACTNUM", 1)[0]
    p = tmp_path / "bad.grdecl"
    p.write_text(
        head
        + "ACTNUM\n 8*1 /\n"
        + "PORO\n 0.1 0.2 0.3 /\n"   # 3 values, need 8
    )
    with pytest.raises(GrdeclConsistencyError):
        read_grdecl(p)


def test_duplicate_property_warns(tmp_path):
    p = tmp_path / "dup.grdecl"
    base = (FIXTURES / "simple_2x2x2.grdecl").read_text()
    p.write_text(base + "\nPORO\n 8*0.10 /\nPORO\n 8*0.20 /\n")
    with pytest.warns(GrdeclDuplicatePropertyWarning):
        g = read_grdecl(p)
    np.testing.assert_allclose(g.props["PORO"], 0.20)


def test_include_quotes_stripped(tmp_path):
    p = tmp_path / "q.grdecl"
    sub = tmp_path / "sub.inc"
    sub.write_text((FIXTURES / "simple_2x2x2.grdecl").read_text())
    p.write_text(f"INCLUDE\n \"{sub.name}\" /\n")
    g = read_grdecl(p)
    assert g.cart_dims == (2, 2, 2)


def test_include_whitespace_in_filename_raises(tmp_path):
    p = tmp_path / "bad_inc.grdecl"
    p.write_text("INCLUDE\n 'has space.inc' /\n")
    with pytest.raises(GrdeclSyntaxError):
        read_grdecl(p)


def test_int_property_uses_int32_not_int8(tmp_path):
    # Region IDs > 127 must survive — int8 would overflow.
    p = tmp_path / "regions.grdecl"
    base = (FIXTURES / "simple_2x2x2.grdecl").read_text()
    p.write_text(base + "\nSATNUM\n 100 110 120 130  140 150 160 200 /\n")
    g = read_grdecl(p)
    assert "SATNUM" in g.props
    assert g.props["SATNUM"].dtype == np.int32
    assert g.props["SATNUM"].tolist() == [100, 110, 120, 130, 140, 150, 160, 200]
