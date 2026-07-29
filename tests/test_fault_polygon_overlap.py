"""A fault face is the OVERLAP of two cell faces, never their union envelope.

`compute_face_geometry` takes its corner columns SMALLER-Z FIRST. Its MRST-derived
names for them are "bot" and "top", which reads backwards under Eclipse's Z-down
convention and is exactly how the callers came to pass them the other way round.

Passing them reversed does not raise. It silently returns the union envelope of
the two cell faces instead of their overlap — a fault face covering one and a half
cells in a grid of unit cells. These tests pin the orientation at both levels.
"""
from __future__ import annotations

import numpy as np
import pytest

from cpg2unstructured import from_grdecl, read_grdecl
from cpg2unstructured._fault_geometry import compute_face_geometry
from cpg2unstructured.unstructured import find_i_faces, find_j_faces

FIXTURES = "tests/fixtures"


def _two_faces(a_lo, a_hi, b_lo, b_hi, *, smaller_z_first=True):
    """One pillar pair carrying face A over [a_lo, a_hi] and B over [b_lo, b_hi]."""
    pts: list[list[float]] = []

    def add(y, z):
        pts.append([0.0, y, z])
        return len(pts) - 1

    if smaller_z_first:
        pa = [add(0.0, a_lo), add(1.0, a_lo), add(0.0, a_hi), add(1.0, a_hi)]
        pb = [add(0.0, b_lo), add(1.0, b_lo), add(0.0, b_hi), add(1.0, b_hi)]
    else:
        pa = [add(0.0, a_hi), add(1.0, a_hi), add(0.0, a_lo), add(1.0, a_lo)]
        pb = [add(0.0, b_hi), add(1.0, b_hi), add(0.0, b_lo), add(1.0, b_lo)]
    return np.array([pa]), np.array([pb]), np.asarray(pts, dtype=np.float64)


def _polygon(points, num_nodes, corners):
    return points[corners[: int(num_nodes[0])]]


def _area(poly):
    """Planar polygon area, by the shoelace on (y, z) — every x here is equal."""
    y, z = poly[:, 1], poly[:, 2]
    return 0.5 * abs(np.dot(y, np.roll(z, 1)) - np.dot(z, np.roll(y, 1)))


class TestOrientation:
    """A spans z in [0, 1], B spans [0.5, 1.5]. Overlap = [0.5, 1.0], area 0.5."""

    def test_smaller_z_first_gives_the_overlap(self):
        pa, pb, pts = _two_faces(0.0, 1.0, 0.5, 1.5)
        poly = _polygon(*compute_face_geometry(pa, pb, pts))
        assert poly[:, 2].min() == pytest.approx(0.5)
        assert poly[:, 2].max() == pytest.approx(1.0)
        assert _area(poly) == pytest.approx(0.5)

    def test_reversed_gives_the_UNION_and_does_not_raise(self):
        """Documents the trap: wrong order is silent, and 3x too large here."""
        pa, pb, pts = _two_faces(0.0, 1.0, 0.5, 1.5, smaller_z_first=False)
        poly = _polygon(*compute_face_geometry(pa, pb, pts))
        assert poly[:, 2].min() == pytest.approx(0.0)
        assert poly[:, 2].max() == pytest.approx(1.5)
        assert _area(poly) == pytest.approx(1.5)

    @pytest.mark.parametrize(
        "a_lo,a_hi,b_lo,b_hi,lo,hi",
        [(0.0, 1.0, 0.5, 1.5, 0.5, 1.0),      # partial, B deeper
         (0.0, 1.0, -0.5, 0.5, 0.0, 0.5),     # partial, B shallower
         (0.0, 1.0, 0.25, 0.75, 0.25, 0.75),  # B inside A
         (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)],     # exact match
        ids=["B-deeper", "B-shallower", "B-inside-A", "identical"],
    )
    def test_the_overlap_is_correct_for_every_arrangement(
        self, a_lo, a_hi, b_lo, b_hi, lo, hi
    ):
        pa, pb, pts = _two_faces(a_lo, a_hi, b_lo, b_hi)
        poly = _polygon(*compute_face_geometry(pa, pb, pts))
        assert poly[:, 2].min() == pytest.approx(lo)
        assert poly[:, 2].max() == pytest.approx(hi)


class TestOnRealFixtures:
    """Every cell is a unit cube, so no fault face may span more than one."""

    @staticmethod
    def _max_z_span(cube, finder):
        nodes = np.asarray(cube.nodes, dtype=np.float64)
        faces = finder(cube)
        spans = []
        for q in np.asarray(faces.nodes):
            ids = [int(x) for x in q if 0 <= int(x) < len(nodes)]
            if len(ids) >= 3:
                z = nodes[ids][:, 2]
                spans.append(float(z.max() - z.min()))
        return max(spans) if spans else 0.0

    @pytest.mark.parametrize("name", ["simple_faulted_2x2x2", "simple_faulted_3x3x2"])
    def test_no_i_face_spans_more_than_one_cell(self, name):
        """1.5 and 1.6 before the fix, against a cell height of exactly 1.0."""
        cube = from_grdecl(read_grdecl(f"{FIXTURES}/{name}.grdecl"))
        assert self._max_z_span(cube, find_i_faces) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("name", ["simple_faulted_2x2x2_j",
                                      "simple_faulted_3x3x2_jfault"])
    def test_no_j_face_spans_more_than_one_cell(self, name):
        cube = from_grdecl(read_grdecl(f"{FIXTURES}/{name}.grdecl"))
        assert self._max_z_span(cube, find_j_faces) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("name", ["simple_2x2x2", "simple_3x3x2"])
    def test_unfaulted_decks_are_untouched(self, name):
        cube = from_grdecl(read_grdecl(f"{FIXTURES}/{name}.grdecl"))
        assert self._max_z_span(cube, find_i_faces) == pytest.approx(1.0, abs=1e-9)
