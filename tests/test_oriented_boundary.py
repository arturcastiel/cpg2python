"""Tests for oriented ∂₂: oriented_face_edges + boundary_chain (signed ℤ rim).

PR-001 of the quad-mesh chain-map phase. These are the scaffold invariants the
later fill/assembly steps build on: every face's signed boundary is a cycle
(∂₁∂₂ = 0), the signed chain reduces mod 2 to the existing unsigned rim, and
folds become an even coefficient instead of a silent mod-2 hole.
"""
from __future__ import annotations

import numpy as np

from cpg2unstructured import (
    boundary_chain,
    boundary_edges,
    face_edges,
    mesh_edges,
    oriented_face_edges,
    planar_minimal_surface,
)

BIG = np.array([[-9, -9], [9, -9], [9, 9], [-9, 9]], dtype=float)


def _surface_face_ids(mesh):
    surf = planar_minimal_surface(
        mesh, np.array([1.0, 1, 1]), np.array([1.0, 0, 0]),
        BIG, np.array([0.0, 1, 0]), np.array([0.0, 0, 1]))
    return surf.face_ids


def _directed_endpoints(mesh, eid, signs):
    """Reconstruct directed (src, dst) node pairs from edge ids + signs."""
    edges = mesh_edges(mesh)
    lohi = edges[eid]                          # (..., 2), lo<hi per row
    lo, hi = lohi[..., 0], lohi[..., 1]
    src = np.where(signs > 0, lo, hi)
    dst = np.where(signs > 0, hi, lo)
    return src, dst


# ── exports / shapes ─────────────────────────────────────────────────


def test_exports_are_importable():
    # Importable from the package root (the public surface for PR-002/003).
    assert callable(oriented_face_edges)
    assert callable(boundary_chain)


def test_oriented_face_edges_shapes(mesh_3x3x2):
    eid, signs = oriented_face_edges(mesh_3x3x2)
    nf = mesh_3x3x2.faces.nodes.shape[0]
    assert eid.shape == (nf, 4)
    assert signs.shape == (nf, 4)
    assert eid.dtype == np.int64
    assert set(np.unique(signs).tolist()).issubset({-1, 1})


# ── oriented_face_edges agrees with the unsigned sibling ─────────────


def test_oriented_edges_are_the_same_four_edges(mesh_3x3x2):
    # Same 4 edges as face_edges, just in cyclic (not sorted) order.
    eid, _ = oriented_face_edges(mesh_3x3x2)
    unsigned = face_edges(mesh_3x3x2)
    assert np.array_equal(np.sort(eid, axis=1), unsigned)


def test_each_face_boundary_is_a_cycle(mesh_3x3x2):
    # Walking the 4 oriented edges returns to the start: per face the multiset of
    # sources equals the multiset of destinations (∂₁∂₂ = 0 for one face).
    eid, signs = oriented_face_edges(mesh_3x3x2)
    src, dst = _directed_endpoints(mesh_3x3x2, eid, signs)
    for f in range(eid.shape[0]):
        assert sorted(src[f].tolist()) == sorted(dst[f].tolist())
        # and it is a single 4-cycle: each of 4 distinct nodes in/out once
        assert len(set(src[f].tolist())) == 4


def test_consecutive_edges_share_a_node(mesh_3x3x2):
    # Cyclic order: edge k's dst is edge k+1's src.
    eid, signs = oriented_face_edges(mesh_3x3x2)
    src, dst = _directed_endpoints(mesh_3x3x2, eid, signs)
    assert np.array_equal(dst, np.roll(src, -1, axis=1))


# ── boundary_chain ──────────────────────────────────────────────────


def test_boundary_chain_single_face_four_unit_edges(mesh_3x3x2):
    eid, coeff = boundary_chain(mesh_3x3x2, np.array([0]))
    assert eid.shape == (4,)
    assert set(np.abs(coeff).tolist()) == {1}
    # same 4 edges the unsigned ∂₂ gives for that face
    assert np.array_equal(np.sort(eid), face_edges(mesh_3x3x2)[0])


def test_boundary_chain_is_a_cycle_for_any_face_set(mesh_3x3x2):
    # ∂₁∂₂ = 0: the net 1-chain has zero divergence at every node, regardless of
    # orientation coherence (a sum of cycles is a cycle).
    rng = np.random.default_rng(7)
    nf = mesh_3x3x2.faces.nodes.shape[0]
    edges = mesh_edges(mesh_3x3x2)
    for _ in range(20):
        k = int(rng.integers(1, nf + 1))
        fids = rng.choice(nf, size=k, replace=False)
        eid, coeff = boundary_chain(mesh_3x3x2, fids)
        bal = np.zeros(mesh_3x3x2.nodes.shape[0], dtype=np.int64)
        lo, hi = edges[eid][:, 0], edges[eid][:, 1]
        np.add.at(bal, hi, coeff)
        np.add.at(bal, lo, -coeff)
        assert not bal.any()


def test_boundary_chain_mod2_matches_boundary_edges(mesh_3x3x2):
    # The oriented-ℤ chain, reduced mod 2, is exactly the unsigned rim.
    fids = _surface_face_ids(mesh_3x3x2)
    eid, coeff = boundary_chain(mesh_3x3x2, fids)
    odd = set(eid[coeff % 2 != 0].tolist())
    assert odd == set(boundary_edges(mesh_3x3x2, fids).tolist())


def test_boundary_chain_matches_bruteforce(mesh_3x3x2):
    # Vectorized accumulation == a plain python reference, including signs.
    rng = np.random.default_rng(11)
    nf = mesh_3x3x2.faces.nodes.shape[0]
    eid_all, sign_all = oriented_face_edges(mesh_3x3x2)
    fids = rng.choice(nf, size=8, replace=False)
    fsign = rng.choice([-1, 1], size=8)

    ref: dict[int, int] = {}
    for f, s in zip(fids, fsign):
        for e, g in zip(eid_all[f], sign_all[f]):
            ref[int(e)] = ref.get(int(e), 0) + int(s) * int(g)
    ref = {e: c for e, c in ref.items() if c != 0}

    eid, coeff = boundary_chain(mesh_3x3x2, fids, face_signs=fsign)
    got = dict(zip(eid.tolist(), coeff.tolist()))
    assert got == ref


def test_boundary_chain_empty(mesh_3x3x2):
    eid, coeff = boundary_chain(mesh_3x3x2, np.empty(0, np.int64))
    assert eid.size == 0 and coeff.size == 0


def test_boundary_chain_face_signs_length_checked(mesh_3x3x2):
    import pytest
    with pytest.raises(ValueError):
        boundary_chain(mesh_3x3x2, np.array([0, 1]), face_signs=np.array([1]))
