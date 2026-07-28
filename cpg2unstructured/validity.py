"""Geometric validity of a deformed pillar lattice.

The one hard constraint on any lateral grid deformation is that no cell may break:
no crossing edges, no folded or inverted cells. On the pillar lattice that reduces
to a cheap, fully vectorised test.

**Why not** :func:`cpg2unstructured.geometry.cell_geometry`? It computes volumes by
the divergence theorem inside a Python loop over cells, with an inner loop over
faces. Correct, but interpreted work proportional to cell count — calling it once
per solver iteration on a real grid is prohibitive. The signed-area test below is a
handful of numpy expressions over the ``(NX, NY)`` column grid, and for lateral
motion it detects exactly the same event.

**Both planes are always checked.** A fold at the top plane and one at the bottom
plane are independent once pillars are slanted — and real pillars are slanted
(Norne: 7 of 5,311 are vertical). A single-plane test is only sufficient for the
perfectly vertical case, which is the exception rather than the rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cpg2unstructured.pillars import PillarLattice, local_spacing

__all__ = [
    "ValidityReport",
    "column_signed_areas",
    "check_validity",
    "separation_violations",
    "depth_crossings",
    "corner_folds",
    "proximity_violations",
    "WELD_TOLERANCE",
]

#: Node identity in this library is coordinate-derived — ``from_grdecl`` deduplicates
#: on ``np.round(xyz * 1e9)``. Two pillars closer than this in every coordinate
#: therefore collapse to ONE node, silently welding cells together and changing the
#: topology. It is the absolute floor below which a deformation is not merely poor
#: quality but structurally wrong.
WELD_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ValidityReport:
    """Outcome of a validity check over a candidate lattice."""

    ok: bool
    folded_top: np.ndarray      # (NX, NY) bool — sign flipped or degenerate at top
    folded_bot: np.ndarray      # (NX, NY) bool — ... at bottom
    welded: np.ndarray          # (NX+1, NY+1) bool — pillar within WELD_TOLERANCE of a neighbour
    too_close: np.ndarray       # (NX+1, NY+1) bool — below the separation fraction
    n_folded: int
    n_welded: int
    n_too_close: int
    degenerate: np.ndarray | None = None   # (NX, NY) bool — below min_scaled_jacobian
    n_degenerate: int = 0

    def summary(self) -> str:
        if self.ok:
            return "valid: no folds, no welds, no separation violations"
        bits = [
            f"{self.n_folded} folded column(s)",
            f"{self.n_welded} weld(s)",
            f"{self.n_too_close} separation violation(s)",
        ]
        if self.n_degenerate:
            bits.append(f"{self.n_degenerate} degenerate cell(s)")
        return "INVALID: " + ", ".join(bits)


def column_signed_areas(xy: np.ndarray) -> np.ndarray:
    """``(NX, NY)`` signed area of every column's plan-view quad.

    ``xy`` is ``(NX+1, NY+1, 2)`` pillar positions on one plane. Corners are taken in
    cyclic order ``(i,j) -> (i+1,j) -> (i+1,j+1) -> (i,j+1)`` and the shoelace formula
    applied. The SIGN is what matters: it flips exactly when the quad folds over
    itself, which is the same event as an edge crossing or an inverted cell.
    """
    p00 = xy[:-1, :-1]
    p10 = xy[1:, :-1]
    p11 = xy[1:, 1:]
    p01 = xy[:-1, 1:]
    # shoelace over the 4 corners, taken cyclically
    return 0.5 * (
        (p00[..., 0] * p10[..., 1] - p10[..., 0] * p00[..., 1])
        + (p10[..., 0] * p11[..., 1] - p11[..., 0] * p10[..., 1])
        + (p11[..., 0] * p01[..., 1] - p01[..., 0] * p11[..., 1])
        + (p01[..., 0] * p00[..., 1] - p00[..., 0] * p01[..., 1])
    )


def corner_folds(xy: np.ndarray, xy0: np.ndarray) -> np.ndarray:
    """``(NX, NY)`` bool — cells with a corner that has folded over.

    Uses the PER-CORNER Jacobian sign, not the quad's total signed area. The
    shoelace sum over a quad can stay positive while one corner turns inside out —
    the other corners simply outweigh it — so an area-only test reports a valid
    cell where a corner has inverted. Measured: a run reporting "0 folds" by signed
    area had a scaled Jacobian of -0.23, i.e. a genuinely inverted cell.

    A corner is folded when its cross product changes sign relative to the same
    corner in the original grid.
    """
    from cpg2unstructured.quality import _corners, _cross

    quad_now = list(_corners(xy))
    quad_ref = list(_corners(xy0))
    out = np.zeros(quad_now[0].shape[:2], dtype=bool)
    for k in range(4):
        cur_n, nxt_n, prv_n = quad_now[k], quad_now[(k + 1) % 4], quad_now[(k - 1) % 4]
        cur_r, nxt_r, prv_r = quad_ref[k], quad_ref[(k + 1) % 4], quad_ref[(k - 1) % 4]
        jn = _cross(nxt_n - cur_n, prv_n - cur_n)
        jr = _cross(nxt_r - cur_r, prv_r - cur_r)
        out |= (np.sign(jn) != np.sign(jr)) & (np.abs(jr) > 0)
    return out


def _folded(before: np.ndarray, after: np.ndarray, *, rel_tol: float = 1e-12) -> np.ndarray:
    """Columns whose signed area changed sign or collapsed.

    Compared against the ORIGINAL sign rather than assuming positive orientation —
    a deck may be laid out either way, and a whole-grid convention is not something
    to assume. Columns that were already degenerate before the move are excluded:
    the deformation cannot be blamed for them.
    """
    scale = np.maximum(np.abs(before), 1.0)
    was_degenerate = np.abs(before) <= rel_tol * scale
    flipped = np.sign(after) != np.sign(before)
    collapsed = np.abs(after) <= rel_tol * scale
    return (flipped | collapsed) & ~was_degenerate


def separation_violations(
    lattice: PillarLattice,
    original: PillarLattice,
    *,
    min_fraction: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Pillars that have come too close to a lattice neighbour.

    Returns ``(welded, too_close)``, both ``(NX+1, NY+1)`` bool.

    ``welded`` marks separations at or below :data:`WELD_TOLERANCE`, where node
    deduplication merges the pillars and the mesh topology silently changes.

    ``too_close`` marks separations below ``min_fraction`` of the ORIGINAL spacing.
    That is a quality bound, not a correctness one: transmissibility scales as
    ``A/d``, so a cell squeezed to a sliver wrecks conditioning long before the
    geometry becomes formally invalid.
    """
    welded = np.zeros(lattice.shape, dtype=bool)
    too_close = np.zeros(lattice.shape, dtype=bool)

    for sl_a, sl_b in (
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
    ):
        for plane_now, plane_ref in ((lattice.top, original.top), (lattice.bot, original.bot)):
            d_now = np.linalg.norm(plane_now[sl_b] - plane_now[sl_a], axis=-1)
            d_ref = np.linalg.norm(plane_ref[sl_b] - plane_ref[sl_a], axis=-1)
            w = d_now <= WELD_TOLERANCE
            t = (d_now < min_fraction * d_ref) & ~w
            for mask, sink in ((w, welded), (t, too_close)):
                sink[sl_a] |= mask
                sink[sl_b] |= mask
    return welded, too_close


def _point_segment_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from the ORIGIN to each segment ``a -> b`` (both ``(..., 2)``)."""
    ab = b - a
    denom = np.sum(ab * ab, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(denom > 0, -np.sum(a * ab, axis=-1) / np.where(denom > 0, denom, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[..., None] * ab
    return np.linalg.norm(closest, axis=-1)


def depth_crossings(lattice: PillarLattice, *, tol: float = WELD_TOLERANCE) -> np.ndarray:
    """``(NX+1, NY+1)`` bool — pillars that cross a lattice neighbour *in depth*.

    Two pillars that stay apart in plan view can still intersect somewhere between
    their top and bottom once they tilt independently — the failure mode stage 2
    introduces and a single-plane signed-area test cannot see.

    Both pillars are straight segments, so their horizontal separation is linear in
    depth: ``s(z)`` runs from ``s_top`` to ``s_bot``. They meet iff that 2D segment
    passes through the origin, so the test is the distance from the origin to the
    segment ``[s_top, s_bot]`` — no root-finding, fully vectorised.

    Only lattice-adjacent pairs are examined: those are the ones sharing a cell,
    and therefore the ones whose crossing breaks geometry.
    """
    crossed = np.zeros(lattice.shape, dtype=bool)
    for sl_a, sl_b in (
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
    ):
        s_top = lattice.top[sl_b] - lattice.top[sl_a]
        s_bot = lattice.bot[sl_b] - lattice.bot[sl_a]
        hit = _point_segment_distance(s_top, s_bot) <= tol
        crossed[sl_a] |= hit
        crossed[sl_b] |= hit
    return crossed


def proximity_violations(
    lattice: PillarLattice,
    original: PillarLattice,
    *,
    min_fraction: float = 0.10,
) -> np.ndarray:
    """``(NX+1, NY+1)`` bool — pillars too close to ANY other pillar, not just neighbours.

    :func:`separation_violations` only compares lattice-ADJACENT pairs, which misses
    the shear mode: a band of cells can be flattened so that pillars two rows apart
    nearly coincide while every adjacent pair stays comfortably separated. Measured
    on the 12x12 example, adjacent pairs held at 18 m (of an original 100 m) while
    pillars offset by (1,2) closed to 3.1 m — invisible to the adjacent-only test.

    Uses a KD-tree, so this stays cheap at field scale (Norne: 5,311 pillars).
    """
    from scipy.spatial import cKDTree

    shape = lattice.shape
    bad = np.zeros(shape, dtype=bool)
    h = local_spacing(original)
    radius = float(min_fraction * np.mean(h))
    if radius <= 0:
        return bad

    for plane in (lattice.top, lattice.bot):
        flat = plane.reshape(-1, 2)
        pairs = cKDTree(flat).query_pairs(radius, output_type="ndarray")
        if pairs.size == 0:
            continue
        ia, ja = np.divmod(pairs[:, 0], shape[1])
        ib, jb = np.divmod(pairs[:, 1], shape[1])
        # adjacent pairs are separation_violations' business; report the rest
        far = (np.abs(ia - ib) + np.abs(ja - jb)) > 1
        bad[ia[far], ja[far]] = True
        bad[ib[far], jb[far]] = True
    return bad


def check_validity(
    candidate: PillarLattice,
    original: PillarLattice,
    *,
    min_fraction: float = 0.10,
    min_scaled_jacobian: float | None = None,
    active: np.ndarray | None = None,
) -> ValidityReport:
    """Full validity check of ``candidate`` against the ``original`` it came from.

    Checks the signed-area sign at BOTH the top and bottom planes (independent once
    pillars are slanted), plus weld and separation guards.

    ``active`` is an ``(NX, NY)`` bool mask of columns that hold at least one live
    cell. Without it every check runs over the whole lattice, **including columns
    that contain no cells at all** — and a fold in a column with no cells is not a
    defect, it is arithmetic on absent geometry. Measured on Norne, where 56% of
    columns are dead: the deck SHIPS with 23 inverted columns, 18 of them dead, and
    a morph that flips one more is rejected. The bisection answers the rejection by
    raising the smoothness weight until the morph stops moving — on the `m_east`
    fault it settles at ``w`` above 100 and removes 20% of the misfit, where a
    weight that ignores dead ground removes far more with zero folds in live cells.
    Roughly 4x the fidelity was being surrendered to cells that hold no fluid.

    Pass :func:`cpg2unstructured.pillars.active_columns(grid)`. ``morph_pillars``
    does this for you by default — see its ``validity_scope``.

    ``min_scaled_jacobian`` closes a real hole, and is OFF by default because
    switching it on changes what "valid" means for decks that pass today. Every
    check above tests whether corners CROSS. A cell can be flattened to near-zero
    area without any corner crossing: measured on the benchmark's `zigzag` case, the
    engine returned ``ok=True, n_folded=0`` for a mesh containing a cell at scaled
    Jacobian 5.2e-04, with 14 cells below 0.3. Degenerate is not inverted, so
    nothing here saw it. Pass a threshold — 0.05 is a conservative floor, 0.3 the
    usual reject level — to make that an invalidity rather than a surprise
    downstream.
    """
    # Signed-area sign flip catches whole-cell inversion; corner_folds catches the
    # single-corner fold an area sum can hide. Both are needed.
    folded_top = _folded(
        column_signed_areas(original.top), column_signed_areas(candidate.top)
    ) | corner_folds(candidate.top, original.top)
    folded_bot = _folded(
        column_signed_areas(original.bot), column_signed_areas(candidate.bot)
    ) | corner_folds(candidate.bot, original.bot)
    welded, too_close = separation_violations(
        candidate, original, min_fraction=min_fraction
    )
    live_pillar = None if active is None else _pillars_of(active)
    # Depth crossings: only reachable once top and bottom move independently, and
    # invisible to either plane's signed area on its own.
    crossed = depth_crossings(candidate)
    welded = welded | crossed
    # Non-adjacent proximity — the shear mode the adjacent-only test cannot see.
    too_close = too_close | proximity_violations(
        candidate, original, min_fraction=min_fraction
    )
    if active is not None:
        folded_top = folded_top & active
        folded_bot = folded_bot & active
        welded = welded & live_pillar
        too_close = too_close & live_pillar

    n_folded = int(folded_top.sum() + folded_bot.sum())
    n_welded = int(welded.sum())
    n_too_close = int(too_close.sum())

    degenerate = None
    n_degenerate = 0
    if min_scaled_jacobian is not None:
        from cpg2unstructured.quality import scaled_jacobian

        sj = np.minimum(scaled_jacobian(candidate.top), scaled_jacobian(candidate.bot))
        degenerate = sj < float(min_scaled_jacobian)
        if active is not None:
            degenerate = degenerate & active
        n_degenerate = int(degenerate.sum())

    return ValidityReport(
        ok=(n_folded == 0 and n_welded == 0 and n_too_close == 0
            and n_degenerate == 0),
        folded_top=folded_top,
        folded_bot=folded_bot,
        welded=welded,
        too_close=too_close,
        n_folded=n_folded,
        n_welded=n_welded,
        n_too_close=n_too_close,
        degenerate=degenerate,
        n_degenerate=n_degenerate,
    )


def _pillars_of(active: np.ndarray) -> np.ndarray:
    """``(NX+1, NY+1)`` bool — pillars touching at least one active column.

    A pillar belongs to up to four columns, ``(i-1..i, j-1..j)``. A pillar all of
    whose columns are dead carries no live geometry, so a weld or a separation
    violation there is not a defect either.
    """
    nx, ny = active.shape
    out = np.zeros((nx + 1, ny + 1), dtype=bool)
    for di in (0, 1):
        for dj in (0, 1):
            out[di:di + nx, dj:dj + ny] |= active
    return out


def characteristic_length(lattice: PillarLattice) -> np.ndarray:
    """``(NX+1, NY+1)`` — the per-pillar length scale used to nondimensionalise.

    Thin alias over :func:`cpg2unstructured.pillars.local_spacing`, present so
    validity and the solver agree on what "one cell" means.
    """
    return local_spacing(lattice)
