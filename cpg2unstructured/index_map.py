"""Active-cell id ↔ (i, j, k) translation with a lazy reverse map.

 scope: wrap an earlier `original_ijk` in a small bi-directional class.
"""
from __future__ import annotations

import numpy as np

__all__ = ["IndexMap"]


class IndexMap:
    """Bi-directional active-cell id ↔ (i, j, k) map.

    Parameters
    ----------
    cart_dims
        (NX, NY, NZ).
    original_ijk
        (Ncells_active, 3) int64. Eclipse-natural (i, j, k) per cell —
        produced by `build_topology`.

    Notes
    -----
    Forward lookup (``cell_ijk``) is a simple array index. Reverse lookup
    (``active_id``) builds a dense ``(NX, NY, NZ)`` int64 array on first
    call and caches it. Inactive cells return ``None`` rather
    than raise.
    """

    __slots__ = ("_cart_dims", "_original_ijk", "_linear_to_cell", "_reverse")

    def __init__(self, cart_dims: tuple[int, int, int],
                 original_ijk: np.ndarray) -> None:
        nx, ny, nz = cart_dims
        if original_ijk.ndim != 2 or original_ijk.shape[1] != 3:
            raise ValueError(
                f"original_ijk must be (n_active, 3); got {original_ijk.shape}"
            )
        self._cart_dims = (int(nx), int(ny), int(nz))
        self._original_ijk = original_ijk.astype(np.int64, copy=False)
        i, j, k = original_ijk.T
        self._linear_to_cell = (i + j * nx + k * nx * ny).astype(np.int64)
        self._reverse: np.ndarray | None = None

    # ── forward ──────────────────────────────────────────────────────

    def cell_ijk(self, active_id: int) -> tuple[int, int, int]:
        """Return ``(i, j, k)`` for an active cell id."""
        i, j, k = self._original_ijk[active_id]
        return int(i), int(j), int(k)

    @property
    def linear_to_cell(self) -> np.ndarray:
        """``(N_active,)`` int64 — Eclipse linear index ``i + j*NX + k*NX*NY``."""
        return self._linear_to_cell

    @property
    def n_active(self) -> int:
        return int(self._original_ijk.shape[0])

    @property
    def cart_dims(self) -> tuple[int, int, int]:
        return self._cart_dims

    # ── reverse (lazy) ───────────────────────────────────────────────

    def active_id(self, i: int, j: int, k: int) -> int | None:
        """Return active-cell id for ``(i, j, k)``; ``None`` if inactive.

        Raises ``IndexError`` if ``(i, j, k)`` is outside ``cart_dims``.
        """
        nx, ny, nz = self._cart_dims
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            raise IndexError(
                f"({i},{j},{k}) outside cart_dims {self._cart_dims}"
            )
        rev = self._build_reverse()
        a = int(rev[i, j, k])
        return a if a != -1 else None

    def _build_reverse(self) -> np.ndarray:
        if self._reverse is None:
            nx, ny, nz = self._cart_dims
            rev = np.full((nx, ny, nz), -1, dtype=np.int64)
            i, j, k = self._original_ijk.T
            rev[i, j, k] = np.arange(self._original_ijk.shape[0])
            self._reverse = rev
        return self._reverse

    # ── batch helpers ────────────────────────────────────────────────

    def cells_in_layer(self, k: int) -> np.ndarray:
        """All active-cell ids whose Eclipse ``k`` equals the argument."""
        mask = self._original_ijk[:, 2] == k
        return np.where(mask)[0]

    # ── misc ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"IndexMap(cart_dims={self._cart_dims}, "
                f"n_active={self.n_active})")
