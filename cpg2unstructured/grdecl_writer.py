"""Write CPG grid keywords to a GRDECL text file.

The inverse of :func:`~cpg2unstructured.grdecl_reader.read_grdecl`, deliberately
scoped to the GRID keywords the reader parses: ``SPECGRID``, ``COORD``, ``ZCORN``
and ``ACTNUM``.

**This never rewrites a deck.** ``read_grdecl`` records unconsumed keywords by NAME
only (``state.unrecognized``), never their content, so a read-modify-write over a
real deck would silently drop ``EQUALS``, ``COPY``, ``MULTFLT``, EDIT-section
keywords, regions and schedule data. The supported workflow is therefore to write
the grid keywords to their OWN file and ``INCLUDE`` it from the master deck:

    INCLUDE
      'adapted_grid.grdecl' /

so the original is never touched and nothing can be lost.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cpg2unstructured.corner_point import CornerPointGrid

__all__ = ["write_grid_include", "MapaxesNotInvertedError", "format_keyword"]


class MapaxesNotInvertedError(ValueError):
    """Raised when writing a grid whose COORD has been rotated into the map frame.

    ``read_egrid(..., apply_mapaxes=True)`` (the default) rotates ``coord`` into the
    MAP frame and sets ``mapaxes_applied``. Writing those coordinates back beside the
    original ``MAPAXES`` record would make the next reader rotate them a SECOND time.
    Refusing is the honest option: the caller either supplies raw-frame coordinates
    (``apply_mapaxes=False``) or opts in explicitly via ``allow_mapaxes_applied``.
    """


def _rle(tokens: list[str]) -> list[str]:
    """Collapse runs of equal tokens into Eclipse ``N*VALUE`` form.

    A run of one is emitted bare — ``1*x`` is legal but noisy, and decks in the wild
    do not use it.
    """
    if not tokens:
        return []
    out: list[str] = []
    run_val = tokens[0]
    run_len = 1
    for tok in tokens[1:]:
        if tok == run_val:
            run_len += 1
            continue
        out.append(run_val if run_len == 1 else f"{run_len}*{run_val}")
        run_val, run_len = tok, 1
    out.append(run_val if run_len == 1 else f"{run_len}*{run_val}")
    return out


def _fmt_float(x: float) -> str:
    """Shortest round-trippable text for a float, without exponent clutter.

    ``repr`` already gives the shortest string that round-trips through ``float()``;
    trailing ``.0`` is kept because Eclipse readers accept it and it keeps COORD/ZCORN
    visually column-aligned.
    """
    return repr(float(x))


def format_keyword(name: str, values, *, per_line: int = 8, integer: bool = False) -> str:
    """Render one keyword block: name, run-length-encoded values, ``/`` terminator."""
    arr = np.asarray(values).ravel()
    if integer:
        toks = [str(int(v)) for v in arr]
    else:
        toks = [_fmt_float(v) for v in arr]
    toks = _rle(toks)
    lines = [name]
    for i in range(0, len(toks), per_line):
        lines.append("  " + " ".join(toks[i : i + per_line]))
    lines.append("/")
    return "\n".join(lines) + "\n"


def write_grid_include(
    grid: CornerPointGrid,
    path: str | Path,
    *,
    include_actnum: bool = True,
    allow_mapaxes_applied: bool = False,
    header: str | None = None,
) -> Path:
    """Write ``SPECGRID`` / ``COORD`` / ``ZCORN`` / ``ACTNUM`` to ``path``.

    Parameters
    ----------
    grid
        The grid to write. Only the four GRID keywords are emitted; ``props`` and
        ``unrecognized`` are deliberately NOT written — see the module docstring.
    path
        Destination file. Written atomically via a sibling temp file.
    include_actnum
        Write ``ACTNUM`` (default). Skip it when the master deck supplies its own.
    allow_mapaxes_applied
        Opt in to writing map-frame coordinates. Default ``False`` raises
        :class:`MapaxesNotInvertedError` rather than emit a deck that double-rotates.
    header
        Optional comment text placed at the top; every line is prefixed with ``--``.

    Returns
    -------
    Path
        The path written.
    """
    p = Path(path)
    nx, ny, nz = (int(v) for v in grid.cart_dims)

    if grid.mapaxes is not None and grid.mapaxes_applied and not allow_mapaxes_applied:
        raise MapaxesNotInvertedError(
            "grid.coord is in the MAP frame (mapaxes_applied=True). Writing it beside "
            "the original MAPAXES record would rotate it twice on the next read. Re-read "
            "with apply_mapaxes=False, or pass allow_mapaxes_applied=True if the target "
            "deck carries no MAPAXES."
        )

    coord = np.asarray(grid.coord, dtype=np.float64).reshape(-1, 6)
    n_pillars = (nx + 1) * (ny + 1)
    if coord.shape[0] != n_pillars:
        raise ValueError(
            f"COORD has {coord.shape[0]} pillars, expected {n_pillars} for {nx}x{ny}"
        )
    zcorn = np.asarray(grid.zcorn, dtype=np.float64).ravel()
    if zcorn.size != 8 * nx * ny * nz:
        raise ValueError(
            f"ZCORN has {zcorn.size} values, expected {8 * nx * ny * nz}"
        )

    chunks: list[str] = []
    if header:
        chunks.append("".join(f"-- {line}\n" for line in header.splitlines()))
    chunks.append(
        "-- Written by cpg2unstructured.write_grid_include.\n"
        "-- GRID keywords only — INCLUDE this from the master deck; the deck itself\n"
        "-- is never rewritten, so no unparsed keyword can be lost.\n\n"
    )
    chunks.append(f"SPECGRID\n  {nx} {ny} {nz} 1 F /\n\n")
    chunks.append(format_keyword("COORD", coord, per_line=6))
    chunks.append("\n")
    chunks.append(format_keyword("ZCORN", zcorn, per_line=8))

    if include_actnum and grid.actnum is not None:
        actnum = np.asarray(grid.actnum).ravel()
        if actnum.size != nx * ny * nz:
            raise ValueError(
                f"ACTNUM has {actnum.size} values, expected {nx * ny * nz}"
            )
        chunks.append("\n")
        chunks.append(format_keyword("ACTNUM", actnum, per_line=20, integer=True))

    text = "".join(chunks)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)
    return p
