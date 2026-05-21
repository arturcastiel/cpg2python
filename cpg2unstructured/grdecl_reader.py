"""Eclipse GRDECL reader — DIMENS/SPECGRID + COORD + ZCORN + ACTNUM.

PR-3 scope only. INCLUDE + property pass-through land in PR-4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from cpg2unstructured._errors import (
    GrdeclConsistencyError,
    GrdeclSyntaxError,
)

__all__ = ["read_grdecl", "Grdecl"]


@dataclass(frozen=True)
class Grdecl:
    """Parsed GRDECL deck (PR-3 subset)."""

    cart_dims: tuple[int, int, int]              # (NX, NY, NZ)
    coord: np.ndarray                            # (n_pillars, 6)  float64
    zcorn: np.ndarray                            # (8*NX*NY*NZ,)   float64
    actnum: np.ndarray                           # (NX*NY*NZ,)     int8
    props: dict[str, np.ndarray] = field(default_factory=dict)
    unrecognized: list[str] = field(default_factory=list)

    @property
    def n_cells(self) -> int:
        nx, ny, nz = self.cart_dims
        return nx * ny * nz

    @property
    def n_pillars(self) -> int:
        nx, ny, _ = self.cart_dims
        return (nx + 1) * (ny + 1)


def _tokenize(lines: Iterable[str]) -> Iterator[str]:
    """Strip `--` comments, split on whitespace, expand `N*VALUE` runs."""
    for line in lines:
        i = line.find("--")
        if i >= 0:
            line = line[:i]
        for tok in line.split():
            if "*" in tok and not tok.startswith("*"):
                count_s, _, val = tok.partition("*")
                if not count_s.isdigit():
                    raise GrdeclSyntaxError(f"bad run-length token: {tok!r}")
                yield from [val] * int(count_s)
            else:
                yield tok


def _read_block(tokens: Iterator[str]) -> list[str]:
    """Read tokens up to (and consuming) the next `/` terminator."""
    block: list[str] = []
    for tok in tokens:
        if tok == "/":
            return block
        block.append(tok)
    raise GrdeclSyntaxError("unexpected EOF (missing `/` terminator)")


def _kw_dimens(toks: list[str]) -> tuple[int, int, int]:
    if len(toks) < 3:
        raise GrdeclSyntaxError(f"DIMENS needs 3 ints, got {toks!r}")
    return int(toks[0]), int(toks[1]), int(toks[2])


def _kw_specgrid(toks: list[str]) -> tuple[int, int, int]:
    if len(toks) < 3:
        raise GrdeclSyntaxError(f"SPECGRID needs >=3 ints, got {toks!r}")
    return int(toks[0]), int(toks[1]), int(toks[2])


def _kw_floats(toks: list[str]) -> np.ndarray:
    return np.fromiter((float(t) for t in toks), dtype=np.float64, count=len(toks))


def _kw_ints(toks: list[str]) -> np.ndarray:
    return np.fromiter((int(t) for t in toks), dtype=np.int8, count=len(toks))


def read_grdecl(path: str | Path) -> Grdecl:
    """Parse a GRDECL deck (PR-3 keyword subset)."""
    path = Path(path)
    with path.open("r", encoding="ascii", errors="replace") as fh:
        tokens = _tokenize(iter(fh))

        cart_dims_specgrid: tuple[int, int, int] | None = None
        cart_dims_dimens:   tuple[int, int, int] | None = None
        coord:  np.ndarray | None = None
        zcorn:  np.ndarray | None = None
        actnum: np.ndarray | None = None
        unrecognized: list[str] = []

        for tok in tokens:
            kw = tok.upper()
            if kw == "DIMENS":
                cart_dims_dimens = _kw_dimens(_read_block(tokens))
            elif kw == "SPECGRID":
                cart_dims_specgrid = _kw_specgrid(_read_block(tokens))
            elif kw == "COORD":
                coord = _kw_floats(_read_block(tokens))
            elif kw == "ZCORN":
                zcorn = _kw_floats(_read_block(tokens))
            elif kw == "ACTNUM":
                actnum = _kw_ints(_read_block(tokens))
            elif kw.isalpha() and kw == kw.upper() and len(kw) <= 8:
                _read_block(tokens)
                if kw not in unrecognized:
                    unrecognized.append(kw)
            else:
                raise GrdeclSyntaxError(f"unexpected token outside keyword: {tok!r}")

    if cart_dims_dimens is None and cart_dims_specgrid is None:
        raise GrdeclConsistencyError("neither DIMENS nor SPECGRID found")
    if (cart_dims_dimens is not None and cart_dims_specgrid is not None
            and cart_dims_dimens != cart_dims_specgrid):
        raise GrdeclConsistencyError(
            f"DIMENS={cart_dims_dimens} != SPECGRID={cart_dims_specgrid}"
        )
    cart_dims = cart_dims_dimens or cart_dims_specgrid
    assert cart_dims is not None
    nx, ny, nz = cart_dims

    if coord is None:
        raise GrdeclConsistencyError("COORD missing")
    if zcorn is None:
        raise GrdeclConsistencyError("ZCORN missing")

    n_pillars = (nx + 1) * (ny + 1)
    if coord.size != 6 * n_pillars:
        raise GrdeclConsistencyError(
            f"COORD has {coord.size} floats, expected {6 * n_pillars} for "
            f"NX={nx}, NY={ny}"
        )
    coord = coord.reshape(n_pillars, 6)

    if zcorn.size != 8 * nx * ny * nz:
        raise GrdeclConsistencyError(
            f"ZCORN has {zcorn.size} floats, expected {8 * nx * ny * nz}"
        )

    if actnum is None:
        actnum = np.ones(nx * ny * nz, dtype=np.int8)
    elif actnum.size != nx * ny * nz:
        raise GrdeclConsistencyError(
            f"ACTNUM has {actnum.size} ints, expected {nx * ny * nz}"
        )

    return Grdecl(
        cart_dims=cart_dims,
        coord=coord,
        zcorn=zcorn,
        actnum=actnum,
        props={},
        unrecognized=unrecognized,
    )
