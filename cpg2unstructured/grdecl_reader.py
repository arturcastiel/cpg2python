"""Eclipse GRDECL reader.

 scope: DIMENS / SPECGRID + COORD + ZCORN + ACTNUM.
 scope: INCLUDE recursion + property pass-through (PORO / PERMX / ... ).
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from cpg2unstructured._errors import (
    GrdeclConsistencyError,
    GrdeclDuplicatePropertyWarning,
    GrdeclIncludeError,
    GrdeclSyntaxError,
)
from cpg2unstructured.corner_point import CornerPointGrid, Grdecl

__all__ = ["read_grdecl", "Grdecl"]


_PROP_KEYWORDS_FLOAT = frozenset({
    "PORO", "PERMX", "PERMY", "PERMZ", "NTG",
    "MULTX", "MULTY", "MULTZ",
    "PRESSURE", "SWAT", "SGAS", "SOIL",
})

_PROP_KEYWORDS_INT = frozenset({
    "SATNUM", "PVTNUM", "EQLNUM", "FIPNUM",
})


# `Grdecl` (the dataclass) now lives in cpg2unstructured.corner_point as
# `CornerPointGrid`. The name `Grdecl` is re-exported above as a TypeAlias.
# Kept the import at top of file so all internal references resolve.


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


def _kw_ints(toks: list[str], dtype=np.int8) -> np.ndarray:
    # int8 default keeps ACTNUM (0/1) cheap; SATNUM-class callers pass int32.
    return np.fromiter((int(t) for t in toks), dtype=dtype, count=len(toks))


def _strip_quotes(s: str) -> str:
    if len(s) >= 2 and s[0] in {"'", '"'} and s[-1] == s[0]:
        return s[1:-1]
    return s


class _State:
    """Mutable accumulator shared across recursive _parse_stream calls."""
    __slots__ = (
        "cart_dims_dimens", "cart_dims_specgrid",
        "coord", "zcorn", "actnum",
        "props_float", "props_int",
        "unrecognized",
    )

    def __init__(self) -> None:
        self.cart_dims_dimens: tuple[int, int, int] | None = None
        self.cart_dims_specgrid: tuple[int, int, int] | None = None
        self.coord: np.ndarray | None = None
        self.zcorn: np.ndarray | None = None
        self.actnum: np.ndarray | None = None
        self.props_float: dict[str, np.ndarray] = {}
        self.props_int: dict[str, np.ndarray] = {}
        self.unrecognized: list[str] = []


def _parse_stream(
    path: Path,
    stack: frozenset[Path],
    state: _State,
) -> None:
    """Parse one GRDECL/INC file in-place into `state`; recurse on INCLUDE."""
    with path.open("r", encoding="ascii", errors="replace") as fh:
        tokens = _tokenize(iter(fh))
        for tok in tokens:
            kw = tok.upper()
            if kw == "DIMENS":
                state.cart_dims_dimens = _kw_dimens(_read_block(tokens))
            elif kw == "SPECGRID":
                state.cart_dims_specgrid = _kw_specgrid(_read_block(tokens))
            elif kw == "COORD":
                state.coord = _kw_floats(_read_block(tokens))
            elif kw == "ZCORN":
                state.zcorn = _kw_floats(_read_block(tokens))
            elif kw == "ACTNUM":
                state.actnum = _kw_ints(_read_block(tokens), dtype=np.int8)
            elif kw == "INCLUDE":
                block = _read_block(tokens)
                if not block:
                    raise GrdeclSyntaxError("INCLUDE empty")
                first = block[0]
                # Detect a quote-split filename: the tokenizer splits on
                # whitespace globally, so `'has space.inc'` arrives as two
                # tokens `'has` and `space.inc'`. An open quote that doesn't
                # close on the same token means the filename had embedded
                # whitespace inside the quotes.
                if first[:1] in {"'", '"'} and first[-1:] != first[:1]:
                    raise GrdeclSyntaxError(
                        f"INCLUDE filename contains whitespace inside quotes: {first!r}..."
                    )
                raw = _strip_quotes(first.strip())
                if any(ch.isspace() for ch in raw):
                    raise GrdeclSyntaxError(
                        f"INCLUDE filename contains whitespace: {raw!r}"
                    )
                child = (path.parent / raw).resolve()
                if child in stack:
                    raise GrdeclIncludeError(
                        f"INCLUDE cycle detected: {child}"
                    )
                _parse_stream(child, stack | {child}, state)
            elif kw in _PROP_KEYWORDS_FLOAT:
                arr = _kw_floats(_read_block(tokens))
                if kw in state.props_float or kw in state.props_int:
                    warnings.warn(
                        f"Duplicate property keyword {kw!r}; last-write-wins.",
                        GrdeclDuplicatePropertyWarning,
                        stacklevel=2,
                    )
                state.props_float[kw] = arr
            elif kw in _PROP_KEYWORDS_INT:
                arr = _kw_ints(_read_block(tokens), dtype=np.int32)
                if kw in state.props_int or kw in state.props_float:
                    warnings.warn(
                        f"Duplicate property keyword {kw!r}; last-write-wins.",
                        GrdeclDuplicatePropertyWarning,
                        stacklevel=2,
                    )
                state.props_int[kw] = arr
            elif kw.isalpha() and kw == kw.upper() and len(kw) <= 8:
                _read_block(tokens)
                if kw not in state.unrecognized:
                    state.unrecognized.append(kw)
            else:
                raise GrdeclSyntaxError(
                    f"unexpected token outside keyword: {tok!r}"
                )


def _finalize(state: _State) -> Grdecl:
    """Validate accumulated state and construct the final `Grdecl`."""
    if state.cart_dims_dimens is None and state.cart_dims_specgrid is None:
        raise GrdeclConsistencyError("neither DIMENS nor SPECGRID found")
    if (state.cart_dims_dimens is not None
            and state.cart_dims_specgrid is not None
            and state.cart_dims_dimens != state.cart_dims_specgrid):
        raise GrdeclConsistencyError(
            f"DIMENS={state.cart_dims_dimens} != "
            f"SPECGRID={state.cart_dims_specgrid}"
        )
    cart_dims = state.cart_dims_dimens or state.cart_dims_specgrid
    assert cart_dims is not None
    nx, ny, nz = cart_dims
    n_cells = nx * ny * nz

    if state.coord is None:
        raise GrdeclConsistencyError("COORD missing")
    if state.zcorn is None:
        raise GrdeclConsistencyError("ZCORN missing")

    n_pillars = (nx + 1) * (ny + 1)
    if state.coord.size != 6 * n_pillars:
        raise GrdeclConsistencyError(
            f"COORD has {state.coord.size} floats, expected "
            f"{6 * n_pillars} for NX={nx}, NY={ny}"
        )
    coord = state.coord.reshape(n_pillars, 6)

    if state.zcorn.size != 8 * n_cells:
        raise GrdeclConsistencyError(
            f"ZCORN has {state.zcorn.size} floats, expected {8 * n_cells}"
        )

    actnum = state.actnum
    if actnum is None:
        actnum = np.ones(n_cells, dtype=np.int8)
    elif actnum.size != n_cells:
        raise GrdeclConsistencyError(
            f"ACTNUM has {actnum.size} ints, expected {n_cells}"
        )

    props: dict[str, np.ndarray] = {}
    for kw, arr in state.props_float.items():
        if arr.size != n_cells:
            raise GrdeclConsistencyError(
                f"{kw} has {arr.size} values, expected {n_cells}"
            )
        props[kw] = arr
    for kw, arr in state.props_int.items():
        if arr.size != n_cells:
            raise GrdeclConsistencyError(
                f"{kw} has {arr.size} values, expected {n_cells}"
            )
        props[kw] = arr

    return CornerPointGrid(
        cart_dims=cart_dims,
        coord=coord,
        zcorn=state.zcorn,
        actnum=actnum,
        props=props,
        unrecognized=state.unrecognized,
        source="grdecl",
    )


def read_grdecl(path: str | Path) -> Grdecl:
    """Parse a GRDECL deck. Follows `INCLUDE 'file.inc' /` directives.

    INCLUDE filenames are resolved relative to the file containing the
    directive. A re-include of an already-open file raises
    `GrdeclIncludeError`. Property keywords (PORO / PERMX / SATNUM / ...)
    populate `Grdecl.props`; unknown keywords still land in
    `Grdecl.unrecognized`.
    """
    root = Path(path).resolve()
    state = _State()
    _parse_stream(root, frozenset({root}), state)
    return _finalize(state)
