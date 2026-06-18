"""Streaming short-circuit test (T3).

Asserts ``read_egrid`` stops reading after ENDGRID — any keyword
material AFTER ENDGRID (LGR sections, NNC blocks) must NOT be loaded.

The check is twofold:

1. **Behavioural**: ``read_egrid`` on the padded fixture produces a
   ``CornerPointGrid`` byte-for-byte identical to the same fixture
   without the trailing dummy block.
2. **Mechanical**: we wrap ``resfo.lazy_read`` and assert the reader
   stopped iterating before reaching the dummy keywords.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import resfo

from cpg2unstructured import egrid_reader, read_egrid

DATA = Path(__file__).resolve().parent / "data"


def test_padded_file_produces_same_grid_as_standard():
    std = read_egrid(DATA / "cart_2x2x2.EGRID")
    padded = read_egrid(DATA / "streaming_padded.EGRID")
    assert std.cart_dims == padded.cart_dims
    np.testing.assert_array_equal(std.coord, padded.coord)
    np.testing.assert_array_equal(std.zcorn, padded.zcorn)
    np.testing.assert_array_equal(std.actnum, padded.actnum)


def test_short_circuit_stops_iteration_before_dummy_block(monkeypatch):
    """Monkey-patch ``resfo.lazy_read`` to count how many entries the
    reader actually pulled from the iterator. The padded fixture has the
    order
        FILEHEAD, GRIDHEAD, COORD, ZCORN, ACTNUM, ENDGRID, LGRSGONE, BIGDUMMY
    so the reader must consume at most 6 entries (it can stop earlier if
    the WANT set completes before ENDGRID; here MAPAXES is absent so the
    short-circuit fires on ENDGRID).
    """
    seen: list[str] = []

    real_lazy_read = resfo.lazy_read

    def counting_lazy_read(*args, **kwargs):
        for entry in real_lazy_read(*args, **kwargs):
            seen.append(entry.read_keyword().strip())

            class _Wrap:
                def __init__(self, e, kw):
                    self._e = e
                    self._kw = kw

                def read_keyword(self):
                    return self._kw.ljust(8)[:8]

                def read_array(self):
                    return self._e.read_array()

            yield _Wrap(entry, seen[-1])

    monkeypatch.setattr(egrid_reader.resfo, "lazy_read", counting_lazy_read)
    read_egrid(DATA / "streaming_padded.EGRID")

    assert "ENDGRID" in seen
    assert "LGRSGONE" not in seen, (
        "Reader iterated past ENDGRID into the dummy block — short-circuit broken."
    )
    assert "BIGDUMMY" not in seen
