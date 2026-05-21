"""Domain-specific exceptions for the GRDECL parsing + meshing stack."""
from __future__ import annotations

__all__ = [
    "Cpg2UError",
    "GrdeclError",
    "GrdeclSyntaxError",
    "GrdeclConsistencyError",
    "GrdeclIncludeError",
    "GrdeclDuplicatePropertyWarning",
    "NonMatchingZError",
]


class Cpg2UError(Exception):
    """Root for all package errors."""


class GrdeclError(Cpg2UError):
    """Anything that goes wrong reading a .grdecl file."""


class GrdeclSyntaxError(GrdeclError):
    """Tokenizer-level grammar violation (missing `/`, broken `N*V`, etc.)."""


class GrdeclConsistencyError(GrdeclError):
    """Logically inconsistent file (DIMENS != SPECGRID; ZCORN length wrong; ...)."""


class GrdeclIncludeError(GrdeclError):
    """INCLUDE-graph problem (cycle, missing file, bad filename)."""


class GrdeclDuplicatePropertyWarning(UserWarning):
    """Same property keyword declared more than once; last-write-wins."""


class NonMatchingZError(Cpg2UError):
    """Pillar Z values don't match across cells — fault geometry not v1."""
