# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- Initial repo skeleton (PR-1).
- Toy 2×2×2 cartesian CPG fixture + reference topology pipeline
  (`cpg2unstructured._toy`) with integer-invariant smoke tests (PR-2).
- GRDECL reader (`read_grdecl`, `Grdecl`) supporting DIMENS / SPECGRID /
  COORD / ZCORN / ACTNUM with `--` comments and `N*VALUE` run-length
  expansion. Unknown keywords are collected on `Grdecl.unrecognized`
  rather than raised. Domain exceptions in `cpg2unstructured._errors`
  (PR-3).
- GRDECL reader follows `INCLUDE 'file.inc' /` directives recursively,
  resolving filenames relative to the file containing the directive.
  Cycle detection raises `GrdeclIncludeError`. Property keywords
  (PORO, PERMX/Y/Z, NTG, MULTX/Y/Z, PRESSURE, SWAT, SGAS, SOIL,
  SATNUM, PVTNUM, EQLNUM, FIPNUM) populate `Grdecl.props` as 1-D arrays
  of length `NX·NY·NZ`. Integer properties use `int32` (sufficient for
  region IDs > 127). Duplicate property keywords warn via
  `GrdeclDuplicatePropertyWarning` with last-write-wins semantics
  (PR-4).
