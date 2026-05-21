-- Geometry sub-deck for the with_include PR-4 fixture
-- COORD + ZCORN for a 2x2x2 cartesian grid; no SPECGRID, no ACTNUM.

COORD
  0 0 0  0 0 2
  1 0 0  1 0 2
  2 0 0  2 0 2
  0 1 0  0 1 2
  1 1 0  1 1 2
  2 1 0  2 1 2
  0 2 0  0 2 2
  1 2 0  1 2 2
  2 2 0  2 2 2  /

ZCORN
  16*0   16*1
  16*1   16*2 /
