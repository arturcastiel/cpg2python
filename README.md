# cpg2unstructured

Lightweight Python: read Eclipse CPG (corner-point geometry) `.grdecl`
files → fully unstructured mesh → adjacency graph for shortest-path
algorithms. Bidirectional `(i,j,k) ↔ cell-id` map preserved.

**Status:** alpha — under active development. Matching-Z faults only
in v1 (general fault geometry deferred to v2).

> Private use first; not yet published to PyPI.

## Install (local / editable)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .          # numpy-only
pip install -e .[fast]    # + scipy.sparse for fast Dijkstra
pip install -e .[dev]     # + pytest + build
```

## Quickstart

_(See `examples/` once PR-13 lands.)_

## License
MIT
