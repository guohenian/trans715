# Final Coordinate Topology Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild both scale datasets with zero real-target tokenization failures.

**Architecture:** Repair decoded geometry only after rotation into final projected coordinates, and only when the rotated geometry is invalid. Version the tokenization pipeline so old manifests cannot be reused, then rebuild atomic data, vocabularies, BPE data, and audits.

**Tech Stack:** Python 3.13, Shapely, PyProj, PyShp, PyTorch, pytest.

## Global Constraints

- Do not change the 720 direction bins or 0.2 metre atomic length resolution.
- Do not modify or delete source Shapefiles or `output/`.
- Do not reuse old checkpoints or old BPE vocabularies.
- Real target preprocessing failures must be zero for both cities and scales.

### Task 1: Final-coordinate decode repair

**Files:** `building_simplify/geometry.py`, `tests/test_architecture.py`

- [ ] Add a failing regression test for a polygon that becomes invalid after rotation.
- [ ] Confirm the test fails with the current decoder.
- [ ] Rotate first, return valid Polygons unchanged, and conditionally clean invalid final-coordinate geometry.
- [ ] Require the repaired result to be a valid non-empty Polygon.
- [ ] Run geometry and architecture tests.

### Task 2: Dataset invalidation and hard gates

**Files:** `building_simplify/config.py`, `building_simplify/pipeline.py`, `tests/test_pipeline.py`

- [ ] Add a failing test that a changed tokenization algorithm version invalidates a manifest.
- [ ] Add `TOKENIZATION_VERSION` to scale manifest configuration.
- [ ] Gate both Los Angeles and New York preprocessing failures at zero.
- [ ] Run pipeline tests.

### Task 3: Workspace cleanup

**Files:** `.claude/`, `.pytest_cache/`, `.superpowers/`

- [ ] Verify all three resolved paths are within the workspace.
- [ ] Remove the stale lock, pytest cache, and old task-report directory.

### Task 4: Full data rebuild

**Files:** `datasets/scale5000/`, `datasets/scale10000/`

- [ ] Run `python -m building_simplify.pipeline prepare-all --workspace . --output-root datasets --seed 20260713 --bpe-training-rows 10000 --min-free-gb 0`.
- [ ] Confirm projected Shapefiles are reused and both scale manifests rebuild because the tokenization version changed.
- [ ] Confirm all four preparation failure files contain zero rows.
- [ ] Confirm BPE roundtrip, invalid token, and alignment failures are zero.

### Task 5: Final verification

**Files:** all modified code and generated manifests.

- [ ] Recompute both scale manifest hashes.
- [ ] Run `python -m pytest -q`.
- [ ] Run `python -m compileall -q building_simplify`.
- [ ] Run the one-step training smoke test.
- [ ] Report final row counts, rectangle retention, audit results, and remaining disk space.
