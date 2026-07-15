# Final Coordinate Topology Repair Design

## Goal

All real Los Angeles and New York target polygons must survive atomic tokenization as non-empty valid Polygons. No real target may be silently removed because rotation reintroduced floating-point self-intersection.

## Root Cause

Token rings are closed in local coordinates and repaired before being rotated back to large UTM coordinates. The affine rotation can reintroduce a microscopic self-intersection at the closure location. The existing preparation gate then rejects the target even though a final-coordinate `buffer(0)` repairs every one of the 2,033 observed failures.

## Design

- Keep the existing 720 direction bins, 0.2 metre length token, source frame, and token sequences.
- Build and rotate the decoded polygon exactly as today.
- If the final-coordinate polygon is already a valid non-empty Polygon, return it unchanged.
- Otherwise run the existing polygon cleaner in final coordinates and require a valid non-empty Polygon.
- Raise an explicit error if final-coordinate repair cannot produce a Polygon; never silently accept invalid geometry.
- Add a tokenization algorithm version to scale manifests so existing datasets are rebuilt.
- Relearn both BPE vocabularies from the rebuilt filtered training sets.
- Require zero pairing failures, zero preprocessing failures, and zero BPE audit failures for Los Angeles and New York.

## Data Impact

Existing valid target token sequences remain unchanged. The previously excluded 2,033 real targets become eligible for their deterministic train, validation, or New York test split. Geometry metrics use the conditionally repaired final-coordinate polygon. Formal model training starts from scratch with the rebuilt vocabularies.

## Verification

- Regression test reproduces a rotation-induced invalid polygon and proves final decode is valid.
- Existing valid decoding remains geometrically unchanged.
- Full preparation reports zero preprocessing failures for all four city/scale combinations.
- Atomic/BPE row counts align and BPE roundtrip, invalid token, and alignment failures are all zero.
