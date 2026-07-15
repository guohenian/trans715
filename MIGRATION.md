# Architecture migration

The public workflow is now `python -m building_simplify.pipeline`.

| Removed module | Replacement |
| --- | --- |
| `data.py` BPE JSONL functions | `bpe.py` |
| `data.py` prediction-to-shapefile function | `infer.py` |
| `data.py` ArcPy and hard-coded New York pipeline | `preparation.py` using `pyproj`; old implementation removed |
| `evaluate.py` | `evaluation.py` |
| `evaluate_predictions.py` | `evaluation.py` JSONL grouped evaluation |
| `detailed_evaluation.py` | renamed and expanded as `evaluation.py` |
| `token_audit.py` | `evaluation.audit_bpe_jsonl` |
| `unmatched_evaluation.py` | `evaluation.evaluate_unmatched_shapefile` |

Core modules retained without changing their ownership:

- `geometry.py`: atomic polygon token representation.
- `model.py`: Transformer and decoding.
- `train.py`: model training and full greedy prediction.
- `infer.py`: Shapefile inference and prediction export.
- `preparation.py`: projected dataset preparation internals.
- `pipeline.py`: the only public data/experiment workflow.
