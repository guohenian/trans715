import json
from pathlib import Path

from building_simplify.bpe import BPEVocabulary, build_bpe_jsonl_from_raw_jsonl, select_bpe_training_subset, train_bpe_from_jsonl_corpus
from building_simplify.infer import predictions_jsonl_to_shapefile


def test_bpe_jsonl_io_lives_in_bpe_module(tmp_path: Path):
    raw = tmp_path / "atomic.jsonl"
    vocab_path = tmp_path / "vocab.txt"
    encoded = tmp_path / "bpe.jsonl"
    row = {"sample_id": "a", "scale": 5000, "source_tokens": [725, 723, 1, 1, 722], "target_tokens": [723, 1, 1, 722]}
    raw.write_text(json.dumps(row) + "\n", encoding="utf-8")

    train_bpe_from_jsonl_corpus(raw, vocab_path, vocab_size=1501, max_atomic_len=10, scale=5000)
    report = build_bpe_jsonl_from_raw_jsonl(raw, vocab_path, encoded)
    saved = json.loads(encoded.read_text(encoding="utf-8"))

    assert report.written == 1
    vocab = BPEVocabulary.load(vocab_path)
    assert vocab.decode(saved["source_tokens"]) == row["source_tokens"]


def test_prediction_shapefile_converter_is_exposed_by_infer_module():
    assert callable(predictions_jsonl_to_shapefile)


def test_bpe_training_subset_is_bounded_and_deterministic(tmp_path: Path):
    source = tmp_path / "train.jsonl"
    source.write_text("".join(json.dumps({"osm_id": str(i), "source_tokens": [i], "target_tokens": [i]}) + "\n" for i in range(20)), encoding="utf-8")
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    assert select_bpe_training_subset(source, first, count=5, seed=20260713) == 5
    assert select_bpe_training_subset(source, second, count=5, seed=20260713) == 5
    assert first.read_bytes() == second.read_bytes()


def test_bpe_jsonl_keeps_previous_complete_output_when_a_row_is_skipped(tmp_path: Path):
    training = tmp_path / "training.jsonl"
    vocab_path = tmp_path / "vocab.txt"
    valid = {"sample_id": "ok", "scale": 5000, "source_tokens": [725, 723, 1, 722], "target_tokens": [723, 1, 722]}
    training.write_text(json.dumps(valid) + "\n", encoding="utf-8")
    train_bpe_from_jsonl_corpus(training, vocab_path, vocab_size=1501, max_atomic_len=10, scale=5000)

    raw = tmp_path / "atomic.jsonl"
    invalid = {"sample_id": "bad", "scale": 5000, "source_tokens": [725, 723, 1, 722]}
    raw.write_text(json.dumps(valid) + "\n" + json.dumps(invalid) + "\n", encoding="utf-8")
    encoded = tmp_path / "bpe.jsonl"
    encoded.write_text("previous complete output\n", encoding="utf-8")

    report = build_bpe_jsonl_from_raw_jsonl(raw, vocab_path, encoded)

    assert report.written == 1
    assert report.skipped == 1
    assert encoded.read_text(encoding="utf-8") == "previous complete output\n"
    assert Path(report.failures_path).exists()
    assert not encoded.with_suffix(".jsonl.tmp").exists()
