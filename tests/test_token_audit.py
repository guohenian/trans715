import json
from pathlib import Path

from building_simplify.bpe import BPEVocabulary
from building_simplify.evaluation import audit_bpe_jsonl


def test_bpe_audit_requires_lossless_roundtrip(tmp_path: Path):
    raw_path = tmp_path / "raw.jsonl"
    bpe_path = tmp_path / "bpe.jsonl"
    vocab_path = tmp_path / "vocab.txt"
    vocab = BPEVocabulary(token_atoms={1500: (1, 1)})
    vocab.save(vocab_path)
    raw = {"sample_id": "a", "source_tokens": [725, 723, 1, 1, 722], "target_tokens": [723, 1, 1, 722]}
    bpe = {"sample_id": "a", "source_tokens": [725, 723, 1500, 722], "target_tokens": [723, 1500, 722]}
    raw_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    bpe_path.write_text(json.dumps(bpe) + "\n", encoding="utf-8")

    report = audit_bpe_jsonl(raw_path, bpe_path, vocab_path)

    assert report["rows"] == 1
    assert report["roundtrip_failures"] == 0
    assert report["invalid_token_count"] == 0


def test_bpe_audit_rejects_unused_atomic_gap(tmp_path: Path):
    raw_path = tmp_path / "raw.jsonl"
    bpe_path = tmp_path / "bpe.jsonl"
    vocab_path = tmp_path / "vocab.txt"
    BPEVocabulary(token_atoms={}).save(vocab_path)
    row = {"sample_id": "a", "source_tokens": [725, 723, 800, 722], "target_tokens": [723, 722]}
    raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    bpe_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert audit_bpe_jsonl(raw_path, bpe_path, vocab_path)["invalid_token_count"] == 1
