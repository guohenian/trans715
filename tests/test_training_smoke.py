import json
from pathlib import Path

from building_simplify.bpe import BPEVocabulary
from building_simplify.train import evaluate_full_greedy_from_config, train_from_config


def test_one_step_training_checkpoint_and_prediction_output(tmp_path: Path):
    dataset = tmp_path / "scale5000" / "train" / "bpe.jsonl"
    test_dataset = tmp_path / "scale5000" / "test" / "bpe.jsonl"
    vocab_path = dataset.parent / "bpe_vocab.txt"
    dataset.parent.mkdir(parents=True)
    test_dataset.parent.mkdir(parents=True)
    rows = [
        {"sample_id": "a", "osm_id": "a", "scale": 5000, "difficulty": "complex", "source_vertex_count": 5,
         "source_tokens": [725, 723, 0, 0, 722], "target_tokens": [723, 0, 0, 722], "source_frame": {}},
        {"sample_id": "b", "osm_id": "b", "scale": 5000, "difficulty": "complex", "source_vertex_count": 6,
         "source_tokens": [725, 723, 180, 180, 722], "target_tokens": [723, 180, 180, 722], "source_frame": {}},
    ]
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    dataset.write_text(payload, encoding="utf-8")
    test_dataset.write_text(payload, encoding="utf-8")
    BPEVocabulary(token_atoms={}).save(vocab_path)
    output_dir = tmp_path / "run"

    checkpoint = train_from_config(
        dataset, output_dir, vocab_path=vocab_path, test_dataset_path=test_dataset,
        epochs=1, batch_size=1, eval_batch_size=1, device="cpu", max_steps=1,
        d_model=16, nhead=4, num_layers=1, dim_feedforward=32, dropout=0.0,
        weight_decay=0.0, scheduled_sampling_max=0.0, max_source_len=32, max_target_len=32,
        show_progress=False,
    )
    predictions = tmp_path / "predictions.jsonl"
    metrics = evaluate_full_greedy_from_config(
        test_dataset, checkpoint, vocab_path=vocab_path, batch_size=1, device="cpu",
        show_progress=False, prediction_output=predictions, max_target_len=8,
    )

    assert checkpoint.exists()
    assert predictions.exists()
    assert metrics["sample_count"] == 2
