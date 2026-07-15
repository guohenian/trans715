from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Sequence

import torch
import warnings
warnings.filterwarnings("ignore", message=".*nested tensor.*")
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .bpe import BPEVocabulary
from .config import (
    BPE_VOCAB_10000_PATH,
    BPE_VOCAB_5000_PATH,
    DEFAULT_CONFIG,
    DEFAULT_MODEL_D_MODEL,
    DEFAULT_MODEL_DIM_FEEDFORWARD,
    DEFAULT_MODEL_DROPOUT,
    DEFAULT_MODEL_NHEAD,
    DEFAULT_MODEL_NUM_LAYERS,
    TOKEN_BOS,
    TOKEN_EOS,
    TOKEN_PAD,
)
from .model import BuildingTransformer, _decode_next_token, _encode_source, generate_greedy_batch


DEFAULT_TRAIN_EPOCHS = 20
DEFAULT_TRAIN_BATCH_SIZE = 32
DEFAULT_TRAIN_MAX_SOURCE_LEN = 2000
DEFAULT_TRAIN_MAX_TARGET_LEN = 2000
DEFAULT_TRAIN_SAVE_EVERY_STEPS = 0
DEFAULT_EOS_LOSS_WEIGHT = 2.0
DEFAULT_LENGTH_LOSS_WEIGHT = 0.05
DEFAULT_METRIC_PROGRESS_EVERY = 0
DEFAULT_NUM_WORKERS = 0
DEFAULT_PROGRESS = True
DEFAULT_SEED = 20260713


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TokenJsonlDataset(Dataset):
    def __init__(
        self,
        path: Path | str,
        max_source_len: int | None = DEFAULT_CONFIG.max_source_len,
        max_target_len: int | None = DEFAULT_CONFIG.max_target_len,
    ) -> None:
        self.path = Path(path)
        self.offsets: list[int] = []
        self.skipped_too_long = 0
        self._handle = None
        with self.path.open("r", encoding="utf-8") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    row = json.loads(line)
                    source_len = len(row["source_tokens"])
                    target_len = len(row["target_tokens"])
                    if max_source_len is not None and source_len > max_source_len:
                        self.skipped_too_long += 1
                        continue
                    if max_target_len is not None and target_len > max_target_len:
                        self.skipped_too_long += 1
                        continue
                    self.offsets.append(offset)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def _reader(self):
        if self._handle is None or self._handle.closed:
            self._handle = self.path.open("r", encoding="utf-8")
        return self._handle

    def __getitem__(self, index: int) -> dict:
        handle = self._reader()
        handle.seek(self.offsets[index])
        return json.loads(handle.readline())


def _pad(sequences: list[list[int]]) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    padded = torch.full((len(sequences), max_len), TOKEN_PAD, dtype=torch.long)
    for row_index, seq in enumerate(sequences):
        padded[row_index, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return padded


def collate_batch(batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = _pad([row["source_tokens"] for row in batch])
    targets = [row["target_tokens"] for row in batch]
    decoder_input = _pad([tokens[:-1] for tokens in targets])
    labels = _pad([tokens[1:] for tokens in targets])
    return source, decoder_input, labels


def strip_pad_tokens(tokens: Sequence[int], pad_token_id: int = TOKEN_PAD) -> list[int]:
    stripped = [int(token) for token in tokens]
    while stripped and stripped[-1] == pad_token_id:
        stripped.pop()
    return stripped


def tokens_equal_ignoring_pad(
    prediction: Sequence[int],
    reference: Sequence[int],
    pad_token_id: int = TOKEN_PAD,
) -> bool:
    return strip_pad_tokens(prediction, pad_token_id) == strip_pad_tokens(reference, pad_token_id)


def normalize_labels_for_metric(labels: torch.Tensor, pad_token_id: int = TOKEN_PAD) -> torch.Tensor:
    return torch.where(labels.eq(-100), torch.full_like(labels, pad_token_id), labels)


def exact_match_rate_from_token_batches(
    predictions: Sequence[Sequence[int]],
    references: Sequence[Sequence[int]],
) -> dict[str, float | int]:
    if len(predictions) != len(references):
        raise ValueError(f"Prediction/reference count mismatch: {len(predictions)} != {len(references)}")
    exact = 0
    for prediction, reference in zip(predictions, references):
        if tokens_equal_ignoring_pad(prediction, reference):
            exact += 1
    sample_count = len(references)
    return {
        "greedy_exact_match_rate": exact / sample_count if sample_count else 0.0,
        "exact_match_count": exact,
        "sample_count": sample_count,
    }



def _target_lengths(labels: torch.Tensor) -> torch.Tensor:
    return labels.ne(TOKEN_PAD).sum(dim=1)


def _count_early_stop_greedy_exact_batch(
    model: BuildingTransformer,
    source: torch.Tensor,
    labels: torch.Tensor,
    max_len: int,
) -> int:
    batch_size = labels.size(0)
    if batch_size == 0:
        return 0

    active_indices = torch.arange(batch_size, dtype=torch.long, device=source.device)
    exact = torch.zeros(batch_size, dtype=torch.bool, device=source.device)
    target_lengths = _target_lengths(labels)
    active_indices = active_indices[target_lengths.gt(0)]
    if active_indices.numel() == 0:
        return 0

    memory, mem_pad = _encode_source(model, source)
    generated = torch.full((active_indices.numel(), 1), TOKEN_BOS, dtype=torch.long, device=source.device)

    max_decode_steps = min(labels.size(1), max(0, max_len - 1))
    for step in range(max_decode_steps):
        if active_indices.numel() == 0:
            break
        active_memory = memory.index_select(0, active_indices)
        active_mem_pad = mem_pad.index_select(0, active_indices)
        next_token = _decode_next_token(model, generated, active_memory, active_mem_pad)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        expected = labels.index_select(0, active_indices)[:, step]
        active_target_lengths = target_lengths.index_select(0, active_indices)
        matched = next_token.eq(expected)
        ended = expected.eq(TOKEN_EOS)
        completed = matched & ended & active_target_lengths.eq(step + 1)
        if bool(completed.any().item()):
            exact[active_indices[completed]] = True
        keep = matched & ~ended & active_target_lengths.gt(step + 1)
        active_indices = active_indices[keep]
        generated = generated[keep]

    return int(exact.sum().item())


@torch.inference_mode()
def compute_exact_match_rate(
    model: BuildingTransformer,
    loader: DataLoader,
    device: str,
    max_len: int,
    metric_progress_every: int = DEFAULT_METRIC_PROGRESS_EVERY,
    split_name: str = "eval",
    show_progress: bool = DEFAULT_PROGRESS,
) -> dict[str, float | int]:
    model.eval()
    exact = 0
    sample_count = 0
    iterator = _progress_iter(loader, total=len(loader), desc=f"{split_name} greedy exact", enabled=show_progress)
    for batch_index, (source, _decoder_input, labels) in enumerate(iterator, start=1):
        source = _to_device(source, device)
        labels = _to_device(normalize_labels_for_metric(labels), device)
        exact += _count_early_stop_greedy_exact_batch(model, source, labels, max_len=max_len)
        sample_count += int(labels.size(0))
        if metric_progress_every and batch_index % metric_progress_every == 0:
            _log(f"[{split_name}] greedy exact batches={batch_index} samples={sample_count}")
    return {
        "greedy_exact_match_rate": exact / sample_count if sample_count else 0.0,
        "exact_match_count": exact,
        "sample_count": sample_count,
    }


@torch.inference_mode()
def compute_teacher_forced_token_accuracy(
    model: BuildingTransformer,
    loader: DataLoader,
    device: str,
    metric_progress_every: int = DEFAULT_METRIC_PROGRESS_EVERY,
    split_name: str = "eval",
    show_progress: bool = DEFAULT_PROGRESS,
) -> dict[str, float | int]:
    model.eval()
    correct_count = 0
    token_count = 0
    iterator = _progress_iter(loader, total=len(loader), desc=f"{split_name} token acc", enabled=show_progress)
    for batch_index, (source, decoder_input, labels) in enumerate(iterator, start=1):
        source = _to_device(source, device)
        decoder_input = _to_device(decoder_input, device)
        labels = _to_device(normalize_labels_for_metric(labels), device)
        predictions = model(source, decoder_input).argmax(dim=-1)
        mask = labels.ne(TOKEN_PAD)
        correct_count += int((predictions.eq(labels) & mask).sum().item())
        token_count += int(mask.sum().item())
        if metric_progress_every and batch_index % metric_progress_every == 0:
            _log(f"[{split_name}] teacher-forced token accuracy batches={batch_index} tokens={token_count}")
    return {
        "teacher_forced_token_accuracy": correct_count / token_count if token_count else 0.0,
        "teacher_forced_token_correct_count": correct_count,
        "teacher_forced_token_count": token_count,
    }


def _append_metric(metrics_path: Path, payload: dict) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _default_test_dataset_path(dataset_path: Path) -> Path:
    text = str(dataset_path).replace("\\", "/")
    if "/train/" not in text:
        raise ValueError(f"Cannot infer test dataset from non-train path: {dataset_path}")
    return Path(text.replace("/train/", "/test/", 1))


def _should_save_step_checkpoint(save_every_steps: int | None, global_step: int) -> bool:
    return save_every_steps is not None and save_every_steps > 0 and global_step % save_every_steps == 0


def _training_limit_reached(global_step: int, max_steps: int | None) -> bool:
    return max_steps is not None and global_step >= max_steps


def _average_completed_loss(total_loss: float, completed_steps: int) -> float:
    return total_loss / max(1, completed_steps)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _resolve_device(device: str | None) -> str:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(resolved_device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested, but this Python environment cannot access CUDA")
    return resolved_device


def _dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_batch,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def _to_device(tensor: torch.Tensor, device: str) -> torch.Tensor:
    return tensor.to(device, non_blocking=str(device).startswith("cuda"))


def _progress_iter(iterable, *, total: int, desc: str, enabled: bool):
    if not enabled:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit="batch",
        ascii=True,
        dynamic_ncols=True,
        leave=False,
        file=sys.stderr,
    )


def train_from_config(
    dataset_path: Path,
    output_dir: Path,
    vocab_path: Path | None = None,
    test_dataset_path: Path | None = None,
    epochs: int = DEFAULT_TRAIN_EPOCHS,
    batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    eval_batch_size: int | None = None,
    lr: float = 1e-4,
    device: str | None = None,
    max_source_len: int | None = DEFAULT_TRAIN_MAX_SOURCE_LEN,
    max_target_len: int | None = DEFAULT_TRAIN_MAX_TARGET_LEN,
    save_every_steps: int | None = DEFAULT_TRAIN_SAVE_EVERY_STEPS,
    resume_from: Path | None = None,
    eos_loss_weight: float = DEFAULT_EOS_LOSS_WEIGHT,
    length_loss_weight: float = DEFAULT_LENGTH_LOSS_WEIGHT,
    d_model: int = DEFAULT_MODEL_D_MODEL,
    nhead: int = DEFAULT_MODEL_NHEAD,
    num_layers: int = DEFAULT_MODEL_NUM_LAYERS,
    dim_feedforward: int = DEFAULT_MODEL_DIM_FEEDFORWARD,
    dropout: float = DEFAULT_MODEL_DROPOUT,
    pre_ln: bool = False,
    weight_decay: float = 0.01,
    scheduled_sampling_max: float = 0.25,
    max_steps: int | None = None,
    seed: int = DEFAULT_SEED,
    metric_progress_every: int = DEFAULT_METRIC_PROGRESS_EVERY,
    num_workers: int = DEFAULT_NUM_WORKERS,
    show_progress: bool = DEFAULT_PROGRESS,
    use_amp: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)
    resolved_device = _resolve_device(device)

    dataset = TokenJsonlDataset(
        dataset_path,
        max_source_len=max_source_len,
        max_target_len=max_target_len,
    )
    if len(dataset) == 0:
        raise ValueError(
            f"Dataset is empty after length filtering: {dataset_path} "
            f"(max_source_len={max_source_len}, max_target_len={max_target_len})"
        )
    resolved_eval_batch_size = eval_batch_size or batch_size
    pin_memory = str(resolved_device).startswith("cuda")
    loader_generator = torch.Generator().manual_seed(seed)
    loader = _dataloader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, generator=loader_generator)
    metric_loader = _dataloader(
        dataset,
        batch_size=resolved_eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    resolved_test_dataset = test_dataset_path or _default_test_dataset_path(dataset_path)
    test_dataset: TokenJsonlDataset | None = None
    test_loader: DataLoader | None = None
    if resolved_test_dataset.exists():
        candidate_test_dataset = TokenJsonlDataset(
            resolved_test_dataset,
            max_source_len=max_source_len,
            max_target_len=max_target_len,
        )
        if len(candidate_test_dataset) > 0:
            test_dataset = candidate_test_dataset
            test_loader = _dataloader(
                test_dataset,
                batch_size=resolved_eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            _log(f"Test dataset is empty after filtering, skipping per-epoch metrics: {resolved_test_dataset}")
    else:
        _log(f"Test dataset not found, skipping per-epoch metrics: {resolved_test_dataset}")

    bpe_vocab = BPEVocabulary.load(vocab_path or _default_vocab_path(dataset_path))

    model_config = {
        "vocab_size": bpe_vocab.vocab_size,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "pre_ln": pre_ln,
    }
    model = BuildingTransformer(**model_config).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_epochs = 3
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    class_weights = torch.ones(bpe_vocab.vocab_size, dtype=torch.float, device=resolved_device)
    class_weights[TOKEN_EOS] = eos_loss_weight
    loss_fn = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD, weight=class_weights)
    length_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    global_step = 0
    start_epoch = 0
    metrics_path = output_dir / "metrics.jsonl"

    if resume_from is not None:
        payload = torch.load(resume_from, map_location=resolved_device)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", 0))
        global_step = int(payload.get("global_step", 0))
    if _training_limit_reached(global_step, max_steps):
        return resume_from or (output_dir / f"checkpoint_epoch_{start_epoch}.pt")

    def save_checkpoint(path: Path, epoch: int, avg_loss: float) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": model_config,
                "epoch": epoch,
                "loss": avg_loss,
                "dataset_path": str(dataset_path),
                "max_source_len": max_source_len,
                "max_target_len": max_target_len,
                "skipped_too_long": dataset.skipped_too_long,
                "global_step": global_step,
                "eos_loss_weight": eos_loss_weight,
                "length_loss_weight": length_loss_weight,
                "seed": seed,
            },
            path,
        )

    _log(
        f"Training: epochs={epochs} batch_size={batch_size} eval_batch_size={resolved_eval_batch_size} "
        f"num_workers={num_workers} len={len(dataset)} steps/epoch~{len(dataset)//batch_size} device={resolved_device}"
    )
    checkpoint_path = output_dir / "checkpoint_epoch_0.pt"
    for epoch in range(start_epoch, epochs):
        if _training_limit_reached(global_step, max_steps):
            break
        scheduled_p = min(scheduled_sampling_max, (epoch / max(1, epochs - 1)) * scheduled_sampling_max)
        model.train()
        total_loss = 0.0
        completed_steps = 0
        _log(f"[Epoch {epoch+1}/{epochs}] starting...")
        train_iterator = _progress_iter(
            loader,
            total=len(loader),
            desc=f"epoch {epoch+1}/{epochs} train",
            enabled=show_progress,
        )
        for step_index, (source, decoder_input, labels) in enumerate(train_iterator, start=1):
            source = _to_device(source, resolved_device)
            decoder_input = _to_device(decoder_input, resolved_device)
            labels = _to_device(labels, resolved_device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(source, decoder_input)
                if scheduled_p > 0:
                    with torch.no_grad():
                        preds = logits.argmax(dim=-1)
                        pred_shifted = torch.cat([decoder_input[:, :1], preds[:, :-1]], dim=1)
                    mask = torch.rand_like(decoder_input.float()) < scheduled_p
                    mask[:, 0] = False
                    mask = mask & decoder_input.ne(TOKEN_PAD)
                    if mask.any():
                        mixed = torch.where(mask, pred_shifted, decoder_input)
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            logits = model(source, mixed)
                token_loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                if length_loss_weight > 0:
                    valid_mask = labels.ne(TOKEN_PAD)
                    eos_targets = labels.eq(TOKEN_EOS).float()
                    eos_logits = logits[:, :, TOKEN_EOS]
                    eos_loss = length_loss_fn(eos_logits, eos_targets)
                    eos_loss = (eos_loss * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1.0)
                    loss = token_loss + (length_loss_weight * eos_loss)
                else:
                    loss = token_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            completed_steps += 1
            global_step += 1
            if show_progress and hasattr(train_iterator, "set_postfix"):
                train_iterator.set_postfix(loss=f"{total_loss / step_index:.4f}")

            if _should_save_step_checkpoint(save_every_steps, global_step):
                save_checkpoint(
                    output_dir / f"checkpoint_step_{global_step}.pt",
                    epoch + (step_index / max(1, len(loader))),
                    total_loss / step_index,
                )
            if max_steps is not None and global_step >= max_steps:
                break

        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        avg_loss = _average_completed_loss(total_loss, completed_steps)
        save_checkpoint(checkpoint_path, epoch + 1, avg_loss)
        train_exact = compute_exact_match_rate(
            model,
            metric_loader,
            resolved_device,
            max_len=max_target_len or DEFAULT_CONFIG.max_target_len,
            metric_progress_every=metric_progress_every,
            split_name="train",
            show_progress=show_progress,
        )
        train_token = compute_teacher_forced_token_accuracy(
            model,
            metric_loader,
            resolved_device,
            metric_progress_every=metric_progress_every,
            split_name="train",
            show_progress=show_progress,
        )
        _log(
            f"[Epoch {epoch+1}/{epochs}] loss={avg_loss:.4f} "
            f"greedy_exact={train_exact['greedy_exact_match_rate']:.4f} "
            f"teacher_forced_token_acc={train_token['teacher_forced_token_accuracy']:.4f}"
        )
        metric_payload = {
            "split": "train",
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "greedy_exact_match_rate": train_exact["greedy_exact_match_rate"],
            "exact_match_count": train_exact["exact_match_count"],
            "sample_count": train_exact["sample_count"],
            "teacher_forced_token_accuracy": train_token["teacher_forced_token_accuracy"],
            "teacher_forced_token_correct_count": train_token["teacher_forced_token_correct_count"],
            "teacher_forced_token_count": train_token["teacher_forced_token_count"],
            "skipped_too_long": dataset.skipped_too_long,
            "global_step": global_step,
        }
        _append_metric(metrics_path, metric_payload)
        print(json.dumps(metric_payload, ensure_ascii=False), flush=True)

        if test_dataset is not None and test_loader is not None:
            test_exact = compute_exact_match_rate(
                model,
                test_loader,
                resolved_device,
                max_len=max_target_len or DEFAULT_CONFIG.max_target_len,
                metric_progress_every=metric_progress_every,
                split_name="test",
                show_progress=show_progress,
            )
            test_token = compute_teacher_forced_token_accuracy(
                model,
                test_loader,
                resolved_device,
                metric_progress_every=metric_progress_every,
                split_name="test",
                show_progress=show_progress,
            )
            test_payload = {
                "split": "test",
                "epoch": epoch + 1,
                "test_dataset_path": str(resolved_test_dataset),
                "greedy_exact_match_rate": test_exact["greedy_exact_match_rate"],
                "exact_match_count": test_exact["exact_match_count"],
                "sample_count": test_exact["sample_count"],
                "teacher_forced_token_accuracy": test_token["teacher_forced_token_accuracy"],
                "teacher_forced_token_correct_count": test_token["teacher_forced_token_correct_count"],
                "teacher_forced_token_count": test_token["teacher_forced_token_count"],
                "skipped_too_long": test_dataset.skipped_too_long,
                "global_step": global_step,
            }
            _append_metric(metrics_path, test_payload)
            print(json.dumps(test_payload, ensure_ascii=False), flush=True)
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        if max_steps is not None and global_step >= max_steps:
            break

    return checkpoint_path


def _default_vocab_path(dataset_path: Path) -> Path:
    text = str(dataset_path).replace("\\", "/")
    if "scale5000" in text:
        return BPE_VOCAB_5000_PATH
    if "scale10000" in text:
        return BPE_VOCAB_10000_PATH
    raise ValueError(f"Cannot infer vocab path from dataset: {dataset_path}")


def evaluate_full_greedy_from_config(
    dataset_path: Path, checkpoint_path: Path, output_dir: Path | None = None,
    vocab_path: Path | None = None, batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    device: str | None = None, show_progress: bool = DEFAULT_PROGRESS,
    prediction_output: Path | None = None,
    max_target_len: int | None = None,
    **_
) -> dict[str, float | int | str]:
    import tempfile
    tmp = prediction_output or Path(tempfile.mktemp(suffix=".jsonl"))
    n = _full_greedy_save_predictions(
        dataset_path=dataset_path, checkpoint_path=checkpoint_path, output_path=tmp,
        vocab_path=vocab_path, batch_size=batch_size, device=device, show_progress=show_progress,
        max_target_len=max_target_len,
    )
    if n == 0:
        return {"error": "no samples"}
    # 从保存的预测计算指标
    import json as _json
    exact = 0; tok_correct = 0; tok_total = 0
    with open(tmp, encoding="utf-8") as f:
        for line in f:
            r = _json.loads(line)
            pred = r["pred_tokens"]; tgt = r["target_tokens"]
            if pred == tgt: exact += 1
            for i in range(min(len(pred), len(tgt))):
                if pred[i] == tgt[i]: tok_correct += 1
            tok_total += max(len(pred), len(tgt))
    if prediction_output is None:
        tmp.unlink()
    result = {"split":"eval","dataset_path":str(dataset_path),"checkpoint_path":str(checkpoint_path),
              "greedy_exact_match_rate":exact/n,"exact_match_count":exact,"sample_count":n,
              "greedy_token_accuracy":tok_correct/tok_total if tok_total else 0}
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir/"full_greedy_metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, default=None)
    parser.add_argument("--test-dataset", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-source-len", type=int, default=DEFAULT_TRAIN_MAX_SOURCE_LEN)
    parser.add_argument("--max-target-len", type=int, default=DEFAULT_TRAIN_MAX_TARGET_LEN)
    parser.add_argument("--save-every-steps", type=int, default=DEFAULT_TRAIN_SAVE_EVERY_STEPS)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--eval-only-checkpoint", type=Path, default=None)
    parser.add_argument("--prediction-output", type=Path, default=None)
    parser.add_argument("--auto-eval", action="store_true", help="训练后自动跑 full greedy 评估+推理")
    parser.add_argument("--eval-raw-shp", type=Path, default=None, help="推理用 raw shapefile")
    parser.add_argument("--eval-scale", type=int, choices=[5000, 10000], default=5000)
    parser.add_argument("--auto-eval-batch-size", type=int, default=16, help="auto_eval full greedy 的 batch size")
    parser.add_argument("--eval-infer-limit", type=int, default=None)
    parser.add_argument("--metric-progress-every", type=int, default=DEFAULT_METRIC_PROGRESS_EVERY)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--no-progress", dest="show_progress", action="store_false")
    parser.set_defaults(show_progress=DEFAULT_PROGRESS)
    parser.add_argument("--eos-loss-weight", type=float, default=DEFAULT_EOS_LOSS_WEIGHT)
    parser.add_argument("--length-loss-weight", type=float, default=DEFAULT_LENGTH_LOSS_WEIGHT)
    parser.add_argument("--d-model", type=int, default=DEFAULT_MODEL_D_MODEL)
    parser.add_argument("--nhead", type=int, default=DEFAULT_MODEL_NHEAD)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_MODEL_NUM_LAYERS)
    parser.add_argument("--dim-feedforward", type=int, default=DEFAULT_MODEL_DIM_FEEDFORWARD)
    parser.add_argument("--dropout", type=float, default=DEFAULT_MODEL_DROPOUT)
    parser.add_argument("--pre-ln", action="store_true")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false", help="Disable automatic mixed precision")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--scheduled-sampling-max", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    if args.eval_only_checkpoint is not None:
        metrics = evaluate_full_greedy_from_config(
            dataset_path=args.dataset,
            checkpoint_path=args.eval_only_checkpoint,
            output_dir=args.output_dir,
            vocab_path=args.vocab_path,
            batch_size=args.eval_batch_size or args.batch_size,
            device=args.device,
            max_source_len=args.max_source_len,
            max_target_len=args.max_target_len,
            metric_progress_every=args.metric_progress_every,
            num_workers=args.num_workers,
            show_progress=args.show_progress,
            prediction_output=args.prediction_output,
        )
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        return

    checkpoint = train_from_config(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        device=args.device,
        vocab_path=args.vocab_path,
        test_dataset_path=args.test_dataset,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        save_every_steps=args.save_every_steps,
        resume_from=args.resume_from,
        eos_loss_weight=args.eos_loss_weight,
        length_loss_weight=args.length_loss_weight,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pre_ln=args.pre_ln,
        weight_decay=args.weight_decay,
        scheduled_sampling_max=args.scheduled_sampling_max,
        max_steps=args.max_steps,
        seed=args.seed,
        metric_progress_every=args.metric_progress_every,
        num_workers=args.num_workers,
        show_progress=args.show_progress,
        use_amp=args.use_amp,
    )
    print(checkpoint)

    # -- 训练后自动评估 --
    if args.auto_eval:
        _auto_eval(args, checkpoint)


def _full_greedy_save_predictions(
    dataset_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    vocab_path: Path | None = None,
    batch_size: int = 32,
    device: str | None = None,
    max_target_len: int | None = None,
    show_progress: bool = True,
) -> int:
    """Full greedy 解码并保存预测 token 序列到 JSONL。

    每条: {raw_fid, osm_id, scale, source_frame, pred_tokens, target_tokens}
    后续配合 bpe_vocab.txt 即可在 CPU 上转 shp, 无需 GPU。
    """
    import torch
    resolved_device = _resolve_device(device)
    dataset = TokenJsonlDataset(dataset_path, max_source_len=None, max_target_len=None)
    if len(dataset) == 0:
        raise ValueError(f"Dataset empty: {dataset_path}")
    pin_memory = str(resolved_device).startswith("cuda")
    loader = _dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    bpe_vocab = BPEVocabulary.load(vocab_path or _default_vocab_path(dataset_path))
    payload = torch.load(checkpoint_path, map_location=resolved_device)
    model_config = dict(payload.get("model_config") or {})
    model_config.setdefault("vocab_size", bpe_vocab.vocab_size)
    model = BuildingTransformer(**model_config).to(resolved_device)
    model.load_state_dict(payload["model"])
    model.eval()
    _log(f"Full greedy save: loaded {checkpoint_path}")

    max_len = max_target_len or DEFAULT_CONFIG.max_target_len
    total = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iterator = _progress_iter(loader, total=len(loader), desc="full greedy save", enabled=show_progress)
    with open(output_path, "w", encoding="utf-8") as fout:
        for source, _, labels in iterator:
            source = _to_device(source, resolved_device)
            labels = _to_device(normalize_labels_for_metric(labels), resolved_device)
            generated = generate_greedy_batch(model, source, max_len=max_len)[:, 1:]
            # 读取原始 JSONL 行获取 frame 等元数据
            for i in range(source.size(0)):
                idx = total + i
                if idx >= len(dataset):
                    break
                row = dataset[idx]
                pred = strip_pad_tokens(generated[i].cpu().tolist())
                tgt = strip_pad_tokens(labels[i].cpu().tolist())
                record = {
                    "raw_fid": row.get("raw_fid", idx),
                    "osm_id": row.get("osm_id", ""),
                    "scale": row.get("scale", 0),
                    "source_frame": row.get("source_frame", {}),
                    "difficulty": row.get("difficulty", "unknown"),
                    "source_vertex_count": row.get("source_vertex_count", 0),
                    "pred_tokens": pred,
                    "target_tokens": tgt,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += source.size(0)
    _log(f"Full greedy save: {total} samples → {output_path}")
    return total


def _auto_eval(args, _checkpoint_path):
    """训练结束后找最佳 checkpoint 跑 full greedy 评估 + 推理。"""
    import glob as _glob
    metrics_path = args.output_dir / "metrics.jsonl"
    if not metrics_path.exists():
        _log("auto-eval: no metrics.jsonl, skipping")
        return

    # 找 greedy_exact 最高且 split="test" 的 epoch（没有则用 train）
    best_epoch = None
    best_score = -1.0
    split_type = "test"
    with open(metrics_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            m = json.loads(line)
            if m.get("split") == "test" and m.get("greedy_exact_match_rate", -1) > best_score:
                best_score = m["greedy_exact_match_rate"]
                best_epoch = m["epoch"]
    if best_epoch is None:
        with open(metrics_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                m = json.loads(line)
                if m.get("split") == "train" and m.get("greedy_exact_match_rate", -1) > best_score:
                    best_score = m["greedy_exact_match_rate"]
                    best_epoch = m["epoch"]
        split_type = "train"

    if best_epoch is None:
        _log("auto-eval: no metrics found")
        return

    best_ckpt = args.output_dir / f"checkpoint_epoch_{best_epoch}.pt"
    if not best_ckpt.exists():
        _log(f"auto-eval: checkpoint {best_ckpt} not found")
        return

    _log(f"auto-eval: best epoch={best_epoch} ({split_type} greedy_exact={best_score:.4f}) using {best_ckpt}")

    eval_dir = args.output_dir / "auto_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 用最优 checkpoint 跑 full greedy, 保存预测序列
    _log(f"auto-eval: full greedy save predictions (best, epoch={best_ckpt.stem}) ...")
    for dataset_path, label in [
        (args.test_dataset or args.dataset, "test"),
        (args.dataset, "train"),
    ]:
        if dataset_path is None or not Path(str(dataset_path)).exists():
            continue
        out_file = eval_dir / f"preds_{label}.jsonl"
        _full_greedy_save_predictions(
            dataset_path=dataset_path,
            checkpoint_path=best_ckpt,
            output_path=out_file,
            vocab_path=args.vocab_path,
            batch_size=args.auto_eval_batch_size,
            device=args.device,
            max_target_len=512,
            show_progress=args.show_progress,
        )
        _log(f"auto-eval: {label} → {out_file}")
        # 计算 full greedy 指标
        t_exact = tok_corr = tok_total = t_total = 0
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                t_total += 1
                if r["pred_tokens"] == r["target_tokens"]: t_exact += 1
                for i in range(min(len(r["pred_tokens"]), len(r["target_tokens"]))):
                    if r["pred_tokens"][i] == r["target_tokens"][i]: tok_corr += 1
                tok_total += max(len(r["pred_tokens"]), len(r["target_tokens"]))
        _log(f"auto-eval: {label} greedy_exact={t_exact/t_total:.4f} greedy_token_acc={tok_corr/tok_total:.4f}")

    # 用保存的预测序列直接转 shp (CPU, 无 GPU)
    from .infer import predictions_jsonl_to_shapefile
    for label in ["test", "train"]:
        preds_file = eval_dir / f"preds_{label}.jsonl"
        if not preds_file.exists():
            _log(f"auto-eval: skip shp {label} — {preds_file} not found")
            continue
        _log(f"auto-eval: converting {label} predictions to shp ...")
        shp_file = eval_dir / f"pred_{label}.shp"
        n = predictions_jsonl_to_shapefile(preds_file, args.vocab_path or _default_vocab_path(args.dataset), shp_file)
        _log(f"auto-eval: {label} {n} features → {shp_file}")

    _log("auto-eval done")
if __name__ == "__main__":
    main()
