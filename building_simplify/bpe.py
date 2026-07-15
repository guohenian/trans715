from __future__ import annotations

import heapq
import hashlib
import json
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Iterable, Sequence

from .config import BPE_MAX_ATOMIC_LEN, BPE_TOKEN_START, BPE_VOCAB_SIZE, SPECIAL_TOKENS


@dataclass(frozen=True)
class DatasetBuildReport:
    written: int
    skipped: int
    failures_path: str | None


@dataclass(frozen=True)
class BPETrainingReport:
    scale: int
    source_train_dir: str
    vocab_path: str
    vocab_size: int
    token_start: int
    max_atomic_len: int
    sequences: int


@dataclass(frozen=True)
class BPEVocabulary:
    start_id: int = BPE_TOKEN_START
    vocab_size: int = BPE_VOCAB_SIZE
    token_atoms: dict[int, tuple[int, ...]] | None = None
    _trie: dict[int, dict] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_atoms", dict(self.token_atoms or {}))
        self._validate()
        object.__setattr__(self, "_trie", self._build_trie())

    def _validate(self) -> None:
        for token_id, atoms in self.token_atoms.items():
            if token_id < self.start_id:
                raise ValueError(f"Composite token id must be >= start_id: {token_id}")
            if token_id >= self.vocab_size:
                raise ValueError(f"Composite token id exceeds vocab size: {token_id}")
            if not atoms:
                raise ValueError(f"Composite token {token_id} has no atoms")
            if len(atoms) == 1:
                raise ValueError(f"Composite token {token_id} must contain more than one atom")

    @property
    def composite_ids(self) -> list[int]:
        return sorted(self.token_atoms)

    def composite_lengths(self) -> dict[int, int]:
        return {token_id: len(atoms) for token_id, atoms in self.token_atoms.items()}

    def token_length(self, token_id: int) -> int:
        atoms = self.token_atoms.get(token_id)
        if atoms is None:
            return 1
        return len(atoms)

    def _build_trie(self) -> dict[int, dict]:
        root: dict[int, dict] = {}
        end_key = -1
        for token_id in sorted(self.token_atoms):
            node = root
            for atom in self.token_atoms[token_id]:
                node = node.setdefault(atom, {})
            node[end_key] = token_id
        return root

    def encode(self, sequence: Sequence[int]) -> list[int]:
        trie = self._trie
        end_key = -1
        encoded: list[int] = []
        i = 0
        while i < len(sequence):
            token = sequence[i]
            if token in SPECIAL_TOKENS:
                encoded.append(int(token))
                i += 1
                continue

            node = trie
            best_id = None
            best_end = None
            j = i
            while j < len(sequence):
                token_j = sequence[j]
                if token_j in SPECIAL_TOKENS:
                    break
                node = node.get(token_j)
                if node is None:
                    break
                j += 1
                if end_key in node:
                    best_id = node[end_key]
                    best_end = j
            if best_id is None or best_end is None:
                encoded.append(int(token))
                i += 1
            else:
                encoded.append(int(best_id))
                i = best_end
        return encoded

    def decode(self, sequence: Sequence[int]) -> list[int]:
        decoded: list[int] = []
        for token in sequence:
            if token >= self.start_id and token in self.token_atoms:
                decoded.extend(self.token_atoms[token])
            else:
                decoded.append(int(token))
        return decoded

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(
                f"# start_id={self.start_id} vocab_size={self.vocab_size} "
                f"max_atomic_len={BPE_MAX_ATOMIC_LEN}\n"
            )
            for token_id in sorted(self.token_atoms):
                atoms = " ".join(str(atom) for atom in self.token_atoms[token_id])
                handle.write(f"{token_id}\t{atoms}\n")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "BPEVocabulary":
        path = Path(path)
        start_id = BPE_TOKEN_START
        vocab_size = BPE_VOCAB_SIZE
        token_atoms: dict[int, tuple[int, ...]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    parts = dict(part.split("=", 1) for part in line[1:].strip().split() if "=" in part)
                    start_id = int(parts.get("start_id", start_id))
                    vocab_size = int(parts.get("vocab_size", vocab_size))
                    continue
                token_text, atoms_text = line.split("\t", 1)
                token_atoms[int(token_text)] = tuple(int(part) for part in atoms_text.split() if part)
        return cls(start_id=start_id, vocab_size=vocab_size, token_atoms=token_atoms)


def _is_weighted_sequence(item: object) -> bool:
    return (
        isinstance(item, tuple)
        and len(item) == 2
        and not isinstance(item[0], int)
        and isinstance(item[1], int)
    )


def _iter_weighted_sequences(
    corpus: Iterable[Sequence[int] | tuple[Sequence[int], int]],
) -> Iterable[tuple[Sequence[int], int]]:
    for item in corpus:
        if _is_weighted_sequence(item):
            sequence, weight = item
            yield sequence, int(weight)
        else:
            yield item, 1


def _iter_trainable_segments(sequence: Sequence[int]) -> Iterable[tuple[int, ...]]:
    segment: list[int] = []
    for token in sequence:
        token = int(token)
        if token in SPECIAL_TOKENS:
            if len(segment) >= 2:
                yield tuple(segment)
            segment = []
            continue
        segment.append(token)
    if len(segment) >= 2:
        yield tuple(segment)


def _normalize_segments(
    corpus: Iterable[Sequence[int] | tuple[Sequence[int], int]],
) -> list[tuple[tuple[int, ...], int]]:
    counts: Counter[tuple[int, ...]] = Counter()
    for sequence, weight in _iter_weighted_sequences(corpus):
        if weight <= 0:
            continue
        for segment in _iter_trainable_segments(sequence):
            counts[segment] += weight
    return list(counts.items())


def _pair_occurrences(sequence: Sequence[int]) -> Counter[tuple[int, int]]:
    return Counter(zip(sequence, sequence[1:]))


def _merge_pair(sequence: Sequence[int], pair: tuple[int, int], new_token_id: int) -> tuple[int, ...]:
    merged: list[int] = []
    i = 0
    left, right = pair
    while i < len(sequence):
        if i + 1 < len(sequence) and sequence[i] == left and sequence[i + 1] == right:
            merged.append(new_token_id)
            i += 2
            continue
        merged.append(int(sequence[i]))
        i += 1
    return tuple(merged)


class _OccurrenceBPETrainer:
    def __init__(self, weighted_segments: Iterable[tuple[tuple[int, ...], int]]) -> None:
        self.tokens = array("i")
        self.next_links = array("i")
        self.prev_links = array("i")
        self.weights = array("i")
        self.pair_counts: Counter[tuple[int, int]] = Counter()
        self.pair_positions: defaultdict[tuple[int, int], array] = defaultdict(lambda: array("I"))
        self.pair_order: dict[tuple[int, int], int] = {}
        self.active_nodes = 0
        self.active_segments = 0

        for segment, weight in weighted_segments:
            self.add_segment(segment, weight)

    def add_segment(self, segment: tuple[int, ...], weight: int) -> None:
        if weight <= 0 or len(segment) < 2:
            return
        start = len(self.tokens)
        n = len(segment)
        self.tokens.extend(segment)
        self.weights.extend(array("i", [weight]) * n)
        self.prev_links.extend(array("i", [-1] + list(range(start, start + n - 1))))
        self.next_links.extend(array("i", list(range(start + 1, start + n)) + [-1]))
        self.active_nodes += n
        self.active_segments += 1
        for offset in range(n - 1):
            idx = start + offset
            self._add_pair((segment[offset], segment[offset + 1]), weight, idx)

    def _ensure_pair_order(self, pair: tuple[int, int]) -> None:
        if pair not in self.pair_order:
            self.pair_order[pair] = len(self.pair_order)

    def _add_pair(self, pair: tuple[int, int], weight: int, position: int) -> None:
        if weight <= 0:
            return
        self._ensure_pair_order(pair)
        self.pair_counts[pair] += weight
        self.pair_positions[pair].append(position)

    def _subtract_pair(self, pair: tuple[int, int], weight: int) -> None:
        if weight <= 0:
            return
        new_count = self.pair_counts[pair] - weight
        if new_count > 0:
            self.pair_counts[pair] = new_count
        else:
            self.pair_counts.pop(pair, None)

    def heap(self) -> list[tuple[int, int, tuple[int, int]]]:
        heap = [
            (-count, self.pair_order[pair], pair)
            for pair, count in self.pair_counts.items()
            if count > 0
        ]
        heapq.heapify(heap)
        return heap

    def push_pair(
        self,
        heap: list[tuple[int, int, tuple[int, int]]],
        pair: tuple[int, int],
        blocked_pairs: set[tuple[int, int]],
    ) -> None:
        count = self.pair_counts.get(pair, 0)
        if count > 0 and pair not in blocked_pairs:
            heapq.heappush(heap, (-count, self.pair_order[pair], pair))

    def valid_pair_at(self, position: int, pair: tuple[int, int]) -> bool:
        if position < 0 or position >= len(self.tokens):
            return False
        right = self.next_links[position]
        return (
            right != -1
            and self.tokens[position] == pair[0]
            and self.tokens[right] == pair[1]
        )

    def merge_pair(
        self,
        pair: tuple[int, int],
        new_token_id: int,
        heap: list[tuple[int, int, tuple[int, int]]],
        blocked_pairs: set[tuple[int, int]],
    ) -> int:
        positions = self.pair_positions.pop(pair, array("I"))
        changed_pairs: set[tuple[int, int]] = set()
        merged = 0

        for position in positions:
            if not self.valid_pair_at(position, pair):
                continue

            right = self.next_links[position]
            left = self.prev_links[position]
            after = self.next_links[right]
            weight = self.weights[position]

            if left != -1:
                left_pair = (self.tokens[left], self.tokens[position])
                self._subtract_pair(left_pair, weight)
                changed_pairs.add(left_pair)

            self._subtract_pair(pair, weight)
            changed_pairs.add(pair)

            if after != -1:
                right_pair = (self.tokens[right], self.tokens[after])
                self._subtract_pair(right_pair, weight)
                changed_pairs.add(right_pair)

            self.tokens[position] = new_token_id
            self.tokens[right] = -1
            self.prev_links[right] = -1
            self.next_links[right] = -1
            self.next_links[position] = after
            if after != -1:
                self.prev_links[after] = position
            self.active_nodes -= 1
            merged += weight

            if left != -1:
                new_left_pair = (self.tokens[left], new_token_id)
                self._add_pair(new_left_pair, weight, left)
                changed_pairs.add(new_left_pair)
            if after != -1:
                new_right_pair = (new_token_id, self.tokens[after])
                self._add_pair(new_right_pair, weight, position)
                changed_pairs.add(new_right_pair)

        self.pair_counts.pop(pair, None)
        for changed_pair in changed_pairs:
            self.push_pair(heap, changed_pair, blocked_pairs)
        return merged


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _merged_atoms(
    pair: tuple[int, int],
    token_atoms: dict[int, tuple[int, ...]],
) -> tuple[int, ...]:
    return token_atoms.get(pair[0], (pair[0],)) + token_atoms.get(pair[1], (pair[1],))


def _pop_best_pair(
    heap: list[tuple[int, int, tuple[int, int]]],
    pair_counts: Counter[tuple[int, int]],
    pair_order: dict[tuple[int, int], int],
    token_atoms: dict[int, tuple[int, ...]],
    max_atomic_len: int,
    blocked_pairs: set[tuple[int, int]],
) -> tuple[tuple[int, int] | None, tuple[int, ...] | None, int]:
    while heap:
        neg_count, order, pair = heapq.heappop(heap)
        count = pair_counts.get(pair, 0)
        if count <= 0 or count != -neg_count or order != pair_order.get(pair) or pair in blocked_pairs:
            continue
        atoms = _merged_atoms(pair, token_atoms)
        if len(atoms) > max_atomic_len:
            blocked_pairs.add(pair)
            continue
        return pair, atoms, count
    return None, None, 0


def train_bpe_vocabulary(
    corpus: Iterable[Sequence[int] | tuple[Sequence[int], int]],
    target_size: int = BPE_VOCAB_SIZE,
    max_atomic_len: int = BPE_MAX_ATOMIC_LEN,
    start_id: int = BPE_TOKEN_START,
) -> BPEVocabulary:
    weighted_segments = _normalize_segments(corpus)
    trainer = _OccurrenceBPETrainer(weighted_segments)
    del weighted_segments
    heap = trainer.heap()
    blocked_pairs: set[tuple[int, int]] = set()
    token_atoms: dict[int, tuple[int, ...]] = {}
    next_token_id = start_id
    _log(
        f"Prepared BPE corpus: unique_segments={trainer.active_segments} "
        f"tokens={trainer.active_nodes} pairs={len(trainer.pair_counts)}"
    )

    while next_token_id < target_size:
        selected_pair, merged_atoms, _ = _pop_best_pair(
            heap,
            trainer.pair_counts,
            trainer.pair_order,
            token_atoms,
            max_atomic_len,
            blocked_pairs,
        )
        if selected_pair is None or merged_atoms is None:
            break

        token_atoms[next_token_id] = merged_atoms
        merged_count = trainer.merge_pair(selected_pair, next_token_id, heap, blocked_pairs)
        if (next_token_id - start_id + 1) % 10 == 0 or next_token_id + 1 == target_size:
            _log(
                f"BPE merges: {next_token_id - start_id + 1}/{target_size - start_id} "
                f"merged_occurrences={merged_count} active_tokens={trainer.active_nodes} "
                f"pairs={len(trainer.pair_counts)}"
            )
        next_token_id += 1

    return BPEVocabulary(start_id=start_id, vocab_size=target_size, token_atoms=token_atoms)


def build_atomic_corpus_from_sequences(sequences: Iterable[Sequence[int]]) -> list[tuple[tuple[int, ...], int]]:
    counts: Counter[tuple[int, ...]] = Counter()
    for sequence in sequences:
        counts[tuple(int(token) for token in sequence)] += 1
    return list(counts.items())


def train_bpe_from_jsonl_corpus(
    raw_jsonl_path: Path | str,
    vocab_path: Path | str,
    vocab_size: int = BPE_VOCAB_SIZE,
    start_id: int = BPE_TOKEN_START,
    max_atomic_len: int = BPE_MAX_ATOMIC_LEN,
    scale: int | None = None,
) -> BPETrainingReport:
    raw_jsonl_path = Path(raw_jsonl_path)
    vocab_path = Path(vocab_path)
    inferred_scale = scale
    sequences = 0

    def iter_training_sequences():
        nonlocal inferred_scale, sequences
        with raw_jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                inferred_scale = inferred_scale or row.get("scale")
                for field in ("source_tokens", "target_tokens"):
                    yield row[field]
                    sequences += 1

    vocab = train_bpe_vocabulary(
        iter_training_sequences(),
        target_size=vocab_size,
        start_id=start_id,
        max_atomic_len=max_atomic_len,
    )
    vocab_work = vocab_path.with_suffix(vocab_path.suffix + ".tmp")
    vocab.save(vocab_work)
    vocab_work.replace(vocab_path)
    if inferred_scale is None:
        raise ValueError(f"Cannot infer scale from {raw_jsonl_path}")
    return BPETrainingReport(
        scale=int(inferred_scale),
        source_train_dir=str(raw_jsonl_path.parent),
        vocab_path=str(vocab_path),
        vocab_size=vocab.vocab_size,
        token_start=vocab.start_id,
        max_atomic_len=max_atomic_len,
        sequences=sequences,
    )


def build_bpe_jsonl_from_raw_jsonl(
    raw_jsonl_path: Path | str,
    vocab_path: Path | str,
    output_jsonl: Path | str,
    limit: int | None = None,
) -> DatasetBuildReport:
    raw_jsonl_path = Path(raw_jsonl_path)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_work = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")
    failures_path = output_jsonl.with_suffix(".failures.jsonl")
    failures_work = failures_path.with_suffix(failures_path.suffix + ".tmp")
    failures_work.unlink(missing_ok=True)
    vocab = BPEVocabulary.load(vocab_path)
    written = skipped = 0
    failure_handle = None
    try:
        with raw_jsonl_path.open("r", encoding="utf-8") as source, output_work.open("w", encoding="utf-8") as output:
            for line_number, line in enumerate(source, start=1):
                if limit is not None and written >= limit:
                    break
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    encoded = dict(row)
                    encoded["source_tokens"] = vocab.encode(row["source_tokens"])
                    encoded["target_tokens"] = vocab.encode(row["target_tokens"])
                    output.write(json.dumps(encoded, ensure_ascii=False) + "\n")
                    written += 1
                except Exception as exc:
                    skipped += 1
                    if failure_handle is None:
                        failure_handle = failures_work.open("w", encoding="utf-8")
                    failure_handle.write(json.dumps({"line": line_number, "error": str(exc)}, ensure_ascii=False) + "\n")
    except BaseException:
        if failure_handle is not None:
            failure_handle.close()
        output_work.unlink(missing_ok=True)
        failures_work.unlink(missing_ok=True)
        raise
    if failure_handle is not None:
        failure_handle.close()
    if skipped:
        output_work.unlink(missing_ok=True)
        failures_work.replace(failures_path)
    else:
        output_work.replace(output_jsonl)
        failures_path.unlink(missing_ok=True)
    return DatasetBuildReport(
        written=written,
        skipped=skipped,
        failures_path=str(failures_path) if skipped else None,
    )


def select_bpe_training_subset(
    input_jsonl: Path | str,
    output_jsonl: Path | str,
    count: int = 10_000,
    seed: int = 20260713,
) -> int:
    if count <= 0:
        raise ValueError("count must be positive")
    selected: list[tuple[int, str, str]] = []
    with Path(input_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("osm_id", row.get("sample_id", "")))
            score = int.from_bytes(hashlib.sha256(f"{seed}:bpe:{key}".encode("utf-8")).digest()[:8], "big")
            item = (-score, key, line.rstrip("\n"))
            if len(selected) < count:
                heapq.heappush(selected, item)
            elif item > selected[0]:
                heapq.heapreplace(selected, item)
    ordered = sorted(selected, key=lambda item: (-item[0], item[1]))
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for _, _, line in ordered:
            output.write(line + "\n")
    temporary.replace(output_jsonl)
    return len(ordered)
