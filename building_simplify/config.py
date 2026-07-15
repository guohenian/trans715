from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TokenConfig:
    directions: int = 720
    map_scale: int = 2000
    map_precision_mm: float = 0.1
    max_source_len: int = 2000
    max_target_len: int = 2000

    @property
    def token_length_m(self) -> float:
        return (self.map_precision_mm / 1000.0) * self.map_scale

    @property
    def bin_width_rad(self) -> float:
        return 2.0 * math.pi / self.directions


DEFAULT_CONFIG = TokenConfig()
TOKENIZATION_VERSION = 2

TOKEN_PAD = 720
TOKEN_SEP = 721
TOKEN_EOS = 722
TOKEN_BOS = 723
TOKEN_INNER = 724
TOKEN_SCALE_5000 = 725
TOKEN_SCALE_10000 = 726
TOKEN_BPE_START = 1500
BPE_TOKEN_START = TOKEN_BPE_START
BPE_VOCAB_SIZE = 6000
BPE_MAX_ATOMIC_LEN = 10

SPECIAL_TOKENS = {
    TOKEN_PAD,
    TOKEN_SEP,
    TOKEN_EOS,
    TOKEN_BOS,
    TOKEN_INNER,
    TOKEN_SCALE_5000,
    TOKEN_SCALE_10000,
}

VOCAB_SIZE = BPE_VOCAB_SIZE

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = WORKSPACE_ROOT / "datasets"
BPE_VOCAB_5000_PATH = DATASETS_ROOT / "scale5000" / "vocab" / "bpe_vocab.txt"
BPE_VOCAB_10000_PATH = DATASETS_ROOT / "scale10000" / "vocab" / "bpe_vocab.txt"

DEFAULT_MODEL_D_MODEL = 256
DEFAULT_MODEL_NHEAD = 8
DEFAULT_MODEL_NUM_LAYERS = 4
DEFAULT_MODEL_DIM_FEEDFORWARD = 1024
DEFAULT_MODEL_DROPOUT = 0.1
