from __future__ import annotations

import math

import torch
import warnings
warnings.filterwarnings("ignore", message=".*nested tensor.*")
from torch import nn

from .config import TOKEN_BOS, TOKEN_EOS, TOKEN_PAD, TOKEN_SCALE_5000, TOKEN_SCALE_10000


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class BuildingTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pre_ln: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=TOKEN_PAD)
        self.position = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=pre_ln,
        )
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_key_padding_mask = src.eq(TOKEN_PAD)
        tgt_key_padding_mask = tgt.eq(TOKEN_PAD)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device, dtype=torch.bool)
        src_emb = self.position(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.position(self.embedding(tgt) * math.sqrt(self.d_model))
        hidden = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output(hidden)


@torch.no_grad()
def generate_greedy(
    model: BuildingTransformer,
    source: torch.Tensor,
    bos_token: int,
    eos_token: int,
    max_len: int,
) -> torch.Tensor:
    """贪婪解码（encoder 输出缓存, 避免重复编码）。"""
    model.eval()
    device = source.device
    src_pad = source.eq(TOKEN_PAD)

    # 一次编码源序列, 后续复用
    src_emb = model.position(model.embedding(source) * math.sqrt(model.d_model))
    memory = model.transformer.encoder(src_emb, src_key_padding_mask=src_pad)
    mem_pad = src_pad

    generated = torch.full((source.size(0), 1), bos_token, dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        tgt_pad = generated.eq(TOKEN_PAD)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(generated.size(1), device=device, dtype=torch.bool)
        tgt_emb = model.position(model.embedding(generated) * math.sqrt(model.d_model))
        # 直接调 decoder, 复用缓存的 memory
        hidden = model.transformer.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=mem_pad,
        )
        logits = model.output(hidden)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if torch.all(next_token.squeeze(1).eq(eos_token)):
            break
    return generated


@torch.no_grad()
def generate_beam_search(
    model: BuildingTransformer,
    source: torch.Tensor,
    bos_token: int,
    eos_token: int,
    max_len: int,
    beam_size: int = 5,
    length_penalty: float = 0.8,
    min_len: int = 4,
) -> list[tuple[list[int], float]]:
    if source.size(0) != 1:
        raise ValueError("Beam search currently expects a single source sequence")

    model.eval()
    device = source.device
    beams: list[tuple[list[int], float, bool]] = [([bos_token], 0.0, False)]
    blocked_tokens = [TOKEN_PAD, TOKEN_BOS, TOKEN_SCALE_5000, TOKEN_SCALE_10000]

    def normalized_score(tokens: list[int], score: float) -> float:
        length = max(1, len(tokens) - 1)
        if length_penalty <= 0:
            return score
        return score / (length**length_penalty)

    # 一次编码 source
    src_pad = source.eq(TOKEN_PAD)
    src_emb = model.position(model.embedding(source) * math.sqrt(model.d_model))
    memory = model.transformer.encoder(src_emb, src_key_padding_mask=src_pad)
    mem_pad = src_pad

    for _ in range(max_len - 1):
        active = [(idx, tokens, score) for idx, (tokens, score, ended) in enumerate(beams) if not ended]
        if not active:
            break

        decoder = torch.tensor([tokens for _, tokens, _ in active], dtype=torch.long, device=device)
        mem_expanded = memory.expand(len(active), -1, -1)
        mem_pad_expanded = mem_pad.expand(len(active), -1) if mem_pad.size(0) == 1 else mem_pad
        tgt_pad = decoder.eq(TOKEN_PAD)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder.size(1), device=device, dtype=torch.bool)
        tgt_emb = model.position(model.embedding(decoder) * math.sqrt(model.d_model))
        hidden = model.transformer.decoder(
            tgt_emb, mem_expanded,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=mem_pad_expanded,
        )
        logits = model.output(hidden)[:, -1, :]
        logits[:, blocked_tokens] = -torch.inf
        if decoder.size(1) < min_len:
            logits[:, eos_token] = -torch.inf
        log_probs = torch.log_softmax(logits, dim=-1)
        top_scores, top_tokens = torch.topk(log_probs, k=min(beam_size, log_probs.size(-1)), dim=-1)

        candidates: list[tuple[list[int], float, bool]] = [
            (tokens, score, ended) for tokens, score, ended in beams if ended
        ]
        for row, (_, tokens, score) in enumerate(active):
            for col in range(top_tokens.size(1)):
                token = int(top_tokens[row, col].item())
                token_score = float(top_scores[row, col].item())
                if not math.isfinite(token_score):
                    continue
                next_tokens = [*tokens, token]
                candidates.append((next_tokens, score + token_score, token == eos_token))

        candidates.sort(key=lambda item: normalized_score(item[0], item[1]), reverse=True)
        beams = candidates[:beam_size]
        if all(ended for _, _, ended in beams):
            break

    return [(tokens, normalized_score(tokens, score)) for tokens, score, _ in beams]


# ── 批量贪婪解码 (encoder 缓存, 给 train.py 用) ──

def _encode_source(model: BuildingTransformer, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    src_pad = source.eq(TOKEN_PAD)
    src_emb = model.position(model.embedding(source) * math.sqrt(model.d_model))
    memory = model.transformer.encoder(src_emb, src_key_padding_mask=src_pad)
    return memory, src_pad


def _decode_next_token(
    model: BuildingTransformer,
    generated: torch.Tensor,
    memory: torch.Tensor,
    memory_key_padding_mask: torch.Tensor,
) -> torch.Tensor:
    tgt_pad = generated.eq(TOKEN_PAD)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(
        generated.size(1), device=generated.device, dtype=torch.bool,
    )
    tgt_emb = model.position(model.embedding(generated) * math.sqrt(model.d_model))
    hidden = model.transformer.decoder(
        tgt_emb, memory, tgt_mask=tgt_mask,
        tgt_key_padding_mask=tgt_pad,
        memory_key_padding_mask=memory_key_padding_mask,
    )
    logits = model.output(hidden)
    return logits[:, -1].argmax(dim=-1)


@torch.no_grad()
def generate_greedy_batch(
    model: BuildingTransformer,
    source: torch.Tensor,
    max_len: int,
    bos_token: int = TOKEN_BOS,
    eos_token: int = TOKEN_EOS,
) -> torch.Tensor:
    """批量贪婪解码, 缓存 encoder 输出。"""
    model.eval()
    memory, mem_pad = _encode_source(model, source)
    generated = torch.full((source.size(0), 1), bos_token, dtype=torch.long, device=source.device)
    finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
    for _ in range(max(0, max_len - 1)):
        next_token = _decode_next_token(model, generated, memory, mem_pad)
        next_token = torch.where(finished, torch.full_like(next_token, TOKEN_PAD), next_token)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        finished = finished | next_token.eq(eos_token)
        if bool(torch.all(finished).item()):
            break
    return generated
