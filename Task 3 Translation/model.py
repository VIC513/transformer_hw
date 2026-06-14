from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(float(dropout))

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / float(d_model))
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones((sz, sz), dtype=torch.bool, device=device),
        diagonal=1,
    )


@dataclass(frozen=True)
class TransformerMasks:
    tgt_mask: torch.Tensor
    src_key_padding_mask: torch.Tensor
    tgt_key_padding_mask: torch.Tensor
    memory_key_padding_mask: torch.Tensor


def build_transformer_masks(
    *,
    src: torch.Tensor,
    tgt_inp: torch.Tensor,
    src_key_padding_mask: torch.Tensor,
    tgt_key_padding_mask: torch.Tensor,
) -> TransformerMasks:
    device = src.device
    tgt_len = int(tgt_inp.size(1))
    tgt_mask = generate_square_subsequent_mask(tgt_len, device=device)
    return TransformerMasks(
        tgt_mask=tgt_mask,
        src_key_padding_mask=src_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=src_key_padding_mask,
    )


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        *,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
        pad_id_src: int,
        pad_id_tgt: int,
    ):
        super().__init__()
        self.pad_id_src = int(pad_id_src)
        self.pad_id_tgt = int(pad_id_tgt)

        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=self.pad_id_src)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=self.pad_id_tgt)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = int(d_model)

    def forward(
        self,
        *,
        src: torch.Tensor,
        tgt_inp: torch.Tensor,
        masks: TransformerMasks,
    ) -> torch.Tensor:
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt_inp) * math.sqrt(self.d_model))
        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=masks.tgt_mask,
            src_key_padding_mask=masks.src_key_padding_mask,
            tgt_key_padding_mask=masks.tgt_key_padding_mask,
            memory_key_padding_mask=masks.memory_key_padding_mask,
        )
        return self.generator(out)

    @torch.no_grad()
    def encode(
        self,
        *,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    @torch.no_grad()
    def decode(
        self,
        *,
        tgt_inp: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt_inp) * math.sqrt(self.d_model))
        return self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
