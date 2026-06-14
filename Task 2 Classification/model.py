from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 130
    num_classes: int = 4
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 512
    pooling: str = "cls"
    pad_id: int = 0
    add_cls: bool = True
    include_cls_in_mean: bool = False


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0, dtype=torch.float32)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("x must be [batch, seq, d_model]")
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].unsqueeze(0).to(dtype=x.dtype)
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")
        if cfg.pooling == "cls" and not cfg.add_cls:
            raise ValueError("pooling='cls' requires add_cls=True")

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_enc = PositionalEncoding(cfg.d_model, dropout=cfg.dropout, max_len=cfg.max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        try:
            self.encoder = nn.TransformerEncoder(
                enc_layer, num_layers=cfg.num_layers, enable_nested_tensor=False
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be [batch, seq]")

        if key_padding_mask is None:
            key_padding_mask = input_ids.eq(self.cfg.pad_id)

        x = self.token_emb(input_ids)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        if self.cfg.pooling == "cls":
            pooled = x[:, 0, :]
        else:
            valid = ~key_padding_mask
            if self.cfg.add_cls and not self.cfg.include_cls_in_mean:
                valid = valid.clone()
                valid[:, 0] = False
            denom = valid.sum(dim=1).clamp_min(1).unsqueeze(-1)
            pooled = (x * valid.unsqueeze(-1)).sum(dim=1) / denom

        logits = self.head(pooled)
        return logits


def build_default_model(
    seq_len: int = 300,
    num_classes: int = 4,
    vocab_size: int = 130,
    pooling: str = "cls",
) -> TransformerClassifier:
    cfg = TransformerConfig(
        vocab_size=vocab_size,
        num_classes=num_classes,
        max_len=max(64, int(seq_len)),
        pooling=pooling,
    )
    return TransformerClassifier(cfg)


if __name__ == "__main__":
    model = build_default_model(seq_len=300, num_classes=4, vocab_size=130, pooling="cls")
    input_ids = torch.randint(0, 130, (2, 300))
    input_ids[:, -10:] = 0
    logits = model(input_ids)
    print(logits.shape)
