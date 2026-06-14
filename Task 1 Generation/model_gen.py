from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 130
    # 作业一不再需要 num_classes，输出就是 vocab_size
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 512
    pad_id: int = 0

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)

class MusicTransformerGenerator(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
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
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        
        # 【关键改动】输出头现在对应词表大小，预测下一个音高
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def generate_causal_mask(self, sz: int, device: torch.device):
        """生成下三角掩码，防止模型看到未来信息"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # 1. 生成因果掩码 (Causal Mask)
        causal_mask = self.generate_causal_mask(seq_len, input_ids.device)

        # 2. Embedding + Position
        x = self.token_emb(input_ids)
        x = self.pos_enc(x)

        # 3. Transformer Encoder (同时传入两种 Mask)
        # src_mask 用于屏蔽未来音符，src_key_padding_mask 用于屏蔽 Padding
        x = self.encoder(
            x, 
            mask=causal_mask, 
            src_key_padding_mask=key_padding_mask
        )
        
        x = self.norm(x)
        logits = self.head(x) # 输出形状: [batch, seq, vocab_size]
        return logits

def build_gen_model(vocab_size: int = 130) -> MusicTransformerGenerator:
    cfg = TransformerConfig(vocab_size=vocab_size)
    return MusicTransformerGenerator(cfg)

if __name__ == "__main__":
    model = build_gen_model()
    # 模拟输入 2 条数据，每条 50 个音符
    test_input = torch.randint(0, 130, (2, 50))
    logits = model(test_input)
    print(f"输出形状 (应为 [2, 50, 130]): {logits.shape}")