from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
import warnings

import torch
from torch.utils.data import DataLoader

from config import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN, SPECIAL_TOKENS, UNK_TOKEN

try:
    import jieba  # type: ignore
except ImportError:
    jieba = None

_JIEBA_WARNING_SHOWN = False


def tokenize_en(text: str) -> List[str]:
    return [t for t in text.strip().lower().split() if t]


def tokenize_zh(text: str) -> List[str]:
    global _JIEBA_WARNING_SHOWN
    text = text.strip()
    if not text:
        return []
    if jieba is not None:
        return [t for t in jieba.lcut(text) if t and not t.isspace()]
    # Fallback for environments without jieba so training can still run.
    if not _JIEBA_WARNING_SHOWN:
        warnings.warn(
            "jieba is not installed; falling back to character-level Chinese tokenization. "
            "This is fine for debugging, but translation quality may drop.",
            stacklevel=2,
        )
        _JIEBA_WARNING_SHOWN = True
    return [ch for ch in text if not ch.isspace()]


@dataclass(frozen=True)
class Vocab:
    token_to_id: Dict[str, int]
    id_to_token: List[str]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def sos_id(self) -> int:
        return self.token_to_id[SOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.id_to_token)


def build_vocab(
    token_seqs: Iterable[Sequence[str]],
    *,
    min_freq: int,
) -> Vocab:
    counter: Counter[str] = Counter()
    for seq in token_seqs:
        counter.update(seq)

    id_to_token: List[str] = list(SPECIAL_TOKENS)
    for tok, freq in counter.most_common():
        if freq < min_freq:
            continue
        if tok in SPECIAL_TOKENS:
            continue
        id_to_token.append(tok)

    token_to_id = {t: i for i, t in enumerate(id_to_token)}
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token)


class TextProcessor:
    def __init__(
        self,
        *,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_seq_len: int,
    ):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_seq_len = int(max_seq_len)

    @classmethod
    def from_sentence_pairs(
        cls,
        pairs: Sequence[Tuple[str, str]],
        *,
        max_seq_len: int,
        min_freq: int,
    ) -> "TextProcessor":
        src_tokens = [tokenize_en(en) for en, _ in pairs]
        tgt_tokens = [tokenize_zh(zh) for _, zh in pairs]
        src_vocab = build_vocab(src_tokens, min_freq=min_freq)
        tgt_vocab = build_vocab(tgt_tokens, min_freq=min_freq)
        return cls(src_vocab=src_vocab, tgt_vocab=tgt_vocab, max_seq_len=max_seq_len)

    def encode_src(self, text: str) -> List[int]:
        tokens = tokenize_en(text)
        ids = [self.src_vocab.sos_id]
        ids += [self.src_vocab.token_to_id.get(t, self.src_vocab.unk_id) for t in tokens]
        ids.append(self.src_vocab.eos_id)
        return ids[: self.max_seq_len]

    def encode_tgt(self, text: str) -> List[int]:
        tokens = tokenize_zh(text)
        ids = [self.tgt_vocab.sos_id]
        ids += [self.tgt_vocab.token_to_id.get(t, self.tgt_vocab.unk_id) for t in tokens]
        ids.append(self.tgt_vocab.eos_id)
        return ids[: self.max_seq_len]

    def decode_tgt(self, ids: Sequence[int], *, remove_special: bool = True) -> str:
        tokens: List[str] = []
        for i in ids:
            if 0 <= int(i) < len(self.tgt_vocab.id_to_token):
                tok = self.tgt_vocab.id_to_token[int(i)]
            else:
                tok = UNK_TOKEN

            if remove_special and tok in {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN}:
                continue
            tokens.append(tok)
        return "".join(tokens)

    def build_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        *,
        batch_size: int,
        shuffle: bool,
        num_workers: int = 0,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            num_workers=int(num_workers),
            collate_fn=self._collate,
            drop_last=False,
        )

    def _collate(self, batch: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        src_list = [torch.tensor(x["src_ids"], dtype=torch.long) for x in batch]
        tgt_list = [torch.tensor(x["tgt_ids"], dtype=torch.long) for x in batch]

        src = torch.nn.utils.rnn.pad_sequence(
            src_list,
            batch_first=True,
            padding_value=self.src_vocab.pad_id,
        )
        tgt = torch.nn.utils.rnn.pad_sequence(
            tgt_list,
            batch_first=True,
            padding_value=self.tgt_vocab.pad_id,
        )

        src = src[:, : self.max_seq_len]
        tgt = tgt[:, : self.max_seq_len]

        src_key_padding_mask = src.eq(self.src_vocab.pad_id)
        tgt_key_padding_mask = tgt.eq(self.tgt_vocab.pad_id)

        return {
            "src": src,
            "tgt": tgt,
            "src_key_padding_mask": src_key_padding_mask,
            "tgt_key_padding_mask": tgt_key_padding_mask,
        }
