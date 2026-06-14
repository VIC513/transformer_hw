from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from io_utils import ensure_file
from text_processor import TextProcessor


def read_tsv_pairs(
    path: str | Path,
    *,
    max_rows: int | None = None,
    en_col: int = 0,
    zh_col: int = 1,
    delimiter: str = "\t",
) -> List[Tuple[str, str]]:
    p = ensure_file(
        path,
        hint=(
            "Put a TSV file at the path above. Expected columns: EN<TAB>ZH. "
            "You can run `python Task 3 Translation/prepare_tatoeba.py` to download and prepare."
        ),
    )
    pairs: List[Tuple[str, str]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            cols = line.split(delimiter)
            if len(cols) <= max(en_col, zh_col):
                continue
            en = cols[en_col].strip()
            zh = cols[zh_col].strip()
            if not en or not zh:
                continue
            pairs.append((en, zh))
            if max_rows is not None and len(pairs) >= int(max_rows):
                break
    if len(pairs) == 0:
        raise ValueError(f"No valid pairs found in {p.resolve()}")
    return pairs


class ParallelTextDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        processor: TextProcessor,
    ):
        self.pairs = list(pairs)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        en, zh = self.pairs[idx]
        return {
            "src_ids": self.processor.encode_src(en),
            "tgt_ids": self.processor.encode_tgt(zh),
        }


@dataclass(frozen=True)
class DataBundle:
    processor: TextProcessor
    train_ds: ParallelTextDataset
    val_ds: ParallelTextDataset


def build_datasets(
    pairs: Sequence[Tuple[str, str]],
    *,
    max_seq_len: int,
    min_freq: int,
    train_split: float,
    seed: int = 42,
) -> DataBundle:
    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)

    n_train = max(1, int(len(pairs) * float(train_split)))
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:] if n_train < len(pairs) else pairs[: min(1000, len(pairs))]

    processor = TextProcessor.from_sentence_pairs(
        train_pairs,
        max_seq_len=max_seq_len,
        min_freq=min_freq,
    )

    return DataBundle(
        processor=processor,
        train_ds=ParallelTextDataset(train_pairs, processor),
        val_ds=ParallelTextDataset(val_pairs, processor),
    )

