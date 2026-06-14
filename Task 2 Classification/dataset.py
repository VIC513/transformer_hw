from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


try:
    import pretty_midi
except Exception:  # pragma: no cover
    pretty_midi = None


PAD_ID = 0
CLS_ID = 1
PITCH_OFFSET = 2
VOCAB_SIZE = PITCH_OFFSET + 128


GENRES = ("classical", "pop", "jazz", "electronic")
GENRE_TO_LABEL = {g: i for i, g in enumerate(GENRES)}
LABEL_TO_GENRE = {i: g for g, i in GENRE_TO_LABEL.items()}


@dataclass(frozen=True)
class MidiExample:
    path: str
    label: int


def _list_midi_files(folder: Path) -> List[Path]:
    exts = {".mid", ".midi"}
    files: List[Path] = []
    for root, _, filenames in os.walk(folder):
        for name in filenames:
            p = Path(root) / name
            if p.suffix.lower() in exts:
                files.append(p)
    files.sort(key=lambda p: str(p).lower())
    return files


def build_index(data_root: str | Path = "./data/genres") -> List[MidiExample]:
    root = Path(data_root)
    examples: List[MidiExample] = []
    for genre in GENRES:
        genre_dir = root / genre
        if not genre_dir.exists():
            continue
        for midi_path in _list_midi_files(genre_dir):
            examples.append(MidiExample(path=str(midi_path), label=GENRE_TO_LABEL[genre]))
    return examples


def parse_midi_to_pitch_tokens(
    midi_path: str | Path,
    seq_len: int,
    add_cls: bool = True,
    ignore_drums: bool = True,
    min_notes: int = 1,
) -> torch.Tensor:
    if pretty_midi is None:
        raise ImportError(
            "pretty_midi is not available. Install with `pip install pretty_midi`."
        )

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes: List[pretty_midi.Note] = []

    for inst in midi.instruments:
        if ignore_drums and getattr(inst, "is_drum", False):
            continue
        notes.extend(inst.notes)

    notes.sort(key=lambda n: (float(n.start), float(n.end), int(n.pitch)))

    pitches: List[int] = []
    for n in notes:
        p = int(n.pitch)
        if 0 <= p <= 127:
            pitches.append(PITCH_OFFSET + p)

    if len(pitches) < min_notes:
        raise ValueError(f"too few notes: {len(pitches)}")

    if add_cls:
        max_notes = max(0, seq_len - 1)
        ids = [CLS_ID] + pitches[:max_notes]
    else:
        ids = pitches[:seq_len]

    if len(ids) < seq_len:
        ids = ids + [PAD_ID] * (seq_len - len(ids))

    return torch.tensor(ids, dtype=torch.long)


class GenreMIDIDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path = "./data/genres",
        seq_len: int = 300,
        add_cls: bool = True,
        ignore_drums: bool = True,
        on_error: str = "return_none",
    ):
        self.data_root = str(data_root)
        self.seq_len = int(seq_len)
        self.add_cls = bool(add_cls)
        self.ignore_drums = bool(ignore_drums)
        if on_error not in {"return_none", "raise"}:
            raise ValueError("on_error must be 'return_none' or 'raise'")
        self.on_error = on_error

        self.examples = build_index(self.data_root)
        if len(self.examples) == 0:
            raise FileNotFoundError(
                f"No MIDI files found under {Path(self.data_root).resolve()}"
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        try:
            input_ids = parse_midi_to_pitch_tokens(
                ex.path,
                seq_len=self.seq_len,
                add_cls=self.add_cls,
                ignore_drums=self.ignore_drums,
            )
        except Exception:
            if self.on_error == "raise":
                raise
            return None

        return {
            "input_ids": input_ids,
            "label": torch.tensor(ex.label, dtype=torch.long),
            "path": ex.path,
        }


def collate_skip_bad(batch: Sequence[Optional[Dict]]) -> Optional[Dict[str, object]]:
    batch_ok = [b for b in batch if b is not None]
    if len(batch_ok) == 0:
        return None

    input_ids = torch.stack([b["input_ids"] for b in batch_ok], dim=0)
    labels = torch.stack([b["label"] for b in batch_ok], dim=0)
    paths = [str(b["path"]) for b in batch_ok]
    key_padding_mask = input_ids.eq(PAD_ID)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "key_padding_mask": key_padding_mask,
        "paths": paths,
    }

