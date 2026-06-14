from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import torch
from torch.utils.data import Dataset

try:
    import pretty_midi
except Exception:
    pretty_midi = None

# 保持词表约定一致
PAD_ID = 0
# 生成任务通常不需要 CLS，但为了兼容之前的 Embedding 层，我们保留偏移量
PITCH_OFFSET = 2 
VOCAB_SIZE = PITCH_OFFSET + 128

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

def parse_midi_for_generation(
    midi_path: str | Path,
    seq_len: int,
    ignore_drums: bool = True,
) -> torch.Tensor:
    """将 MIDI 转换为音高序列"""
    if pretty_midi is None:
        raise ImportError("Install pretty_midi with `pip install pretty_midi`.")

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes: List[pretty_midi.Note] = []

    for inst in midi.instruments:
        if ignore_drums and getattr(inst, "is_drum", False):
            continue
        notes.extend(inst.notes)

    # 按时间排序
    notes.sort(key=lambda n: (float(n.start), float(n.end), int(n.pitch)))

    pitches: List[int] = []
    for n in notes:
        p = int(n.pitch)
        if 0 <= p <= 127:
            pitches.append(PITCH_OFFSET + p)

    # 【核心改动】生成任务需要多取一个音符，用来构造错位的标签
    # 如果 seq_len 是 100，我们需要 101 个音符来形成 (0-99) 输入和 (1-100) 输出
    required_len = seq_len + 1
    
    if len(pitches) < required_len:
        # 如果长度不足，用 PAD 填充
        pitches = pitches + [PAD_ID] * (required_len - len(pitches))
    else:
        # 如果长度超过，进行截断
        pitches = pitches[:required_len]

    return torch.tensor(pitches, dtype=torch.long)

class MusicGenDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path = "../data/genres", # 注意这里用了 .. 返回上级目录找 data
        seq_len: int = 300,
        ignore_drums: bool = True,
    ):
        self.data_root = str(data_root)
        self.seq_len = int(seq_len)
        self.ignore_drums = bool(ignore_drums)

        # 扫描所有子目录下的 MIDI 文件作为语料
        self.midi_files = _list_midi_files(Path(self.data_root))
        if len(self.midi_files) == 0:
            raise FileNotFoundError(f"No MIDI files found under {self.data_root}")

    def __len__(self) -> int:
        return len(self.midi_files)

    def __getitem__(self, idx: int):
        path = self.midi_files[idx]
        try:
            full_seq = parse_midi_for_generation(
                path,
                seq_len=self.seq_len,
                ignore_drums=self.ignore_drums,
            )
            # 【自回归切分】
            # input_ids: [0, 1, 2, ..., N-1]
            # target_ids: [1, 2, 3, ..., N]
            return {
                "input_ids": full_seq[:-1],
                "target_ids": full_seq[1:],
                "path": str(path)
            }
        except Exception:
            return None

def collate_gen(batch: Sequence[Optional[Dict]]) -> Optional[Dict]:
    batch_ok = [b for b in batch if b is not None]
    if not batch_ok: return None

    input_ids = torch.stack([b["input_ids"] for b in batch_ok])
    target_ids = torch.stack([b["target_ids"] for b in batch_ok])
    
    # 生成 Padding Mask 用于 Transformer
    key_padding_mask = input_ids.eq(PAD_ID)
    
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "key_padding_mask": key_padding_mask,
    }

if __name__ == "__main__":
    # 测试代码
    try:
        ds = MusicGenDataset(data_root="../data/genres", seq_len=10)
        item = ds[0]
        if item:
            print(f"输入长度: {len(item['input_ids'])}")
            print(f"目标长度: {len(item['target_ids'])}")
            print(f"第一个目标音符是输入序列的第二个音符: {item['input_ids'][1] == item['target_ids'][0]}")
    except Exception as e:
        print(f"测试失败（可能是路径问题）: {e}")