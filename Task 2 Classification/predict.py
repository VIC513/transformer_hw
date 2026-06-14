from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from dataset import LABEL_TO_GENRE, PAD_ID, VOCAB_SIZE, parse_midi_to_pitch_tokens
from model import build_default_model


def load_checkpoint(path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict) and "model_state" in obj:
        meta = {k: v for k, v in obj.items() if k != "model_state"}
        return obj["model_state"], meta
    if isinstance(obj, dict):
        return obj, {}
    raise ValueError("Unsupported checkpoint format")


def main():
    parser = argparse.ArgumentParser(description="Predict genre for a single MIDI file")
    parser.add_argument("midi_path", type=str)
    parser.add_argument("--model", type=str, default="best_model.pth")
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--pooling", type=str, choices=["cls", "mean"], default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.model)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path.resolve()}")

    state_dict, meta = load_checkpoint(ckpt_path)
    seq_len = int(args.seq_len) if args.seq_len is not None else int(meta.get("seq_len", 300))
    pooling = str(args.pooling) if args.pooling is not None else str(meta.get("pooling", "cls"))

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(args.device))

    model = build_default_model(
        seq_len=seq_len,
        num_classes=4,
        vocab_size=VOCAB_SIZE,
        pooling=pooling,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    midi_path = Path(args.midi_path)
    if not midi_path.exists():
        raise FileNotFoundError(f"midi not found: {midi_path.resolve()}")

    input_ids = parse_midi_to_pitch_tokens(midi_path, seq_len=seq_len, add_cls=True)
    input_ids = input_ids.unsqueeze(0).to(device)
    key_padding_mask = input_ids.eq(PAD_ID)

    with torch.no_grad():
        logits = model(input_ids, key_padding_mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu()
        pred = int(probs.argmax().item())

    genre = LABEL_TO_GENRE.get(pred, str(pred))
    print(f"pred: {pred} ({genre})")
    print("probs:")
    for i in range(len(probs)):
        name = LABEL_TO_GENRE.get(i, str(i))
        print(f"  {i} {name}: {float(probs[i]):.4f}")


if __name__ == "__main__":
    main()

