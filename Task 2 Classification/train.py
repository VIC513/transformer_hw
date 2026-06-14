from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import (
    GENRES,
    LABEL_TO_GENRE,
    VOCAB_SIZE,
    GenreMIDIDataset,
    collate_skip_bad,
)
from model import build_default_model


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    for batch in loader:
        if batch is None:
            continue
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        key_padding_mask = batch["key_padding_mask"].to(device)

        logits = model(input_ids, key_padding_mask)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=-1)
        total_loss += float(loss.item()) * int(labels.size(0))
        total_correct += int((preds == labels).sum().item())
        total_count += int(labels.size(0))

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    if total_count == 0:
        return 0.0, 0.0, y_true, y_pred
    return total_loss / total_count, total_correct / total_count, y_true, y_pred


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        if batch is None:
            continue
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        key_padding_mask = batch["key_padding_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, key_padding_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = logits.argmax(dim=-1)
        total_loss += float(loss.item()) * int(labels.size(0))
        total_correct += int((preds == labels).sum().item())
        total_count += int(labels.size(0))

    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


def split_indices_stratified(labels: List[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    try:
        from sklearn.model_selection import train_test_split

        idx = list(range(len(labels)))
        train_idx, val_idx = train_test_split(
            idx, test_size=val_ratio, random_state=seed, stratify=labels
        )
        return list(train_idx), list(val_idx)
    except Exception:
        idx = list(range(len(labels)))
        rnd = random.Random(seed)
        rnd.shuffle(idx)
        cut = max(1, int(round(len(idx) * (1.0 - val_ratio))))
        return idx[:cut], idx[cut:]


def main():
    parser = argparse.ArgumentParser(description="Train Transformer MIDI genre classifier")
    parser.add_argument("--data_root", type=str, default="./data/genres")
    parser.add_argument("--seq_len", type=int, default=300)
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--best_path", type=str, default="best_model.pth")
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = GenreMIDIDataset(
        data_root=args.data_root,
        seq_len=int(args.seq_len),
        add_cls=True,
        on_error="return_none",
    )
    labels = [ex.label for ex in ds.examples]
    train_idx, val_idx = split_indices_stratified(labels, float(args.val_ratio), int(args.seed))

    train_set = Subset(ds, train_idx)
    val_set = Subset(ds, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_skip_bad,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_skip_bad,
        drop_last=False,
    )

    model = build_default_model(
        seq_len=int(args.seq_len),
        num_classes=len(GENRES),
        vocab_size=VOCAB_SIZE,
        pooling=str(args.pooling),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )

    best_acc = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_path = Path(args.best_path)

    print(f"device: {device}")
    print(f"train/val: {len(train_idx)}/{len(val_idx)}")
    print(f"vocab_size: {VOCAB_SIZE} (from dataset.VOCAB_SIZE)")

    for epoch in range(1, int(args.epochs) + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optimizer)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, device, criterion)

        print(
            f"epoch {epoch:03d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f}"
        )

        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state": best_state,
                    "seq_len": int(args.seq_len),
                    "pooling": str(args.pooling),
                    "vocab_size": int(VOCAB_SIZE),
                    "num_classes": int(len(GENRES)),
                    "label_to_genre": dict(LABEL_TO_GENRE),
                },
                str(best_path),
            )

    try:
        from sklearn.metrics import classification_report, confusion_matrix
    except Exception as e:
        raise ImportError(
            "scikit-learn is required for classification_report/confusion_matrix. Install with `pip install scikit-learn`."
        ) from e

    if best_state is not None:
        model.load_state_dict(best_state)
    _, _, y_true, y_pred = evaluate(model, val_loader, device, criterion)

    print("\nClassification Report (val):")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(GENRES))),
            target_names=list(GENRES),
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(GENRES))))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=list(range(len(GENRES))),
            yticks=list(range(len(GENRES))),
            xticklabels=list(GENRES),
            yticklabels=list(GENRES),
            ylabel="True",
            xlabel="Pred",
            title="Confusion Matrix (val)",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        thresh = cm.max() / 2.0 if cm.size else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        fig.tight_layout()
        out_png = best_path.with_suffix(".confusion_matrix.png")
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        print(f"saved: {out_png}")
    except Exception:
        pass

    print(f"best val acc: {best_acc:.4f}")
    print(f"saved: {best_path.resolve()}")


if __name__ == "__main__":
    main()

