from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None

from config import (
    BATCH_SIZE,
    D_MODEL,
    DIM_FEEDFORWARD,
    DROPOUT,
    EPOCHS,
    LR,
    MAX_SEQ_LEN,
    MIN_FREQ,
    NHEAD,
    NUM_DECODER_LAYERS,
    NUM_ENCODER_LAYERS,
    WEIGHT_DECAY,
)
from data import ParallelTextDataset, read_tsv_pairs
from model import Seq2SeqTransformer, build_transformer_masks, generate_square_subsequent_mask
from text_processor import TextProcessor


Pair = Tuple[str, str]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device_arg: str) -> torch.device:
    arg = str(device_arg).strip().lower()
    if arg == "cpu":
        return torch.device("cpu")

    if arg == "auto":
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            return torch.device("cuda:0")
        return torch.device("cpu")

    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        torch.cuda.set_device(0)
        return torch.device("cuda:0")

    if arg.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        idx_str = arg.split(":", 1)[1]
        idx = int(idx_str)
        if idx < 0 or idx >= int(torch.cuda.device_count()):
            raise RuntimeError(
                f"Requested cuda:{idx}, but device_count={torch.cuda.device_count()}."
            )
        torch.cuda.set_device(idx)
        return torch.device(f"cuda:{idx}")

    raise ValueError(f"Unknown --device value: {device_arg}")


def build_writer(*, logdir: str, run_name: str) -> object | None:
    if not logdir or str(logdir).strip().lower() in {"none", "off", "false", "0"}:
        return None
    if SummaryWriter is None:
        print("TensorBoard SummaryWriter is unavailable (tensorboard not installed). Skipping logs.")
        return None
    base = logdir
    if base == "auto":
        base = str(Path(__file__).resolve().parent / "runs")
    run_dir = Path(base) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(run_dir))


def train_step(
    *,
    model: Seq2SeqTransformer,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    src = batch["src"].to(device)
    tgt = batch["tgt"].to(device)
    src_kpm = batch["src_key_padding_mask"].to(device)
    tgt_kpm = batch["tgt_key_padding_mask"].to(device)

    tgt_inp = tgt[:, :-1]
    tgt_out = tgt[:, 1:]
    tgt_kpm_inp = tgt_kpm[:, :-1]

    masks = build_transformer_masks(
        src=src,
        tgt_inp=tgt_inp,
        src_key_padding_mask=src_kpm,
        tgt_key_padding_mask=tgt_kpm_inp,
    )

    logits = model(src=src, tgt_inp=tgt_inp, masks=masks)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def eval_loss(
    *,
    model: Seq2SeqTransformer,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses: List[float] = []
    for batch in loader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        src_kpm = batch["src_key_padding_mask"].to(device)
        tgt_kpm = batch["tgt_key_padding_mask"].to(device)

        tgt_inp = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_kpm_inp = tgt_kpm[:, :-1]
        masks = build_transformer_masks(
            src=src,
            tgt_inp=tgt_inp,
            src_key_padding_mask=src_kpm,
            tgt_key_padding_mask=tgt_kpm_inp,
        )
        logits = model(src=src, tgt_inp=tgt_inp, masks=masks)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        losses.append(float(loss.item()))
    return float(sum(losses) / max(1, len(losses)))


@dataclass(frozen=True)
class DecodeDiagnostics:
    avg_entropy: float
    steps: int


def _entropy(probs: torch.Tensor) -> float:
    p = probs.clamp_min(1e-12)
    return float((-p * torch.log(p)).sum().item())


@torch.no_grad()
def translate_sentence(
    *,
    model: Seq2SeqTransformer,
    processor: TextProcessor,
    sentence_en: str,
    device: torch.device,
    max_len: int,
    debug: bool = False,
) -> Tuple[str, DecodeDiagnostics]:
    model.eval()

    src_ids = processor.encode_src(sentence_en)
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_kpm = src.eq(processor.src_vocab.pad_id)
    memory = model.encode(src=src, src_key_padding_mask=src_kpm)

    ys = torch.tensor([[processor.tgt_vocab.sos_id]], dtype=torch.long, device=device)
    entropies: List[float] = []

    for step in range(max_len - 1):
        tgt_kpm = ys.eq(processor.tgt_vocab.pad_id)
        tgt_mask = generate_square_subsequent_mask(int(ys.size(1)), device=device)
        out = model.decode(
            tgt_inp=ys,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=src_kpm,
        )
        logits = model.generator(out[:, -1, :])

        logits[:, processor.tgt_vocab.pad_id] = float("-inf")
        logits[:, processor.tgt_vocab.sos_id] = float("-inf")
        if step == 0:
            logits[:, processor.tgt_vocab.eos_id] = float("-inf")

        probs = torch.softmax(logits[0], dim=-1)
        ent = _entropy(probs)
        entropies.append(ent)

        next_id = int(torch.argmax(logits, dim=-1).item())

        if debug and step < 8:
            top_k = min(5, logits.size(-1))
            top_logits, top_ids = torch.topk(logits[0], top_k)
            top_probs = torch.softmax(top_logits, dim=0).tolist()
            top_tokens = [processor.tgt_vocab.id_to_token[int(i)] for i in top_ids]
            print(
                f"  Step {step}: entropy={ent:.4f} top_tokens={top_tokens}, "
                f"relative_probs={[round(float(p), 4) for p in top_probs]}"
            )

        ys = torch.cat([ys, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if next_id == processor.tgt_vocab.eos_id:
            break

    pred = processor.decode_tgt(ys.squeeze(0).tolist(), remove_special=True)
    avg_ent = float(sum(entropies) / max(1, len(entropies)))
    return pred, DecodeDiagnostics(avg_entropy=avg_ent, steps=len(entropies))


def _highlight_diff(pred: str, ref: str) -> str:
    if pred == ref:
        return pred
    import difflib

    sm = difflib.SequenceMatcher(a=pred, b=ref)
    out: List[str] = []
    for tag, i1, i2, _, _ in sm.get_opcodes():
        chunk = pred[i1:i2]
        if not chunk:
            continue
        if tag == "equal":
            out.append(chunk)
        else:
            out.append(f"<<{chunk}>>")
    return "".join(out)


def append_translation_log(
    *,
    log_path: Path,
    epoch: int,
    device: torch.device,
    tests: List[str],
    model: Seq2SeqTransformer,
    processor: TextProcessor,
    max_len: int,
    train_ref: Dict[str, str],
    entropy_warn_threshold: float,
    debug: bool,
) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] epoch={epoch} device={device}\n")
        for en in tests:
            ref = train_ref.get(en, "")
            pred, diag = translate_sentence(
                model=model,
                processor=processor,
                sentence_en=en,
                device=device,
                max_len=max_len,
                debug=debug,
            )
            highlighted = _highlight_diff(pred, ref) if ref else pred
            f.write(f"[{en}] | [{highlighted}] | [{ref}]\n")
            if ref and pred != ref and diag.avg_entropy < float(entropy_warn_threshold):
                f.write(
                    f"  WARN: low_entropy_bias? avg_entropy={diag.avg_entropy:.4f} steps={diag.steps}\n"
                )
        f.write("\n")


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float
    best: float = float("inf")
    bad_epochs: int = 0

    def update(self, val_loss: float) -> bool:
        if val_loss < self.best - float(self.min_delta):
            self.best = float(val_loss)
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= int(self.patience)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train + diagnostics harness (v4)")
    p.add_argument("--train-tsv", default="../data/tatoeba_en_zh_train.tsv")
    p.add_argument("--val-tsv", default="../data/tatoeba_en_zh_val.tsv")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--logdir", default="auto")
    p.add_argument("--translation-log", default="log_v4.txt")
    p.add_argument("--entropy-warn", type=float, default=0.8)
    p.add_argument("--early-patience", type=int, default=3)
    p.add_argument("--early-min-delta", type=float, default=1e-4)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--translation-debug", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(int(args.seed))
    device = pick_device(str(args.device))

    train_pairs: List[Pair] = read_tsv_pairs(Path(args.train_tsv), max_rows=None)
    val_pairs: List[Pair] = read_tsv_pairs(Path(args.val_tsv), max_rows=None)
    processor = TextProcessor.from_sentence_pairs(
        train_pairs,
        max_seq_len=MAX_SEQ_LEN,
        min_freq=MIN_FREQ,
    )

    train_ds = ParallelTextDataset(train_pairs, processor)
    val_ds = ParallelTextDataset(val_pairs, processor)

    train_loader = processor.build_dataloader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
    )
    val_loader = processor.build_dataloader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )

    if device.type == "cuda":
        idx = int(device.index or 0)
        print(f"Device: {device} | name={torch.cuda.get_device_name(idx)} | count={torch.cuda.device_count()}")
    else:
        print(f"Device: {device}")
    print(f"Train rows: {len(train_pairs)} | Val rows: {len(val_pairs)}")
    print(f"Vocab: src={len(processor.src_vocab)} tgt={len(processor.tgt_vocab)}")

    model = Seq2SeqTransformer(
        src_vocab_size=len(processor.src_vocab),
        tgt_vocab_size=len(processor.tgt_vocab),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_len=MAX_SEQ_LEN,
        pad_id_src=processor.src_vocab.pad_id,
        pad_id_tgt=processor.tgt_vocab.pad_id,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=processor.tgt_vocab.pad_id)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(LR),
        weight_decay=float(WEIGHT_DECAY),
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    train_ref: Dict[str, str] = {}
    for en, zh in train_pairs:
        train_ref.setdefault(en, zh)

    tests = [
        "I am a student.",
        "I am a doctor.",
        "Tom likes rice.",
        "Lucy eats noodles.",
        "Where is the library?",
        "Where is the bathroom?",
        "I want pizza.",
        "They like dumplings.",
        "We want soup.",
        "Mary is a teacher.",
    ]

    run_name = datetime.now().strftime("mt4_%Y%m%d_%H%M%S")
    writer = build_writer(logdir=str(args.logdir), run_name=run_name)
    log_path = (Path(__file__).resolve().parent / str(args.translation_log)).resolve()

    early = EarlyStopping(patience=int(args.early_patience), min_delta=float(args.early_min_delta))

    if not bool(args.skip_train):
        for epoch in range(1, int(args.epochs) + 1):
            losses: List[float] = []
            for step, batch in enumerate(train_loader, start=1):
                loss = train_step(
                    model=model,
                    batch=batch,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    device=device,
                )
                losses.append(loss)
                if step == 1 or step == len(train_loader):
                    print(f"  Epoch {epoch:02d} step {step:03d}/{len(train_loader):03d} loss={loss:.4f}")

            train_loss = float(sum(losses) / max(1, len(losses)))
            val_loss = eval_loss(model=model, loader=val_loader, loss_fn=loss_fn, device=device)
            print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

            if writer is not None:
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("loss/val", val_loss, epoch)
                writer.flush()

            append_translation_log(
                log_path=log_path,
                epoch=epoch,
                device=device,
                tests=tests,
                model=model,
                processor=processor,
                max_len=MAX_SEQ_LEN,
                train_ref=train_ref,
                entropy_warn_threshold=float(args.entropy_warn),
                debug=bool(args.translation_debug),
            )

            if early.update(val_loss):
                print(
                    f"Early stopping at epoch={epoch} (best_val={early.best:.4f}, "
                    f"bad_epochs={early.bad_epochs}, min_delta={early.min_delta})"
                )
                break

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()

