from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
    TRAIN_SPLIT,
    WEIGHT_DECAY,
    DataPaths,
)
from data import build_datasets, read_tsv_pairs
from model import Seq2SeqTransformer, build_transformer_masks, generate_square_subsequent_mask


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_pairs_path(path: str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return (Path(__file__).resolve().parent / raw_path).resolve()


def summarize_pairs(pairs: List[tuple[str, str]], limit: int = 3) -> None:
    preview = pairs[:limit]
    if not preview:
        return
    print("Sample pairs:")
    for idx, (en, zh) in enumerate(preview, start=1):
        print(f"  [{idx}] EN: {en}")
        print(f"      ZH: {zh}")


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


@torch.no_grad()
def translate_sentence(
    *,
    model: Seq2SeqTransformer,
    processor,
    sentence_en: str,
    device: torch.device,
    max_len: int,
    debug: bool = False,
) -> str:
    model.eval()

    src_ids = processor.encode_src(sentence_en)
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_kpm = src.eq(processor.src_vocab.pad_id)
    memory = model.encode(src=src, src_key_padding_mask=src_kpm)

    ys = torch.tensor([[processor.tgt_vocab.sos_id]], dtype=torch.long, device=device)

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

        # Avoid selecting special tokens as normal output tokens.
        logits[:, processor.tgt_vocab.pad_id] = float("-inf")
        logits[:, processor.tgt_vocab.sos_id] = float("-inf")
        if step == 0:
            logits[:, processor.tgt_vocab.eos_id] = float("-inf")

        next_id = int(torch.argmax(logits, dim=-1).item())

        if debug and step < 5:
            top_k = min(5, logits.size(-1))
            top_logits, top_ids = torch.topk(logits[0], top_k)
            top_probs = torch.softmax(top_logits, dim=0).tolist()
            top_tokens = [processor.tgt_vocab.id_to_token[int(id_)] for id_ in top_ids]
            print(
                f"  Step {step}: top_tokens={top_tokens}, "
                f"relative_probs={[round(float(p), 4) for p in top_probs]}"
            )

        ys = torch.cat(
            [ys, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )
        if next_id == processor.tgt_vocab.eos_id:
            break

    return processor.decode_tgt(ys.squeeze(0).tolist(), remove_special=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a simple EN->ZH transformer.")
    parser.add_argument("--pairs-tsv", default=DataPaths().pairs_tsv, help="Path to EN-ZH TSV data.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit the number of pairs for fast debugging. Default: use all rows.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto | cpu | cuda | cuda:N.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Use 0 for Windows debugging; try 2-8 for speed.",
    )
    parser.add_argument(
        "--logdir",
        default="auto",
        help="TensorBoard log directory. Use empty string to disable.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only inspect data/model setup plus sample translations.",
    )
    parser.add_argument(
        "--translation-debug",
        action="store_true",
        help="Print top token candidates during all translation test cases.",
    )
    parser.add_argument(
        "--translation-log",
        default="log.txt",
        help="Path to translation log file (appended each epoch).",
    )
    return parser


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
        try:
            idx = int(idx_str)
        except ValueError as e:
            raise ValueError(f"Invalid cuda device index: {device_arg}") from e
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


def append_translation_log(
    *,
    log_path: Path,
    epoch: int,
    train_loss: float | None,
    val_loss: float | None,
    device: torch.device,
    tests: List[str],
    model: Seq2SeqTransformer,
    processor,
    max_len: int,
) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = [
        f"[{stamp}] epoch={epoch} device={device}",
    ]
    if train_loss is not None:
        header.append(f"train_loss={train_loss:.6f}")
    if val_loss is not None:
        header.append(f"val_loss={val_loss:.6f}")
    text = " | ".join(header)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")
        for sent in tests:
            pred = translate_sentence(
                model=model,
                processor=processor,
                sentence_en=sent,
                device=device,
                max_len=max_len,
                debug=False,
            )
            f.write(f"EN: {sent}\n")
            f.write(f"ZH: {pred}\n")
        f.write("\n")


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    pairs_path = resolve_pairs_path(args.pairs_tsv)
    pairs = read_tsv_pairs(pairs_path, max_rows=args.max_rows)
    bundle = build_datasets(
        pairs,
        max_seq_len=MAX_SEQ_LEN,
        min_freq=MIN_FREQ,
        train_split=TRAIN_SPLIT,
        seed=args.seed,
    )

    if device.type == "cuda":
        idx = int(device.index or 0)
        print(f"Device: {device} | name={torch.cuda.get_device_name(idx)} | count={torch.cuda.device_count()}")
    else:
        print(f"Device: {device}")
    print(f"Data file: {pairs_path}")
    print(f"Loaded {len(pairs)} sentence pairs")
    print(f"Source vocab size: {len(bundle.processor.src_vocab)}")
    print(f"Target vocab size: {len(bundle.processor.tgt_vocab)}")
    print(f"Train set size: {len(bundle.train_ds)}")
    print(f"Validation set size: {len(bundle.val_ds)}")
    summarize_pairs(pairs)
    print()

    train_loader = bundle.processor.build_dataloader(
        bundle.train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = bundle.processor.build_dataloader(
        bundle.val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    first_batch = next(iter(train_loader))
    print(
        "First batch shapes: "
        f"src={tuple(first_batch['src'].shape)}, "
        f"tgt={tuple(first_batch['tgt'].shape)}"
    )

    model = Seq2SeqTransformer(
        src_vocab_size=len(bundle.processor.src_vocab),
        tgt_vocab_size=len(bundle.processor.tgt_vocab),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_len=MAX_SEQ_LEN,
        pad_id_src=bundle.processor.src_vocab.pad_id,
        pad_id_tgt=bundle.processor.tgt_vocab.pad_id,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=bundle.processor.tgt_vocab.pad_id)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(LR),
        weight_decay=float(WEIGHT_DECAY),
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    best_val = float("inf")
    ckpt_path = Path(__file__).resolve().parent / "best_mt_transformer.pth"
    run_name = datetime.now().strftime("mt_%Y%m%d_%H%M%S")
    writer = build_writer(logdir=str(args.logdir), run_name=run_name)
    log_path = (Path(__file__).resolve().parent / str(args.translation_log)).resolve()

    tests = [
        "I love you",
        "This is a book",
        "How are you",
        "I am a student",
        "What is your name",
        "Thank you",
        "Good morning",
        "See you tomorrow",
        "I want to eat",
        "Where is the bathroom",
    ]

    if not args.skip_train:
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
            print(
                f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"src_vocab={len(bundle.processor.src_vocab)} | tgt_vocab={len(bundle.processor.tgt_vocab)}"
            )

            if writer is not None:
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("loss/val", val_loss, epoch)
                writer.flush()

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "src_vocab": bundle.processor.src_vocab,
                        "tgt_vocab": bundle.processor.tgt_vocab,
                        "max_seq_len": bundle.processor.max_seq_len,
                    },
                    ckpt_path,
                )
                print(f"  Saved checkpoint to {ckpt_path}")

            append_translation_log(
                log_path=log_path,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                device=device,
                tests=tests,
                model=model,
                processor=bundle.processor,
                max_len=MAX_SEQ_LEN,
            )

        if writer is not None:
            writer.close()

    print("\n===== Greedy Translation Tests =====")
    for idx, sentence in enumerate(tests):
        debug_mode = args.translation_debug or idx == 0
        print(f"EN: {sentence}")
        if debug_mode:
            print("  [Debug mode enabled - showing top predictions]")
        pred = translate_sentence(
            model=model,
            processor=bundle.processor,
            sentence_en=sentence,
            device=device,
            max_len=MAX_SEQ_LEN,
            debug=debug_mode,
        )
        print(f"ZH: {pred}\n")

    append_translation_log(
        log_path=log_path,
        epoch=int(args.epochs) if not args.skip_train else 0,
        train_loss=None,
        val_loss=None,
        device=device,
        tests=tests,
        model=model,
        processor=bundle.processor,
        max_len=MAX_SEQ_LEN,
    )


if __name__ == "__main__":
    main()
