from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(args: list[str], *, cwd: Path) -> None:
    print("\n$ " + " ".join(args))
    proc = subprocess.run(args, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    root = Path(__file__).resolve().parent
    train_tsv = root / ".." / "data" / "tatoeba_en_zh_train.tsv"
    val_tsv = root / ".." / "data" / "tatoeba_en_zh_val.tsv"

    _run(
        [
            sys.executable,
            "generate_data_4.py",
            "--out-train",
            str(train_tsv),
            "--out-val",
            str(val_tsv),
            "--target-rows",
            "6000",
        ],
        cwd=root,
    )

    _run(
        [
            sys.executable,
            "debug_data_2.py",
            "--train-tsv",
            str(train_tsv),
            "--val-tsv",
            str(val_tsv),
            "--keyword-threshold",
            "5",
        ],
        cwd=root,
    )

    _run(
        [
            sys.executable,
            "train_mt_4.py",
            "--train-tsv",
            str(train_tsv),
            "--val-tsv",
            str(val_tsv),
            "--device",
            "auto",
            "--epochs",
            "8",
            "--early-patience",
            "3",
            "--early-min-delta",
            "0.0001",
            "--entropy-warn",
            "0.8",
            "--translation-log",
            "log_v4.txt",
        ],
        cwd=root,
    )

    _run(
        [
            sys.executable,
            "debug_data_1.py",
            "--train-tsv",
            str(train_tsv),
            "--val-tsv",
            str(val_tsv),
            "--keyword-threshold",
            "5",
        ],
        cwd=root,
    )

    print("\nHarness done.")
    print(f"train_tsv: {train_tsv.resolve()}")
    print(f"val_tsv:   {val_tsv.resolve()}")
    print(f"log:       {(root / 'log_v4.txt').resolve()}")
    print(f"runs:      {(root / 'runs').resolve()}")


if __name__ == "__main__":
    main()

