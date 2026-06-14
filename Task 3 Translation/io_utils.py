from __future__ import annotations

from pathlib import Path


def ensure_file(path: str | Path, *, hint: str = "") -> Path:
    p = Path(path)
    if p.exists() and p.is_file():
        return p
    msg = f"File not found: {p.resolve()}"
    if hint:
        msg = msg + f"\n{hint}"
    raise FileNotFoundError(msg)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    if p.exists() and p.is_dir():
        return p
    raise FileNotFoundError(f"Directory not found: {p.resolve()}")

