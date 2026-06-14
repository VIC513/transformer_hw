from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen


def download_and_prepare(
    *,
    out_tsv: str | Path = "../data/tatoeba_en_zh.tsv",
    max_rows: int = 500000,
) -> Path:
    out_path = Path(out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    url = "https://www.manythings.org/anki/cmn-eng.zip"
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        },
    )
    with urlopen(req, timeout=60) as resp:
        data = resp.read()

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        candidates = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        if not candidates:
            raise FileNotFoundError("No .txt found in zip")
        name = candidates[0]
        raw = zf.read(name).decode("utf-8")

    n = 0
    with out_path.open("w", encoding="utf-8") as w:
        for line in raw.splitlines():
            cols = line.split("\t")
            if len(cols) < 2:
                continue
            zh = cols[0].strip()
            en = cols[1].strip()
            if not zh or not en:
                continue
            w.write(en.replace("\t", " ") + "\t" + zh.replace("\t", " ") + "\n")
            n += 1
            if n >= int(max_rows):
                break
    return out_path


def main() -> None:
    p = download_and_prepare()
    print(f"Prepared: {p.resolve()}")


if __name__ == "__main__":
    main()
