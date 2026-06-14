from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from config import MAX_SEQ_LEN, MIN_FREQ
from data import read_tsv_pairs
from text_processor import TextProcessor, tokenize_en, tokenize_zh


Pair = Tuple[str, str]


@dataclass(frozen=True)
class CoverageReport:
    src_vocab_size: int
    tgt_vocab_size: int
    train_rows: int
    val_rows: int
    src_unk_ratio: float
    tgt_unk_ratio: float


def _count_unk_ratio(*, processor: TextProcessor, pairs: Sequence[Pair]) -> Tuple[float, float]:
    src_total = 0
    src_unk = 0
    tgt_total = 0
    tgt_unk = 0
    for en, zh in pairs:
        src_ids = processor.encode_src(en)
        tgt_ids = processor.encode_tgt(zh)
        src_total += len(src_ids)
        tgt_total += len(tgt_ids)
        src_unk += sum(1 for i in src_ids if int(i) == processor.src_vocab.unk_id)
        tgt_unk += sum(1 for i in tgt_ids if int(i) == processor.tgt_vocab.unk_id)
    return (src_unk / max(1, src_total), tgt_unk / max(1, tgt_total))


def _warn_low_freq_keywords(
    *,
    processor: TextProcessor,
    train_pairs: Sequence[Pair],
    keywords_zh: Sequence[str],
    threshold: int,
) -> None:
    counts: Counter[str] = Counter()
    for _, zh in train_pairs:
        counts.update(tokenize_zh(zh))

    bad: List[Tuple[str, int, bool]] = []
    for kw in keywords_zh:
        c = int(counts.get(kw, 0))
        in_vocab = kw in processor.tgt_vocab.token_to_id
        if c < int(threshold):
            bad.append((kw, c, in_vocab))

    if bad:
        print("\n⚠️  Low-frequency keyword warning (tgt/train)")
        for kw, c, in_vocab in sorted(bad, key=lambda x: (x[1], x[0])):
            print(f"  token='{kw}' count={c} in_vocab={in_vocab}")


def _stress_tests(*, processor: TextProcessor, seen_en: set[str], seen_zh: set[str]) -> None:
    tests: List[Pair] = []
    tests.append(
        (
            "I want to eat rice and noodles and dumplings and bread and pizza and soup and salad every day because I am very hungry.",
            "我想吃米饭面条饺子面包披萨汤沙拉，因为我今天非常饿。",
        )
    )
    tests.append(
        (
            "Tom is a student in 2026 and he likes zzzzq-unknown-token.",
            "汤姆是学生，他喜欢未知词zzzzq。",
        )
    )
    tests.append(
        (
            "Where is the library near the subway station?",
            "地铁站附近的图书馆在哪里？",
        )
    )
    tests.append(
        (
            "Mary is a doctor and she wants pizza, but she is busy.",
            "玛丽是医生，她想要披萨，但是她很忙。",
        )
    )
    tests.append(
        (
            "They want ice cream, and we want soup.",
            "他们想要冰淇淋，我们想要汤。",
        )
    )

    print("\n== Stress Tests (encoding stability) ==")
    for idx, (en, zh) in enumerate(tests, start=1):
        tag = []
        if en in seen_en:
            tag.append("EN_seen")
        if zh in seen_zh:
            tag.append("ZH_seen")
        try:
            src_ids = processor.encode_src(en)
            tgt_ids = processor.encode_tgt(zh)
            src_unk = sum(1 for i in src_ids if int(i) == processor.src_vocab.unk_id)
            tgt_unk = sum(1 for i in tgt_ids if int(i) == processor.tgt_vocab.unk_id)
            print(
                f"  [{idx}] tags={','.join(tag) or 'none'} | "
                f"src_len={len(src_ids)} tgt_len={len(tgt_ids)} | "
                f"src_unk={src_unk} tgt_unk={tgt_unk} | max_seq_len={MAX_SEQ_LEN}"
            )
        except Exception as e:
            print(f"  [{idx}] ERROR: {type(e).__name__}: {e}")


def run(*, train_tsv: Path, val_tsv: Path, keyword_threshold: int = 5) -> CoverageReport:
    train_pairs = read_tsv_pairs(train_tsv, max_rows=None)
    val_pairs = read_tsv_pairs(val_tsv, max_rows=None)
    processor = TextProcessor.from_sentence_pairs(
        train_pairs,
        max_seq_len=MAX_SEQ_LEN,
        min_freq=MIN_FREQ,
    )

    src_unk_ratio, tgt_unk_ratio = _count_unk_ratio(processor=processor, pairs=train_pairs + val_pairs)

    keywords_zh = [
        "学生",
        "老师",
        "医生",
        "工程师",
        "设计师",
        "厨师",
        "司机",
        "护士",
        "米饭",
        "面条",
        "饺子",
        "面包",
        "披萨",
        "汤",
        "沙拉",
        "冰淇淋",
    ]
    _warn_low_freq_keywords(
        processor=processor,
        train_pairs=train_pairs,
        keywords_zh=keywords_zh,
        threshold=keyword_threshold,
    )

    src_vocab = processor.src_vocab
    tgt_vocab = processor.tgt_vocab
    print("\n== Vocab Summary ==")
    print(f"src_vocab_size={len(src_vocab)} tgt_vocab_size={len(tgt_vocab)}")
    print(f"src_special={src_vocab.id_to_token[:4]}")
    print(f"tgt_special={tgt_vocab.id_to_token[:4]}")
    print(f"unk_ratio: src={src_unk_ratio:.4f} tgt={tgt_unk_ratio:.4f}")

    seen_en = {en for en, _ in train_pairs + val_pairs}
    seen_zh = {zh for _, zh in train_pairs + val_pairs}
    _stress_tests(processor=processor, seen_en=seen_en, seen_zh=seen_zh)

    return CoverageReport(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        train_rows=len(train_pairs),
        val_rows=len(val_pairs),
        src_unk_ratio=float(src_unk_ratio),
        tgt_unk_ratio=float(tgt_unk_ratio),
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Deep diagnostics: vocab coverage + stress tests")
    p.add_argument("--train-tsv", default="../data/tatoeba_en_zh_train.tsv")
    p.add_argument("--val-tsv", default="../data/tatoeba_en_zh_val.tsv")
    p.add_argument("--keyword-threshold", type=int, default=5)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report = run(
        train_tsv=Path(args.train_tsv),
        val_tsv=Path(args.val_tsv),
        keyword_threshold=int(args.keyword_threshold),
    )
    print("\n== Report ==")
    print(report)


if __name__ == "__main__":
    main()

