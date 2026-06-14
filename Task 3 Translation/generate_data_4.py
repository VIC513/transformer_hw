from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


Pair = Tuple[str, str]
Signature = Tuple[str, ...]


@dataclass(frozen=True)
class Sample:
    en: str
    zh: str
    signature: Signature
    meta: Dict[str, str]


def _clean_text(s: str) -> str:
    s = s.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _has_leftover_placeholder(s: str) -> bool:
    return "{" in s or "}" in s


def _has_weird_repetition(s: str) -> bool:
    if not s:
        return True
    if re.search(r"(.)\1\1\1\1", s):
        return True
    tokens = s.split()
    run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            run += 1
            if run >= 4:
                return True
        else:
            run = 1
    return False


def _looks_like_en(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9 ,.!?'\-]+", s))


def _looks_like_zh(s: str) -> bool:
    return bool(re.fullmatch(r"[\u4e00-\u9fff0-9，。！？；：、（）()“”\- ]+", s))


def _is_bad_sample(en: str, zh: str) -> str | None:
    en2, zh2 = _clean_text(en), _clean_text(zh)
    if not en2 or not zh2:
        return "empty"
    if len(en2) < 2 or len(zh2) < 1:
        return "too_short"
    if _has_leftover_placeholder(en2) or _has_leftover_placeholder(zh2):
        return "leftover_placeholder"
    if _has_weird_repetition(en2) or _has_weird_repetition(zh2):
        return "repetition"
    if not _looks_like_en(en2):
        return "bad_en_charset"
    if not _looks_like_zh(zh2):
        return "bad_zh_charset"
    return None


def _dedup(samples: Iterable[Sample]) -> Tuple[List[Sample], Dict[str, int]]:
    seen: set[Tuple[str, str, Signature]] = set()
    out: List[Sample] = []
    removed: Dict[str, int] = {}

    for s in samples:
        en = _clean_text(s.en)
        zh = _clean_text(s.zh)
        reason = _is_bad_sample(en, zh)
        if reason is not None:
            removed[reason] = removed.get(reason, 0) + 1
            continue
        key = (en, zh, s.signature)
        if key in seen:
            removed["duplicate"] = removed.get("duplicate", 0) + 1
            continue
        seen.add(key)
        out.append(Sample(en=en, zh=zh, signature=s.signature, meta=s.meta))
    return out, removed


def _histogram(title: str, counts: Dict[str, int], *, width: int = 20, top_n: int = 12) -> None:
    if not counts:
        print(f"{title}: (empty)")
        return
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    max_v = max(v for _, v in items)
    tail = sum(1 for _, v in items if v <= 2)
    print(f"\n== {title} ==")
    print(f"Total unique: {len(items)} | Max freq: {max_v} | Tail(<=2): {tail}")
    for k, v in items[:top_n]:
        bar_len = int(round((v / max_v) * width)) if max_v > 0 else 0
        bar = "#" * bar_len
        print(f"{k:<16} | {bar:<{width}} | {v}")


def build_slots() -> Dict[str, List[Tuple[str, str]]]:
    subjects = [
        ("I", "我"),
        ("You", "你"),
        ("We", "我们"),
        ("They", "他们"),
        ("Tom", "汤姆"),
        ("Lucy", "露西"),
        ("John", "约翰"),
        ("Mary", "玛丽"),
        ("David", "大卫"),
        ("Anna", "安娜"),
        ("Mike", "迈克"),
        ("Emma", "艾玛"),
    ]
    occupations = [
        ("a student", "学生"),
        ("a teacher", "老师"),
        ("a doctor", "医生"),
        ("an engineer", "工程师"),
        ("a designer", "设计师"),
        ("a cook", "厨师"),
        ("a driver", "司机"),
        ("a nurse", "护士"),
    ]
    foods = [
        ("rice", "米饭"),
        ("noodles", "面条"),
        ("dumplings", "饺子"),
        ("bread", "面包"),
        ("pizza", "披萨"),
        ("soup", "汤"),
        ("salad", "沙拉"),
        ("ice cream", "冰淇淋"),
    ]
    places = [
        ("the bathroom", "洗手间"),
        ("the restaurant", "餐厅"),
        ("the library", "图书馆"),
        ("the subway station", "地铁站"),
        ("the hospital", "医院"),
        ("the supermarket", "超市"),
    ]
    return {
        "subjects": subjects,
        "occupations": occupations,
        "foods": foods,
        "places": places,
    }


def _id_of(en: str) -> str:
    return re.sub(r"\s+", "_", en.strip().lower())


def generate_samples(*, seed: int = 42) -> List[Sample]:
    rng = random.Random(seed)
    slots = build_slots()
    subjects = slots["subjects"]
    occupations = slots["occupations"]
    foods = slots["foods"]
    places = slots["places"]

    samples: List[Sample] = []

    for sub_en, sub_zh in subjects:
        for occ_en, occ_zh in occupations:
            template_id = "SVO_be_occupation"
            en = f"{sub_en} am {occ_en}." if sub_en == "I" else f"{sub_en} are {occ_en}." if sub_en in {"You", "We", "They"} else f"{sub_en} is {occ_en}."
            zh = f"{sub_zh}是{occ_zh}。"
            signature: Signature = (template_id, _id_of(sub_en), _id_of("be"), _id_of(occ_en))
            meta = {
                "template_id": template_id,
                "subject": sub_en,
                "occupation": occ_en,
            }
            samples.append(Sample(en=en, zh=zh, signature=signature, meta=meta))

    verbs = [
        ("like", "喜欢"),
        ("eat", "吃"),
        ("want", "想要"),
    ]
    objects = foods

    for sub_en, sub_zh in subjects:
        for v_en, v_zh in verbs:
            for obj_en, obj_zh in objects:
                template_id = f"SVO_{v_en}_food"
                if v_en == "want":
                    en = f"{sub_en} want {obj_en}." if sub_en != "I" else f"I want {obj_en}."
                    zh = f"{sub_zh}想要{obj_zh}。"
                    verb_id = "want"
                else:
                    en = f"{sub_en} {v_en} {obj_en}."
                    zh = f"{sub_zh}{v_zh}{obj_zh}。"
                    verb_id = v_en
                signature = (template_id, _id_of(sub_en), _id_of(verb_id), _id_of(obj_en))
                meta = {
                    "template_id": template_id,
                    "subject": sub_en,
                    "verb": verb_id,
                    "food": obj_en,
                }
                samples.append(Sample(en=en, zh=zh, signature=signature, meta=meta))

    for place_en, place_zh in places:
        template_id = "Q_where_is_place"
        en = f"Where is {place_en}?"
        zh = f"{place_zh}在哪里？"
        signature = (template_id, _id_of(place_en),)
        meta = {
            "template_id": template_id,
            "place": place_en,
        }
        samples.append(Sample(en=en, zh=zh, signature=signature, meta=meta))

    rng.shuffle(samples)
    return samples


def split_disjoint_by_signature(
    *,
    samples: Sequence[Sample],
    train_ratio: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample]]:
    sig2: Dict[Signature, List[Sample]] = {}
    for s in samples:
        sig2.setdefault(s.signature, []).append(s)

    sigs = list(sig2.keys())
    rng = random.Random(seed)
    rng.shuffle(sigs)

    n_train = max(1, int(len(sigs) * float(train_ratio)))
    train_sigs = set(sigs[:n_train])
    val_sigs = set(sigs[n_train:])

    if not train_sigs.isdisjoint(val_sigs):
        overlap = list(train_sigs.intersection(val_sigs))[:5]
        raise RuntimeError(f"Signature sets are not disjoint. overlap={overlap}")

    train: List[Sample] = []
    val: List[Sample] = []
    for sig, group in sig2.items():
        (train if sig in train_sigs else val).extend(group)

    tset = {s.signature for s in train}
    vset = {s.signature for s in val}
    if not tset.isdisjoint(vset):
        overlap = list(tset.intersection(vset))[:5]
        raise RuntimeError(f"Sample-level signature overlap detected. overlap={overlap}")

    print(f"Disjoint signatures: train={len(train_sigs)} val={len(val_sigs)}")
    print(f"Samples: train={len(train)} val={len(val)}")
    return train, val


def materialize(
    *,
    train: Sequence[Sample],
    val: Sequence[Sample],
    target_total_rows: int,
    seed: int,
) -> Tuple[List[Pair], List[Pair]]:
    rng = random.Random(seed)
    train_pairs = [(s.en, s.zh) for s in train]
    val_pairs = [(s.en, s.zh) for s in val]
    if len(train_pairs) == 0 or len(val_pairs) == 0:
        raise RuntimeError("Empty split; adjust train_ratio or templates.")

    total = max(int(target_total_rows), 5000)
    target_train = int(total * 0.98)
    target_val = total - target_train

    def _expand(pairs: List[Pair], target: int) -> List[Pair]:
        if len(pairs) >= target:
            return pairs[:target]
        out = list(pairs)
        while len(out) < target:
            en, zh = pairs[rng.randrange(0, len(pairs))]
            out.append((en, zh))
        return out

    return _expand(train_pairs, target_train), _expand(val_pairs, target_val)


def write_tsv(pairs: Sequence[Pair], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for en, zh in pairs:
            f.write(f"{_clean_text(en)}\t{_clean_text(zh)}\n")
    return path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate disjoint-combination synthetic EN-ZH data")
    p.add_argument("--out-train", default="../data/tatoeba_en_zh_train.tsv")
    p.add_argument("--out-val", default="../data/tatoeba_en_zh_val.tsv")
    p.add_argument("--target-rows", type=int, default=6000)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    raw = generate_samples(seed=int(args.seed))
    cleaned, removed = _dedup(raw)
    print(f"Generated raw samples: {len(raw)}")
    print(f"Cleaned samples: {len(cleaned)}")
    if removed:
        top = sorted(removed.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        print("Removed (top reasons): " + ", ".join([f"{k}={v}" for k, v in top]))

    train_s, val_s = split_disjoint_by_signature(
        samples=cleaned,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
    )

    subj_counts: Dict[str, int] = {}
    occ_counts: Dict[str, int] = {}
    food_counts: Dict[str, int] = {}
    for s in cleaned:
        if "subject" in s.meta:
            subj_counts[s.meta["subject"]] = subj_counts.get(s.meta["subject"], 0) + 1
        if "occupation" in s.meta:
            occ_counts[s.meta["occupation"]] = occ_counts.get(s.meta["occupation"], 0) + 1
        if "food" in s.meta:
            food_counts[s.meta["food"]] = food_counts.get(s.meta["food"], 0) + 1

    _histogram("subjects", subj_counts)
    _histogram("occupations", occ_counts)
    _histogram("foods", food_counts)

    train_pairs, val_pairs = materialize(
        train=train_s,
        val=val_s,
        target_total_rows=int(args.target_rows),
        seed=int(args.seed),
    )

    out_train = write_tsv(train_pairs, Path(args.out_train))
    out_val = write_tsv(val_pairs, Path(args.out_val))
    print(f"\nWrote train TSV: {out_train.resolve()} ({len(train_pairs)} rows)")
    print(f"Wrote val TSV:   {out_val.resolve()} ({len(val_pairs)} rows)")


if __name__ == "__main__":
    main()

