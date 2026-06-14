"""生成可训练的英中翻译数据 (TSV: EN<TAB>ZH)。

目标:
- 基础模板至少 200 对不同句子
- 通过同义词替换 / 主宾对调自动扩增变体
- 最终输出 tatoeba_en_zh.tsv 总行数 >= 5000
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

Pair = Tuple[str, str]


def _clean_text(s: str) -> str:
    s = s.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _dedup(pairs: Iterable[Pair]) -> List[Pair]:
    seen: set[Pair] = set()
    out: List[Pair] = []
    for en, zh in pairs:
        en2, zh2 = _clean_text(en), _clean_text(zh)
        if not en2 or not zh2:
            continue
        key = (en2, zh2)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _expand(
    *,
    en_tmpl: str,
    zh_tmpl: str,
    slots: Dict[str, Sequence[Tuple[str, str]]],
    limit: int | None,
    rng: random.Random,
) -> List[Pair]:
    keys = list(slots.keys())
    if not keys:
        return [(en_tmpl, zh_tmpl)]

    combos: List[Dict[str, Tuple[str, str]]] = [{}]
    for k in keys:
        values = list(slots[k])
        rng.shuffle(values)
        new_combos: List[Dict[str, Tuple[str, str]]] = []
        for base in combos:
            for v in values:
                d = dict(base)
                d[k] = v
                new_combos.append(d)
        combos = new_combos

    if limit is not None and len(combos) > int(limit):
        rng.shuffle(combos)
        combos = combos[: int(limit)]

    out: List[Pair] = []
    for m in combos:
        fmt_en = {k: v[0] for k, v in m.items()}
        fmt_zh = {k: v[1] for k, v in m.items()}
        out.append((en_tmpl.format(**fmt_en), zh_tmpl.format(**fmt_zh)))
    return out


def build_base_pairs(*, seed: int = 42) -> List[Pair]:
    rng = random.Random(seed)

    names = [
        ("Tom", "汤姆"),
        ("Lucy", "露西"),
        ("John", "约翰"),
        ("Mary", "玛丽"),
        ("David", "大卫"),
        ("Anna", "安娜"),
        ("Mike", "迈克"),
        ("Emma", "艾玛"),
        ("Peter", "彼得"),
        ("Linda", "琳达"),
        ("Robert", "罗伯特"),
        ("Sophia", "索菲亚"),
        ("James", "詹姆斯"),
        ("Olivia", "奥利维亚"),
        ("Daniel", "丹尼尔"),
        ("Grace", "格蕾丝"),
        ("Henry", "亨利"),
        ("Chloe", "克洛伊"),
        ("Kevin", "凯文"),
        ("Sarah", "莎拉"),
        ("Brian", "布莱恩"),
        ("Mia", "米娅"),
        ("Jason", "杰森"),
        ("Lily", "莉莉"),
        ("Chris", "克里斯"),
        ("Nancy", "南希"),
        ("Eric", "埃里克"),
        ("Alice", "爱丽丝"),
        ("Frank", "弗兰克"),
        ("Julia", "朱莉娅"),
    ]
    pronouns = [("I", "我"), ("You", "你"), ("We", "我们"), ("They", "他们")]
    subjects = pronouns + names

    foods = [
        ("rice", "米饭"),
        ("noodles", "面条"),
        ("dumplings", "饺子"),
        ("bread", "面包"),
        ("salad", "沙拉"),
        ("chicken", "鸡肉"),
        ("beef", "牛肉"),
        ("fish", "鱼"),
        ("an apple", "一个苹果"),
        ("a banana", "一根香蕉"),
        ("an orange", "一个橙子"),
        ("a sandwich", "一个三明治"),
        ("soup", "汤"),
        ("pizza", "披萨"),
        ("ice cream", "冰淇淋"),
        ("chocolate", "巧克力"),
        ("cake", "蛋糕"),
        ("breakfast", "早餐"),
        ("lunch", "午餐"),
        ("dinner", "晚餐"),
    ]
    drinks = [
        ("water", "水"),
        ("tea", "茶"),
        ("coffee", "咖啡"),
        ("milk", "牛奶"),
        ("juice", "果汁"),
        ("soda", "汽水"),
        ("hot chocolate", "热巧克力"),
        ("sparkling water", "苏打水"),
    ]
    places = [
        ("the bathroom", "洗手间"),
        ("the restroom", "卫生间"),
        ("the restaurant", "餐厅"),
        ("the subway station", "地铁站"),
        ("the hotel", "酒店"),
        ("the hospital", "医院"),
        ("the bank", "银行"),
        ("the library", "图书馆"),
        ("the supermarket", "超市"),
        ("the school", "学校"),
        ("the airport", "机场"),
        ("the police station", "派出所"),
        ("the post office", "邮局"),
        ("the bus stop", "公交站"),
        ("the park", "公园"),
        ("the museum", "博物馆"),
        ("the cinema", "电影院"),
        ("the coffee shop", "咖啡店"),
        ("the pharmacy", "药店"),
        ("the train station", "火车站"),
    ]
    items = [
        ("this", "这个"),
        ("that", "那个"),
        ("this book", "这本书"),
        ("this pen", "这支笔"),
        ("this phone", "这部手机"),
        ("a ticket", "一张票"),
        ("a coffee", "一杯咖啡"),
        ("a bottle of water", "一瓶水"),
        ("a bag", "一个包"),
        ("a charger", "一个充电器"),
        ("a map", "一张地图"),
        ("a table", "一张桌子"),
        ("a seat", "一个座位"),
    ]
    langs = [("Chinese", "中文"), ("English", "英文"), ("Japanese", "日文"), ("Korean", "韩文")]
    feelings = [("happy", "高兴"), ("tired", "累"), ("hungry", "饿"), ("thirsty", "渴"), ("busy", "忙")]
    times = [("seven", "七点"), ("eight", "八点"), ("nine", "九点"), ("ten", "十点"), ("five", "五点")]
    weekdays = [
        ("Monday", "周一"),
        ("Tuesday", "周二"),
        ("Wednesday", "周三"),
        ("Thursday", "周四"),
        ("Friday", "周五"),
        ("Saturday", "周六"),
        ("Sunday", "周日"),
    ]
    weather = [("sunny", "晴朗"), ("rainy", "下雨"), ("cloudy", "多云"), ("windy", "刮风"), ("cold", "冷"), ("hot", "热")]
    transport = [
        ("bus", "公交车"),
        ("subway", "地铁"),
        ("taxi", "出租车"),
        ("bike", "自行车"),
        ("car", "车"),
        ("train", "火车"),
        ("walk", "走路"),
    ]
    hobbies = [
        ("music", "音乐"),
        ("sports", "运动"),
        ("cooking", "做饭"),
        ("reading", "读书"),
        ("movies", "电影"),
        ("gaming", "打游戏"),
        ("traveling", "旅行"),
        ("photography", "摄影"),
        ("swimming", "游泳"),
        ("running", "跑步"),
    ]
    occupations = [("a student", "学生"), ("a teacher", "老师"), ("a doctor", "医生"), ("an engineer", "工程师"), ("a designer", "设计师")]

    manual_core: List[Pair] = [
        ("Hello", "你好"),
        ("Hi", "你好"),
        ("Good morning", "早上好"),
        ("Good afternoon", "下午好"),
        ("Good evening", "晚上好"),
        ("Good night", "晚安"),
        ("Thank you", "谢谢"),
        ("Thanks", "谢谢"),
        ("You're welcome", "不客气"),
        ("Excuse me", "打扰一下"),
        ("I'm sorry", "对不起"),
        ("No problem", "没问题"),
        ("I don't understand", "我不明白"),
        ("Could you repeat that", "你能再说一遍吗"),
        ("Please speak slowly", "请说慢一点"),
        ("Nice to meet you", "很高兴认识你"),
        ("See you tomorrow", "明天见"),
        ("See you later", "待会见"),
        ("What is your name", "你叫什么名字"),
        ("My name is Tom", "我叫汤姆"),
    ]

    templated: List[Pair] = []
    templated += _expand(
        en_tmpl="How are you?",
        zh_tmpl="你好吗？",
        slots={},
        limit=None,
        rng=rng,
    )
    templated += _expand(
        en_tmpl="{subj} like {hobby}.",
        zh_tmpl="{subj}喜欢{hobby}。",
        slots={"subj": subjects, "hobby": hobbies},
        limit=None,
        rng=rng,
    )
    templated += _expand(
        en_tmpl="{subj} want to eat {food}.",
        zh_tmpl="{subj}想吃{food}。",
        slots={"subj": pronouns, "food": foods},
        limit=None,
        rng=rng,
    )
    templated += _expand(
        en_tmpl="{subj} want to drink {drink}.",
        zh_tmpl="{subj}想喝{drink}。",
        slots={"subj": pronouns, "drink": drinks},
        limit=None,
        rng=rng,
    )
    templated += _expand(
        en_tmpl="Where is {place}?",
        zh_tmpl="{place}在哪里？",
        slots={"place": places},
        limit=None,
        rng=rng,
    )
    templated += _expand(
        en_tmpl="How much is {item}?",
        zh_tmpl="{item}多少钱？",
        slots={"item": items},
        limit=None,
        rng=rng,
    )
    templated += _expand(
        en_tmpl="Do you speak {lang}?",
        zh_tmpl="你会说{lang}吗？",
        slots={"lang": langs},
        limit=None,
        rng=rng,
    )
    templated += _expand(
        en_tmpl="{subj} am learning {lang}.",
        zh_tmpl="{subj}在学{lang}。",
        slots={"subj": [("I", "我"), ("We", "我们")], "lang": langs},
        limit=None,
        rng=rng,
    )
    templated += _expand(
        en_tmpl="{lang} is difficult.",
        zh_tmpl="{lang}很难。",
        slots={"lang": langs},
        limit=None,
        rng=rng,
    )
    templated += _expand(
        en_tmpl="{subj} feel {adj} today.",
        zh_tmpl="{subj}今天觉得{adj}。",
        slots={"subj": subjects, "adj": feelings},
        limit=None,
        rng=rng,
    )
    templated += _expand(
        en_tmpl="What time is it?",
        zh_tmpl="现在几点？",
        slots={},
        limit=None,
        rng=rng,
    )
    templated += _expand(
        en_tmpl="It is {time} o'clock.",
        zh_tmpl="现在{time}。",
        slots={"time": times},
        limit=None,
        rng=rng,
    )
    templated += _expand(
        en_tmpl="Today is {day}.",
        zh_tmpl="今天是{day}。",
        slots={"day": weekdays},
        limit=None,
        rng=rng,
    )
    templated += _expand(
        en_tmpl="It is {w} today.",
        zh_tmpl="今天{w}。",
        slots={"w": weather},
        limit=None,
        rng=rng,
    )
    templated += _expand(
        en_tmpl="{subj} go to {place} by {t}.",
        zh_tmpl="{subj}坐{t}去{place}。",
        slots={"subj": pronouns, "place": places, "t": transport},
        limit=None,
        rng=rng,
    )

    templated += _expand(
        en_tmpl="{subj} need to go to {place}.",
        zh_tmpl="{subj}得去{place}。",
        slots={"subj": pronouns, "place": places},
        limit=None,
        rng=rng,
    )

    templated += _expand(
        en_tmpl="{subj} would like {item}.",
        zh_tmpl="{subj}想要{item}。",
        slots={"subj": pronouns, "item": items},
        limit=None,
        rng=rng,
    )
    templated += _expand(
        en_tmpl="{subj} am {job}.",
        zh_tmpl="{subj}是{job}。",
        slots={"subj": [("I", "我"), ("You", "你"), ("We", "我们")], "job": occupations},
        limit=None,
        rng=rng,
    )

    swap_names = names[:]
    swap_pairs: List[Pair] = []
    for a_en, a_zh in swap_names:
        for b_en, b_zh in swap_names:
            if a_en == b_en:
                continue
            swap_pairs.append((f"{a_en} helps {b_en}.", f"{a_zh}帮助{b_zh}。"))
            swap_pairs.append((f"{a_en} likes {b_en}.", f"{a_zh}喜欢{b_zh}。"))
            swap_pairs.append((f"{a_en} calls {b_en}.", f"{a_zh}给{b_zh}打电话。"))
            swap_pairs.append((f"{a_en} meets {b_en}.", f"{a_zh}认识{b_zh}。"))

    base = _dedup(manual_core + templated + swap_pairs)

    if len(base) < 200:
        raise RuntimeError(f"Base pairs too small: {len(base)} (<200)")
    return base


def _apply_contractions(en: str) -> List[str]:
    variants = [en]
    rules = [
        ("I am ", "I'm "),
        ("You are ", "You're "),
        ("We are ", "We're "),
        ("They are ", "They're "),
        ("It is ", "It's "),
        ("What is ", "What's "),
        ("do not", "don't"),
        ("does not", "doesn't"),
        ("cannot", "can't"),
    ]
    for a, b in rules:
        if a in en:
            variants.append(en.replace(a, b))
    seen: set[str] = set()
    out: List[str] = []
    for v in variants:
        v2 = _clean_text(v)
        if not v2 or v2 in seen:
            continue
        seen.add(v2)
        out.append(v2)
    return out


def _apply_bilingual_synonyms(en: str, zh: str) -> List[Pair]:
    out: List[Pair] = [(en, zh)]
    rules = [
        ("Thank you", "Thanks", "谢谢", "谢谢"),
        ("Please ", "Could you please ", "请", "请"),
        ("Where is the bathroom", "Where is the restroom", "洗手间", "卫生间"),
        ("the bathroom", "the restroom", "洗手间", "卫生间"),
        ("I want to", "I would like to", "我想", "我想"),
        ("I like", "I love", "我喜欢", "我爱"),
        ("Do you", "Would you", "你", "你"),
    ]
    for en_a, en_b, zh_a, zh_b in rules:
        if en_a in en and zh_a in zh:
            out.append((en.replace(en_a, en_b), zh.replace(zh_a, zh_b)))
        if en_b in en and zh_b in zh:
            out.append((en.replace(en_b, en_a), zh.replace(zh_b, zh_a)))
    return _dedup(out)


def _try_swap_subject_object(en: str, zh: str) -> List[Pair]:
    en = en.strip()
    zh = zh.strip()
    out: List[Pair] = []

    en_patterns = [
        re.compile(r"^(?P<a>[A-Za-z]+) helps (?P<b>[A-Za-z]+)\.$"),
        re.compile(r"^(?P<a>[A-Za-z]+) likes (?P<b>[A-Za-z]+)\.$"),
        re.compile(r"^(?P<a>[A-Za-z]+) calls (?P<b>[A-Za-z]+)\.$"),
        re.compile(r"^(?P<a>[A-Za-z]+) meets (?P<b>[A-Za-z]+)\.$"),
    ]
    zh_patterns = [
        re.compile(r"^(?P<a>.+?)帮助(?P<b>.+?)。$"),
        re.compile(r"^(?P<a>.+?)喜欢(?P<b>.+?)。$"),
        re.compile(r"^(?P<a>.+?)给(?P<b>.+?)打电话。$"),
        re.compile(r"^(?P<a>.+?)认识(?P<b>.+?)。$"),
    ]

    en_swap: str | None = None
    for pat in en_patterns:
        m = pat.match(en)
        if not m:
            continue
        a = m.group("a")
        b = m.group("b")
        verb = en.split(" ")[1]
        en_swap = f"{b} {verb} {a}."
        break

    zh_swap: str | None = None
    for pat in zh_patterns:
        m = pat.match(zh)
        if not m:
            continue
        a = m.group("a")
        b = m.group("b")
        if "打电话" in zh:
            zh_swap = f"{b}给{a}打电话。"
        elif "帮助" in zh:
            zh_swap = f"{b}帮助{a}。"
        elif "喜欢" in zh:
            zh_swap = f"{b}喜欢{a}。"
        elif "认识" in zh:
            zh_swap = f"{b}认识{a}。"
        break

    if en_swap is not None and zh_swap is not None:
        out.append((en_swap, zh_swap))
    return _dedup(out)


def generate_variants(*, pairs: Sequence[Pair], target_size: int, seed: int = 42) -> List[Pair]:
    rng = random.Random(seed)
    base = _dedup(pairs)
    seen: set[Pair] = set(base)
    pool: List[Pair] = list(base)

    source: List[Pair] = list(base)
    max_attempts = max(50_000, int(target_size) * 25)
    attempts = 0

    while len(pool) < int(target_size) and attempts < max_attempts:
        attempts += 1
        en, zh = source[rng.randrange(0, len(source))]
        candidates: List[Pair] = []

        for en2 in _apply_contractions(en):
            candidates.append((en2, zh))

        candidates.extend(_apply_bilingual_synonyms(en, zh))
        candidates.extend(_try_swap_subject_object(en, zh))

        if rng.random() < 0.35:
            punct = rng.choice([".", "!", "?"])
            en3 = re.sub(r"[.!?]+$", punct, en)
            zh3 = re.sub(r"[。！？]+$", {".": "。", "!": "！", "?": "？"}[punct], zh)
            candidates.append((en3, zh3))

        if rng.random() < 0.25 and en.startswith("How much is ") and en.endswith("?"):
            en4 = en.replace("How much is ", "How much does ").replace("?", " cost?")
            candidates.append((en4, zh))

        if rng.random() < 0.25 and en.startswith("Where is ") and en.endswith("?"):
            en5 = en.replace("Where is ", "How do I get to ")
            candidates.append((en5, zh))

        for cand_en, cand_zh in candidates:
            key = (_clean_text(cand_en), _clean_text(cand_zh))
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)
            pool.append(key)
            source.append(key)
            if len(pool) >= int(target_size):
                break

    if len(pool) < int(target_size):
        raise RuntimeError(
            f"Could not reach target_size={target_size}. got={len(pool)} after {attempts} attempts"
        )
    return pool


def write_tsv(*, pairs: Sequence[Pair], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for en, zh in pairs:
            f.write(f"{_clean_text(en)}\t{_clean_text(zh)}\n")
    return out_path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate synthetic EN-ZH TSV with augmentation.")
    p.add_argument("--out", default="../data/tatoeba_en_zh.tsv", help="Output TSV path")
    p.add_argument("--target-lines", type=int, default=6000, help="Target total TSV rows (>=5000)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_path = Path(args.out)
    base = build_base_pairs(seed=int(args.seed))
    full = generate_variants(pairs=base, target_size=max(int(args.target_lines), 5000), seed=int(args.seed))
    write_tsv(pairs=full, out_path=out_path)
    print(f"Base pairs: {len(base)}")
    print(f"Generated rows: {len(full)}")
    print(f"Saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()




