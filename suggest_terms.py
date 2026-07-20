#!/usr/bin/env python3
"""Предлагает кандидатов в словарь terms.txt по истории диктовок.

Ищет в history.sqlite3 слова, которые LLM-чистка регулярно исправляла
(сырой текст Whisper -> чистый текст): такие исправления — сигнал, что Whisper
не знает слово, и его стоит добавить в словарь, чтобы ошибка чинилась уже
на этапе распознавания.

  uv run suggest_terms.py          # показать и спросить по каждому
  uv run suggest_terms.py --list   # только показать
  uv run suggest_terms.py --min 1  # порог повторов (по умолчанию 2)
"""
import argparse
import difflib
import os
import re
import sqlite3
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
FILLERS = {"эээ", "эм", "ну", "короче", "м", "m", "типа", "мда", "в", "общем",
           "как", "бы", "это", "самое", "значит"}


def words(s: str) -> list:
    return re.findall(r"[\w-]+", s)


def correction_pairs(raw: str, clean: str):
    """Пары (сырое слово -> чистое слово) из пословного диффа одной записи."""
    a, b = words(raw), words(clean)
    sm = difflib.SequenceMatcher(a=[w.lower() for w in a], b=[w.lower() for w in b])
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "replace":
            continue  # удаления — это паразиты, вставок чистка не делает
        src, dst = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if src.lower() == dst.lower():
            continue  # правка регистра — не словарный случай
        if all(w.lower() in FILLERS for w in words(src)):
            continue
        # созвучность: иначе это перефраз, а не исправление слова
        if difflib.SequenceMatcher(a=src.lower(), b=dst.lower()).ratio() < 0.5:
            continue
        yield src, dst


def load_terms() -> set:
    try:
        with open(os.path.join(BASE, "terms.txt")) as f:
            return {line.strip().lower() for line in f
                    if line.strip() and not line.startswith("#")}
    except FileNotFoundError:
        return set()


def suggestions(min_count: int) -> list:
    db = sqlite3.connect(os.path.join(BASE, "history.sqlite3"))
    rows = db.execute("SELECT raw_text, text FROM transcriptions "
                      "WHERE raw_text != text").fetchall()
    db.close()
    counts = Counter()
    for raw, clean in rows:
        for src, dst in correction_pairs(raw, clean):
            counts[(src.lower(), dst)] += 1
    known = load_terms()
    out = []
    for (src, dst), n in counts.most_common():
        if n < min_count or dst.lower() in known or len(dst) < 3:
            continue
        if len(words(dst)) > 2:
            continue  # длинные перефразы — не словарный случай
        out.append((dst, src, n))
    return out


AUTO_PATH = os.path.join(BASE, "auto_terms.txt")


def build_auto_terms(llm_run=None, cap=40) -> list:
    """Собирает автослой словаря из истории. Три источника:
    латиница в русской речи, повторившиеся исправления LLM,
    редкие русские слова (отбирает LLM, если передан llm_run)."""
    db = sqlite3.connect(os.path.join(BASE, "history.sqlite3"))
    raws = [r[0] for r in db.execute("SELECT raw_text FROM transcriptions")]
    db.close()
    known = load_terms()
    picked = []

    def add(w):
        if w and w.lower() not in known and w.lower() not in {p.lower() for p in picked}:
            picked.append(w)

    # 1) латинские слова в русской диктовке (Kubernetes, Redis...), >=2 раз
    lat = Counter(w for t in raws for w in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", t))
    for w, n in lat.most_common():
        if n >= 2:
            add(w)
    # 2) устоявшиеся исправления чистки
    for dst, _src, _n in suggestions(min_count=2):
        add(dst)
    # 3) редкие русские слова — отбирает LLM
    if llm_run is not None:
        cyr = Counter(w.lower() for t in raws
                      for w in re.findall(r"[А-Яа-яЁё][а-яё-]{4,}", t))
        frequent = [w for w, n in cyr.most_common(80) if n >= 3]
        if frequent:
            try:
                out = llm_run(
                    "Из списка слов выбери ТОЛЬКО узкоспециальные термины, жаргон "
                    "и имена собственные, которых может не знать система "
                    "распознавания речи (пример: задеплоить, фейловер). Обычные "
                    "слова, даже айтишные (сервер, статус, кнопка, словарь), "
                    "НЕ бери. Максимум 10. По одному на строку, без пояснений.",
                    ", ".join(frequent))
                for line in out.splitlines():
                    w = line.strip(" -•,.").strip()
                    if w.lower() in frequent:
                        add(w)
            except Exception:
                pass
    picked = picked[:cap]
    with open(AUTO_PATH, "w") as f:
        f.write("\n".join(picked) + ("\n" if picked else ""))
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="только показать")
    ap.add_argument("--min", type=int, default=2, help="минимум повторов")
    args = ap.parse_args()

    cands = suggestions(args.min)
    if not cands:
        print(f"Кандидатов нет (исправлений, повторившихся ≥{args.min} раз, не найдено).")
        return
    print(f"Кандидаты в словарь (по {len(cands)} исправлениям истории):\n")
    to_add = []
    for dst, src, n in cands:
        line = f"  {src} → {dst}   ({n} раз)"
        if args.list:
            print(line)
            continue
        ans = input(f"{line}   добавить «{dst}»? [y/N] ").strip().lower()
        if ans in ("y", "д", "да", "yes"):
            to_add.append(dst)
    if to_add:
        with open(os.path.join(BASE, "terms.txt"), "a") as f:
            f.write("\n".join(to_add) + "\n")
        print(f"\nДобавлено в terms.txt: {', '.join(to_add)} — действует со следующей диктовки.")


if __name__ == "__main__":
    main()
