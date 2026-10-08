#!/usr/bin/env python3
"""Сравнение моделей распознавания на сохранённом аудио диктовок.

Нужна галка «Хранить аудио диктовок» в меню: тогда каждая диктовка лежит в
audio/<id>.wav, где <id> — строка history.sqlite3 (текст, приложение, дата).

  uv run compare_asr.py                      # последние 30 диктовок; активная модель + скачанные из каталога
  uv run compare_asr.py --last 50 --models mlx-community/whisper-large-v3-turbo,mlx-community/parakeet-tdt-0.6b-v3
  uv run compare_asr.py --ids 812,815        # конкретные диктовки
  uv run compare_asr.py --wav a.wav b.wav    # произвольные файлы (16 кГц mono; иначе — пересэмплируем)
  uv run compare_asr.py --out my.html --no-open

Для Whisper-моделей словарь-подсказка берётся как в приложении (слой по
приложению из истории), у Parakeet подсказки нет. Эталона нет: таблица
показывает, что выдала каждая модель, время и расхождения с первой колонкой —
решает человек. Все расчёты локальные, звук машину не покидает.
"""
import argparse
import difflib
import html
import os
import re
import sqlite3
import subprocess
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dictate  # noqa: E402  — конфиг, словарь, движки; приложение при импорте не стартует

BASE = dictate.BASE


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise SystemExit(f"{path}: нужен PCM16, а тут {sw * 8} бит")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != dictate.SAMPLE_RATE:  # простой линейный ресэмплинг — для сравнения достаточно
        idx = np.linspace(0, len(x) - 1, int(len(x) * dictate.SAMPLE_RATE / sr))
        x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    return x


def samples_from_history(ids=None, last=30) -> list:
    db = sqlite3.connect(os.path.join(BASE, "history.sqlite3"))
    db.row_factory = sqlite3.Row
    if ids:
        rows = db.execute("SELECT id, ts, app, raw_text, text, duration FROM transcriptions "
                          f"WHERE id IN ({','.join('?' * len(ids))}) ORDER BY id", ids).fetchall()
    else:
        rows = db.execute("SELECT id, ts, app, raw_text, text, duration FROM transcriptions "
                          "ORDER BY id DESC LIMIT ?", (last * 4,)).fetchall()
    out = []
    for r in rows:
        path = os.path.join(dictate.AUDIO_DIR, f"{r['id']}.wav")
        if os.path.exists(path):
            out.append({"id": r["id"], "ts": r["ts"], "app": r["app"] or "", "raw": r["raw_text"] or "",
                        "text": r["text"] or "", "path": path})
    if not ids:
        out = sorted(out, key=lambda s: s["id"])[-last:]
    return out


def default_models() -> list:
    active = dictate.CONFIG["asr_model"]
    repos = [active]
    for repo, full, _ in dictate.ROLES["asr"][1]:
        if repo != active and dictate._repo_status(repo, full)["state"] == "done":
            repos.append(repo)
    return repos


def run_model(repo: str, audio: np.ndarray, app: str) -> tuple:
    t = time.time()
    if dictate.asr_engine(repo) == "parakeet":
        dictate.parakeet_load(repo)
        res = dictate.parakeet_transcribe(audio)
    else:
        import mlx_whisper
        res = mlx_whisper.transcribe(audio, path_or_hf_repo=repo,
                                     language=dictate.CONFIG["asr_language"] or None,
                                     initial_prompt=dictate.asr_hint(app) or None,
                                     word_timestamps=True)
    return res["text"].strip(), time.time() - t


def words(text: str) -> list:
    return re.findall(r"[\w'’-]+|[^\w\s]", text)


def mark_diff(base: str, other: str) -> str:
    """HTML текста other с подсветкой слов, которых нет на тех же местах в base."""
    a, b = words(base.lower()), words(other)
    bl = [w.lower() for w in b]
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, bl, autojunk=False).get_opcodes():
        chunk = html.escape(" ".join(b[j1:j2]))
        out.append(chunk if op == "equal" else f"<mark>{chunk}</mark>")
    return " ".join(x for x in out if x)


CSS = """
body{font:14px -apple-system,system-ui;margin:24px;color:#222;background:#fff}
@media(prefers-color-scheme:dark){body{color:#ddd;background:#1b1b1d}th{background:#2a2a2d}mark{background:#6b4e00;color:#fff}}
table{border-collapse:collapse;width:100%}th,td{border:1px solid #8884;padding:6px 8px;vertical-align:top}
th{background:#f3f3f5;position:sticky;top:0}td.meta{white-space:nowrap;color:#888;font-size:12px}
mark{background:#ffe08a;padding:0 2px;border-radius:2px}.t{color:#888;font-size:11px}
h1{font-size:18px}p{max-width:900px}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="репо через запятую; по умолчанию активная + скачанные из каталога")
    ap.add_argument("--last", type=int, default=30, help="сколько последних диктовок с аудио (30)")
    ap.add_argument("--ids", help="id строк истории через запятую")
    ap.add_argument("--wav", nargs="*", help="произвольные wav вместо истории")
    ap.add_argument("--app", default="", help="приложение для слоя словаря при --wav (по умолчанию общий)")
    ap.add_argument("--out", default=os.path.join(BASE, "compare.html"))
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    dictate.load_config()
    models = [m.strip() for m in args.models.split(",")] if args.models else default_models()
    if args.wav:
        samples = [{"id": os.path.basename(p), "ts": os.path.getmtime(p), "app": args.app,
                    "raw": "", "text": "", "path": p} for p in args.wav]
    else:
        samples = samples_from_history([int(x) for x in args.ids.split(",")] if args.ids else None,
                                       args.last)
    if not samples:
        raise SystemExit("Нет аудио: включи в меню «Хранить аудио диктовок» и надиктуй, "
                         f"или укажи --wav. Папка: {dictate.AUDIO_DIR}")
    print(f"Диктовок: {len(samples)} · модели: {', '.join(m.split('/')[-1] for m in models)}", flush=True)

    results = {m: [] for m in models}  # repo -> [(text, sec)]
    audios = [read_wav(s["path"]) for s in samples]
    for repo in models:  # по модели, не по диктовке: модель грузится один раз
        short = repo.split("/")[-1]
        t0 = time.time()  # загрузка и прогрев — отдельно, чтобы первая диктовка не платила за них
        if dictate.asr_engine(repo) == "parakeet":
            dictate.parakeet_load(repo)
        else:
            import mlx.core as mx
            from mlx_whisper.transcribe import ModelHolder
            ModelHolder.get_model(repo, mx.float16)
            run_model(repo, audios[0][: dictate.SAMPLE_RATE], "")
        print(f"  {short}: загрузка {time.time() - t0:.1f}s", flush=True)
        for i, (s, audio) in enumerate(zip(samples, audios), 1):
            text, sec = run_model(repo, audio, s["app"])
            results[repo].append((text, sec))
            print(f"  {short} {i}/{len(samples)} · {sec:.2f}s · {text[:80]}", flush=True)

    # сводка
    rows = []
    for repo in models:
        secs = [sec for _, sec in results[repo]]
        dur = sum(len(a) / dictate.SAMPLE_RATE for a in audios)
        same = sum(1 for (t, _), s in zip(results[repo], samples) if s["raw"] and t == s["raw"])
        rows.append((repo, sum(secs), dur / sum(secs) if sum(secs) else 0, same))
    print("\nМодель · время на всё · ×реалтайм · совпало с тем, что было тогда")
    for repo, tot, rtf, same in rows:
        print(f"  {repo.split('/')[-1]:40} {tot:7.1f}s  ×{rtf:4.0f}  {same}/{len(samples)}")

    # HTML
    base = models[0]
    h = [f"<!doctype html><meta charset=utf-8><title>Сравнение моделей — dictate</title><style>{CSS}</style>",
         "<h1>Сравнение моделей распознавания на сохранённом аудио</h1>",
         f"<p>{len(samples)} диктовок · {time.strftime('%d.%m.%Y %H:%M')} · подсветка — слова, "
         f"отличающиеся от первой колонки ({html.escape(base.split('/')[-1])}). Эталона нет: "
         "правильность решает человек. Строка «тогда» — сырой текст, который активная модель выдала при диктовке.</p>",
         "<table><tr><th>Диктовка</th>"]
    h.append("".join(f"<th>{html.escape(r.split('/')[-1])}<div class=t>"
                     f"{tot:.1f}s · ×{rtf:.0f} · совпало {same}/{len(samples)}</div></th>"
                     for r, tot, rtf, same in rows))
    h.append("</tr>")
    for i, s in enumerate(samples):
        when = time.strftime("%d.%m %H:%M", time.localtime(s["ts"]))
        dur = len(audios[i]) / dictate.SAMPLE_RATE
        meta = (f"#{s['id']}<br>{when}<br>{html.escape(s['app'])}<br>{dur:.1f}s"
                + (f"<br><br><b>тогда:</b><br>{html.escape(s['raw'])}" if s["raw"] else ""))
        h.append(f"<tr><td class=meta>{meta}</td>")
        base_text = results[base][i][0]
        for repo in models:
            text, sec = results[repo][i]
            cell = html.escape(text) if repo == base else mark_diff(base_text, text)
            h.append(f"<td>{cell}<div class=t>{sec:.2f}s</div></td>")
        h.append("</tr>")
    h.append("</table>")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(h))
    print(f"\nОтчёт: {args.out}")
    if not args.no_open:
        subprocess.run(["open", args.out])


if __name__ == "__main__":
    main()
