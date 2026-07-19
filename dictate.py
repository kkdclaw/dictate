#!/usr/bin/env python3
"""Push-to-talk диктовка: зажми правый Option, говори, отпусти — текст вставится в активное поле.

Пайплайн: микрофон → whisper-large-v3-turbo (MLX) → LLM-чистка (Qwen3-4B) → вставка + история.
Словарь терминов — terms.txt рядом со скриптом. История — history.sqlite3.
"""
import os
import queue
import re
import sqlite3
import subprocess
import threading
import time

import numpy as np
import sounddevice as sd
import mlx_whisper
from mlx_whisper.transcribe import ModelHolder
import mlx.core as mx
from pynput import keyboard
import Quartz
from AppKit import NSWorkspace

BASE = os.path.dirname(os.path.abspath(__file__))
ASR_MODEL = "mlx-community/whisper-large-v3-turbo"
LLM_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
ENHANCE = True  # False — вставлять сырой текст Whisper без LLM-чистки
LANGUAGE = None  # None = автоопределение; "ru" — жёстко русский
HOTKEY = keyboard.Key.alt_r  # правый Option
SAMPLE_RATE = 16000
MIN_DURATION = 0.4  # сек; короче — случайное нажатие, игнорируем

recording = False
chunks = []
lock = threading.Lock()
jobs = queue.Queue()  # аудио -> единственный ML-поток (MLX не переживает смену потока)
stream_holder = {}  # текущий InputStream; пересоздаётся при смене устройства/тишине


def probe_rms(device) -> float:
    try:
        a = sd.rec(int(0.3 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1,
                   dtype="float32", device=device)
        sd.wait()
        return float(np.sqrt((a ** 2).mean()))
    except Exception:
        return 0.0


def pick_device():
    """Вход по умолчанию, а если он молчит (AirPods в кейсе) — первый живой микрофон."""
    default = sd.query_devices(kind="input")
    default_idx = default["index"]
    if probe_rms(default_idx) > 1e-5:
        return default_idx, default["name"], True
    best_idx, best_name, best_rms = None, None, 0.0
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1 or i == default_idx or "NoMachine" in d["name"]:
            continue
        rms = probe_rms(i)
        if rms > max(best_rms, 1e-5):
            best_idx, best_name, best_rms = i, d["name"], rms
    if best_idx is not None:
        return best_idx, best_name, False
    return default_idx, default["name"], True  # все молчат — остаёмся на дефолте


def open_stream():
    old = stream_holder.pop("stream", None)
    if old:
        old.stop()
        old.close()
    dev, name, is_default = pick_device()
    s = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                       device=dev, callback=audio_callback)
    s.start()
    stream_holder["stream"] = s
    note = "" if is_default else "  (вход по умолчанию молчит — взял живой)"
    print(f"Микрофон: {name}{note}", flush=True)


def load_terms() -> str:
    try:
        with open(os.path.join(BASE, "terms.txt")) as f:
            return ", ".join(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return ""


def asr_hint() -> str:
    # словарь в initial_prompt: Whisper подхватывает термины прямо при распознавании
    terms = load_terms()
    return f"Словарь: {terms}. Глаголы: задеплой, задеплоить." if terms else ""


def system_prompt() -> str:
    return (
        "Убери из надиктованного текста слова-паразиты (эээ, ну, короче, эм) и оговорки, "
        "поправь пунктуацию. Больше НИЧЕГО не меняй: грамматику, термины, смысл и порядок "
        "слов сохрани. Если исправлять нечего — верни текст дословно. "
        "Выведи ТОЛЬКО итоговый текст."
    )


def paste_text(text: str) -> None:
    subprocess.run(["pbcopy"], input=text.encode())
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    for down in (True, False):
        ev = Quartz.CGEventCreateKeyboardEvent(src, 9, down)  # 9 = kVK_ANSI_V
        Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)


def frontmost_app() -> str:
    try:
        return NSWorkspace.sharedWorkspace().frontmostApplication().localizedName()
    except Exception:
        return ""


def history_db() -> sqlite3.Connection:
    db = sqlite3.connect(os.path.join(BASE, "history.sqlite3"))
    db.execute(
        "CREATE TABLE IF NOT EXISTS transcriptions ("
        "id INTEGER PRIMARY KEY, ts REAL, text TEXT, raw_text TEXT, "
        "duration REAL, app TEXT)"
    )
    return db


def audio_callback(indata, frames, t, status):
    with lock:
        if recording:
            chunks.append(indata.copy())


FILLERS = re.compile(
    r"\b(э+м*|а+м+|мда+|ну|короче|как бы|типа|это самое|в общем|значит)\b|\b[mм]\b",
    re.IGNORECASE)


def needs_enhance(raw: str) -> bool:
    # LLM зовём только если есть что чистить — иначе вставляем сырой текст мгновенно
    return bool(FILLERS.search(raw))


def strip_short_period(text: str) -> str:
    # короткая фраза без внутренней пунктуации — скорее команда/фрагмент, точка не нужна
    if text.endswith(".") and "." not in text[:-1] and len(text) < 60:
        return text[:-1]
    return text


def ml_worker(ready: threading.Event):
    ModelHolder.get_model(ASR_MODEL, mx.float16)
    llm = tok = None
    if ENHANCE:
        from mlx_lm import load, generate
        llm, tok = load(LLM_MODEL)
        generate(llm, tok, prompt=tok.apply_chat_template(
            [{"role": "user", "content": "ок"}], add_generation_prompt=True),
            max_tokens=4, verbose=False)  # прогрев, чтобы первая диктовка была быстрой
    db = history_db()
    ready.set()

    def enhance(raw: str) -> str:
        msgs = [{"role": "system", "content": system_prompt()},
                {"role": "user", "content": raw}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
        out = generate(llm, tok, prompt=prompt,
                       max_tokens=len(tok.encode(raw)) * 2 + 50, verbose=False).strip()
        # деградация LLM (пусто / разнесло в разы) — откатываемся на сырой текст
        if not out or len(out) > len(raw) * 2 + 40:
            return raw
        return out

    while True:
        audio = jobs.get()
        duration = len(audio) / SAMPLE_RATE
        rms = float(np.sqrt((audio ** 2).mean()))
        if rms < 1e-4:
            print("  ✗ запись тихая (AirPods в кейсе? крышка закрыта?) — "
                  "ищу живой микрофон, попробуй ещё раз", flush=True)
            try:
                old = stream_holder.pop("stream", None)
                if old:
                    old.stop(); old.close()
                sd._terminate(); sd._initialize()  # перечитать список устройств CoreAudio
                open_stream()
            except Exception as e:
                print(f"  не удалось переоткрыть: {e}", flush=True)
            continue
        t0 = time.time()
        try:
            raw = mlx_whisper.transcribe(
                audio, path_or_hf_repo=ASR_MODEL, language=LANGUAGE,
                initial_prompt=asr_hint() or None)["text"].strip()
        except Exception as e:
            print(f"  ошибка распознавания: {e}", flush=True)
            continue
        t_asr = time.time() - t0
        if not raw:
            continue
        # тихое аудио + initial_prompt => Whisper галлюцинирует куски словаря
        raw_words = set(re.findall(r"\w+", raw.lower()))
        hint_words = set(re.findall(r"\w+", asr_hint().lower()))
        if raw_words and raw_words <= hint_words:
            print(f"  ✗ похоже на эхо словаря, не вставляю: {raw}", flush=True)
            continue
        text = raw
        t_llm = 0.0
        if ENHANCE and needs_enhance(raw):
            t1 = time.time()
            try:
                text = enhance(raw)
            except Exception as e:
                print(f"  ошибка чистки (вставляю сырой): {e}", flush=True)
            t_llm = time.time() - t1
        text = strip_short_period(text)
        app = frontmost_app()
        paste_text(text)
        db.execute("INSERT INTO transcriptions (ts, text, raw_text, duration, app) "
                   "VALUES (?, ?, ?, ?, ?)", (time.time(), text, raw, duration, app))
        db.commit()
        mark = "" if text == strip_short_period(raw) else f"  (сырой: {raw})"
        print(f"  [{duration:.1f}s аудио → asr {t_asr:.1f}s + llm {t_llm:.1f}s → {app}] "
              f"{text}{mark}", flush=True)


def on_press(key):
    global recording
    if key == HOTKEY and not recording:
        with lock:
            chunks.clear()
        recording = True
        print("● запись...", flush=True)


def on_release(key):
    global recording
    if key == HOTKEY and recording:
        recording = False
        with lock:
            if not chunks:
                return
            audio = np.concatenate(chunks).flatten().astype(np.float32)
            chunks.clear()
        if len(audio) / SAMPLE_RATE >= MIN_DURATION:
            jobs.put(audio)


def main():
    print(f"Прогреваю модели ({ASR_MODEL.split('/')[-1]}"
          f"{' + ' + LLM_MODEL.split('/')[-1] if ENHANCE else ''})...")
    ready = threading.Event()
    threading.Thread(target=ml_worker, args=(ready,), daemon=True).start()
    ready.wait()
    open_stream()
    print("Готово. Зажми правый Option и говори; отпусти — текст вставится. Ctrl+C — выход.")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
