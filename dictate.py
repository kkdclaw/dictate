#!/usr/bin/env python3
"""Push-to-talk диктовка: зажми правый Option, говори, отпусти — текст вставится в активное поле.

Пайплайн: микрофон → whisper-large-v3-turbo (MLX) → LLM-чистка (Qwen3-4B) → вставка + история.
Словарь терминов — terms.txt рядом со скриптом. История — history.sqlite3.
"""
import collections
import json
import os
import queue
import re
import sqlite3
import subprocess
import threading
import time

import numpy as np
import rumps
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
LANGUAGE = None  # None = автоопределение; "ru" — жёстко русский
HOTKEY = keyboard.Key.alt_r  # правый Option
SAMPLE_RATE = 16000
MIN_DURATION = 0.4  # сек; короче — случайное нажатие, игнорируем

STATE = {"loading": True, "mic": "…", "enhance": True, "app": ""}

CONFIG_PATH = os.path.join(BASE, "config.json")
VOICEPRINT_PATH = os.path.join(BASE, "voiceprint.npy")
STYLES = {  # ключ -> подпись в меню
    "clean": "Чистка (по умолчанию)",
    "casual": "Разговорный (без точек)",
    "formal": "Строгий (письменный)",
    "raw": "Как сказано (без LLM)",
    "translate": "Перевод → EN",
}
CONFIG = {"default_style": "clean", "profiles": {}, "only_my_voice": False,
          "translate_all": False, "vp_threshold": 0.40}


def load_config():
    try:
        with open(CONFIG_PATH) as f:
            CONFIG.update(json.load(f))
    except (FileNotFoundError, ValueError):
        pass


def save_config():
    with open(CONFIG_PATH, "w") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)


def style_for(app: str) -> str:
    if CONFIG["translate_all"]:
        return "translate"
    return CONFIG["profiles"].get(app, CONFIG["default_style"])


recording = False
chunks = []
PREROLL_SEC = 0.5  # секунды звука ДО нажатия, подклеиваемые к записи
preroll = collections.deque(maxlen=64)  # кольцевой буфер последних блоков микрофона
lock = threading.Lock()
jobs = queue.Queue()  # аудио -> единственный ML-поток (MLX не переживает смену потока)
stream_holder = {}  # текущий InputStream; пересоздаётся при смене устройства/тишине


import ctypes

_coreaudio = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")


class _PropAddr(ctypes.Structure):
    _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32),
                ("element", ctypes.c_uint32)]


_LISTENER_T = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_uint32, ctypes.c_uint32,
                               ctypes.POINTER(_PropAddr), ctypes.c_void_p)
_listener_refs = []  # защита колбэка и адреса от сборщика мусора


def _fourcc(s: str) -> int:
    return int.from_bytes(s.encode(), "big")


def watch_default_input(on_change):
    """CoreAudio-событие «вход по умолчанию сменился» (надели AirPods и т.п.)."""
    addr = _PropAddr(_fourcc("dIn "), _fourcc("glob"), 0)

    def _cb(obj, n, a, ctx):
        on_change()
        return 0

    cb = _LISTENER_T(_cb)
    _listener_refs.extend([cb, addr])
    _coreaudio.AudioObjectAddPropertyListener(1, ctypes.byref(addr), cb, None)


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


def reopen_stream(follow_default=False):
    """Полный перезапуск аудио: закрыть поток, перечитать устройства CoreAudio, открыть заново."""
    old = stream_holder.pop("stream", None)
    if old:
        try:
            old.stop(); old.close()
        except Exception:
            pass
    try:
        sd._terminate(); sd._initialize()
    except Exception:
        pass
    open_stream(follow_default=follow_default)


def ensure_stream():
    """Перед записью: если поток умер или пульс пропал (микрофон отвалился) — переоткрыть."""
    s = stream_holder.get("stream")
    alive = False
    try:
        alive = bool(s) and s.active and time.time() - stream_holder.get("last_cb", 0) < 2.0
    except Exception:
        pass
    if not alive:
        print("  микрофон пропал — переоткрываю...", flush=True)
        try:
            # follow_default: без проб устройств, окно потери звука минимально;
            # если дефолт окажется мёртвым, сработает фолбэк по тихой записи
            reopen_stream(follow_default=True)
        except Exception as e:
            print(f"  не удалось открыть микрофон: {e}", flush=True)


def open_stream(follow_default=False):
    old = stream_holder.pop("stream", None)
    if old:
        old.stop()
        old.close()
    if follow_default:
        # смена входа по умолчанию — это действие пользователя, верим без проб
        d = sd.query_devices(kind="input")
        dev, name, is_default = d["index"], d["name"], True
    else:
        dev, name, is_default = pick_device()
    s = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                       device=dev, callback=audio_callback)
    s.start()
    stream_holder["stream"] = s
    note = "" if is_default else "  (вход по умолчанию молчит — взял живой)"
    STATE["mic"] = name
    print(f"Микрофон: {name}{note}", flush=True)


mic_changed = threading.Event()


def mic_watcher():
    """Ловит смену входа по умолчанию и пересаживает поток на новое устройство."""
    while True:
        mic_changed.wait()
        time.sleep(0.7)  # дебаунс: при переключении CoreAudio сыплет пачку событий
        mic_changed.clear()
        while recording:  # не дёргать поток посреди записи
            time.sleep(0.2)
        try:
            print("Сменился вход по умолчанию — переключаюсь...", flush=True)
            reopen_stream(follow_default=True)
        except Exception as e:
            print(f"  не удалось переключить микрофон: {e}", flush=True)


def load_terms() -> str:
    # ручное ядро + автослой из истории; лимит ~60 слов (у initial_prompt Whisper
    # потолок 224 токена), ручные — в приоритете
    words, seen = [], set()
    for fname in ("terms.txt", "auto_terms.txt"):
        try:
            with open(os.path.join(BASE, fname)) as f:
                for line in f:
                    w = line.strip()
                    if w and not w.startswith("#") and w.lower() not in seen:
                        words.append(w)
                        seen.add(w.lower())
        except FileNotFoundError:
            pass
        if len(words) >= 60:
            break
    return ", ".join(words[:60])



def asr_hint() -> str:
    # словарь в initial_prompt: Whisper подхватывает термины прямо при распознавании
    terms = load_terms()
    return f"Словарь: {terms}. Глаголы: задеплой, задеплоить." if terms else ""


def system_prompt() -> str:
    return (
        "Ты корректор надиктованного текста. Правила:\n"
        "1. Убери слова-паразиты (эээ, ну, короче, эм) и оговорки. Значимые слова "
        "(нужно, надо, давай, проверь) паразитами НЕ являются — сохраняй их.\n"
        "2. Исправляй ТОЛЬКО искажённые распознаванием слова. Грамматику, падежи, "
        "наклонение, порядок слов и смысл НЕ меняй. Ничего не добавляй и не пересказывай.\n"
        f"3. Термины пользователя: {load_terms()}. Искажённое слово, созвучное термину, "
        "исправь на термин. Правильно написанный термин НЕ заменяй на другой термин.\n"
        "4. Слитные глаголы, разбитые на части, склей: «За деплой сервис» → «Задеплой сервис».\n"
        "5. Если исправлять нечего — верни текст дословно.\n"
        "Примеры: «филовер настроен» → «фейловер настроен»; «проверь зиро тир» → "
        "«проверь ZeroTier»; «MTG работает» → «MTG работает» (не менять!).\n"
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
    stream_holder["last_cb"] = time.time()  # пульс: колбэки идут, пока устройство живо
    with lock:
        preroll.append(indata.copy())
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


TRANSLATE_PROMPT = (
    "Translate the dictated Russian text into natural, fluent English. "
    "Keep the meaning, tone and technical terms. Output ONLY the translation."
)
FORMAL_PROMPT_ADDON = (
    "\nДополнительно: оформи как аккуратный письменный текст — законченные "
    "предложения, правильная пунктуация, без разговорных огрызков."
)


def ml_worker(ready: threading.Event):
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps
    vad = load_silero_vad(onnx=True)
    from speechbrain.inference.speaker import EncoderClassifier
    spk = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                         savedir=os.path.join(BASE, "models/ecapa"))
    ModelHolder.get_model(ASR_MODEL, mx.float16)
    from mlx_lm import load, generate
    llm, tok = load(LLM_MODEL)
    generate(llm, tok, prompt=tok.apply_chat_template(
        [{"role": "user", "content": "ок"}], add_generation_prompt=True),
        max_tokens=4, verbose=False)  # прогрев, чтобы первая диктовка была быстрой
    db = history_db()
    voiceprint = np.load(VOICEPRINT_PATH) if os.path.exists(VOICEPRINT_PATH) else None
    STATE["loading"] = False
    ready.set()

    def rebuild_autodict():
        try:
            import suggest_terms
            added = suggest_terms.build_auto_terms(llm_run=llm_run)
            print(f"Автословарь обновлён: {', '.join(added) if added else 'пусто'}",
                  flush=True)
        except Exception as e:
            print(f"  автословарь не собрался: {e}", flush=True)

    def embed(audio: np.ndarray) -> np.ndarray:
        e = spk.encode_batch(torch.from_numpy(audio).unsqueeze(0)).squeeze().numpy()
        return e / np.linalg.norm(e)

    def llm_run(system: str, user: str, max_factor: int = 2) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
        return generate(llm, tok, prompt=prompt,
                        max_tokens=len(tok.encode(user)) * max_factor + 60,
                        verbose=False).strip()

    def enhance(raw: str, formal: bool = False, doubtful=None) -> str:
        system = system_prompt() + (FORMAL_PROMPT_ADDON if formal else "")
        if doubtful:
            system += ("\nДополнительно: распознаватель не уверен в словах: "
                       + ", ".join(f"«{w}»" for w in doubtful[:8])
                       + " — это возможные ослышки (города, имена, термины, "
                       "разорванные слова). Исправь их правдоподобно по контексту; "
                       "если слово выглядит верным — оставь.")
        out = llm_run(system, raw)
        # деградация LLM (пусто / разнесло в разы) — откатываемся на сырой текст
        if not out or len(out) > len(raw) * 2 + 40:
            return raw
        return out

    rebuild_autodict()

    while True:
        kind, audio = jobs.get()
        if kind == "autodict":
            rebuild_autodict()
            continue
        if kind == "enroll":
            nonlocal_vp = embed(audio)
            np.save(VOICEPRINT_PATH, nonlocal_vp)
            voiceprint = nonlocal_vp
            print("Отпечаток голоса сохранён.", flush=True)
            continue
        duration = len(audio) / SAMPLE_RATE
        rms = float(np.sqrt((audio ** 2).mean()))
        if rms < 1e-4:
            print("  ✗ запись тихая (AirPods в кейсе? крышка закрыта?) — "
                  "ищу живой микрофон, попробуй ещё раз", flush=True)
            try:
                reopen_stream()
            except Exception as e:
                print(f"  не удалось переоткрыть: {e}", flush=True)
            continue
        # диагностика: цифровые нули в начале = устройство ещё не отдавало звук
        nz = np.flatnonzero(audio)
        if len(nz) and nz[0] > SAMPLE_RATE * 0.25:
            print(f"  ⚠ микрофон молчал первые {nz[0] / SAMPLE_RATE:.2f}с записи "
                  f"(просыпался после переключения?)", flush=True)
        # VAD: есть ли вообще речь, и если есть — обрезать тишину по краям
        spans = get_speech_timestamps(torch.from_numpy(audio), vad,
                                      sampling_rate=SAMPLE_RATE, speech_pad_ms=150)
        if not spans:
            print("  ✗ речи не слышно — не вставляю", flush=True)
            continue
        audio = audio[max(0, spans[0]["start"] - SAMPLE_RATE // 4):
                      spans[-1]["end"] + SAMPLE_RATE // 10]
        # отпечаток голоса: чужую речь (ТВ, коллеги) не транскрибируем
        if CONFIG["only_my_voice"] and voiceprint is not None:
            sim = float(embed(audio) @ voiceprint)
            if sim < CONFIG["vp_threshold"]:
                print(f"  ✗ не твой голос (сходство {sim:.2f} < "
                      f"{CONFIG['vp_threshold']}) — не вставляю", flush=True)
                continue
        t0 = time.time()
        try:
            result = mlx_whisper.transcribe(
                audio, path_or_hf_repo=ASR_MODEL, language=LANGUAGE,
                initial_prompt=asr_hint() or None, word_timestamps=True)
            raw = result["text"].strip()
        except Exception as e:
            print(f"  ошибка распознавания: {e}", flush=True)
            continue
        # слова, в которых Whisper сам не уверен, — кандидаты на ослышку
        doubtful = [w["word"].strip() for s in result["segments"]
                    for w in s.get("words", []) if w["probability"] < 0.6]
        t_asr = time.time() - t0
        if not raw:
            continue
        # тихое аудио + initial_prompt => Whisper галлюцинирует куски словаря
        raw_words = set(re.findall(r"\w+", raw.lower()))
        hint_words = set(re.findall(r"\w+", asr_hint().lower()))
        if raw_words and raw_words <= hint_words:
            print(f"  ✗ похоже на эхо словаря, не вставляю: {raw}", flush=True)
            continue
        app = frontmost_app()
        style = style_for(app)
        text = raw
        t_llm = 0.0
        t1 = time.time()
        try:
            if style == "translate":
                text = llm_run(TRANSLATE_PROMPT, raw, max_factor=3) or raw
            elif style == "formal":
                text = enhance(raw, formal=True, doubtful=doubtful)
            elif style == "raw":
                pass
            elif STATE["enhance"] and (needs_enhance(raw) or doubtful):  # clean / casual
                text = enhance(raw, doubtful=doubtful)
        except Exception as e:
            print(f"  ошибка обработки (вставляю сырой): {e}", flush=True)
        t_llm = time.time() - t1
        if style == "casual":
            text = text.rstrip(".")
        else:
            text = strip_short_period(text)
        paste_text(text)
        db.execute("INSERT INTO transcriptions (ts, text, raw_text, duration, app) "
                   "VALUES (?, ?, ?, ?, ?)", (time.time(), text, raw, duration, app))
        db.commit()
        mark = "" if text == strip_short_period(raw) else f"  (сырой: {raw})"
        doubt = f"  [сомнения: {', '.join(doubtful[:5])}]" if doubtful else ""
        print(f"  [{duration:.1f}s аудио → asr {t_asr:.1f}s + llm {t_llm:.1f}s → "
              f"{app}/{style}] {text}{mark}{doubt}", flush=True)


TAP_MAX = 0.35  # сек: короче — «тап» (toggle-режим), дольше — классический push-to-talk

toggle_mode = False
press_time = 0.0


def stop_and_submit():
    global recording, toggle_mode
    recording = False
    toggle_mode = False
    with lock:
        if not chunks:
            return
        audio = np.concatenate(chunks).flatten().astype(np.float32)
        chunks.clear()
    if len(audio) / SAMPLE_RATE >= MIN_DURATION:
        jobs.put(("dictate", audio))


def cancel_recording():
    global recording, toggle_mode
    recording = False
    toggle_mode = False
    with lock:
        chunks.clear()
    print("  ✗ запись отменена (Esc)", flush=True)


def on_press(key):
    global recording, press_time
    if key == keyboard.Key.esc and recording:
        cancel_recording()
        return
    if key != HOTKEY:
        return
    if not recording:
        # флаг записи — сразу, проверка/оживление потока — в фоне: колбэки начнут
        # наполнять chunks в ту же миллисекунду, как поток жив
        threading.Thread(target=ensure_stream, daemon=True).start()
        with lock:
            chunks.clear()
            # подклеиваем последние PREROLL_SEC до нажатия — первое слово не режется,
            # даже если начал говорить одновременно с клавишей
            need = int(PREROLL_SEC * SAMPLE_RATE)
            got = 0
            for block in reversed(preroll):
                chunks.insert(0, block)
                got += len(block)
                if got >= need:
                    break
        press_time = time.time()
        recording = True
        print("● запись...", flush=True)
    elif toggle_mode:
        stop_and_submit()  # второй тап — стоп


def on_release(key):
    global toggle_mode
    if key != HOTKEY or not recording:
        return
    if time.time() - press_time < TAP_MAX:
        toggle_mode = True  # короткий тап: пишем дальше до второго тапа или Esc
        print("  … toggle-режим: говори, ещё один тап Option — стоп, Esc — отмена", flush=True)
    else:
        stop_and_submit()  # классика: отпустил — обрабатываем


class DictateApp(rumps.App):
    def __init__(self):
        super().__init__("Dictate", title="⏳", quit_button=rumps.MenuItem("Выход"))
        self.mic_item = rumps.MenuItem("Микрофон: …")
        self.recent = rumps.MenuItem("Последние (клик — скопировать)")
        self.recent.add(rumps.MenuItem("пусто"))
        self.enh_item = rumps.MenuItem("LLM-чистка паразитов", callback=self.toggle_enhance)
        self.enh_item.state = int(STATE["enhance"])

        self.profile = rumps.MenuItem("Профиль: …")
        for key, label in [("default", "По умолчанию")] + list(STYLES.items()):
            it = rumps.MenuItem(label, callback=self.set_profile)
            it._style_key = key
            self.profile.add(it)
        self.default_style = rumps.MenuItem("Стиль по умолчанию")
        for key, label in STYLES.items():
            it = rumps.MenuItem(label, callback=self.set_default_style)
            it._style_key = key
            self.default_style.add(it)
        self.translate_item = rumps.MenuItem("Перевод → EN (везде)", callback=self.toggle_translate)
        self.translate_item.state = int(CONFIG["translate_all"])
        self.vp_item = rumps.MenuItem("Только мой голос", callback=self.toggle_voice)
        self.vp_item.state = int(CONFIG["only_my_voice"])

        self.menu = [self.mic_item, self.recent, None,
                     self.profile, self.default_style, self.translate_item, None,
                     self.vp_item,
                     rumps.MenuItem("Записать отпечаток голоса (5 с)", callback=self.enroll),
                     None,
                     self.enh_item,
                     rumps.MenuItem("Словарь терминов…", callback=self.open_terms),
                     rumps.MenuItem("Обновить автословарь из истории", callback=self.suggest),
                     rumps.MenuItem("Лог…", callback=self.open_log), None]
        rumps.Timer(self.refresh_title, 0.3).start()
        rumps.Timer(self.refresh_recent, 3.0).start()

    def refresh_title(self, _):
        self.title = "⏳" if STATE["loading"] else ("🔴" if recording else "🎤")
        self.mic_item.title = f"Микрофон: {STATE['mic']}"
        app = frontmost_app() or STATE["app"]
        STATE["app"] = app
        cur = CONFIG["profiles"].get(app)
        self.profile.title = f"Профиль «{app}»: " + (STYLES[cur] if cur else "по умолчанию")
        for it in self.profile.values():
            it.state = int((cur is None and it._style_key == "default") or it._style_key == cur)
        for it in self.default_style.values():
            it.state = int(it._style_key == CONFIG["default_style"])

    def set_profile(self, sender):
        app = STATE["app"]
        if not app:
            return
        if sender._style_key == "default":
            CONFIG["profiles"].pop(app, None)
        else:
            CONFIG["profiles"][app] = sender._style_key
        save_config()

    def set_default_style(self, sender):
        CONFIG["default_style"] = sender._style_key
        save_config()

    def toggle_translate(self, sender):
        CONFIG["translate_all"] = not CONFIG["translate_all"]
        sender.state = int(CONFIG["translate_all"])
        save_config()

    def toggle_voice(self, sender):
        if not os.path.exists(VOICEPRINT_PATH) and not CONFIG["only_my_voice"]:
            rumps.alert("Только мой голос",
                        "Сначала запиши отпечаток: пункт «Записать отпечаток голоса (5 с)».")
            return
        CONFIG["only_my_voice"] = not CONFIG["only_my_voice"]
        sender.state = int(CONFIG["only_my_voice"])
        save_config()

    def enroll(self, _):
        rumps.alert("Отпечаток голоса",
                    "После ОК говори 5 секунд обычным голосом — любую фразу.")
        def rec():
            a = sd.rec(int(5 * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32")
            sd.wait()
            jobs.put(("enroll", a.flatten().astype(np.float32)))
        threading.Thread(target=rec, daemon=True).start()

    def refresh_recent(self, _):
        try:
            db = sqlite3.connect(os.path.join(BASE, "history.sqlite3"))
            rows = db.execute("SELECT text FROM transcriptions "
                              "ORDER BY id DESC LIMIT 5").fetchall()
            db.close()
        except Exception:
            return
        self.recent.clear()
        if not rows:
            self.recent.add(rumps.MenuItem("пусто"))
            return
        for (text,) in rows:
            label = text if len(text) <= 60 else text[:57] + "…"
            item = rumps.MenuItem(label, callback=self.copy_item)
            item._full_text = text
            self.recent.add(item)

    def copy_item(self, sender):
        subprocess.run(["pbcopy"], input=sender._full_text.encode())

    def toggle_enhance(self, sender):
        STATE["enhance"] = not STATE["enhance"]
        sender.state = int(STATE["enhance"])

    def open_terms(self, _):
        subprocess.run(["open", "-t", os.path.join(BASE, "terms.txt")])

    def suggest(self, _):
        jobs.put(("autodict", None))
        rumps.alert("Автословарь", "Пересборка запущена в фоне — результат "
                    "появится в логе строкой «Автословарь обновлён: …». "
                    "Он также пересобирается сам при каждом старте.")

    def open_log(self, _):
        subprocess.run(["open", "-t", os.path.join(BASE, "dictate.log")])


def request_permissions():
    """При старте: проверить все три TCC-разрешения и запросить недостающие системными диалогами."""
    import ctypes
    from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
    from AVFoundation import AVCaptureDevice
    missing = []
    # Микрофон: 3 = granted; запрос покажет системный диалог
    if AVCaptureDevice.authorizationStatusForMediaType_("soun") != 3:
        AVCaptureDevice.requestAccessForMediaType_completionHandler_("soun", lambda ok: None)
        missing.append("Микрофон")
    # Мониторинг ввода: 0 = granted; запрос добавит python3 в список и покажет диалог
    iokit = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
    if iokit.IOHIDCheckAccess(1) != 0:
        iokit.IOHIDRequestAccess(1)
        missing.append("Мониторинг ввода")
    # Универсальный доступ: диалог со ссылкой в настройки
    if not AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True}):
        missing.append("Универсальный доступ")
    if missing:
        print(f"⚠ Нет разрешений: {', '.join(missing)}. Выдай в Настройках → "
              f"Конфиденциальность и перезапусти (daemon.sh restart).", flush=True)
    else:
        print("Разрешения: все выданы.", flush=True)


def main():
    load_config()
    request_permissions()
    print(f"Прогреваю модели ({ASR_MODEL.split('/')[-1]} + {LLM_MODEL.split('/')[-1]})...")
    ready = threading.Event()
    threading.Thread(target=ml_worker, args=(ready,), daemon=True).start()
    threading.Thread(target=open_stream, daemon=True).start()
    threading.Thread(target=mic_watcher, daemon=True).start()
    watch_default_input(mic_changed.set)
    keyboard.Listener(on_press=on_press, on_release=on_release).start()
    print("Меню-бар запущен. Зажми правый Option и говори; отпусти — текст вставится.")
    DictateApp().run()


if __name__ == "__main__":
    main()
