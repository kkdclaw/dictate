#!/usr/bin/env python3
"""Push-to-talk диктовка: зажми правый Option, говори, отпусти — текст вставится в активное поле.

Пайплайн: микрофон → whisper-large-v3-turbo (MLX) → LLM-чистка (Qwen3-4B) → вставка + история.
Словарь терминов — terms.txt рядом со скриптом. История — history.sqlite3.
"""
import collections
import json
import os
import platform
import queue
import re
import sqlite3
import subprocess
import sys
import tempfile
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
import webwindow
import hud
import statuspanel

BASE = os.path.dirname(os.path.abspath(__file__))
# Семантическая версия. Держим синхронно с pyproject.toml и git-тегом vX.Y.Z:
#   MAJOR — несовместимые изменения (формат конфига/истории, смена хоткея по умолчанию)
#   MINOR — новые возможности
#   PATCH — исправления без новых возможностей
# Тег ставится на релизном коммите: git tag -a v0.4.0 -m "…" && git push --tags
VERSION = "0.5.0"
ASR_MODEL = "mlx-community/whisper-large-v3-turbo"
LLM_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
LANGUAGE = None  # None = автоопределение; "ru" — жёстко русский
HOTKEY = keyboard.Key.alt_r  # правый Option
SAMPLE_RATE = 16000
MIN_DURATION = 0.4  # сек; короче — случайное нажатие, игнорируем

STATE = {"loading": True, "mic": "…", "enhance": True, "app": "", "error": "",
         "perms": {}, "last_hotkey": 0.0, "started": time.time(), "perms_ts": 0.0}

ROLES = {  # роль -> (заголовок раздела, [(HF-репозиторий, ~полный размер МБ, подпись)])
    "asr": ("Распознавание", [
        ("mlx-community/whisper-large-v3-turbo", 1600,
         "Whisper large-v3-turbo — быстрая"),
        ("mlx-community/whisper-large-v3-mlx", 3100,
         "Whisper large-v3 — точнее, в ~2 раза медленнее"),
        ("mlx-community/whisper-medium-mlx", 1500,
         "Whisper medium — лёгкая, качество ниже"),
        ("mlx-community/whisper-large-v3-turbo-q4", 500,
         "Whisper turbo 4-bit — для слабых машин, качество почти turbo"),
        ("mlx-community/whisper-small-mlx", 500,
         "Whisper small — совсем лёгкая, русский заметно хуже"),
    ]),
    "llm": ("Чистка текста", [
        ("mlx-community/Qwen3-4B-Instruct-2507-4bit", 2300,
         "Qwen3-4B — быстрая"),
        ("mlx-community/Qwen3-1.7B-4bit", 1000,
         "Qwen3-1.7B — для слабых машин"),
        ("RockTalk/GigaChat3.1-10B-A1.8B-MLX-4bit", 6000,
         "GigaChat 3.1 Lightning (MoE) — русскоцентричная, быстрая"),
        ("mlx-community/Qwen3-14B-4bit", 8300,
         "Qwen3-14B — качественнее, медленнее"),
        ("mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit", 17200,
         "Qwen3-30B-A3B (MoE) — лучшее качество"),
    ]),
}
ROLE_CFG = {"asr": "asr_model", "llm": "llm_model"}  # роль -> ключ в config.json

CONFIG_PATH = os.path.join(BASE, "config.json")
VOICEPRINT_PATH = os.path.join(BASE, "voiceprint.npz")
VOICEPRINT_OLD = os.path.join(BASE, "voiceprint.npy")  # формат до 0.5.0: только вектор
VP_RECORD_SEC = 12   # столько пишем при записи отпечатка
VP_CHECK_SEC = 4     # столько пишем при проверке «мой ли голос»
VP_MIN_SPEECH = 4.0  # минимум чистой речи (после VAD) для годного отпечатка
STYLES = {  # ключ -> подпись в меню
    "clean": "Чистка (по умолчанию)",
    "casual": "Разговорный (без точек)",
    "formal": "Строгий (письменный)",
    "raw": "Как сказано (без LLM)",
    "translate": "Перевод → EN",
}
CONFIG = {"default_style": "clean", "profiles": {}, "only_my_voice": False,
          "translate_all": False, "vp_threshold": 0.40, "enhance": True,
          "asr_model": ASR_MODEL, "llm_model": LLM_MODEL,
          **hud.DEFAULTS}  # индикатор записи и звуки


def notify_ui(title: str, message: str):
    """Показать окно с результатом из любого потока (ML-поток — не главный)."""
    from PyObjCTools import AppHelper
    AppHelper.callAfter(lambda: rumps.alert(title, message))


def load_voiceprint():
    """Отпечаток с диска: dict с вектором и данными записи, или None.

    Старый формат (.npy, один вектор без метаданных) читаем как есть — иначе
    после обновления у людей молча пропадал бы записанный отпечаток."""
    try:
        if os.path.exists(VOICEPRINT_PATH):
            d = np.load(VOICEPRINT_PATH, allow_pickle=False)
            return {"vec": d["vec"], "ts": float(d["ts"]), "device": str(d["device"]),
                    "windows": int(d["windows"]), "speech_sec": float(d["speech_sec"]),
                    "self_min": float(d["self_min"]), "self_mean": float(d["self_mean"])}
        if os.path.exists(VOICEPRINT_OLD):
            return {"vec": np.load(VOICEPRINT_OLD), "ts": os.path.getmtime(VOICEPRINT_OLD),
                    "device": "?", "windows": 1, "speech_sec": 0.0,
                    "self_min": 0.0, "self_mean": 0.0, "legacy": True}
    except Exception as e:
        print(f"  отпечаток голоса не прочитался ({e}) — запиши заново", flush=True)
    return None


def voiceprint_summary() -> str:
    """Строка для меню и панели: когда и на каком микрофоне записан."""
    vp = load_voiceprint()
    if vp is None:
        return "не записан"
    when = time.strftime("%d.%m.%Y", time.localtime(vp["ts"]))
    if vp.get("legacy"):
        return f"{when}, старый формат — перезапиши"
    warn = "" if vp["device"] == STATE.get("mic") else "  ⚠️ сейчас другой микрофон"
    return (f"{when}, {vp['device']}, {vp['speech_sec']:.0f}с речи, "
            f"порог {CONFIG['vp_threshold']}{warn}")


def load_config():
    """Прочитать config.json поверх дефолтов, отбросив мусор.

    Битый/частичный конфиг не должен ронять приложение: тип каждого значения
    сверяем с дефолтом, неизвестный стиль профиля выкидываем."""
    global ASR_MODEL, LLM_MODEL
    defaults = dict(CONFIG)
    try:
        with open(CONFIG_PATH) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("config.json — не объект")
        for key, val in data.items():
            ref = defaults.get(key)
            if ref is not None and not isinstance(val, type(ref)) and not (
                    isinstance(ref, (int, float)) and isinstance(val, (int, float))):
                print(f"  config: «{key}» неверного типа — беру дефолт", flush=True)
                continue
            CONFIG[key] = val
    except FileNotFoundError:
        pass
    except (ValueError, OSError, UnicodeDecodeError) as e:
        print(f"  config.json не прочитан ({e}) — работаю на дефолтах", flush=True)
    if not isinstance(CONFIG.get("profiles"), dict):
        CONFIG["profiles"] = {}
    CONFIG["profiles"] = {a: st for a, st in CONFIG["profiles"].items()
                          if isinstance(a, str) and st in STYLES}
    if CONFIG.get("default_style") not in STYLES:
        CONFIG["default_style"] = "clean"
    for role, cfg_key in ROLE_CFG.items():
        if not isinstance(CONFIG.get(cfg_key), str) or "/" not in CONFIG[cfg_key]:
            CONFIG[cfg_key] = defaults[cfg_key]
    ASR_MODEL = CONFIG["asr_model"]
    LLM_MODEL = CONFIG["llm_model"]
    STATE["enhance"] = bool(CONFIG["enhance"])
    hud.configure(CONFIG)


def save_config():
    # атомарно: обрыв на записи не должен оставить битый JSON (иначе при старте
    # слетают модели и снова спрашивается первая закачка)
    tmp = CONFIG_PATH + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(CONFIG, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, CONFIG_PATH)
    except OSError as e:
        print(f"  не сохранил config.json: {e}", flush=True)


def _dir_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            p = os.path.join(root, name)
            try:
                if not os.path.islink(p):  # snapshots в кэше HF — симлинки на blobs
                    total += os.path.getsize(p)
            except OSError:
                pass
    return total / 1e6


def _fmt_mb(mb: float) -> str:
    return f"{mb / 1000:.1f} ГБ" if mb >= 1000 else f"{mb:.0f} МБ"


def _repo_dir(repo: str) -> str:
    from huggingface_hub.constants import HF_HUB_CACHE
    return os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--"))


_repo_cache = {}  # repo -> (время, статус): os.walk по кэшу HF раз в 3 с, не чаще


def _repo_status(repo: str, full_mb, max_age=3.0) -> dict:
    """Модель в кэше HF: скачана / качается / нет, размер на диске, путь.

    «Качается» — в blobs есть свежий *.incomplete (осиротевший после прерванной
    закачки не считаем); «скачана» — в snapshots есть файлы и закачек нет."""
    import glob
    cached = _repo_cache.get(repo)
    if cached and time.time() - cached[0] < max_age:
        st = dict(cached[1])
        st["full"] = full_mb
        return st
    d = _repo_dir(repo)
    if not os.path.isdir(d):
        st = {"path": d, "state": "none", "mb": 0.0, "full": full_mb}
        _repo_cache[repo] = (time.time(), st)
        return st
    incomplete = [p for p in glob.glob(os.path.join(d, "blobs", "*.incomplete"))
                  if time.time() - os.path.getmtime(p) < 300]
    snaps = glob.glob(os.path.join(d, "snapshots", "*", "*"))
    # «скачана» — только если на месте веса и конфиг: HF кладёт симлинки по мере
    # загрузки, и прерванная закачка иначе выглядит готовой (а падает при старте)
    names = {os.path.basename(x) for x in snaps}
    has_weights = any(n.endswith((".safetensors", ".npz", ".bin", ".gguf"))
                      for n in names)
    has_cfg = any(n in ("config.json", "params.json") or n.endswith(".json")
                  for n in names)
    complete = has_weights and has_cfg
    if incomplete:
        state = "loading"
    elif complete:
        state = "done"
    elif snaps:
        state = "partial"
    else:
        state = "none"
    st = {"path": d, "state": state, "mb": _dir_size_mb(d), "full": full_mb}
    _repo_cache[repo] = (time.time(), st)
    return st


def _model_row(label: str, st: dict, active: bool) -> str:
    mark = "●" if active else "○"
    if st["state"] == "done":
        size = _fmt_mb(st["mb"])
    elif st["state"] == "loading":
        size = f"⏳ {_fmt_mb(st['mb'])} из ~{_fmt_mb(st['full'])}"
    elif st["state"] == "partial":
        size = f"⚠️ скачана частично ({_fmt_mb(st['mb'])} из ~{_fmt_mb(st['full'])})"
    else:
        size = f"не скачана (~{_fmt_mb(st['full'])})"
    return f"{mark} {label} · {size}"


def style_for(app: str) -> str:
    if CONFIG["translate_all"]:
        return "translate"
    return CONFIG["profiles"].get(app, CONFIG["default_style"])


recording = False
chunks = []
PREROLL_SEC = 0.5  # секунды звука ДО нажатия, подклеиваемые к записи
preroll = collections.deque(maxlen=64)  # (время, блок) — последние блоки микрофона
rec_seq = [0]  # номер текущей записи: капсулу прячет только «своя» диктовка
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


SKIP_MICS = ("NoMachine",)  # виртуальные/нежелательные устройства в запасном выборе


class NoMicrophone(Exception):
    """В системе нет ни одного пригодного входа (Mac Studio без встроенного
    микрофона, AirPods в кейсе). Не ошибка — просто ждём появления."""


def usable_inputs() -> list:
    """(индекс, устройство) — реальные входы, без виртуальных адаптеров."""
    try:
        devs = list(enumerate(sd.query_devices()))
    except Exception:
        return []
    return [(i, d) for i, d in devs
            if d["max_input_channels"] > 0
            and not any(x in d["name"] for x in SKIP_MICS)]


def _default_input(retries=6, delay=0.4):
    """Вход по умолчанию. При смене BT-профиля (A2DP↔HFP) устройство на миг
    исчезает и запрос падает с «device -1» — пробуем несколько раз,
    перечитывая список устройств между попытками."""
    with reopen_lock:
        for i in range(retries):
            try:
                return sd.query_devices(kind="input")
            except Exception:
                if i == retries - 1:
                    raise
                time.sleep(delay)
                _pa_recycle()


def pick_device():
    """Доверяем системному входу по умолчанию (macOS сам переключает при AirPods
    в кейсе). Пробуем его несколько раз — Bluetooth-микрофон просыпается не сразу.
    Если он всё же мёртв — предпочитаем ВСТРОЕННЫЙ микрофон, а не Continuity-iPhone.

    Если входа по умолчанию нет вовсе (Mac Studio, наушники в кейсе) —
    NoMicrophone: вочдог перейдёт в тихое ожидание вместо циклов переоткрытия."""
    try:
        default = _default_input()
    except Exception:
        default = None  # «device -1»: система не знает входа по умолчанию
    default_idx = default["index"] if default else None
    if default is not None:
        for _ in range(4):  # ~1.4 c на пробуждение BT-микрофона
            if probe_rms(default_idx) > 1e-5:
                return default_idx, default["name"], True
    live = usable_inputs()
    if not live and default is None:
        raise NoMicrophone("нет входных устройств")
    # запасной приоритет: встроенный микрофон MacBook, потом остальные живые
    for i, d in live:
        if "MacBook" in d["name"] and probe_rms(i) > 1e-5:
            return i, d["name"], False
    best_idx, best_name, best_rms = None, None, 0.0
    for i, d in live:
        if i == default_idx:
            continue
        rms = probe_rms(i)
        if rms > max(best_rms, 1e-5):
            best_idx, best_name, best_rms = i, d["name"], rms
    if best_idx is not None:
        return best_idx, best_name, False
    if default is not None:
        return default_idx, default["name"], True  # все молчат — остаёмся на дефолте
    i, d = live[0]  # вход есть, но молчит: откроем его, вочдог присмотрит
    return i, d["name"], False


# Переоткрытия идут из разных потоков (вотчер смены входа, нажатие Option,
# вочдог, стартовый open_stream): параллельные sd._terminate/_initialize ломают
# PortAudio так, что все запросы устройств возвращают -1 до рестарта. RLock —
# потому что reopen_stream зовёт open_stream, который берёт замок сам.
class _ReopenLock:
    """RLock, который умеет отвечать «сейчас переоткрывает ДРУГОЙ поток».

    threading.RLock.locked() появился только в Python 3.14, а знать это нужно:
    трогать PortAudio, пока другой поток делает _terminate/_initialize, — путь
    к падению процесса."""

    def __init__(self):
        self._lock = threading.RLock()
        self._owner = None
        self._depth = 0

    def acquire(self, blocking=True):
        got = self._lock.acquire(blocking)
        if got:
            self._owner = threading.get_ident()
            self._depth += 1
        return got

    def release(self):
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()

    def busy(self) -> bool:
        return self._depth > 0 and self._owner != threading.get_ident()


reopen_lock = _ReopenLock()
_devices_cache = {"names": [], "ts": 0.0}  # для UI: без похода в PortAudio


def _close_stream():
    old = stream_holder.pop("stream", None)
    if old:
        try:
            old.stop()
            old.close()
        except Exception:
            pass


def _pa_recycle():
    """Перечитать список устройств CoreAudio. Только под reopen_lock!

    Pa_Terminate закрывает все потоки и освобождает память — свой стрим
    обязаны закрыть до этого, иначе последующий close() читает освобождённое."""
    _close_stream()
    try:
        sd._terminate()
    except Exception:
        pass
    try:
        sd._initialize()
    except Exception as e:
        print(f"  PortAudio не переинициализировался: {e}", flush=True)


def reopen_stream(follow_default=False):
    """Полный перезапуск аудио: закрыть поток, перечитать устройства CoreAudio, открыть заново."""
    with reopen_lock:
        _pa_recycle()
        open_stream(follow_default=follow_default)


def stream_alive() -> bool:
    """Поток открыт и колбэки идут (пульс не старше 2 с)."""
    if reopen_lock.busy():
        return False  # идёт переоткрытие: трогать закрываемый поток нельзя
    s = stream_holder.get("stream")
    try:
        return bool(s) and s.active and time.time() - stream_holder.get("last_cb", 0) < 2.0
    except Exception:
        return False


def ensure_stream():
    """Перед записью: если поток умер или пульс пропал (микрофон отвалился) — переоткрыть."""
    if stream_alive():
        return
    if reopen_lock.busy():
        return  # кто-то уже переоткрывает — не вставать в очередь
    print("  микрофон пропал — переоткрываю...", flush=True)
    try:
        # follow_default: без проб устройств, окно потери звука минимально;
        # если дефолт окажется мёртвым, сработает фолбэк по тихой записи
        reopen_stream(follow_default=True)
    except NoMicrophone:
        STATE["mic"] = "нет — подключи микрофон"
        print("  микрофона нет — запись не пойдёт, подключи AirPods или USB-микрофон",
              flush=True)
    except Exception as e:
        print(f"  не удалось открыть микрофон: {e}", flush=True)


def open_stream(follow_default=False):
    """Открыть входной поток. Замок берём сам: стартовый вызов из main() иначе
    гонится с вотчером смены входа (AirPods переключают профиль при открытии)."""
    with reopen_lock:
        _close_stream()
        if follow_default:
            # смена входа по умолчанию — это действие пользователя, верим без проб
            try:
                d = _default_input(retries=2)
                dev, name, is_default = d["index"], d["name"], True
            except Exception:
                dev, name, is_default = pick_device()  # дефолта нет — ищем сами
        else:
            dev, name, is_default = pick_device()
        s = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                           device=dev, callback=audio_callback)
        s.start()
        stream_holder["stream"] = s
        try:  # список входов для окна состояния — обновляем, пока PortAudio наш
            _devices_cache["names"] = [d["name"] for d in sd.query_devices()
                                       if d["max_input_channels"] > 0]
            _devices_cache["ts"] = time.time()
        except Exception:
            pass
    note = "" if is_default else "  (вход по умолчанию молчит — взял живой)"
    STATE["mic"] = name
    print(f"Микрофон: {name}{note}", flush=True)


mic_changed = threading.Event()


def stream_watchdog():
    """Мёртвый поток чинится сам, не дожидаясь событий CoreAudio или нажатия.

    Если входов нет вообще (Mac Studio без встроенного микрофона, AirPods
    в кейсе) — тихо ждём появления, опрашивая раз в 3 с. Иначе — полное
    переоткрытие с пробами устройств; интервал неудачных попыток растёт
    до 30 с, чтобы не заливать лог."""
    fails = 0
    waiting = False
    while True:
        time.sleep(min(30.0, 3.0 * (fails + 1)))
        if recording or stream_alive() or reopen_lock.busy():
            fails = 0
            waiting = False
            continue
        with reopen_lock:  # перечитка списка устройств, сериализовано
            _pa_recycle()
            live = usable_inputs()
            try:
                _devices_cache["names"] = [d["name"] for d in sd.query_devices()
                                           if d["max_input_channels"] > 0]
                _devices_cache["ts"] = time.time()
            except Exception:
                pass
        if not live:
            if not waiting:
                print("  входных устройств нет (AirPods в кейсе?) — жду появления...",
                      flush=True)
                waiting = True
            STATE["mic"] = "нет — подключи микрофон"
            fails = 0  # ждём молча, опрос каждые 3 с
            continue
        waiting = False
        try:
            print("  поток мёртв — вочдог переоткрывает...", flush=True)
            reopen_stream()
            fails = 0
        except NoMicrophone:
            if not waiting:
                print("  микрофона нет — жду появления...", flush=True)
                waiting = True
            STATE["mic"] = "нет — подключи микрофон"
            fails = 0
        except Exception as e:
            fails += 1
            print(f"  вочдог: не удалось ({e}), следующая попытка через "
                  f"{min(30, 3 * (fails + 1))} с", flush=True)


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
        # открытие потока на AirPods само переводит их A2DP→HFP, и CoreAudio
        # сыплет новые события «вход сменился» — глотаем их, иначе цикл
        time.sleep(2.0)
        mic_changed.clear()


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
    """Словарь в initial_prompt: Whisper подхватывает термины при распознавании.

    Служебных слов («Словарь:», «Глаголы:») в подсказке быть не должно — на
    тихих записях Whisper выдаёт их эхом и они протекают в готовый текст."""
    terms = load_terms()
    return f"{terms}, задеплоить." if terms else ""


def system_prompt() -> str:
    return (
        "Ты корректор надиктованного текста. Правила:\n"
        "1. Убери слова-паразиты (эээ, ну, короче, эм) и оговорки. Значимые слова "
        "(нужно, надо, давай, проверь) паразитами НЕ являются — сохраняй их.\n"
        "2. Исправляй ТОЛЬКО искажённые распознаванием слова. Грамматику, падежи, "
        "наклонение, порядок слов и смысл НЕ меняй. Ничего не добавляй и не пересказывай.\n"
        f"3. Термины пользователя (только контекст): {load_terms()}. НИКОГДА не "
        "заменяй обычное слово термином из списка и один термин другим — исправляй "
        "словом из списка только явную ослышку, созвучную ему почти целиком.\n"
        "4. Слитные глаголы, разбитые на части, склей: «За деплой сервис» → «Задеплой сервис».\n"
        "5. Числа, цифры, номера, IP-адреса, суммы НИКОГДА не меняй — "
        "переноси ровно как в исходнике.\n"
        "6. Если исправлять нечего — верни текст дословно.\n"
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
    # метрики скорости — добавляем к существующей таблице, если их ещё нет
    have = {r[1] for r in db.execute("PRAGMA table_info(transcriptions)")}
    for col in ("style TEXT", "asr_ms REAL", "llm_ms REAL",
                "gen_tps REAL", "gen_tokens INTEGER", "vp_sim REAL"):
        if col.split()[0] not in have:
            db.execute(f"ALTER TABLE transcriptions ADD COLUMN {col}")
    return db


MAX_REC_SEC = 600  # предохранитель: забытый toggle не пишет вечно — авто-стоп и обработка
enroll_buf = {"on": False, "chunks": []}  # захват отпечатка/проверки голоса
rec_frames = [0]  # счётчик сэмплов текущей записи (под lock)
overflow_sent = [False]  # авто-стоп ставится один раз на запись, а не на каждый колбэк

def audio_callback(indata, frames, t, status):
    now = time.time()
    stream_holder["last_cb"] = now  # пульс: колбэки идут, пока устройство живо
    overflow = False
    with lock:
        preroll.append((now, indata.copy()))
        if enroll_buf["on"]:
            # отпечаток пишем ИЗ ЭТОГО ЖЕ потока, а не отдельным sd.rec: иначе
            # запись шла с другого устройства, чем диктовка, и эмбеддинги
            # разных микрофонов не сходились ни при каком пороге
            enroll_buf["chunks"].append(indata.copy())
            hud.push_level(float(np.sqrt((indata ** 2).mean())))
        if recording:
            chunks.append(indata.copy())
            rec_frames[0] += len(indata)
            hud.push_level(float(np.sqrt((indata ** 2).mean())))
            if rec_frames[0] > MAX_REC_SEC * SAMPLE_RATE and not overflow_sent[0]:
                overflow_sent[0] = True
                overflow = True
    if overflow:
        print(f"  ⏱ запись дольше {MAX_REC_SEC // 60} мин — авто-стоп и обработка", flush=True)
        threading.Thread(target=stop_and_submit, daemon=True).start()


FILLERS = re.compile(
    r"\b(э+м*|а+м+|мда+|ну|короче|как бы|типа|это самое|в общем|значит)\b|(?<![\w'’])м(?![\w'’])",
    re.IGNORECASE)
FILLER_WORDS = {"эээ", "эм", "ну", "короче", "типа", "мда", "м", "m", "как", "бы",
                "это", "самое", "в", "общем", "значит", "а", "э"}
# частые слова: низкая пословная уверенность на них — не повод чинить фразу
STOP_DOUBT = {"давай", "можно", "вообще", "просто", "очень", "когда", "если",
              "чтобы", "нужно", "надо", "есть", "было", "этот", "который"}


def _norm(s: str) -> str:
    return re.sub(r"[^\wёЁ-]+", " ", s).strip().lower()


_TRANSLIT = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
             "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
             "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
             "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
             "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya", " ": ""}


def _roman(s: str) -> str:
    # грубая романизация для сравнения кириллицы с латинскими терминами
    # (зиро тир ~ zerotier, флинт ~ flint)
    return "".join(_TRANSLIT.get(c, c) for c in s.lower())


def guard_correction(raw: str, out: str, terms_lower: set) -> str:
    """Детерминированный пост-контроль LLM: замену слова принимаем, только если
    она созвучна исходному (обычный порог 0.55; если подставлен термин из
    словаря — строгий 0.75). Удаления принимаем только для слов-паразитов.
    Всё отклонённое откатывается к сырому тексту Whisper."""
    import difflib
    a, b = raw.split(), out.split()
    sm = difflib.SequenceMatcher(a=[_norm(w) for w in a], b=[_norm(w) for w in b])
    result = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            result.extend(b[j1:j2])
        elif op == "delete":
            src_words = [_norm(w) for w in a[i1:i2]]
            if not all(w in FILLER_WORDS for w in src_words if w):
                result.extend(a[i1:i2])  # удалили не паразит — вернуть
        elif op == "insert":
            pass  # LLM не имеет права дописывать сама по себе
        else:  # replace
            src_raw, dst_raw = " ".join(a[i1:i2]), " ".join(b[j1:j2])
            # числа неприкосновенны: Whisper их берёт точно, «ремонт по созвучию»
            # только портит (162→616). Изменились цифры — откат к сырому.
            if re.findall(r"\d", src_raw + dst_raw) and \
                    re.findall(r"\d+", src_raw) != re.findall(r"\d+", dst_raw):
                result.extend(a[i1:i2])
                continue
            src, dst = _norm(src_raw), _norm(dst_raw)
            sim = difflib.SequenceMatcher(a=src, b=dst).ratio()
            # кросс-скрипт: «флинт»/«зиро тир» vs латинский термин — сравним романизацию
            sim = max(sim, difflib.SequenceMatcher(a=_roman(src), b=_roman(dst)).ratio())
            # подстановка термина (в любой словоформе: «роутера» ~ «роутер») —
            # строгая планка созвучности
            is_term = any(t and (dw.startswith(t) or t.startswith(dw))
                          for dw in dst.split() for t in terms_lower)
            bar = 0.75 if is_term else 0.55
            if sim >= bar or all(w in FILLER_WORDS for w in src.split()):
                result.extend(b[j1:j2])
            else:
                result.extend(a[i1:i2])  # несозвучная замена — откат
    return " ".join(result)


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
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps
        vad = load_silero_vad(onnx=True)
        from speechbrain.inference.speaker import EncoderClassifier
        spk = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                             savedir=os.path.join(BASE, "models/ecapa"))
        ModelHolder.get_model(ASR_MODEL, mx.float16)
        from mlx_lm import load, stream_generate
        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
        llm, tok = load(LLM_MODEL)
        last_stats = {}  # заполняется llm_run: gen_tps, prompt_tps, gen_tokens
        pcache = {"cache": make_prompt_cache(llm), "tokens": []}  # KV-кэш префикса промпта

        for _ in stream_generate(llm, tok, prompt=tok.apply_chat_template(
                [{"role": "user", "content": "ок"}], add_generation_prompt=True),
                max_tokens=4):
            pass  # прогрев, чтобы первая диктовка была быстрой
        db = history_db()
    except Exception as e:
        # без моделей диктовать нечем: показываем причину в меню и окне
        # состояния (иначе иконка вечно «⏳», а нажатия копятся в очереди)
        STATE["error"] = f"модели не загрузились: {e}"
        STATE["loading"] = False
        print(f"✗ Модели не загрузились: {e}\n  Проверь сеть и «Модели» в меню; "
              f"после починки — «Перезапустить» в окне состояния.", flush=True)
        import traceback
        traceback.print_exc()
        ready.set()
        return
    _vp = load_voiceprint()
    voiceprint = _vp["vec"] if _vp else None
    STATE["loading"] = False
    ready.set()

    def rebuild_autodict():
        try:
            import suggest_terms
            # llm_run отключён: отбор жаргона 4B-моделью тянул обычные слова,
            # а замусоренный словарь провоцировал подстановки при чистке
            added = suggest_terms.build_auto_terms(llm_run=None)
            print(f"Автословарь обновлён: {', '.join(added) if added else 'пусто'}",
                  flush=True)
        except Exception as e:
            print(f"  автословарь не собрался: {e}", flush=True)

    def embed(audio: np.ndarray) -> np.ndarray:
        e = spk.encode_batch(torch.from_numpy(audio).unsqueeze(0)).squeeze().numpy()
        n = np.linalg.norm(e)
        if not np.isfinite(n) or n < 1e-9:  # тишина -> нули -> деление дало бы NaN
            raise ValueError("пустой эмбеддинг (тишина?)")
        return e / n

    def speech_only(audio: np.ndarray) -> np.ndarray:
        """Оставить только речь. Отпечаток и проверка ОБЯЗАНЫ идти через это:
        при сравнении диктовка уже обрезана по VAD, и эмбеддинг «5 секунд, где
        речи две» с ней не сходился — из-за этого «только мой голос» отсекал
        собственного хозяина."""
        spans = get_speech_timestamps(torch.from_numpy(audio), vad,
                                      sampling_rate=SAMPLE_RATE, speech_pad_ms=150,
                                      threshold=0.35)
        if not spans:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([audio[s["start"]:s["end"]] for s in spans])

    def embed_windows(speech: np.ndarray) -> list:
        """Эмбеддинги окон по 3 с с шагом 1.5 с — устойчивее одного снимка."""
        win, hop = int(3 * SAMPLE_RATE), int(1.5 * SAMPLE_RATE)
        if len(speech) <= win:
            return [embed(speech)]
        return [embed(speech[i:i + win])
                for i in range(0, len(speech) - win + 1, hop)]

    def make_voiceprint(audio: np.ndarray) -> dict:
        """Отпечаток из записи: усреднённый эмбеддинг окон + разброс между ними.

        Разброс — это «насколько мой голос похож сам на себя» в этих условиях;
        по нему подбираем порог, а не берём константу с потолка."""
        speech = speech_only(audio)
        secs = len(speech) / SAMPLE_RATE
        if secs < VP_MIN_SPEECH:
            raise ValueError(f"речи всего {secs:.1f}с из нужных {VP_MIN_SPEECH:.0f}с — "
                             "говори подряд, без длинных пауз")
        embs = embed_windows(speech)
        mean = np.mean(embs, axis=0)
        mean /= np.linalg.norm(mean)
        sims = [float(e @ mean) for e in embs]
        low = min(sims)
        # Порог. Окна одной записи похожи между собой куда сильнее, чем записи
        # разных дней (другой микрофон, простуда, расстояние до рта), поэтому
        # строже 0.40 не берём никогда — иначе назавтра фильтр отбросит хозяина.
        # Разброс окон используем только чтобы ОСЛАБИТЬ порог, если запись вышла
        # шумной. Замер на синтезированной речи: чужой голос даёт 0.04…0.11,
        # запас до 0.40 огромный, а вот свой после смены микрофона проседает.
        thr = round(min(0.40, max(0.28, low - 0.30)), 2)
        return {"vec": mean, "speech_sec": secs, "windows": len(embs),
                "self_min": low, "self_mean": float(np.mean(sims)),
                "threshold": thr, "device": STATE["mic"], "ts": time.time()}

    def save_voiceprint(vp: dict):
        np.savez(VOICEPRINT_PATH, vec=vp["vec"], ts=vp["ts"], device=vp["device"],
                 windows=vp["windows"], speech_sec=vp["speech_sec"],
                 self_min=vp["self_min"], self_mean=vp["self_mean"])

    def llm_run(system: str, user: str, max_factor: int = 2) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
        cache, cached = pcache["cache"], pcache["tokens"]
        # общий префикс с тем, что уже лежит в KV-кэше (константные правила+словарь)
        common = 0
        for x, y in zip(cached, prompt):
            if x != y:
                break
            common += 1
        # хотя бы один токен должен уйти в генерацию: при полном совпадении
        # (повторили ту же фразу) stream_generate с пустым промптом падает
        common = min(common, len(prompt) - 1)
        try:
            if len(cached) > common:  # откатить кэш до общего префикса
                trim_prompt_cache(cache, len(cached) - common)
            feed = prompt[common:]  # обрабатываем только новый хвост
            parts, resp = [], None
            for resp in stream_generate(llm, tok, prompt=feed, prompt_cache=cache,
                                        max_tokens=len(tok.encode(user)) * max_factor + 40):
                parts.append(resp.text)
            if resp is not None:
                trim_prompt_cache(cache, resp.generation_tokens)  # снять сгенерённое
                pcache["tokens"] = prompt
                last_stats.update(gen_tps=resp.generation_tps, prompt_tps=resp.prompt_tps,
                                  gen_tokens=resp.generation_tokens)
            return "".join(parts).strip()
        except Exception as e:  # кэш рассинхронился — сбрасываем и идём без него
            print(f"  кэш промпта сброшен: {e}", flush=True)
            pcache["cache"], pcache["tokens"] = make_prompt_cache(llm), []
            parts, resp = [], None
            for resp in stream_generate(llm, tok, prompt=prompt,
                                        max_tokens=len(tok.encode(user)) * max_factor + 40):
                parts.append(resp.text)
            if resp is not None:
                last_stats.update(gen_tps=resp.generation_tps, prompt_tps=resp.prompt_tps,
                                  gen_tokens=resp.generation_tokens)
            return "".join(parts).strip()

    def enhance(raw: str, formal: bool = False, doubtful=None) -> str:
        system = system_prompt() + (FORMAL_PROMPT_ADDON if formal else "")
        if doubtful:
            system += ("\nДополнительно: распознаватель не уверен в словах: "
                       + ", ".join(f"«{w}»" for w in doubtful[:8])
                       + " — возможны ослышки (города, имена, разорванные слова). "
                       "Исправляй, только если ПО КОНТЕКСТУ очевидно, что было "
                       "сказано на самом деле, и исправление созвучно исходному. "
                       "Сомневаешься — оставь как есть. Остальные слова не трогай.")
        out = llm_run(system, raw)
        # деградация LLM (пусто / разнесло в разы) — откатываемся на сырой текст
        if not out or len(out) > len(raw) * 2 + 40:
            return raw
        return out

    rebuild_autodict()

    while True:
        kind, payload = jobs.get()
        if kind == "autodict":
            rebuild_autodict()
            continue
        if kind == "enroll":
            try:
                vp = make_voiceprint(payload)
                save_voiceprint(vp)
                voiceprint = vp["vec"]
                CONFIG["vp_threshold"] = vp["threshold"]  # порог от разброса, не с потолка
                save_config()
                msg = (f"Отпечаток записан: {vp['speech_sec']:.0f}с речи, "
                       f"{vp['windows']} окон, микрофон «{vp['device']}».\n"
                       f"Похожесть окон на себя: {vp['self_min']:.2f}…{vp['self_mean']:.2f}.\n"
                       f"Порог подобран автоматически: {vp['threshold']}.")
                print(f"Отпечаток голоса сохранён ({vp['speech_sec']:.0f}с речи, "
                      f"окон {vp['windows']}, само-похожесть {vp['self_min']:.2f}…"
                      f"{vp['self_mean']:.2f}, порог {vp['threshold']})", flush=True)
                notify_ui("Отпечаток голоса записан", msg)
            except Exception as e:
                print(f"  ✗ отпечаток не записан: {e}", flush=True)
                notify_ui("Отпечаток не записан", f"{e}\n\nПопробуй ещё раз: говори "
                          "непрерывно, обычным голосом, в тот же микрофон.")
            continue
        if kind == "vpcheck":
            try:
                if voiceprint is None:
                    raise ValueError("отпечаток ещё не записан")
                speech = speech_only(payload)
                if len(speech) < 1.0 * SAMPLE_RATE:
                    raise ValueError("речи не слышно — скажи фразу вслух")
                sim = float(embed(speech) @ voiceprint)
                thr = CONFIG["vp_threshold"]
                verdict = ("✅ Узнаю — это твой голос" if sim >= thr else
                           "❌ Не узнаю — такую диктовку я бы отбросил")
                print(f"  проверка голоса: сходство {sim:.2f} при пороге {thr}", flush=True)
                notify_ui("Проверка голоса",
                          f"{verdict}\n\nСходство {sim:.2f}, порог {thr}.\n"
                          f"Микрофон: {STATE['mic']}.\n\n"
                          + ("Запас хороший." if sim >= thr + 0.1 else
                             "Запас маленький: перезапиши отпечаток на этом микрофоне "
                             "или сделай строгость мягче."))
            except Exception as e:
                print(f"  ✗ проверка голоса не вышла: {e}", flush=True)
                notify_ui("Проверка голоса", f"Не получилось: {e}")
            continue
        audio, rec_app, token = payload
        ok = False
        try:
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
            # провалы в нули ВНУТРИ записи = Bluetooth перещёлкивал профиль
            # (звук в AirPods, вызов и т.п.) — речь в этих местах потеряна
            if len(nz) > 1:
                gaps = np.diff(nz)
                big = gaps[gaps > SAMPLE_RATE * 0.2]
                if len(big):
                    print(f"  ⚠ внутри записи {len(big)} провал(а) в нули, суммарно "
                          f"{big.sum() / SAMPLE_RATE:.1f}с — микрофон отваливался "
                          f"(Bluetooth-профиль?)", flush=True)
            # VAD: есть ли вообще речь, и если есть — обрезать тишину по краям
            spans = get_speech_timestamps(torch.from_numpy(audio), vad,
                                          sampling_rate=SAMPLE_RATE, speech_pad_ms=150,
                                          threshold=0.35)
            if not spans:
                rms = float(np.sqrt((audio ** 2).mean()))
                peak = float(np.abs(audio).max())
                hint = ("захват почти пустой — микрофон не тот/тихий, "
                        if peak < 0.02 else "сигнал есть, но VAD не распознал речь, ")
                print(f"  ✗ речи не слышно ({hint}RMS={rms:.4f} peak={peak:.3f}) — "
                      f"не вставляю", flush=True)
                continue
            audio = audio[max(0, spans[0]["start"] - SAMPLE_RATE // 4):
                          spans[-1]["end"] + SAMPLE_RATE // 10]
            # отпечаток голоса: чужую речь (ТВ, коллеги) не транскрибируем
            vp_sim = None
            if CONFIG["only_my_voice"] and voiceprint is not None:
                try:
                    vp_sim = float(embed(audio) @ voiceprint)
                except Exception as e:
                    # сбой сравнения не повод терять диктовку — пропускаем дальше
                    print(f"  ⚠ отпечаток не сравнился ({e}) — вставляю без проверки",
                          flush=True)
                if vp_sim is not None:
                    STATE["vp_last"] = vp_sim  # видно в панели: есть ли запас до порога
                if vp_sim is not None and vp_sim < CONFIG["vp_threshold"]:
                    near = ("  Почти прошло — «Мой голос → Строгость» мягче или "
                            "перезапиши отпечаток."
                            if vp_sim > CONFIG["vp_threshold"] - 0.1 else "")
                    print(f"  ✗ не твой голос (сходство {vp_sim:.2f} < "
                          f"{CONFIG['vp_threshold']}) — не вставляю.{near}", flush=True)
                    hud.play("error")
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
            # слова, в которых Whisper сам не уверен, — кандидаты на ослышку.
            # Первое слово, короткие и частые слова не считаем: у них низкая
            # вероятность в норме, а «ремонт» по ним переписывает смысл
            all_words = [w for s in result["segments"] for w in s.get("words", [])]
            doubtful = []
            for i, w in enumerate(all_words):
                word = w["word"].strip()
                core = re.sub(r"[^\wёЁ-]", "", word)
                if (w["probability"] < 0.6 and i > 0 and len(core) >= 4
                        and core.lower() not in STOP_DOUBT
                        and not re.search(r"\d", core)):  # числа не «чиним»
                    doubtful.append(word)
            t_asr = time.time() - t0
            if not raw:
                continue
            # тихое аудио + initial_prompt => Whisper галлюцинирует куски словаря
            raw_words = re.findall(r"\w+", raw.lower())
            hint_words = set(re.findall(r"\w+", asr_hint().lower()))
            # эхо словаря — это перечисление НЕСКОЛЬКИХ терминов подряд на тихой
            # записи; одиночный термин («задеплой», «ZeroTier») — нормальная
            # диктовка, её раньше молча съедали
            if (len(raw_words) >= 2 and set(raw_words) <= hint_words
                    and len(set(raw_words)) >= 2):
                print(f"  ✗ похоже на эхо словаря, не вставляю: {raw}", flush=True)
                continue
            app = rec_app or frontmost_app()
            style = style_for(app)
            text = raw
            t_llm = 0.0
            last_stats.clear()  # сбрасываем перед возможным запуском LLM
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
                    terms_lower = {t.strip().lower() for t in load_terms().split(",")}
                    guarded = guard_correction(raw, text, terms_lower)
                    if guarded != text:
                        print(f"  ⛔ пост-контроль откатил часть правок LLM", flush=True)
                        text = guarded
            except Exception as e:
                print(f"  ошибка обработки (вставляю сырой): {e}", flush=True)
            t_llm = time.time() - t1
            if style == "casual":
                text = text.rstrip(".")
            else:
                text = strip_short_period(text)
            paste_text(text)
            ok = True
            gen_tps = last_stats.get("gen_tps")
            gen_tokens = last_stats.get("gen_tokens")
            db.execute(
                "INSERT INTO transcriptions (ts, text, raw_text, duration, app, "
                "style, asr_ms, llm_ms, gen_tps, gen_tokens, vp_sim) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), text, raw, duration, app, style,
                 round(t_asr * 1000), round(t_llm * 1000), gen_tps, gen_tokens, vp_sim))
            db.commit()
            mark = "" if text == strip_short_period(raw) else f"  (сырой: {raw})"
            doubt = f"  [сомнения: {', '.join(doubtful[:5])}]" if doubtful else ""
            speed = f" @{gen_tps:.0f}т/с" if gen_tps else ""
            # сходство печатаем и при успехе: иначе непонятно, есть ли запас до порога
            vp = f" голос {vp_sim:.2f}" if vp_sim is not None else ""
            rtf = duration / t_asr if t_asr else 0
            print(f"  [{duration:.1f}s аудио → asr {t_asr:.1f}s (×{rtf:.0f}) + "
                  f"llm {t_llm:.1f}s{speed}{vp} → {app}/{style}] {text}{mark}{doubt}",
                  flush=True)
        except Exception as e:
            # любое необработанное исключение раньше убивало поток навсегда:
            # иконка «готов», хоткей пишет, а текст не вставляется никогда
            print(f"  ✗ ошибка обработки диктовки: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            # обратная связь: прячем капсулу ТОЛЬКО своей записи (пока шло
            # распознавание, пользователь мог начать новую), звук — по итогу
            hud.hide(token)
            hud.play("done" if ok else "error")


TAP_MAX = 0.35  # сек: короче — «тап» (toggle-режим), дольше — классический push-to-talk

toggle_mode = False
press_time = 0.0
_RALT_MASK = 0x40  # NX_DEVICERALTKEYMASK — бит именно правого Option


def stop_and_submit():
    """Закончить запись и отдать аудио в ML-поток. Идемпотентна: авто-стоп по
    таймеру и отпускание клавиши могут прийти одновременно."""
    global recording, toggle_mode
    with lock:  # снимаем флаг и забираем аудио атомарно, иначе новое нажатие
        if not recording:  # (chunks.clear() в on_press) съедает готовую запись
            return
        recording = False
        toggle_mode = False
        audio = (np.concatenate(chunks).flatten().astype(np.float32)
                 if chunks else np.zeros(0, dtype=np.float32))
        spoken = rec_frames[0] / SAMPLE_RATE  # без приклеенного preroll
        chunks.clear()
        token = rec_seq[0]
    if spoken >= MIN_DURATION and len(audio):
        hud.show("busy", token)  # капсула: «распознаю…» до вставки/отказа
        # приложение-получатель фиксируем СЕЙЧАС: пока идёт распознавание,
        # фронтальным может стать другое окно, и профиль/вставка уедут не туда
        jobs.put(("dictate", (audio, frontmost_app(), token)))
    else:
        hud.hide(token)


def cancel_recording():
    global recording, toggle_mode
    with lock:
        recording = False
        toggle_mode = False
        chunks.clear()
        token = rec_seq[0]
    hud.hide(token)
    print("  ✗ запись отменена (Esc)", flush=True)


def start_recording():
    global recording, press_time
    if enroll_buf["on"]:
        # идёт запись отпечатка/проверки: два захвата с одного потока перепутают
        # звук между собой
        print("  ⏸ сейчас пишется отпечаток голоса — договорим и диктуй", flush=True)
        hud.play("error")
        return
    if STATE["loading"]:
        # модели ещё греются: записанное всё равно вставится минут через
        # несколько и не туда — честно отказываем сразу
        print("  ⏳ модели ещё грузятся — диктовка будет доступна через несколько секунд",
              flush=True)
        hud.play("error")
        return
    # флаг записи — сразу, проверка/оживление потока — в фоне: колбэки начнут
    # наполнять chunks в ту же миллисекунду, как поток жив
    threading.Thread(target=ensure_stream, daemon=True).start()
    now = time.time()
    with lock:
        chunks.clear()
        rec_frames[0] = 0
        overflow_sent[0] = False
        rec_seq[0] += 1
        token = rec_seq[0]
        # подклеиваем последние PREROLL_SEC до нажатия — первое слово не режется,
        # даже если начал говорить одновременно с клавишей. Блоки старше секунды
        # не берём: после мёртвого потока в буфере лежит речь минутной давности
        need = int(PREROLL_SEC * SAMPLE_RATE)
        got = 0
        for ts, block in reversed(preroll):
            if now - ts > 1.0:
                break
            chunks.insert(0, block)
            got += len(block)
            if got >= need:
                break
        recording = True
    press_time = now
    STATE["last_hotkey"] = now
    hud.play("start")
    hud.show("rec", token)
    print("● запись...", flush=True)


def on_press(key):
    try:
        if key == keyboard.Key.esc and recording:
            cancel_recording()
            return
        if key != HOTKEY:
            return
        if not recording:
            start_recording()
        else:
            # второй тап в toggle-режиме — стоп; в push-to-talk сюда попадаем,
            # если отпускание потерялось (был зажат левый Option: macOS отдаёт
            # общую маску Alternate) — тоже трактуем как стоп, иначе запись висит
            if not toggle_mode:
                print("  (отпускание не пришло — останавливаю по повторному нажатию)",
                      flush=True)
            stop_and_submit()
    except Exception as e:  # исключение в колбэке останавливает слушателя клавиш
        print(f"  ошибка обработки нажатия: {e}", flush=True)


def on_release(key):
    global toggle_mode
    try:
        if key != HOTKEY or not recording:
            return
        hold = time.time() - press_time
        if hold < TAP_MAX:
            toggle_mode = True  # короткий тап: пишем дальше до второго тапа или Esc
            print(f"  … toggle-режим (тап {hold:.2f}s): говори, "
                  "ещё один тап Option — стоп, Esc — отмена", flush=True)
        else:
            stop_and_submit()  # классика: отпустил — обрабатываем
    except Exception as e:
        print(f"  ошибка обработки отпускания: {e}", flush=True)


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
        self.voice_menu = self._build_voice_menu()

        self.models_menu = rumps.MenuItem("Модели")
        # версия в меню: видно без открытия панели, клик — копия для отчёта об ошибке
        self.about_item = rumps.MenuItem(f"Версия {app_version()}", callback=self.copy_version)
        self.status_item = rumps.MenuItem("Состояние и разрешения…", callback=self.open_status)
        self.perm_item = rumps.MenuItem("Настроить разрешения…", callback=self.open_perm_wizard)
        self.hud_menu = self._build_hud_menu()

        self.menu = [self.status_item, self.perm_item, self.mic_item, self.recent, None,
                     self.profile, self.default_style, self.translate_item, None,
                     self.voice_menu,
                     None,
                     self.enh_item,
                     self.hud_menu,
                     self.models_menu,
                     rumps.MenuItem("Статистика…", callback=self.open_stats),
                     rumps.MenuItem("Поиск истории…", callback=self.open_search),
                     rumps.MenuItem("Словарь терминов…", callback=self.open_terms),
                     rumps.MenuItem("Обновить автословарь из истории", callback=self.suggest),
                     rumps.MenuItem("Лог…", callback=self.open_log), None,
                     self.about_item]
        rumps.Timer(self.refresh_title, 0.3).start()
        rumps.Timer(self.refresh_recent, 3.0).start()
        self.refresh_models(None)
        rumps.Timer(self.refresh_models, 5.0).start()
        rumps.Timer(self.refresh_status, 1.0).start()
        self.refresh_voice_menu()
        rumps.Timer(self.refresh_voice_menu, 3.0).start()
        self._version = app_version()
        if STATE.get("show_status_on_start"):
            # первый запуск / нет разрешений / модели ещё качаются — открываем окно
            # состояния, когда NSApp уже крутит цикл (не из __init__)
            self._boot = rumps.Timer(self._boot_show_status, 1.5)
            self._boot.start()

    def _boot_show_status(self, _):
        self._boot.stop()
        self.open_status(None, activate=False)  # при логине фокус не отбираем

    # --- мой голос ------------------------------------------------------------
    def _build_voice_menu(self):
        """Всё про отпечаток в одном месте: состояние, запись, проверка, строгость."""
        m = rumps.MenuItem("Мой голос")
        self.vp_status = rumps.MenuItem("Отпечаток: …")  # без callback — просто строка
        self.vp_item = rumps.MenuItem("Пропускать только мой голос",
                                      callback=self.toggle_voice)
        self.vp_check = rumps.MenuItem("Проверить: узнаю ли я тебя…")
        m.add(self.vp_status)
        m.add(rumps.separator)
        m.add(self.vp_item)
        m.add(rumps.MenuItem("Записать отпечаток заново…", callback=self.enroll))
        m.add(self.vp_check)
        strict = rumps.MenuItem("Строгость")
        self.vp_strict = {}
        for label, thr in [("Строго — чужих точно нет", 0.55),
                           ("Обычно", 0.40),
                           ("Мягко — лишь бы своё не терять", 0.28)]:
            it = rumps.MenuItem(label, callback=self.set_vp_strictness)
            it._thr = thr
            strict.add(it)
            self.vp_strict[label] = it
        m.add(strict)
        return m

    # --- индикатор записи и звуки -------------------------------------------
    def _build_hud_menu(self):
        m = rumps.MenuItem("Индикатор и звуки")
        self.hud_on = rumps.MenuItem("Показывать капсулу при записи", callback=self.toggle_hud)
        m.add(self.hud_on)
        m.add(rumps.MenuItem("Показать пример капсулы (2 с)", callback=lambda _: hud.preview()))
        self.hud_groups = {}
        groups = [("hud_size", "Размер", {k: v[0] for k, v in hud.SIZES.items()}),
                  ("hud_icon", "Иконка", hud.ICONS),
                  ("hud_color", "Цвет", {k: v[0] for k, v in hud.COLORS.items()}),
                  ("hud_bg", "Фон", hud.BACKGROUNDS),
                  ("hud_pos", "Положение", hud.POSITIONS)]
        for key, title, options in groups:
            sub = rumps.MenuItem(title)
            for k, label in options.items():
                it = rumps.MenuItem(label, callback=self.set_hud_opt)
                it._opt = (key, k)
                sub.add(it)
            self.hud_groups[key] = sub
            m.add(sub)
        m.add(None)
        self.sounds_item = rumps.MenuItem("Звуки включены", callback=self.toggle_sounds)
        m.add(self.sounds_item)
        names = hud.system_sounds()
        for ev, (title, _default) in hud.SOUND_EVENTS.items():
            sub = rumps.MenuItem(title)
            for name in ["none"] + names:
                it = rumps.MenuItem("Без звука" if name == "none" else name,
                                    callback=self.set_sound)
                it._opt = (f"sound_{ev}", name)
                sub.add(it)
            self.hud_groups[f"sound_{ev}"] = sub
            m.add(sub)
        self._sync_hud_menu()
        return m

    def set_sound(self, sender):
        key, name = sender._opt
        CONFIG[key] = name
        save_config()
        hud.configure(CONFIG)
        self._sync_hud_menu()
        hud.preview_sound(name)

    def _sync_hud_menu(self):
        self.hud_on.state = int(CONFIG["hud"])
        self.sounds_item.state = int(CONFIG["sounds"])
        for key, sub in self.hud_groups.items():
            for it in sub.values():
                it.state = int(it._opt[1] == CONFIG.get(key))

    def toggle_hud(self, sender):
        CONFIG["hud"] = not CONFIG["hud"]
        save_config()
        hud.configure(CONFIG)
        self._sync_hud_menu()

    def set_hud_opt(self, sender):
        key, val = sender._opt
        CONFIG[key] = val
        save_config()
        hud.configure(CONFIG)
        self._sync_hud_menu()
        hud.preview(1.5)

    def toggle_sounds(self, sender):
        CONFIG["sounds"] = not CONFIG["sounds"]
        save_config()
        hud.configure(CONFIG)
        self._sync_hud_menu()
        if CONFIG["sounds"]:
            hud.play("done")

    def restart_now(self, _):
        threading.Thread(target=restart_app, daemon=True).start()

    def open_perm_wizard(self, _):
        # диалоги ждут пользователя минутами — главный поток занимать нельзя
        threading.Thread(target=permissions_wizard, daemon=True).start()

    # --- окно состояния -------------------------------------------------------
    def open_status(self, _, activate=True):
        statuspanel.show(self.status_snapshot, activate=activate, actions={
            "restart": restart_app,
            "copy_version": lambda: self.copy_version(None),
            "vp_enroll": lambda: self.enroll(None),
            "vp_check": lambda: self.vp_check_run(None),
            "log": lambda: self.open_log(None),
            "reopen": lambda: threading.Thread(target=reopen_stream, daemon=True).start(),
            "reveal": reveal_binary,
            "cache": lambda: subprocess.run(["open", os.path.dirname(_repo_dir("x/y"))]),
            "perm:Микрофон": lambda: request_permission("Микрофон"),
            "perm:Мониторинг ввода": lambda: request_permission("Мониторинг ввода"),
            "perm:Универсальный доступ": lambda: request_permission("Универсальный доступ"),
            "dl:asr": lambda: self._download_role("asr"),
            "dl:llm": lambda: self._download_role("llm"),
            "dir:asr": lambda: subprocess.run(["open", _repo_dir(CONFIG["asr_model"])]),
            "dir:llm": lambda: subprocess.run(["open", _repo_dir(CONFIG["llm_model"])]),
            "hud_test": lambda: hud.preview(2.5),
            "wizard": lambda: self.open_perm_wizard(None),
        })

    def _download_role(self, role):
        repo = CONFIG[ROLE_CFG[role]]
        it = rumps.MenuItem(repo)
        it._repo = repo
        self.download_model(it)

    def refresh_status(self, _):
        # окно закрыто — снимок не собираем (в нём вызовы TCC/PortAudio),
        # но сами разрешения перечитываем раз в 10 с, чтобы ⚠️ снималось
        if statuspanel.is_visible():
            statuspanel.refresh()
        elif time.time() - STATE["perms_ts"] > 10:
            STATE["perms_ts"] = time.time()
            try:
                perm_status()
            except Exception:
                pass
        perms = STATE["perms"]
        need_restart = perms_need_restart()
        issues = any(v != "ok" for v in perms.values())
        self.status_item.title = ("⚠️ Состояние и разрешения…" if issues
                                  else "Состояние и разрешения…")
        if need_restart:
            title = "⚠️ Разрешения выданы — перезапустить службу"
            cb = self.restart_now
        elif issues:
            title = "Настроить разрешения…"
            cb = self.open_perm_wizard
        else:
            title = "Разрешения: все выданы"
            cb = self.open_perm_wizard
        if self.perm_item.title != title:
            self.perm_item.title = title
        if getattr(self, "_perm_cb", None) is not cb:  # не перерегистрируем каждую секунду
            self._perm_cb = cb
            self.perm_item.set_callback(cb)

    def status_snapshot(self):
        perms = perm_status()
        STATE["perms_ts"] = time.time()
        # --- служба ---
        need_restart = perms_need_restart()
        if STATE["error"]:
            svc = f"❌ {STATE['error']}"
        elif need_restart:
            svc = ("⚠️ Разрешения выданы, но не применены: " + ", ".join(need_restart)
                   + " — нажми «Перезапустить»")
        elif STATE["loading"]:
            svc = ("⏳ Модели загружаются в память (после старта ~20–30 с, при первом "
                   "запуске — скачиваются)")
        elif recording:
            svc = "🔴 Идёт запись"
        elif stream_alive():
            svc = "✅ Готов: зажми правый Option и говори"
        elif "нет" in STATE["mic"]:
            svc = "⚠️ Нет ни одного микрофона — подключи AirPods или USB-микрофон"
        else:
            svc = "⚠️ Аудиопоток не отдаёт данные — переоткрываю…"
        up = int(time.time() - STATE["started"])
        upt = f"{up // 3600} ч {up % 3600 // 60} мин" if up >= 3600 else f"{up // 60} мин {up % 60} с"
        plist = os.path.expanduser("~/Library/LaunchAgents/com.kkd.dictate.plist")
        how = "демон launchd (com.kkd.dictate, автозапуск при входе)" if os.path.exists(plist) \
            else "вручную (без демона — после выхода не поднимется сам)"
        stale = " · ⬆️ на диске новее — перезапусти" if code_updated_on_disk() else ""
        service = [
            ("Состояние", svc, "Перезапустить", "restart"),
            ("Версия", f"{self._version}{stale}", "Скопировать", "copy_version"),
            ("Процесс", f"PID {os.getpid()} · работает {upt} · {how}", "Лог…", "log"),
        ]
        # --- разрешения ---
        icons = {"ok": "✅ выдано",
                 "restart": "☑️ галка включена, но применится после перезапуска службы",
                 "ask": "❔ ещё не спрашивали — нажми «Запросить»",
                 "denied": "❌ нет — включи галку в Настройках для процесса ниже"}
        prows = []
        missing = [n for n in PERM_ORDER if perms[n] != "ok"]
        if missing:
            prows.append(("Мастер настройки",
                          "Проведу по недостающим разрешениям по одному, с "
                          "объяснением и ожиданием — вместо трёх запросов разом",
                          "Настроить по очереди", "wizard"))
        for name in PERM_ORDER:
            st = perms[name]
            btn = {"ok": None, "restart": "Перезапустить", "ask": "Запросить",
                   "denied": "Открыть Настройки"}[st]
            act = "restart" if st == "restart" else f"perm:{name}"
            prows.append((name, f"{icons[st]} · нужно, чтобы {PERM_WHY[name]}", btn, act))
        binary = os.path.realpath(sys.executable)
        prows.append(("Кому выдавать", f"{binary}\nВ списке Настроек это файл python3.12. "
                      "«Мониторинг ввода» и «Универсальный доступ» macOS отдаёт только "
                      "новому процессу — после включения галки нужен перезапуск.",
                      "Показать в Finder", "reveal"))
        # --- микрофон ---
        devs = input_devices()
        mic = [
            ("Пишем с", STATE["mic"], "Переоткрыть поток", "reopen"),
            ("Входы в системе", ", ".join(devs) if devs else "нет ни одного входного устройства",
             None, None),
        ]
        # --- модели ---
        mrows = []
        for role, (title, options) in ROLES.items():
            repo = CONFIG[ROLE_CFG[role]]
            full = next((f for r, f, _ in options if r == repo), 0)
            label = next((l for r, _, l in options if r == repo), repo)
            st = _repo_status(repo, full)
            if st["state"] == "done":
                txt, btn, act = f"● скачана · {_fmt_mb(st['mb'])} · {label}", "Папка", f"dir:{role}"
            elif st["state"] == "loading":
                txt, btn, act = (f"⏳ скачивается: {_fmt_mb(st['mb'])} из ~{_fmt_mb(full)} · {label}",
                                 None, None)
            elif st["state"] == "partial":
                txt, btn, act = (f"⚠️ скачана частично ({_fmt_mb(st['mb'])} из ~{_fmt_mb(full)}) · "
                                 f"{label} — закачка обрывалась", "Докачать", f"dl:{role}")
            else:
                txt, btn, act = (f"○ не скачана (~{_fmt_mb(full)}) · {label} — скачается при "
                                 f"первом запуске или по кнопке", "Скачать", f"dl:{role}")
            mrows.append((title, f"{repo.split('/')[-1]}\n{txt}", btn, act))
        ec = _repo_status("speechbrain/spkrec-ecapa-voxceleb", 90)
        ec_txt = ("● " + _fmt_mb(ec["mb"]) if ec["state"] == "done"
                  else "⏳ скачивается" if ec["state"] == "loading" else "○ не скачан (~90 МБ)")
        vad_txt = "● загружен" if not STATE["loading"] else "⏳"
        mrows.append(("Служебные", f"Отпечаток голоса ECAPA: {ec_txt} · Silero VAD: {vad_txt}",
                      "Открыть кэш", "cache"))
        # --- хоткей ---
        last = STATE["last_hotkey"]
        if last:
            ago = int(time.time() - last)
            hk = f"✅ обнаружен {ago} с назад" if ago < 3600 else "✅ обнаружен (давно)"
        elif perms["Мониторинг ввода"] != "ok":
            hk = "⏸ ждёт разрешения «Мониторинг ввода»"
        else:
            hk = "нажми правый Option — здесь появится ✅"
        snd = "выключены" if not CONFIG["sounds"] else hud.sound_route_info()
        hot = [("Правый Option", hk, "Проверить индикатор", "hud_test"),
               ("Звуки", snd, None, None)]
        # --- мой голос ---
        vp = load_voiceprint()
        if not CONFIG["only_my_voice"]:
            vtxt = ("фильтр выключен — вставляю любую речь"
                    + ("" if vp is None else f" (отпечаток есть: {voiceprint_summary()})"))
        elif vp is None:
            vtxt = "⚠️ фильтр включён, но отпечатка нет — запиши, иначе он ничего не делает"
        else:
            vtxt = f"✅ пропускаю только мой голос · {voiceprint_summary()}"
        vrows = [("Состояние", vtxt, "Записать заново", "vp_enroll")]
        if vp is not None:
            last = STATE.get("vp_last")
            vrows.append(("Последнее сравнение",
                          f"сходство {last:.2f} при пороге {CONFIG['vp_threshold']}"
                          if last is not None else "ещё не сравнивал — продиктуй что-нибудь",
                          "Проверить голос", "vp_check"))
        return [("Служба", service), ("Разрешения", prows), ("Микрофон", mic),
                ("Мой голос", vrows), ("Модели", mrows), ("Хоткей и индикатор", hot)]

    def refresh_models(self, _):
        try:
            snapshot = []  # (роль, заголовок, [(репо, подпись, статус, активна)])
            for role, (title, options) in ROLES.items():
                active = CONFIG[ROLE_CFG[role]]
                rows = [(repo, label, _repo_status(repo, full), repo == active)
                        for repo, full, label in options]
                snapshot.append((role, title, rows))
            aux = [("ECAPA-voxceleb — отпечаток голоса",
                    _repo_status("speechbrain/spkrec-ecapa-voxceleb", 90))]
            try:  # Silero VAD едет внутри pip-пакета, отдельно не скачивается
                import silero_vad
                d = os.path.dirname(silero_vad.__file__)
                aux.append(("Silero VAD — детектор речи (в пакете)",
                            {"path": d, "state": "done", "mb": _dir_size_mb(d),
                             "full": None}))
            except ImportError:
                pass
        except Exception:
            return
        sig = repr(snapshot) + repr(aux)
        if sig == getattr(self, "_models_sig", ""):
            return  # ничего не изменилось — не перестраиваем открытое меню
        self._models_sig = sig
        all_rows = [r for _, _, rows in snapshot for r in rows]
        self.models_menu.title = (
            "Модели: скачиваются…" if any(st["state"] == "loading"
                                          for *_, st, _ in all_rows) else
            "Модели" if all(st["state"] == "done"
                            for *_, st, act in all_rows if act) else
            "Модели: активная не скачана")
        if self.models_menu._menu is not None:  # NSMenu появляется после первого add
            self.models_menu.clear()
        for role, title, rows in snapshot:
            role_item = rumps.MenuItem(
                f"{title}: {CONFIG[ROLE_CFG[role]].split('/')[-1]}")
            for repo, label, st, is_active in rows:
                row = rumps.MenuItem(_model_row(label, st, is_active))
                if is_active:
                    row.add(rumps.MenuItem("Активная модель"))
                elif st["state"] == "done":
                    act = rumps.MenuItem("Сделать активной (перезапуск)",
                                         callback=self.activate_model)
                    act._cfg_key, act._repo = ROLE_CFG[role], repo
                    row.add(act)
                    rm = rumps.MenuItem("Удалить с диска", callback=self.delete_model)
                    rm._repo, rm._path = repo, st["path"]
                    row.add(rm)
                elif st["state"] in ("none", "partial"):
                    dl = rumps.MenuItem(
                        ("Докачать" if st["state"] == "partial" else "Скачать")
                        + f" (~{_fmt_mb(st['full'])})", callback=self.download_model)
                    dl._repo = repo
                    row.add(dl)
                else:
                    row.add(rumps.MenuItem("Скачивается…"))
                if os.path.isdir(st["path"]):
                    op = rumps.MenuItem("Открыть папку", callback=self.open_model_dir)
                    op._model_path = st["path"]
                    row.add(op)
                role_item.add(row)
            self.models_menu.add(role_item)
        self.models_menu.add(None)
        for label, st in aux:
            item = rumps.MenuItem(f"✓ {label} · {_fmt_mb(st['mb'])}",
                                  callback=self.open_model_dir)
            item._model_path = st["path"]
            self.models_menu.add(item)
        from huggingface_hub.constants import HF_HUB_CACHE
        cache_item = rumps.MenuItem(f"Кэш: {HF_HUB_CACHE.replace(os.path.expanduser('~'), '~')}",
                                    callback=self.open_model_dir)
        cache_item._model_path = HF_HUB_CACHE
        self.models_menu.add(cache_item)

    def activate_model(self, sender):
        CONFIG[sender._cfg_key] = sender._repo
        save_config()
        print(f"Активная модель теперь {sender._repo} — перезапускаюсь...", flush=True)
        restart_app()  # умеет и launchd, и запуск вручную (execv)

    def download_model(self, sender):
        repo = sender._repo
        if not hasattr(self, "_downloading"):
            self._downloading = set()
        if repo in self._downloading:
            return
        self._downloading.add(repo)

        def dl():
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo)
                print(f"Модель {repo} скачана.", flush=True)
            except Exception as e:
                print(f"  модель {repo} не скачалась: {e}", flush=True)
            finally:
                self._downloading.discard(repo)
                _repo_cache.pop(repo, None)
                self._models_sig = ""
        threading.Thread(target=dl, daemon=True).start()
        self._models_sig = ""  # прогресс появится при следующем обновлении

    def delete_model(self, sender):
        if sender._repo in (CONFIG["asr_model"], CONFIG["llm_model"]):
            rumps.alert("Модели", "Эта модель сейчас активна — сначала выбери другую.")
            return
        if getattr(self, "_downloading", set()) & {sender._repo}:
            rumps.alert("Модели", "Эта модель сейчас скачивается — дождись конца.")
            return
        import shutil
        repo, path = sender._repo, sender._path

        def rm():  # rmtree на 17 ГБ в главном потоке морозит меню на секунды
            shutil.rmtree(path, ignore_errors=True)
            print(f"Модель {repo} удалена с диска.", flush=True)
            _repo_cache.pop(repo, None)
            self._models_sig = ""
        threading.Thread(target=rm, daemon=True).start()

    def open_model_dir(self, sender):
        path = sender._model_path
        if not os.path.isdir(path):
            path = os.path.dirname(path)
        subprocess.run(["open", path])

    def refresh_title(self, _):
        # ❌ — модели не загрузились; ⚠️ — поток мёртв/переоткрывается
        title = (STATE.get("vp_countdown")  # запись отпечатка: обратный отсчёт в баре
                 or ("❌" if STATE["error"] else "⏳" if STATE["loading"]
                     else "🟠" if recording else "🎙️" if stream_alive() else "⚠️"))
        if title != self.title:
            self.title = title
        mic = f"Микрофон: {STATE['mic']}"
        if mic != self.mic_item.title:
            self.mic_item.title = mic
        app = frontmost_app() or STATE["app"]
        STATE["app"] = app
        cur = CONFIG["profiles"].get(app)
        prof = f"Профиль «{app}»: " + (STYLES[cur] if cur else "по умолчанию")
        if prof == getattr(self, "_prof_sig", None):
            return  # приложение и стиль не менялись — галки трогать незачем
        self._prof_sig = prof
        self.profile.title = prof
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
        if load_voiceprint() is None and not CONFIG["only_my_voice"]:
            rumps.alert("Только мой голос",
                        "Сначала запиши отпечаток: «Мой голос → Записать отпечаток…».")
            return
        CONFIG["only_my_voice"] = not CONFIG["only_my_voice"]
        sender.state = int(CONFIG["only_my_voice"])
        save_config()
        self.refresh_voice_menu()

    def set_vp_strictness(self, sender):
        CONFIG["vp_threshold"] = sender._thr
        save_config()
        self.refresh_voice_menu()

    def refresh_voice_menu(self, _=None):
        """Подписи пункта «Мой голос»: состояние видно, не открывая панель."""
        self.vp_status.title = f"Отпечаток: {voiceprint_summary()}"
        have = load_voiceprint() is not None
        self.vp_item.state = int(CONFIG["only_my_voice"] and have)
        self.vp_check.set_callback(self.vp_check_run if have else None)
        for it in self.vp_strict.values():
            it.state = int(abs(CONFIG["vp_threshold"] - it._thr) < 0.005)

    def _capture(self, kind, seconds, title, body):
        """Записать кусок с ОСНОВНОГО потока (тот же микрофон, что у диктовки),
        показывая капсулу с уровнем, и отдать в ML-поток."""
        if recording or enroll_buf["on"]:
            rumps.alert(title, "Идёт другая запись — дождись конца и повтори.")
            return
        if STATE["loading"]:
            rumps.alert(title, "Модели ещё грузятся — попробуй через несколько секунд.")
            return
        rumps.alert(title, body)  # ждём ОК: отсчёт начинается, когда человек готов

        def run():
            token = ("vp", time.time())
            try:
                ensure_stream()
                with lock:
                    enroll_buf["chunks"].clear()
                    enroll_buf["on"] = True
                hud.play("start")
                hud.show("rec", token)  # видно, что идёт запись, и как громко
                deadline = time.time() + seconds
                while time.time() < deadline:
                    left = deadline - time.time()
                    STATE["vp_countdown"] = f"🔴 {left:.0f}с"
                    time.sleep(0.1)
                with lock:
                    enroll_buf["on"] = False
                    a = (np.concatenate(enroll_buf["chunks"]).flatten().astype(np.float32)
                         if enroll_buf["chunks"] else np.zeros(0, dtype=np.float32))
                    enroll_buf["chunks"].clear()
                STATE.pop("vp_countdown", None)
                hud.show("busy", token)
                if len(a) < seconds * SAMPLE_RATE * 0.5:
                    raise ValueError("микрофон не отдал звук — проверь вход и повтори")
                jobs.put((kind, a))
            except Exception as e:
                print(f"  ✗ запись голоса не вышла: {e}", flush=True)
                notify_ui(title, f"Не получилось: {e}")
            finally:
                with lock:
                    enroll_buf["on"] = False
                STATE.pop("vp_countdown", None)
                hud.hide(token)
        threading.Thread(target=run, daemon=True).start()

    def enroll(self, _):
        self._capture(
            "enroll", VP_RECORD_SEC, "Запись отпечатка голоса",
            f"После «ОК» говори {VP_RECORD_SEC} секунд обычным голосом, без "
            "длинных пауз — читай любой текст вслух.\n\n"
            "Пока идёт запись, у курсора видна капсула с уровнем звука, а в "
            "меню-баре — обратный отсчёт. По итогу покажу, что получилось.\n\n"
            f"Микрофон сейчас: {STATE['mic']}. Отпечаток привязан к микрофону — "
            "для другого (например, AirPods) запиши заново на нём.")

    def vp_check_run(self, _):
        self._capture(
            "vpcheck", VP_CHECK_SEC, "Проверка голоса",
            f"После «ОК» скажи фразу — {VP_CHECK_SEC} секунды.\n\n"
            "Покажу, узнаю ли я тебя и с каким запасом до порога.")

    def refresh_recent(self, _):
        try:
            db = sqlite3.connect(os.path.join(BASE, "history.sqlite3"))
            rows = db.execute("SELECT id, text FROM transcriptions "
                              "ORDER BY id DESC LIMIT 5").fetchall()
            db.close()
        except Exception:
            return
        if rows == getattr(self, "_recent_sig", None):
            return  # ничего не добавилось — не перестраиваем открытое меню
        self._recent_sig = rows
        self.recent.clear()
        if not rows:
            self.recent.add(rumps.MenuItem("пусто"))
            return
        for i, (_id, text) in enumerate(rows, 1):
            body = text if len(text) <= 60 else text[:57] + "…"
            # нумеруем: rumps держит пункты в словаре по заголовку, и две
            # одинаковые фразы схлопывались в одну строку
            item = rumps.MenuItem(f"{i}. {body}", callback=self.copy_item)
            item._full_text = text
            self.recent.add(item)

    def copy_item(self, sender):
        subprocess.run(["pbcopy"], input=sender._full_text.encode())

    def copy_version(self, _):
        """Клик по версии — в буфер полная справка для отчёта об ошибке."""
        info = "\n".join([
            f"Dictate {app_version()}",
            f"macOS {platform.mac_ver()[0]} · {platform.machine()} · "
            f"Python {platform.python_version()}",
            f"ASR {CONFIG['asr_model']} · LLM {CONFIG['llm_model']}",
            f"Микрофон: {STATE['mic']}",
        ])
        subprocess.run(["pbcopy"], input=info.encode())
        self.about_item.title = "Версия скопирована ✓"
        rumps.Timer(self._restore_about, 2.0).start()

    def _restore_about(self, timer):
        timer.stop()
        self.about_item.title = f"Версия {app_version()}"

    def toggle_enhance(self, sender):
        STATE["enhance"] = not STATE["enhance"]
        sender.state = int(STATE["enhance"])
        CONFIG["enhance"] = STATE["enhance"]  # иначе сбрасывается при рестарте
        save_config()

    def open_terms(self, _):
        subprocess.run(["open", "-t", os.path.join(BASE, "terms.txt")])

    def _dashboard(self, mode):
        # сборка HTML — быстрая, делаем прямо в главном потоке (мы в колбэке меню),
        # окно WKWebView тоже обязано создаваться на главном потоке
        try:
            import importlib, dashboard
            importlib.reload(dashboard)
            # своё имя файла на вкладку: иначе второе окно перетирает первое
            out = dashboard.OUT.replace(".html", f"-{mode}.html")
            with open(out, "w") as f:
                f.write(dashboard.build(mode))
            title = "Статистика — dictate" if mode == "stats" else "Поиск истории — dictate"
            webwindow.show(title, out)
        except Exception as e:
            print(f"  окно не открылось: {e}", flush=True)

    def open_stats(self, _):
        self._dashboard("stats")

    def open_search(self, _):
        self._dashboard("search")

    def suggest(self, _):
        jobs.put(("autodict", None))
        rumps.alert("Автословарь", "Пересборка запущена в фоне — результат "
                    "появится в логе строкой «Автословарь обновлён: …». "
                    "Он также пересобирается сам при каждом старте.")

    def open_log(self, _):
        subprocess.run(["open", "-t", os.path.join(BASE, "dictate.log")])


PRIVACY_PANES = {  # разрешение -> раздел Настроек
    "Микрофон": "Privacy_Microphone",
    "Мониторинг ввода": "Privacy_ListenEvent",
    "Универсальный доступ": "Privacy_Accessibility",
}
PERM_WHY = {
    "Микрофон": "записывать голос во время диктовки",
    "Мониторинг ввода": "видеть глобальный хоткей (правый Option)",
    "Универсальный доступ": "вставлять текст (⌘V) и находить курсор для индикатора",
}


PERM_ORDER = ["Микрофон", "Мониторинг ввода", "Универсальный доступ"]
PERM_HINT = {  # что увидит пользователь в Настройках
    "Микрофон": "Конфиденциальность и безопасность → Микрофон",
    "Мониторинг ввода": "Конфиденциальность и безопасность → Мониторинг ввода",
    "Универсальный доступ": "Конфиденциальность и безопасность → Универсальный доступ",
}
# «Мониторинг ввода» и «Универсальный доступ» кешируются в процессе на всю его
# жизнь: после включения галки текущий процесс продолжает считать, что права
# нет. Свежее состояние узнаём у дочернего процесса того же бинарника (TCC
# смотрит на путь, поэтому ответ — про нас), и предлагаем перезапуск.
_EXTERNAL_PROBE = (
    'import ctypes,json;'
    'io=ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit");'
    'ax=ctypes.CDLL("/System/Library/Frameworks/ApplicationServices.framework'
    '/ApplicationServices");'
    'ax.AXIsProcessTrusted.restype=ctypes.c_bool;'
    'print(json.dumps({"hid":io.IOHIDCheckAccess(1),"ax":bool(ax.AXIsProcessTrusted())}))'
)
_external = {"ts": 0.0, "data": {}}


def _iokit():
    import ctypes
    return ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")


def external_perms(max_age=3.0) -> dict:
    """{'hid': код, 'ax': bool} по мнению СВЕЖЕГО процесса (или {})."""
    if time.time() - _external["ts"] < max_age:
        return _external["data"]
    data = {}
    try:
        out = subprocess.run([sys.executable, "-c", _EXTERNAL_PROBE],
                             capture_output=True, text=True, timeout=10)
        data = json.loads(out.stdout.strip() or "{}")
    except Exception:
        data = {}
    _external.update(ts=time.time(), data=data)
    return data


def perm_status(check_external=True) -> dict:
    """{имя: 'ok' | 'restart' | 'ask' | 'denied'}.

    restart — галка уже включена, но применится только к новому процессу."""
    from ApplicationServices import AXIsProcessTrusted
    from AVFoundation import AVCaptureDevice
    mic = AVCaptureDevice.authorizationStatusForMediaType_("soun")  # 0 не опр., 3 ок
    hid = _iokit().IOHIDCheckAccess(1)  # 0 ок, 1 отказ, 2 не определено
    st = {
        "Микрофон": "ok" if mic == 3 else "ask" if mic == 0 else "denied",
        "Мониторинг ввода": "ok" if hid == 0 else "ask" if hid == 2 else "denied",
        "Универсальный доступ": "ok" if AXIsProcessTrusted() else "denied",
    }
    if check_external and (st["Мониторинг ввода"] != "ok"
                           or st["Универсальный доступ"] != "ok"):
        ext = external_perms()
        if ext.get("hid") == 0 and st["Мониторинг ввода"] != "ok":
            st["Мониторинг ввода"] = "restart"
        if ext.get("ax") and st["Универсальный доступ"] != "ok":
            st["Универсальный доступ"] = "restart"
    STATE["perms"] = st
    STATE["perms_ts"] = time.time()
    return st


def perms_need_restart() -> list:
    return [n for n, v in STATE["perms"].items() if v == "restart"]


def open_privacy_pane(name: str):
    subprocess.run(["open", "x-apple.systempreferences:com.apple.preference."
                    f"security?{PRIVACY_PANES[name]}"])


def reveal_binary():
    subprocess.run(["open", "-R", os.path.realpath(sys.executable)])


def request_permission(name: str, quiet=False):
    """Запросить одно разрешение: системный диалог, если он ещё возможен,
    иначе — открыть панель Настроек (и показать бинарник в Finder)."""
    from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
    from AVFoundation import AVCaptureDevice
    st = perm_status().get(name)
    if st in ("ok", "restart"):
        return
    if name == "Микрофон":
        if st == "ask":
            AVCaptureDevice.requestAccessForMediaType_completionHandler_("soun", lambda ok: None)
            return
    elif name == "Мониторинг ввода":
        if st == "ask":
            _iokit().IOHIDRequestAccess(1)
            return
    elif name == "Универсальный доступ":
        # диалог со ссылкой в Настройки система показывает только в первый раз
        if AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True}):
            return
    # диалога уже не будет — ведём пользователя в Настройки
    if not quiet:
        reveal_binary()
    open_privacy_pane(name)


def _dialog(text: str, buttons, default=None, timeout=300):
    """Модальный диалог через osascript; возвращает нажатую кнопку или None."""
    btns = ", ".join('"%s"' % b.replace('"', "'") for b in buttons)
    script = (f'display dialog "{text}" with title "Dictate — разрешения" '
              f'buttons {{{btns}}} default button "{default or buttons[-1]}" '
              f'giving up after {timeout}')
    try:
        out = subprocess.run(["osascript", "-e", script], capture_output=True,
                             text=True, timeout=timeout + 15).stdout
    except Exception:
        return None
    for b in buttons:
        if f"button returned:{b}" in out:
            return b
    return None


def _wait_granted(name: str, seconds=120) -> bool:
    """Ждём, пока разрешение появится (в процессе или у свежего процесса)."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        if perm_status()[name] in ("ok", "restart"):
            return True
        time.sleep(1.5)
    return False


def permissions_wizard():
    """Проводник по разрешениям: по одному, с объяснением и ожиданием.

    Раньше при каждом старте вылетали все три запроса разом плюс Finder и
    Настройки — теперь это осознанное действие пользователя из меню."""
    if STATE.get("perm_wizard"):
        return
    STATE["perm_wizard"] = True
    try:
        todo = [n for n in PERM_ORDER if perm_status()[n] not in ("ok", "restart")]
        if not todo:
            need = perms_need_restart()
            if need:
                if _dialog("Разрешения выданы, но применятся только после "
                           "перезапуска службы.\n\nПерезапустить сейчас? "
                           "Займёт около 20 секунд.",
                           ["Позже", "Перезапустить"], "Перезапустить", 120) == "Перезапустить":
                    restart_app()
            else:
                _dialog("Все разрешения уже выданы — диктовка готова к работе.",
                        ["ОК"], "ОК", 60)
            return
        binary = os.path.realpath(sys.executable)
        start = _dialog(
            f"Нужно выдать {len(todo)} разрешени(я): {', '.join(todo)}.\n\n"
            "Пойдём по одному: для каждого покажу системный запрос или открою "
            "нужный раздел Настроек. В списке Настроек включи галку для файла "
            f"python3.12 (это и есть служба диктовки).",
            ["Отмена", "Начать"], "Начать", 300)
        if start != "Начать":
            return
        for i, name in enumerate(todo, 1):
            st = perm_status()[name]
            if st in ("ok", "restart"):
                continue
            head = f"Шаг {i} из {len(todo)}: {name}"
            if st == "ask":
                body = (f"{head}\n\nНужно, чтобы {PERM_WHY[name]}.\n\n"
                        "Сейчас появится системный запрос — нажми «Разрешить».")
                if _dialog(body, ["Пропустить", "Показать запрос"],
                           "Показать запрос") != "Показать запрос":
                    continue
                request_permission(name, quiet=True)
            else:
                body = (f"{head}\n\nНужно, чтобы {PERM_WHY[name]}.\n\n"
                        f"Система больше не покажет запрос сама — открою "
                        f"{PERM_HINT[name]} и покажу файл в Finder. Перетащи его "
                        f"в список (или включи галку, если он уже там), потом "
                        f"вернись сюда.\n\n{binary}")
                if _dialog(body, ["Пропустить", "Открыть Настройки"],
                           "Открыть Настройки") != "Открыть Настройки":
                    continue
                reveal_binary()
                open_privacy_pane(name)
            if _wait_granted(name):
                print(f"  ✓ разрешение «{name}» выдано", flush=True)
            else:
                print(f"  … разрешение «{name}» так и не выдано", flush=True)
        left = [n for n in PERM_ORDER if perm_status()[n] not in ("ok", "restart")]
        need = perms_need_restart()
        if need:
            msg = ("Готово: " + ", ".join(n for n in PERM_ORDER if n not in left)
                   + ".\n\nЧтобы «" + "», «".join(need) + "» заработало, службу "
                   "нужно перезапустить (macOS отдаёт эти разрешения только "
                   "новому процессу). Перезапустить сейчас?")
            if _dialog(msg, ["Позже", "Перезапустить"], "Перезапустить", 120) == "Перезапустить":
                restart_app()
        elif left:
            _dialog("Осталось выдать: " + ", ".join(left)
                    + ".\n\nМожно вернуться к этому в меню → «Настроить "
                      "разрешения…».", ["ОК"], "ОК", 60)
        else:
            _dialog("Все разрешения выданы — диктовка готова к работе.",
                    ["ОК"], "ОК", 60)
    except Exception as e:
        print(f"  проводник разрешений: {e}", flush=True)
    finally:
        STATE["perm_wizard"] = False


def check_permissions_at_start():
    """При старте только смотрим и пишем в лог: никаких запросов пачкой.

    Системные запросы всё равно появятся сами в момент использования (открытие
    микрофона, слушатель клавиш), а недостающее пользователь выдаёт из меню
    → «Настроить разрешения…» по очереди."""
    st = perm_status()
    missing = [n for n, v in st.items() if v != "ok"]
    if not missing:
        print("Разрешения: все выданы.", flush=True)
        return []
    need = [n for n in missing if st[n] == "restart"]
    if need:
        print(f"⚠ Разрешения выданы, но применятся после перезапуска: "
              f"{', '.join(need)}. Меню → «Перезапустить службу».", flush=True)
    rest = [n for n in missing if st[n] != "restart"]
    if rest:
        print(f"⚠ Нет разрешений: {', '.join(rest)}. Меню-бар → «Настроить "
              f"разрешения…» проведёт по одному. Процесс, которому их выдавать: "
              f"{os.path.realpath(sys.executable)}", flush=True)
    return missing


def input_devices() -> list:
    """Имена входных устройств для UI.

    Опрос PortAudio параллельно с sd._terminate() в другом потоке — это не
    исключение, а падение процесса, поэтому только под свободным замком;
    иначе отдаём последний известный список."""
    if reopen_lock.acquire(blocking=False):
        try:
            names = [d["name"] for d in sd.query_devices()
                     if d["max_input_channels"] > 0]
            _devices_cache["names"], _devices_cache["ts"] = names, time.time()
        except Exception:
            pass
        finally:
            reopen_lock.release()
    return _devices_cache["names"]


def restart_app():
    """Перезапуск процесса: через launchd, если стоим демоном, иначе exec."""
    label = "com.kkd.dictate"
    plist = os.path.expanduser(f"~/Library/LaunchAgents/{label}.plist")
    print("Перезапуск по запросу из окна состояния…", flush=True)
    if os.path.exists(plist):
        r = subprocess.run(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{label}"])
        if r.returncode == 0:
            return  # нас уже убивают
    os.execv(sys.executable, [sys.executable] + sys.argv)


def _git(*args) -> str:
    try:
        r = subprocess.run(["git", "-C", BASE, *args],
                           capture_output=True, text=True, timeout=3)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def app_version() -> str:
    """Версия работающего кода: «0.4.0+3 · a1b2c3d · 17.08.2026 · правки».

    Считается ОДИН раз при старте и запоминается: после git pull на диске уже
    новый код, а в памяти крутится старый — показывать надо тот, что работает.
    Слагаемые: номер из VERSION (при расхождении с ближайшим тегом верим тегу),
    «+N» — коммитов после тега, хеш и дата коммита, «правки» — есть
    незакоммиченные изменения.
    """
    if "version" in STATE:
        return STATE["version"]
    ver, extra = VERSION, []
    desc = _git("describe", "--tags", "--long", "--dirty", "--match", "v[0-9]*")
    if desc:  # v0.4.0-3-ga1b2c3d[-dirty]
        parts = desc.split("-")
        tag, ahead = parts[0].lstrip("v"), parts[1]
        ver = tag  # тег на коммите — источник правды; VERSION нужен без git
        if ahead != "0":
            ver += f"+{ahead}"
    sha = _git("rev-parse", "--short", "HEAD")
    if sha:
        extra.append(sha)
    date = _git("log", "-1", "--format=%cd", "--date=format:%d.%m.%Y")
    if date:
        extra.append(date)
    if desc.endswith("-dirty") or (not desc and _git("status", "--porcelain")):
        extra.append("правки")
    STATE["version"] = " · ".join([ver] + extra)
    STATE["head"] = sha
    return STATE["version"]


def code_updated_on_disk() -> bool:
    """После git pull код на диске новее работающего — повод перезапустить службу."""
    head = STATE.get("head")
    return bool(head) and _git("rev-parse", "--short", "HEAD") not in ("", head)


def _choose_model_dialog(role: str) -> str | None:
    """Нативный диалог со списком моделей роли; вернёт выбранный репозиторий.

    None — пользователь закрыл диалог (останемся на дефолте)."""
    title, options = ROLES[role]
    active = CONFIG[ROLE_CFG[role]]
    items, default = [], None
    for repo, full, label in options:
        st = _repo_status(repo, full)
        suffix = "скачана" if st["state"] == "done" else f"скачается ~{_fmt_mb(full)}"
        line = f"{label} · {suffix}"
        items.append(line)
        if repo == active:
            default = line
    lst = ", ".join('"%s"' % i.replace('"', "'") for i in items)
    script = (f'choose from list {{{lst}}} with title "Dictate" with prompt '
              f'"Первый запуск: модель для роли «{title}» ещё не скачана.\n'
              f'Какую использовать? (Отмена — предложенная по умолчанию)" '
              f'default items {{"{default}"}}')
    try:
        res = subprocess.run(["osascript", "-e", script], capture_output=True,
                             text=True, timeout=600).stdout.strip()
    except Exception:
        return None
    if res in items:
        return options[items.index(res)][0]
    return None  # false = отмена, пусто = таймаут


def ask_first_download():
    """Первая установка: активные модели не качаем молча — спрашиваем, какие брать."""
    global ASR_MODEL, LLM_MODEL
    changed = False
    for role in ROLES:
        cfg_key = ROLE_CFG[role]
        repo = CONFIG[cfg_key]
        full = next((f for r, f, _ in ROLES[role][1] if r == repo), 0)
        if _repo_status(repo, full)["state"] not in ("none", "partial"):
            continue  # уже на диске (или качается) — вопросов нет
        chosen = _choose_model_dialog(role)
        if chosen and chosen != repo:
            CONFIG[cfg_key] = chosen
            changed = True
    if changed:
        save_config()
        ASR_MODEL, LLM_MODEL = CONFIG["asr_model"], CONFIG["llm_model"]


def _open_stream_quiet():
    """Стартовое открытие потока: отсутствие микрофона — не повод для трейсбека."""
    try:
        open_stream()
    except NoMicrophone:
        STATE["mic"] = "нет — подключи микрофон"
        print("  микрофона нет (Mac Studio без встроенного, AirPods в кейсе?) — "
              "жду появления...", flush=True)
    except Exception as e:
        print(f"  микрофон не открылся: {e} — вочдог попробует ещё раз", flush=True)


def main():
    print(f"Dictate {app_version()}", flush=True)  # первой строкой лога: что именно запустилось
    load_config()
    missing = check_permissions_at_start()
    ask_first_download()
    # окно состояния при старте: нет разрешений или активные модели ещё не на диске
    need_dl = any(_repo_status(CONFIG[ROLE_CFG[r]], 0)["state"] != "done" for r in ROLES)
    for role, cfg_key in ROLE_CFG.items():  # автоподхват докачки после обрыва
        st = _repo_status(CONFIG[cfg_key], 0)["state"]
        if st == "partial":
            print(f"  модель {CONFIG[cfg_key]} скачана не полностью — докачаю при "
                  f"загрузке (или кнопкой «Докачать» в окне состояния)", flush=True)
    STATE["show_status_on_start"] = bool(missing) or need_dl
    print(f"Прогреваю модели ({ASR_MODEL.split('/')[-1]} + {LLM_MODEL.split('/')[-1]})...")
    ready = threading.Event()
    threading.Thread(target=ml_worker, args=(ready,), daemon=True).start()
    threading.Thread(target=_open_stream_quiet, daemon=True).start()
    hud.watch_default_output()  # AirPods подключили — звуки сразу мимо них
    threading.Thread(target=mic_watcher, daemon=True).start()
    threading.Thread(target=stream_watchdog, daemon=True).start()
    watch_default_input(mic_changed.set)
    # без darwin_intercept: с ним pynput регистрирует блокирующий слушатель —
    # каждое нажатие в системе ждёт наш Python-колбэк, и macOS отключает его по
    # таймауту (хоткей переставал работать до перезапуска)
    keyboard.Listener(on_press=on_press, on_release=on_release).start()
    print("Меню-бар запущен. Зажми правый Option и говори; отпусти — текст вставится.")
    DictateApp().run()


if __name__ == "__main__":
    if "--version" in sys.argv or "-V" in sys.argv:
        print(f"Dictate {app_version()}")
    else:
        main()
