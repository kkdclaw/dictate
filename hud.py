"""Капсула-индикатор записи у текстового курсора и короткие звуки-подтверждения.

Индикатор — плавающая NSPanel без рамки поверх всех окон, не берёт фокус и
не ловит мышь. Состояния: «rec» (запись: столбики уровня в акцентном цвете),
«busy» (распознаю: три пульсирующих точки). Позиция — у каретки активного
текстового поля (Accessibility API); если поле не отдаёт координаты — внизу
экрана с курсором мыши. Всё рисование — на главном потоке: вызовы из других
потоков (pynput, ML-воркер, PortAudio) переправляются через callAfter.

Звуки — системные из /System/Library/Sounds: Tink (старт записи), Pop
(текст вставлен), Basso (запись отброшена).
"""
import math
import threading
import time

import AppKit
import objc
from Foundation import NSMakeRect, NSObject, NSTimer
from PyObjCTools import AppHelper

# --- настройки (ключи хранятся в config.json) ---------------------------------
COLORS = {  # ключ -> (подпись, RGB 0..1)
    "red": ("Красный", (1.0, 0.27, 0.23)),
    "orange": ("Оранжевый", (1.0, 0.58, 0.0)),
    "pink": ("Розовый", (1.0, 0.18, 0.33)),
    "purple": ("Фиолетовый", (0.75, 0.35, 0.95)),
    "blue": ("Синий", (0.0, 0.44, 1.0)),
    "cyan": ("Голубой", (0.2, 0.68, 0.9)),
    "green": ("Зелёный", (0.2, 0.78, 0.35)),
    "white": ("Белый", (1.0, 1.0, 1.0)),
}
SIZES = {  # ключ -> (подпись, (ширина, высота) в пунктах)
    "tiny": ("Супермаленький", (30.0, 12.0)),
    "small": ("Маленький", (46.0, 20.0)),
    "normal": ("Обычный", (64.0, 28.0)),
}
_LEGACY_SIZES = {"compact": "tiny", "standard": "small", "large": "normal"}
ICONS = {  # что рисуем внутри капсулы во время записи
    "bars": "Столбики уровня",
    "dot": "Точка (пульсирует от громкости)",
    "ring": "Кольцо",
    "mic": "Микрофон",
    "wave": "Волна",
}
BACKGROUNDS = {"system": "Как в системе", "dark": "Тёмный", "light": "Светлый"}
POSITIONS = {"caret": "У текстового курсора", "bottom": "Внизу экрана"}
SOUND_EVENTS = {  # событие -> (подпись, звук по умолчанию)
    "start": ("Звук старта записи", "Tink"),
    "done": ("Звук вставки", "Pop"),
    "error": ("Звук отказа (пусто/не речь)", "Basso"),
}
DEFAULTS = {"hud": True, "hud_size": "small", "hud_icon": "bars", "hud_color": "red",
            "hud_bg": "system", "hud_pos": "caret", "sounds": True,
            "sound_start": "Tink", "sound_done": "Pop", "sound_error": "Basso"}
SOUNDS_DIR = "/System/Library/Sounds"


def system_sounds() -> list:
    """Имена системных звуков macOS (Basso, Blow, … Tink)."""
    try:
        import os
        return sorted(f[:-5] for f in os.listdir(SOUNDS_DIR) if f.endswith(".aiff"))
    except Exception:
        return ["Tink", "Pop", "Basso"]

BARS = 5
_cfg = dict(DEFAULTS)
_levels = []  # последние RMS-уровни (пишет аудиопоток, читает вью)
_state = {"mode": None, "shown_at": 0.0}


def configure(cfg: dict):
    """Подхватить настройки из CONFIG (вызывать при старте и после смены)."""
    for k in DEFAULTS:
        if k in cfg:
            _cfg[k] = cfg[k]
    if _cfg["hud_size"] in _LEGACY_SIZES:  # старые ключи размера из config.json
        _cfg["hud_size"] = _LEGACY_SIZES[_cfg["hud_size"]]
        cfg["hud_size"] = _cfg["hud_size"]
    if _panel is not None:
        AppHelper.callAfter(_relayout)


def push_level(rms: float):
    """Уровень сигнала для столбиков (вызывается из аудио-колбэка)."""
    _levels.append(min(1.0, rms * 12.0))
    if len(_levels) > 64:
        del _levels[:-64]


# --- звуки --------------------------------------------------------------------
# Важно: если вывод по умолчанию — Bluetooth (AirPods), звук в них НЕЛЬЗЯ:
# наушники перещёлкивают профиль A2DP↔HFP, и микрофон на пару секунд отдаёт
# нули (диктовка «ломается после двух слов»). Тогда играем во встроенный
# динамик Mac; если его нет — молчим.
_sounds = {}  # имя файла -> NSSound
_VOLUME = {"start": 0.55, "done": 0.45, "error": 0.30}
_route = {"ts": 0.0, "uid": None, "reason": ""}


def _sound_route():
    """UID устройства для звуков: None = дефолтный вывод, '' = не играть."""
    now = time.time()
    if now - _route["ts"] < 5.0:
        return _route["uid"]
    uid, reason = None, "вывод по умолчанию"
    try:
        import ctypes
        ca = ctypes.CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        cf.CFStringGetCString.restype = ctypes.c_bool

        class Addr(ctypes.Structure):
            _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32),
                        ("element", ctypes.c_uint32)]

        def fcc(x):
            return int.from_bytes(x.encode(), "big")

        def size(obj, sel, scope="glob"):
            a = Addr(fcc(sel), fcc(scope), 0)
            n = ctypes.c_uint32(0)
            ca.AudioObjectGetPropertyDataSize(obj, ctypes.byref(a), 0, None, ctypes.byref(n))
            return n.value

        def get(obj, sel, buf, scope="glob"):
            a = Addr(fcc(sel), fcc(scope), 0)
            n = ctypes.c_uint32(ctypes.sizeof(buf))
            return ca.AudioObjectGetPropertyData(obj, ctypes.byref(a), 0, None,
                                                 ctypes.byref(n), ctypes.byref(buf))

        def cfstr(ref):
            b = ctypes.create_string_buffer(256)
            cf.CFStringGetCString(ref, b, 256, 0x08000100)
            return b.value.decode()

        dout = ctypes.c_uint32()
        get(1, "dOut", dout)
        tt = ctypes.c_uint32()
        get(dout.value, "tran", tt)
        if tt.value == fcc("blue"):
            uid, reason = "", "вывод — Bluetooth, встроенного динамика нет"
            n = size(1, "dev#") // 4
            ids = (ctypes.c_uint32 * n)()
            get(1, "dev#", ids)
            for d in ids:
                t2 = ctypes.c_uint32()
                get(d, "tran", t2)
                if t2.value == fcc("bltn") and size(d, "stm#", "outp"):
                    ref = ctypes.c_void_p()
                    get(d, "uid ", ref)
                    uid, reason = cfstr(ref), "вывод — Bluetooth, звуки во встроенный динамик"
                    break
    except Exception as e:  # noqa
        uid, reason = None, f"не определил вывод ({e})"
    _route.update(ts=now, uid=uid, reason=reason)
    return uid


def sound_route_info() -> str:
    _sound_route()
    return _route["reason"]


def _play_main(name, volume):
    route = _sound_route()
    if route == "" or not name or name == "none":
        return
    snd = _sounds.get(name)
    if snd is None:
        snd = AppKit.NSSound.alloc().initWithContentsOfFile_byReference_(
            f"{SOUNDS_DIR}/{name}.aiff", True)
        if snd is None:
            return
        _sounds[name] = snd
    snd.setVolume_(volume)
    try:
        snd.setPlaybackDeviceIdentifier_(route)  # None = дефолт
    except Exception:
        pass
    snd.stop()
    snd.play()


def play(kind: str):
    """start | done | error — с любого потока; молчит, если звуки выключены."""
    if _cfg.get("sounds", True) and kind in SOUND_EVENTS:
        AppHelper.callAfter(_play_main, _cfg.get(f"sound_{kind}"), _VOLUME[kind])


def preview_sound(name: str):
    """Проиграть звук по имени (выбор в меню) — с любого потока."""
    AppHelper.callAfter(_play_main, name, 0.5)


# --- позиция: каретка через Accessibility -----------------------------------
def _caret_rect_appkit():
    """Прямоугольник каретки активного поля в координатах AppKit, или None."""
    try:
        import ApplicationServices as AS
        sysw = AS.AXUIElementCreateSystemWide()
        err, el = AS.AXUIElementCopyAttributeValue(sysw, AS.kAXFocusedUIElementAttribute, None)
        if err or el is None:
            return None
        err, rng = AS.AXUIElementCopyAttributeValue(el, AS.kAXSelectedTextRangeAttribute, None)
        if err or rng is None:
            return None
        values = [rng]
        try:  # пустое выделение у некоторых полей не отдаёт границы — пробуем длину 1
            ok, r = AS.AXValueGetValue(rng, AS.kAXValueCFRangeType, None)
            if ok and r.length == 0:
                v1 = AS.AXValueCreate(AS.kAXValueCFRangeType, (r.location, 1))
                if v1 is not None:
                    values.append(v1)
        except Exception:
            pass
        rect = None
        for val in values:
            err, b = AS.AXUIElementCopyParameterizedAttributeValue(
                el, AS.kAXBoundsForRangeParameterizedAttribute, val, None)
            if err or b is None:
                continue
            ok, rc = AS.AXValueGetValue(b, AS.kAXValueCGRectType, None)
            if (ok and rc is not None and 0 < rc.size.height <= 120
                    and rc.size.width <= max(12.0, rc.size.height * 1.5)):
                rect = rc
                break
        if rect is None:
            return None
        # AX: начало координат — левый верхний угол главного экрана, ось Y вниз
        primary = AppKit.NSScreen.screens()[0].frame()
        x = rect.origin.x
        y = primary.size.height - rect.origin.y - rect.size.height
        return NSMakeRect(x, y, max(2.0, rect.size.width), rect.size.height)
    except Exception:
        return None


def _screen_for_point(x, y):
    for s in AppKit.NSScreen.screens():
        if AppKit.NSPointInRect((x, y), s.frame()):
            return s
    return AppKit.NSScreen.mainScreen() or AppKit.NSScreen.screens()[0]


def _target_frame(w, h):
    caret = _caret_rect_appkit() if _cfg["hud_pos"] == "caret" else None
    if caret is not None:
        scr = _screen_for_point(caret.origin.x, caret.origin.y)
        vis = scr.visibleFrame()
        x = caret.origin.x + caret.size.width / 2 - w / 2
        y = caret.origin.y + caret.size.height + 8
        if y + h > vis.origin.y + vis.size.height:  # не влезает сверху — под строкой
            y = caret.origin.y - h - 8
    else:
        loc = AppKit.NSEvent.mouseLocation()
        vis = _screen_for_point(loc.x, loc.y).visibleFrame()
        x = vis.origin.x + vis.size.width / 2 - w / 2
        y = vis.origin.y + 56
    x = min(max(x, vis.origin.x + 6), vis.origin.x + vis.size.width - w - 6)
    y = min(max(y, vis.origin.y + 6), vis.origin.y + vis.size.height - h - 6)
    return NSMakeRect(x, y, w, h)


# --- вью и панель ------------------------------------------------------------
class HUDView(AppKit.NSView):
    def isFlipped(self):
        return False

    def drawRect_(self, rect):
        b = self.bounds()
        w, h = b.size.width, b.size.height
        bg = _cfg["hud_bg"]
        if bg == "system":
            name = AppKit.NSApp.effectiveAppearance().name() if AppKit.NSApp else ""
            bg = "dark" if "Dark" in str(name) else "light"
        if bg == "dark":
            fill = AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.08, 0.86)
            stroke = AppKit.NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.12)
        else:
            fill = AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.98, 0.92)
            stroke = AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.0, 0.10)
        path = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            AppKit.NSInsetRect(b, 1, 1), h / 2, h / 2)
        fill.setFill()
        path.fill()
        stroke.setStroke()
        path.setLineWidth_(1.0)
        path.stroke()

        r, g, bl = COLORS.get(_cfg["hud_color"], COLORS["red"])[1]
        if bg == "light" and _cfg["hud_color"] == "white":
            r, g, bl = (0.25, 0.25, 0.25)  # белый на светлом не виден
        accent = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, bl, 1.0)
        t = time.time()
        mode = _state["mode"]
        tiny = h < 16
        if mode == "rec":
            lv = _levels[-6:] if _levels else [0.0]
            level = max(lv[-1], 0.0)
            icon = _cfg.get("hud_icon", "bars")
            if icon == "bars":
                _draw_bars(w, h, accent, t, tiny)
            elif icon == "dot":
                d = h * (0.36 + 0.42 * min(1.0, level * 1.5))
                accent.setFill()
                AppKit.NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect((w - d) / 2, (h - d) / 2, d, d)).fill()
            elif icon == "ring":
                d0 = h * 0.42
                accent.setFill()
                AppKit.NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect((w - d0) / 2, (h - d0) / 2, d0, d0)).fill()
                ph = (t * 1.6) % 1.0  # расходящееся кольцо
                d1 = d0 + (h * 0.5) * ph
                accent.colorWithAlphaComponent_(1.0 - ph).setStroke()
                ring = AppKit.NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect((w - d1) / 2, (h - d1) / 2, d1, d1))
                ring.setLineWidth_(max(1.0, h * 0.08))
                ring.stroke()
            else:  # mic / wave — SF Symbol, яркость от громкости
                name = "mic.fill" if icon == "mic" else "waveform"
                img = _symbol(name, h * 0.62, accent)
                if img is not None:
                    a = 0.55 + 0.45 * min(1.0, level * 1.5)
                    sz = img.size()
                    img.drawInRect_fromRect_operation_fraction_(
                        NSMakeRect((w - sz.width) / 2, (h - sz.height) / 2, sz.width, sz.height),
                        AppKit.NSZeroRect, AppKit.NSCompositingOperationSourceOver, a)
                else:
                    _draw_bars(w, h, accent, t, tiny)
        elif mode == "busy":
            n = 1 if tiny else 3
            d = h * (0.4 if tiny else 0.2)
            gap = d * 0.9
            total = n * d + (n - 1) * gap
            x0 = (w - total) / 2
            for i in range(n):
                a = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(t * 5 - i * 1.1))
                accent.colorWithAlphaComponent_(a).setFill()
                AppKit.NSBezierPath.bezierPathWithOvalInRect_(
                    NSMakeRect(x0 + i * (d + gap), (h - d) / 2, d, d)).fill()

def _draw_bars(w, h, accent, t, tiny):
    n = 3 if tiny else BARS
    lv = _levels[-n:] if _levels else []
    lv = [0.0] * (n - len(lv)) + lv
    gap = h * 0.14
    bw = max(1.5, min(h * 0.18, (w - h) / n - gap))
    total = n * bw + (n - 1) * gap
    x0 = (w - total) / 2
    for i, v in enumerate(lv):
        pulse = 0.5 + 0.5 * math.sin(t * 6 + i)  # лёгкое дыхание в тишине
        bh = h * 0.2 + (h * 0.55) * max(v, 0.08 * pulse)
        rr = NSMakeRect(x0 + i * (bw + gap), (h - bh) / 2, bw, bh)
        accent.setFill()
        AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            rr, bw / 2, bw / 2).fill()


_symbols = {}


def _symbol(name, size, color):
    key = (name, round(size, 1), color.description())
    if key in _symbols:
        return _symbols[key]
    img = None
    try:
        img = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, None)
        if img is not None:
            cfg = AppKit.NSImageSymbolConfiguration.configurationWithPointSize_weight_(
                size, AppKit.NSFontWeightSemibold)
            try:
                cfg = cfg.configurationByApplyingConfiguration_(
                    AppKit.NSImageSymbolConfiguration.configurationWithHierarchicalColor_(color))
            except Exception:
                pass
            img = img.imageWithSymbolConfiguration_(cfg)
    except Exception:
        img = None
    _symbols[key] = img
    return img


class _Ticker(NSObject):
    def tick_(self, timer):
        if _view is not None:
            _view.setNeedsDisplay_(True)


_panel = None
_view = None
_timer = None
_ticker = _Ticker.alloc().init()


def _size():
    return SIZES.get(_cfg["hud_size"], SIZES["small"])[1]


def _ensure_panel():
    global _panel, _view
    if _panel is not None:
        return
    w, h = _size()
    style = AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel
    p = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(0, 0, w, h), style, AppKit.NSBackingStoreBuffered, False)
    p.setLevel_(AppKit.NSStatusWindowLevel)
    p.setOpaque_(False)
    p.setBackgroundColor_(AppKit.NSColor.clearColor())
    p.setHasShadow_(True)
    p.setIgnoresMouseEvents_(True)
    p.setHidesOnDeactivate_(False)
    p.setReleasedWhenClosed_(False)
    p.setCollectionBehavior_(
        AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
        | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        | AppKit.NSWindowCollectionBehaviorStationary
        | AppKit.NSWindowCollectionBehaviorIgnoresCycle)
    v = HUDView.alloc().initWithFrame_(NSMakeRect(0, 0, w, h))
    v.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
    p.setContentView_(v)
    _panel, _view = p, v


def _relayout():
    if _panel is None or _state["mode"] is None:
        return
    w, h = _size()
    _panel.setFrame_display_(_target_frame(w, h), True)


def _show_main(mode):
    global _timer
    if not _cfg.get("hud", True):
        return
    _ensure_panel()
    first = _state["mode"] is None
    _state["mode"] = mode
    if first:
        _levels.clear()
        _state["shown_at"] = time.time()
        w, h = _size()
        _panel.setFrame_display_(_target_frame(w, h), False)
        _panel.setAlphaValue_(1.0)
        _panel.orderFrontRegardless()
        if _timer is None:
            _timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                1 / 30.0, _ticker, "tick:", None, True)
            AppKit.NSRunLoop.currentRunLoop().addTimer_forMode_(
                _timer, AppKit.NSEventTrackingRunLoopMode)
    _view.setNeedsDisplay_(True)


def _hide_main():
    global _timer
    _state["mode"] = None
    if _timer is not None:
        _timer.invalidate()
        _timer = None
    if _panel is not None:
        _panel.orderOut_(None)


def show(mode: str = "rec"):
    """Показать/переключить капсулу: 'rec' — запись, 'busy' — распознаю."""
    AppHelper.callAfter(_show_main, mode)


def hide():
    AppHelper.callAfter(_hide_main)


def preview(seconds: float = 2.5):
    """Показать капсулу на пару секунд — для проверки настроек из меню."""
    show("rec")

    def _fake():
        t0 = time.time()
        while time.time() - t0 < seconds:
            push_level(0.03 + 0.05 * abs(math.sin(time.time() * 3)))
            time.sleep(0.05)
        hide()
    threading.Thread(target=_fake, daemon=True).start()
