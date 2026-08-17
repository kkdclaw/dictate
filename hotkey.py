"""Горячая клавиша диктовки: строка в config.json ↔ подпись ↔ события pynput.

Формат строки (ключ «hotkey» в config.json): токены через «+», последний —
клавиша, остальные — модификаторы ctrl/alt/shift/cmd. Клавиша — либо имя из
pynput.keyboard.Key (alt_r, cmd_r, space, f13, tab…), либо «fn» (клавиша
Fn/🌐, которую pynput не знает), либо символ US-раскладки (d, 1, `…) — по нему
берётся ФИЗИЧЕСКИЙ код kVK_ANSI_*, поэтому «cmd+shift+d» работает и в русской
раскладке, где та же кнопка печатает «в».

Примеры: «alt_r» (правый Option, по умолчанию), «fn», «ctrl+space», «cmd+shift+d», «f13».

Одиночные модификаторы (Option/Command/Control/Shift/Fn) сопоставляются только
по коду клавиши; для сочетаний дополнительно требуется, чтобы в момент нажатия
был зажат РОВНО указанный набор модификаторов — иначе ⌃⇧Space срабатывал бы как
⌃Space.
"""
from pynput import keyboard
import Quartz

MODS = {  # имя -> маска флагов CGEvent
    "ctrl": Quartz.kCGEventFlagMaskControl,
    "alt": Quartz.kCGEventFlagMaskAlternate,
    "shift": Quartz.kCGEventFlagMaskShift,
    "cmd": Quartz.kCGEventFlagMaskCommand,
}
MOD_ORDER = ["ctrl", "alt", "shift", "cmd"]  # порядок глифов как в меню macOS: ⌃⌥⇧⌘
MOD_GLYPH = {"ctrl": "⌃", "alt": "⌥", "shift": "⇧", "cmd": "⌘"}
ALL_MODS = 0
for _m in MODS.values():
    ALL_MODS |= _m

VK_FN = 0x3F  # kVK_Function — pynput отдаёт его как KeyCode(vk=63) и только на отпускание

# коды клавиш-модификаторов -> имя токена; такие хоткеи ловим сами по себе,
# без проверки флагов (флаг самой клавиши в этот момент уже поднят)
MODIFIER_VKS = {0x3A: "alt_l", 0x3D: "alt_r", 0x37: "cmd_l", 0x36: "cmd_r",
                0x3B: "ctrl_l", 0x3E: "ctrl_r", 0x38: "shift_l", 0x3C: "shift_r",
                VK_FN: "fn"}
MODIFIER_LABELS = {
    "alt_r": "Правый Option ⌥", "alt_l": "Левый Option ⌥",
    "cmd_r": "Правый Command ⌘", "cmd_l": "Левый Command ⌘",
    "ctrl_r": "Правый Control ⌃", "ctrl_l": "Левый Control ⌃",
    "shift_r": "Правый Shift ⇧", "shift_l": "Левый Shift ⇧",
    "fn": "Fn / 🌐",
}
# kVK_ANSI_* — физические коды клавиш, одинаковые в любой раскладке
US_VK = {
    "a": 0x00, "s": 0x01, "d": 0x02, "f": 0x03, "h": 0x04, "g": 0x05, "z": 0x06,
    "x": 0x07, "c": 0x08, "v": 0x09, "b": 0x0B, "q": 0x0C, "w": 0x0D, "e": 0x0E,
    "r": 0x0F, "y": 0x10, "t": 0x11, "1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15,
    "6": 0x16, "5": 0x17, "=": 0x18, "9": 0x19, "7": 0x1A, "-": 0x1B, "8": 0x1C,
    "0": 0x1D, "]": 0x1E, "o": 0x1F, "u": 0x20, "[": 0x21, "i": 0x22, "p": 0x23,
    "l": 0x25, "j": 0x26, "'": 0x27, "k": 0x28, ";": 0x29, "\\": 0x2A, ",": 0x2B,
    "/": 0x2C, "n": 0x2D, "m": 0x2E, ".": 0x2F, "`": 0x32,
}
VK_US = {v: k for k, v in US_VK.items()}

# специальные клавиши pynput: имя -> код (алиасы вроде alt/alt_l дают один код)
SPECIAL_VK = {name: member.value.vk for name, member in keyboard.Key.__members__.items()
              if member.value.vk is not None}
SPECIAL_VK["fn"] = VK_FN
VK_SPECIAL = {}
for _name in ["alt_l", "alt_r", "cmd_l", "cmd_r", "ctrl_l", "ctrl_r", "shift_l", "shift_r",
              "backspace", "caps_lock", "delete", "down", "end", "enter", "esc",
              "home", "left", "page_down", "page_up", "right", "space", "tab", "up",
              "fn"] + [f"f{i}" for i in range(1, 21)]:
    VK_SPECIAL[SPECIAL_VK[_name]] = _name

SPECIAL_LABELS = {"space": "Space", "tab": "Tab", "enter": "Return", "esc": "Esc",
                  "backspace": "Delete ⌫", "delete": "Forward Delete ⌦",
                  "up": "↑", "down": "↓", "left": "←", "right": "→",
                  "home": "Home", "end": "End", "page_up": "PgUp", "page_down": "PgDn",
                  "caps_lock": "Caps Lock"}

DEFAULT = "alt_r"

# готовые варианты в меню: (строка, подпись)
PRESETS = [
    ("alt_r", "Правый Option ⌥  — по умолчанию"),
    ("alt_l", "Левый Option ⌥"),
    ("cmd_r", "Правый Command ⌘"),
    ("ctrl_r", "Правый Control ⌃"),
    ("shift_r", "Правый Shift ⇧"),
    ("fn", "Fn / 🌐"),
    ("ctrl+space", "⌃ Space"),
    ("alt+space", "⌥ Space"),
]


class Hotkey:
    """Разобранное сочетание. mods — маска требуемых модификаторов (0 для
    одиночной клавиши-модификатора), vk — физический код клавиши."""

    def __init__(self, spec: str, mods: int, mod_names: list, vk: int, key: str):
        self.spec = spec
        self.mods = mods
        self.mod_names = mod_names
        self.vk = vk
        self.key = key
        self.is_modifier = vk in MODIFIER_VKS
        self.is_fn = vk == VK_FN

    def matches(self, vk, flags: int) -> bool:
        if vk != self.vk:
            return False
        if self.is_modifier:
            return True
        return (flags & ALL_MODS) == self.mods

    @property
    def label(self) -> str:
        return describe(self.spec)

    def __repr__(self):
        return f"Hotkey({self.spec!r})"


def _key_vk(token: str):
    if token in SPECIAL_VK:
        return SPECIAL_VK[token]
    if len(token) == 1 and token in US_VK:
        return US_VK[token]
    if token.startswith("vk") and token[2:].isdigit():
        return int(token[2:])
    return None


def parse(spec) -> "Hotkey | None":
    """Строка из конфига -> Hotkey; None, если строка бессмысленна (тогда
    вызывающий берёт DEFAULT — битый конфиг не должен оставить без диктовки)."""
    if not isinstance(spec, str) or not spec.strip():
        return None
    tokens = [t.strip().lower() for t in spec.strip().split("+")]
    if not tokens or not tokens[-1]:
        return None
    *mods, key = tokens
    if key == "plus":  # «cmd+plus» — единственный способ записать сам «+»
        key = "="
    mask = 0
    names = []
    for m in mods:
        if m not in MODS or m in names:
            return None
        mask |= MODS[m]
        names.append(m)
    vk = _key_vk(key)
    if vk is None:
        return None
    if vk in MODIFIER_VKS and mods:
        return None  # «cmd+alt_r» — так не работает: одиночный модификатор без сочетаний
    if vk not in MODIFIER_VKS and not mods and not _standalone_ok(vk):
        return None  # голая буква/цифра/пробел — сработает при обычном наборе
    key_name = VK_SPECIAL.get(vk) or VK_US.get(vk) or f"vk{vk}"
    canon = "+".join([m for m in MOD_ORDER if m in names] + [key_name])
    return Hotkey(canon, mask, [m for m in MOD_ORDER if m in names], vk, key_name)


def _standalone_ok(vk: int) -> bool:
    """Какие клавиши годятся хоткеем без модификаторов: F1–F20 (при обычном
    наборе не нужны) — да; буквы, цифры, пробел, стрелки, Return — нет."""
    name = VK_SPECIAL.get(vk, "")
    return name.startswith("f") and name[1:].isdigit()


def describe(spec) -> str:
    """Подпись для меню/окна: «Правый Option ⌥», «⌃ Space», «⌘⇧D», «F13»."""
    hk = parse(spec)
    if hk is None:
        return str(spec)
    if hk.is_modifier:
        return MODIFIER_LABELS.get(hk.key, hk.key)
    glyphs = "".join(MOD_GLYPH[m] for m in hk.mod_names)
    key = _key_label(hk.key)
    return f"{glyphs} {key}".strip() if glyphs else key


def _key_label(key: str) -> str:
    if key in SPECIAL_LABELS:
        return SPECIAL_LABELS[key]
    if key in MODIFIER_LABELS:
        return MODIFIER_LABELS[key]
    return key.upper() if (len(key) == 1 or key.startswith("f")) else key


def key_vk(key):
    """Код клавиши из объекта события pynput (Key или KeyCode); None — неизвестно."""
    if isinstance(key, keyboard.Key):
        return key.value.vk
    return getattr(key, "vk", None)


def current_flags() -> int:
    """Модификаторы, зажатые прямо сейчас (по HID-состоянию, а не по событию —
    в колбэк pynput событие не приходит)."""
    return Quartz.CGEventSourceFlagsState(Quartz.kCGEventSourceStateCombinedSessionState)


def spec_from_press(vk: int, flags: int) -> "str | None":
    """Собрать строку хоткея из нажатия при захвате «своего сочетания».
    None — из этого нажатия хоткей не сделать (например, голая буква)."""
    if vk is None:
        return None
    if vk in MODIFIER_VKS:
        return MODIFIER_VKS[vk]
    names = [m for m in MOD_ORDER if flags & MODS[m]]
    key = VK_SPECIAL.get(vk) or VK_US.get(vk) or f"vk{vk}"
    spec = "+".join(names + [key])
    return spec if parse(spec) else None


def held_description(vk, flags: int) -> str:
    """Что показывать в окне захвата, пока клавиши зажаты: «⌘⇧» + клавиша."""
    if vk in MODIFIER_VKS:  # зажат один модификатор — показываем его по имени
        return MODIFIER_LABELS[MODIFIER_VKS[vk]]
    parts = [MOD_GLYPH[m] for m in MOD_ORDER if flags & MODS[m]]
    if vk is not None:
        parts.append(_key_label(VK_SPECIAL.get(vk) or VK_US.get(vk) or f"vk{vk}"))
    return " ".join(parts) if parts else "…"
