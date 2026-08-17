"""Голосовые команды и сниппеты: короткая фраза вместо диктовки.

Команда срабатывает в двух формах:
  • вся фраза целиком: «удали», «отправь», «новая строка» — текст не
    вставляется, выполняется действие;
  • хвост фразы: «созвон переносим на завтра, отправь» — текст до команды
    вставляется как обычно, затем выполняется действие. Хвостом работают только
    команды, где это осмысленно (отправить, перенос, абзац, сниппет…);
    «удали» в хвосте — просто слово.
Внутри текста команды не ищутся: «удали файл и перезапусти» — обычная диктовка.

Сопоставление — по нормализованным словам (нижний регистр, ё→е, без знаков):
Whisper пишет «Удали.» или «Отправь!», а человеку всё равно.

Пользовательский слой — commands.json рядом со скриптом (см. TEMPLATE):
свои сниппеты, свои формулировки для встроенных команд, отключение команд.
Файл перечитывается на лету по mtime.
"""
import json
import os
import re
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(BASE, "commands.json")
MAX_WORDS = 4  # длиннее — точно не команда, даже если совпало бы

# id -> (фразы, можно ли хвостом, описание для списка в меню)
DEFAULTS = {
    "delete_line": (["удали", "удали строку", "сотри", "сотри строку", "очисти строку"],
                    False, "стереть текущую строку"),
    "undo": (["отмени", "отмена", "откати", "убери"],
             False, "убрать последнюю вставленную диктовку"),
    "delete_word": (["удали слово", "сотри слово"], False, "стереть слово слева (⌥⌫)"),
    "delete_all": (["удали всё", "сотри всё", "очисти всё", "очисти поле"],
                   False, "выделить всё и стереть"),
    "select_all": (["выдели всё", "выделить всё"], False, "выделить всё (⌘A)"),
    "repeat": (["повтори", "ещё раз"], False, "вставить последнюю диктовку ещё раз"),
    "send": (["отправь", "отправить", "отправляй"], True, "Return (в мессенджерах — отправить)"),
    "new_line": (["новая строка", "с новой строки", "перенос", "перенос строки"],
                 True, "перенос строки без отправки (⇧Return)"),
    "paragraph": (["абзац", "новый абзац"], True, "пустая строка (⇧Return ×2)"),
    "tab": (["таб", "табуляция"], True, "Tab"),
    "enter": (["энтер", "ввод"], True, "Return"),
    "esc": (["эскейп", "отбой"], False, "Esc"),
    "copy": (["скопируй", "копировать"], False, "⌘C"),
    "cut": (["вырежи", "вырезать"], False, "⌘X"),
    "paste": (["вставь", "вставить"], False, "⌘V"),
    "translate": (["переведи", "перевод"], True,
                  "перевести на английский: хвостом — текст перед командой, отдельно — "
                  "последнюю вставку (она заменяется переводом)"),
    "style_clean": (["обычный стиль", "стиль чистка", "стиль по умолчанию"], False,
                    "профиль приложения: Чистка"),
    "style_casual": (["разговорный стиль", "стиль разговорный"], False,
                     "профиль приложения: Разговорный"),
    "style_formal": (["строгий стиль", "стиль строгий"], False,
                     "профиль приложения: Строгий"),
    "style_raw": (["как сказано", "стиль как сказано", "без правок"], False,
                  "профиль приложения: Как сказано (без LLM)"),
}

TEMPLATE = {
    "_help": [
        "Сниппеты: фраза -> текст. Сказал «моя почта» — вставилась почта; хвостом тоже: "
        "«пиши на, моя почта». Многострочный текст — через \\n.",
        "phrases: свои формулировки для встроенных команд (список заменяет стандартный). "
        "Идентификаторы: " + ", ".join(DEFAULTS),
        "disabled: список идентификаторов команд, которые выключить.",
        "Файл перечитывается сам, перезапуск не нужен. Меню → Команды → «Список команд…» "
        "показывает, что сейчас действует.",
    ],
    "snippets": {
        "моя почта": "user@example.com",
        "подпись": "С уважением,\nИмя Фамилия",
    },
    "phrases": {},
    "disabled": [],
}

_cache = {"mtime": None, "data": None}


def _norm(s: str) -> list:
    s = s.lower().replace("ё", "е")
    return re.findall(r"[\w'-]+", s)


def _user() -> dict:
    """commands.json (или пустой слой): перечитывается, если файл поменялся."""
    try:
        m = os.path.getmtime(PATH)
    except OSError:
        _cache.update(mtime=None, data={})
        return {}
    if m != _cache["mtime"]:
        try:
            with open(PATH) as f:
                d = json.load(f)
            if not isinstance(d, dict):
                raise ValueError("не объект")
            _cache.update(mtime=m, data=d)
        except (ValueError, OSError, UnicodeDecodeError) as e:
            print(f"  commands.json не прочитан ({e}) — работаю на встроенных", flush=True)
            _cache.update(mtime=m, data={})
    return _cache["data"] or {}


def ensure_file() -> str:
    """Создать commands.json из шаблона, если его нет; вернуть путь."""
    if not os.path.exists(PATH):
        with open(PATH, "w") as f:
            json.dump(TEMPLATE, f, ensure_ascii=False, indent=2)
    return PATH


def table() -> list:
    """Действующие команды: [(id, [фразы], trailing, описание)] + сниппеты
    как ('snippet:<фраза>', [фраза], True, текст)."""
    u = _user()
    disabled = set(u.get("disabled") or []) if isinstance(u.get("disabled"), list) else set()
    over = u.get("phrases") if isinstance(u.get("phrases"), dict) else {}
    rows = []
    for cid, (phrases, trailing, desc) in DEFAULTS.items():
        if cid in disabled:
            continue
        p = over.get(cid)
        if isinstance(p, list) and all(isinstance(x, str) for x in p) and p:
            phrases = p
        rows.append((cid, phrases, trailing, desc))
    snips = u.get("snippets") if isinstance(u.get("snippets"), dict) else {}
    for phrase, text in snips.items():
        if isinstance(phrase, str) and isinstance(text, str) and phrase.strip() and text:
            rows.append((f"snippet:{phrase}", [phrase], True, text))
    return rows


class Match:
    def __init__(self, cid, phrase, head, snippet=None):
        self.cid = cid          # id команды или "snippet:<фраза>"
        self.phrase = phrase    # какой формулировкой сказано
        self.head = head        # текст ДО команды (хвостовая форма) или ""
        self.snippet = snippet  # текст сниппета, если это сниппет

    @property
    def is_snippet(self):
        return self.snippet is not None

    def __repr__(self):
        return f"Match({self.cid!r}, head={self.head!r})"


def match(raw: str) -> "Match | None":
    """Найти команду во фразе: целиком или в хвосте. None — обычная диктовка."""
    words = _norm(raw)
    if not words:
        return None
    rows = table()
    # 1) вся фраза — команда (не длиннее MAX_WORDS)
    if len(words) <= MAX_WORDS:
        for cid, phrases, _tr, desc in rows:
            for ph in phrases:
                if _norm(ph) == words:
                    return Match(cid, ph, "", desc if cid.startswith("snippet:") else None)
    # 2) хвост: последние k слов — команда, перед ними есть текст
    spans = [m for m in re.finditer(r"[\w'-]+", raw.replace("ё", "е").replace("Ё", "Е"))]
    best = None
    for cid, phrases, trailing, desc in rows:
        if not trailing:
            continue
        for ph in phrases:
            pw = _norm(ph)
            k = len(pw)
            if k < len(words) and words[-k:] == pw:
                if best is None or k > best[0]:  # длиннее совпадение точнее
                    best = (k, cid, ph, desc)
    if best is None:
        return None
    k, cid, ph, desc = best
    cut = spans[len(spans) - k].start()
    head = raw[:cut].rstrip(" \t\n,.;:!?…—–-")
    if not head:
        return None
    return Match(cid, ph, head, desc if cid.startswith("snippet:") else None)


def describe_all() -> str:
    """Текст для окна «Список команд»."""
    lines = []
    for cid, phrases, trailing, desc in table():
        if cid.startswith("snippet:"):
            continue
        tail = "  (и хвостом)" if trailing else ""
        lines.append(f"• {' / '.join(phrases)} — {desc}{tail}")
    snips = [(p[0], d) for cid, p, _t, d in table() if cid.startswith("snippet:")]
    if snips:
        lines.append("")
        lines.append("Сниппеты:")
        for ph, text in snips:
            short = text.replace("\n", "⏎")
            if len(short) > 40:
                short = short[:39] + "…"
            lines.append(f"• {ph} → {short}")
    return "\n".join(lines)
