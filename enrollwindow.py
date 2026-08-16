"""Окно записи отпечатка голоса: что читать, сколько осталось, сколько набрано.

Зачем отдельное окно, а не капсула у курсора и отсчёт в меню-баре: запись
отпечатка — редкая процедура на десяток секунд, и человеку нужно видеть сразу
три вещи, которых в капсуле не покажешь: ТЕКСТ, который читать (иначе он сам
придумывает фразу, запинается и молчит), уровень звука (микрофон вообще
слышит?) и сколько НАБРАНО ЧИСТОЙ РЕЧИ — паузы в зачёт не идут, и без счётчика
непонятно, почему двенадцать секунд превратились в три.

Всё рисуется на главном потоке; из потока записи звать update()/finish() —
они сами перекидывают вызов через AppHelper.

Состояния окна: idle (готов начать) -> rec (идёт запись) -> busy (считаю)
-> done (отчёт).
"""
import AppKit
from Foundation import NSMakeRect, NSObject
from PyObjCTools import AppHelper

# Текст для чтения: разные гласные, шипящие, звонкие/глухие пары, «р»/«л»,
# длинные слова — чтобы эмбеддинг захватил голос целиком, а не пару звуков.
READING_TEXT = (
    "Широкая электрификация южных губерний даст мощный толчок подъёму "
    "сельского хозяйства.\n"
    "Съешь ещё этих мягких французских булок да выпей чаю.\n"
    "В чащах юга жил-был цитрусовый жёлтый фрукт, объём которого "
    "человек ценил чрезвычайно высоко.\n"
    "Если текст кончился, а запись идёт — читай сначала, спокойно и без пауз."
)
CHECK_TEXT = "Скажи любую фразу обычным голосом — например, эту."

W = 620
_win = None
_ui = {}
_cb = {}


class _EnrollHandler(NSObject):
    def primary_(self, sender):
        fn = _cb.get("primary")
        if fn:
            fn()

    def secondary_(self, sender):
        fn = _cb.get("secondary")
        if fn:
            fn()


_handler = _EnrollHandler.alloc().init()


def _label(text, size=13.0, bold=False, color=None, wrap=False):
    f = (AppKit.NSTextField.wrappingLabelWithString_(text) if wrap
         else AppKit.NSTextField.labelWithString_(text))
    f.setFont_(AppKit.NSFont.boldSystemFontOfSize_(size) if bold
               else AppKit.NSFont.systemFontOfSize_(size))
    if color is not None:
        f.setTextColor_(color)
    f.setTranslatesAutoresizingMaskIntoConstraints_(False)
    return f


def _build():
    global _win
    style = (AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable)
    _win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(0, 0, W, 460), style, AppKit.NSBackingStoreBuffered, False)
    _win.setTitle_("Отпечаток голоса")
    _win.setReleasedWhenClosed_(False)
    # поверх всего: мы меню-бар без Dock и теряем фокус от любого клика мимо,
    # а человек в это время читает текст с экрана
    _win.setLevel_(AppKit.NSFloatingWindowLevel)
    _win.center()
    content = _win.contentView()

    head = _label("Запись отпечатка голоса", size=17.0, bold=True)
    hint = _label("", size=12.5, color=AppKit.NSColor.secondaryLabelColor(), wrap=True)

    # рамка с текстом для чтения — визуально отделяем «что делать» от служебного
    box = AppKit.NSBox.alloc().initWithFrame_(NSMakeRect(0, 0, W - 40, 150))
    box.setBoxType_(AppKit.NSBoxCustom)
    box.setCornerRadius_(10.0)
    box.setFillColor_(AppKit.NSColor.textBackgroundColor())
    box.setBorderColor_(AppKit.NSColor.separatorColor())
    box.setTitlePosition_(AppKit.NSNoTitle)
    box.setTranslatesAutoresizingMaskIntoConstraints_(False)
    read = _label(READING_TEXT, size=15.0, wrap=True)
    read.setPreferredMaxLayoutWidth_(W - 76)
    box.contentView().addSubview_(read)
    AppKit.NSLayoutConstraint.activateConstraints_([
        read.topAnchor().constraintEqualToAnchor_constant_(
            box.contentView().topAnchor(), 12),
        read.leadingAnchor().constraintEqualToAnchor_constant_(
            box.contentView().leadingAnchor(), 14),
        read.trailingAnchor().constraintEqualToAnchor_constant_(
            box.contentView().trailingAnchor(), -14),
        read.bottomAnchor().constraintLessThanOrEqualToAnchor_constant_(
            box.contentView().bottomAnchor(), -12),
    ])

    # уровень звука: сразу видно, что микрофон слышит (и какой именно микрофон)
    lvl_cap = _label("Микрофон", size=11.5,
                     color=AppKit.NSColor.secondaryLabelColor())
    level = AppKit.NSLevelIndicator.alloc().initWithFrame_(NSMakeRect(0, 0, 260, 18))
    level.setLevelIndicatorStyle_(AppKit.NSLevelIndicatorStyleContinuousCapacity)
    level.setMinValue_(0.0)
    level.setMaxValue_(1.0)
    level.setWarningValue_(0.75)
    level.setCriticalValue_(0.95)
    level.setTranslatesAutoresizingMaskIntoConstraints_(False)

    # главный счётчик: набранная РЕЧЬ, а не прошедшее время
    prog = AppKit.NSProgressIndicator.alloc().initWithFrame_(NSMakeRect(0, 0, W - 40, 8))
    prog.setStyle_(AppKit.NSProgressIndicatorStyleBar)
    prog.setIndeterminate_(False)
    prog.setMinValue_(0.0)
    prog.setMaxValue_(1.0)
    prog.setTranslatesAutoresizingMaskIntoConstraints_(False)
    stat = _label("", size=13.0, wrap=True)

    primary = AppKit.NSButton.buttonWithTitle_target_action_(
        "Начать запись", _handler, "primary:")
    primary.setKeyEquivalent_("\r")
    primary.setTranslatesAutoresizingMaskIntoConstraints_(False)
    secondary = AppKit.NSButton.buttonWithTitle_target_action_(
        "Отмена", _handler, "secondary:")
    secondary.setKeyEquivalent_("\x1b")  # Esc
    secondary.setTranslatesAutoresizingMaskIntoConstraints_(False)

    for v in (head, hint, box, lvl_cap, level, prog, stat, primary, secondary):
        content.addSubview_(v)
    m = 20.0
    C = AppKit.NSLayoutConstraint
    C.activateConstraints_([
        head.topAnchor().constraintEqualToAnchor_constant_(content.topAnchor(), m),
        head.leadingAnchor().constraintEqualToAnchor_constant_(content.leadingAnchor(), m),
        hint.topAnchor().constraintEqualToAnchor_constant_(head.bottomAnchor(), 6),
        hint.leadingAnchor().constraintEqualToAnchor_constant_(content.leadingAnchor(), m),
        hint.trailingAnchor().constraintEqualToAnchor_constant_(content.trailingAnchor(), -m),
        box.topAnchor().constraintEqualToAnchor_constant_(hint.bottomAnchor(), 14),
        box.leadingAnchor().constraintEqualToAnchor_constant_(content.leadingAnchor(), m),
        box.trailingAnchor().constraintEqualToAnchor_constant_(content.trailingAnchor(), -m),
        lvl_cap.topAnchor().constraintEqualToAnchor_constant_(box.bottomAnchor(), 16),
        lvl_cap.leadingAnchor().constraintEqualToAnchor_constant_(content.leadingAnchor(), m),
        level.centerYAnchor().constraintEqualToAnchor_(lvl_cap.centerYAnchor()),
        level.leadingAnchor().constraintEqualToAnchor_constant_(lvl_cap.trailingAnchor(), 10),
        level.trailingAnchor().constraintEqualToAnchor_constant_(content.trailingAnchor(), -m),
        prog.topAnchor().constraintEqualToAnchor_constant_(level.bottomAnchor(), 14),
        prog.leadingAnchor().constraintEqualToAnchor_constant_(content.leadingAnchor(), m),
        prog.trailingAnchor().constraintEqualToAnchor_constant_(content.trailingAnchor(), -m),
        stat.topAnchor().constraintEqualToAnchor_constant_(prog.bottomAnchor(), 10),
        stat.leadingAnchor().constraintEqualToAnchor_constant_(content.leadingAnchor(), m),
        stat.trailingAnchor().constraintEqualToAnchor_constant_(content.trailingAnchor(), -m),
        primary.topAnchor().constraintGreaterThanOrEqualToAnchor_constant_(
            stat.bottomAnchor(), 14),
        primary.trailingAnchor().constraintEqualToAnchor_constant_(
            content.trailingAnchor(), -m),
        primary.bottomAnchor().constraintEqualToAnchor_constant_(
            content.bottomAnchor(), -m),
        secondary.centerYAnchor().constraintEqualToAnchor_(primary.centerYAnchor()),
        secondary.trailingAnchor().constraintEqualToAnchor_constant_(
            primary.leadingAnchor(), -10),
    ])
    _ui.update(head=head, hint=hint, box=box, read=read, level=level,
               prog=prog, stat=stat, primary=primary, secondary=secondary)


def _fit():
    content = _win.contentView()
    content.layoutSubtreeIfNeeded()
    h = max(content.fittingSize().height, 300)
    old = _win.frame()
    if abs(h - old.size.height) > 1:
        _win.setFrame_display_(
            NSMakeRect(old.origin.x, old.origin.y + old.size.height - h, W, h), True)


def _show_main(mode, mic, seconds, need_sec):
    if _win is None:
        _build()
    enroll = mode == "enroll"
    _ui["head"].setStringValue_(
        "Запись отпечатка голоса" if enroll else "Проверка голоса")
    _ui["hint"].setStringValue_(
        (f"Читай текст вслух {seconds} секунд обычным голосом, без длинных пауз. "
         f"Засчитывается только речь: паузы в зачёт не идут.\n"
         f"Микрофон: {mic}. Отпечаток привязан к микрофону — для другого "
         f"(например, AirPods) запиши на нём заново.")
        if enroll else
        (f"Скажи фразу — {seconds} секунды. Покажу, узнаю ли я тебя и с каким "
         f"запасом до порога.\nМикрофон: {mic}."))
    _ui["read"].setStringValue_(READING_TEXT if enroll else CHECK_TEXT)
    _ui["stat"].setStringValue_("Готов. Нажми «Начать запись», когда будешь готов читать."
                               if enroll else "Готов. Нажми «Начать запись».")
    _ui["prog"].setDoubleValue_(0.0)
    _ui["level"].setDoubleValue_(0.0)
    _ui["primary"].setTitle_("Начать запись")
    _ui["primary"].setEnabled_(True)
    _ui["secondary"].setTitle_("Отмена")
    _ui["secondary"].setHidden_(False)
    _ui["_need"] = need_sec
    _fit()
    _win.makeKeyAndOrderFront_(None)
    AppKit.NSApp.activateIgnoringOtherApps_(True)


def _rec_main():
    _ui["primary"].setTitle_("Остановить")
    _ui["primary"].setEnabled_(True)
    _ui["secondary"].setTitle_("Отмена")
    _ui["stat"].setStringValue_("● Идёт запись…")


def _update_main(level, left, speech):
    if _win is None or not _win.isVisible():
        return
    _ui["level"].setDoubleValue_(min(1.0, level))
    need = _ui.get("_need") or 1.0
    _ui["prog"].setDoubleValue_(min(1.0, speech / need))
    enough = "✅" if speech >= need else "…"
    _ui["stat"].setStringValue_(
        f"● Запись: осталось {left:.0f} с   ·   набрано речи {speech:.1f} из "
        f"{need:.0f} с {enough}")


def _busy_main(text):
    if _win is None:
        return
    _ui["primary"].setEnabled_(False)
    _ui["level"].setDoubleValue_(0.0)
    _ui["stat"].setStringValue_(text)


def _finish_main(ok, text, again_title):
    if _win is None:
        return
    _ui["stat"].setStringValue_(("✅ " if ok else "❌ ") + text)
    _ui["primary"].setTitle_(again_title)
    _ui["primary"].setEnabled_(True)
    _ui["secondary"].setTitle_("Закрыть")
    _ui["prog"].setDoubleValue_(1.0 if ok else 0.0)
    _fit()


# --- публичное API (звать откуда угодно) -------------------------------------
def show(mode, mic, seconds, need_sec, on_primary, on_secondary):
    """Открыть окно в состоянии «готов начать».

    on_primary — «Начать»/«Остановить»/«Записать заново» (кнопка сама не знает,
    что именно; смысл задаёт вызывающая сторона по своему состоянию).
    """
    _cb["primary"], _cb["secondary"] = on_primary, on_secondary
    AppHelper.callAfter(_show_main, mode, mic, seconds, need_sec)


def recording():
    AppHelper.callAfter(_rec_main)


def update(level: float, left: float, speech: float):
    AppHelper.callAfter(_update_main, level, left, speech)


def busy(text="Считаю отпечаток…"):
    AppHelper.callAfter(_busy_main, text)


def finish(ok: bool, text: str, again_title="Записать заново"):
    AppHelper.callAfter(_finish_main, ok, text, again_title)


def close():
    def _c():
        if _win is not None:
            _win.orderOut_(None)
    AppHelper.callAfter(_c)


def is_visible() -> bool:
    return _win is not None and _win.isVisible()
