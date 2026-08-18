"""Окно постобработки: какой вариант надиктованной фразы вставить.

Зачем: LLM-чистка иногда меняет смысл — Whisper ослышался, модель «исправила»
по-своему, и в поле прилетает один-единственный вариант без права на апелляцию.
Окно показывает сырой текст, обычный результат и две стилевые переделки; в поле
уходит то, что выбрал человек.

Панель НЕактивирующая (NSWindowStyleMaskNonactivatingPanel + floating): фокус
остаётся в приложении, куда вставляем, поэтому ⌘V после клика попадает ровно
туда, куда попал бы и без окна. Плата за это — у окна нет клавиатурных
сокращений: нажатия ушли бы в чужое поле ввода, поэтому выбор только мышью.

Варианты приходят по мере готовности: сырой и обычный есть сразу, стили
считает LLM по секунде-другой каждый — их строки сначала «считаю…», потом
дозаполняются через update().

Всё рисуется на главном потоке; из ML-потока звать show()/update()/close() —
они сами перекидывают вызов через AppHelper.
"""
import AppKit
from Foundation import NSMakeRect, NSObject
from PyObjCTools import AppHelper

W = 620.0
ROW_W = W - 40.0
MARGIN = 20.0
PENDING = "считаю…"

_win = None
_stack = None
# «видно ли окно» держим обычным флагом, а не спрашиваем у NSPanel: is_visible()
# зовут из ML-потока и из потока горячей клавиши, а AppKit не потокобезопасен
_state = {"variants": [], "on_pick": None, "on_cancel": None,
          "buttons": [], "closing": False, "visible": False}


class _ReviewHandler(NSObject):
    def pick_(self, sender):
        idx = int(sender.tag())
        variants = _state["variants"]
        if not 0 <= idx < len(variants):
            return
        v = variants[idx]
        if v.get("text") is None:
            return  # строка ещё считается — клик по ней ничего не значит
        cb = _state["on_pick"]
        _close_main(cancelled=False)
        if cb:
            cb(v["key"], v["text"])

    def cancel_(self, sender):
        _close_main(cancelled=True)

    def windowWillClose_(self, note):
        # крестик в заголовке — тоже отказ от вставки
        _state["visible"] = False
        if _state["closing"]:
            return  # окно убираем мы сами, отказ уже обработан
        cb = _state["on_cancel"]
        _state["on_pick"] = _state["on_cancel"] = None
        if cb:
            cb()


_handler = _ReviewHandler.alloc().init()


def _attributed(title: str, text: str, pending: bool):
    """Заголовок варианта мелким серым + сам текст. Одной attributed-строкой:
    NSButton умеет многострочный заголовок, а отдельные NSTextField поверх
    кнопки перехватывали бы клик."""
    para = AppKit.NSMutableParagraphStyle.alloc().init()
    para.setLineBreakMode_(AppKit.NSLineBreakByWordWrapping)
    para.setAlignment_(AppKit.NSTextAlignmentLeft)
    para.setParagraphSpacing_(3.0)
    out = AppKit.NSMutableAttributedString.alloc().init()
    head = AppKit.NSAttributedString.alloc().initWithString_attributes_(
        title + "\n", {
            AppKit.NSFontAttributeName: AppKit.NSFont.boldSystemFontOfSize_(11.0),
            AppKit.NSForegroundColorAttributeName: AppKit.NSColor.secondaryLabelColor(),
            AppKit.NSParagraphStyleAttributeName: para})
    body = AppKit.NSAttributedString.alloc().initWithString_attributes_(
        text, {
            AppKit.NSFontAttributeName: AppKit.NSFont.systemFontOfSize_(13.0),
            AppKit.NSForegroundColorAttributeName: (
                AppKit.NSColor.tertiaryLabelColor() if pending
                else AppKit.NSColor.labelColor()),
            AppKit.NSParagraphStyleAttributeName: para})
    out.appendAttributedString_(head)
    out.appendAttributedString_(body)
    return out


def _row(index: int, title: str, text):
    b = AppKit.NSButton.alloc().initWithFrame_(NSMakeRect(0, 0, ROW_W, 44))
    b.setBezelStyle_(AppKit.NSBezelStyleRegularSquare)
    b.setButtonType_(AppKit.NSButtonTypeMomentaryPushIn)
    b.setTarget_(_handler)
    b.setAction_("pick:")
    b.setTag_(index)
    b.cell().setWraps_(True)
    b.setTranslatesAutoresizingMaskIntoConstraints_(False)
    b.widthAnchor().constraintEqualToConstant_(ROW_W).setActive_(True)
    _fill(b, title, text)
    return b


def _fill(button, title: str, text):
    pending = text is None
    button.setAttributedTitle_(_attributed(title, PENDING if pending else text, pending))
    button.setEnabled_(not pending)  # серую строку нельзя выбрать по ошибке


def _build():
    global _win, _stack
    style = (AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable
             | AppKit.NSWindowStyleMaskUtilityWindow
             | AppKit.NSWindowStyleMaskNonactivatingPanel)
    _win = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(0, 0, W, 300), style, AppKit.NSBackingStoreBuffered, False)
    _win.setTitle_("Что вставить?")
    _win.setReleasedWhenClosed_(False)
    _win.setFloatingPanel_(True)
    _win.setLevel_(AppKit.NSFloatingWindowLevel)
    _win.setHidesOnDeactivate_(False)      # мы меню-бар без Dock: окно не должно прятаться
    _win.setBecomesKeyOnlyIfNeeded_(True)  # не отбирать фокус у поля ввода
    _win.setDelegate_(_handler)
    content = _win.contentView()

    _stack = AppKit.NSStackView.alloc().init()
    _stack.setOrientation_(AppKit.NSUserInterfaceLayoutOrientationVertical)
    _stack.setAlignment_(AppKit.NSLayoutAttributeLeading)
    _stack.setSpacing_(8.0)
    _stack.setTranslatesAutoresizingMaskIntoConstraints_(False)

    hint = AppKit.NSTextField.wrappingLabelWithString_(
        "Кликни вариант — он вставится в то же поле. Окно фокус не забирает, "
        "поэтому выбор мышью.")
    hint.setFont_(AppKit.NSFont.systemFontOfSize_(11.5))
    hint.setTextColor_(AppKit.NSColor.secondaryLabelColor())
    hint.setTranslatesAutoresizingMaskIntoConstraints_(False)

    cancel = AppKit.NSButton.buttonWithTitle_target_action_(
        "Не вставлять", _handler, "cancel:")
    cancel.setTranslatesAutoresizingMaskIntoConstraints_(False)

    for v in (_stack, hint, cancel):
        content.addSubview_(v)
    C = AppKit.NSLayoutConstraint
    C.activateConstraints_([
        _stack.topAnchor().constraintEqualToAnchor_constant_(content.topAnchor(), MARGIN),
        _stack.leadingAnchor().constraintEqualToAnchor_constant_(
            content.leadingAnchor(), MARGIN),
        hint.topAnchor().constraintEqualToAnchor_constant_(_stack.bottomAnchor(), 12),
        hint.leadingAnchor().constraintEqualToAnchor_constant_(
            content.leadingAnchor(), MARGIN),
        hint.widthAnchor().constraintEqualToConstant_(ROW_W - 130),
        cancel.centerYAnchor().constraintEqualToAnchor_(hint.centerYAnchor()),
        cancel.trailingAnchor().constraintEqualToAnchor_constant_(
            content.trailingAnchor(), -MARGIN),
        cancel.bottomAnchor().constraintLessThanOrEqualToAnchor_constant_(
            content.bottomAnchor(), -MARGIN),
        hint.bottomAnchor().constraintLessThanOrEqualToAnchor_constant_(
            content.bottomAnchor(), -MARGIN),
    ])


def _place():
    """Показываем на экране, где сейчас мышь (там же, где и поле ввода),
    ближе к низу — чтобы окно не накрывало то, что человек читает."""
    try:
        pt = AppKit.NSEvent.mouseLocation()
        screen = next((s for s in AppKit.NSScreen.screens()
                       if AppKit.NSPointInRect(pt, s.frame())), AppKit.NSScreen.mainScreen())
        vis, f = screen.visibleFrame(), _win.frame()
        x = vis.origin.x + (vis.size.width - f.size.width) / 2
        y = vis.origin.y + vis.size.height * 0.25
        _win.setFrameOrigin_(AppKit.NSMakePoint(x, y))
    except Exception:
        _win.center()


def _fit():
    content = _win.contentView()
    content.layoutSubtreeIfNeeded()
    h = max(content.fittingSize().height, 160.0)
    old = _win.frame()
    if abs(h - old.size.height) > 1:
        _win.setFrame_display_(
            NSMakeRect(old.origin.x, old.origin.y + old.size.height - h, W, h), True)


def _show_main(variants, on_pick, on_cancel):
    if _win is None:
        _build()
    _state.update(variants=list(variants), on_pick=on_pick, on_cancel=on_cancel,
                  buttons=[])
    for old in list(_stack.views()):
        _stack.removeView_(old)
    for i, v in enumerate(_state["variants"]):
        b = _row(i, v["title"], v.get("text"))
        _stack.addView_inGravity_(b, AppKit.NSStackViewGravityTop)
        _state["buttons"].append(b)
    _fit()
    _place()
    _state["visible"] = True
    _win.orderFrontRegardless()  # без активации приложения: фокус остаётся в поле


def _update_main(key, text):
    for i, v in enumerate(_state["variants"]):
        if v["key"] == key:
            v["text"] = text
            if i < len(_state["buttons"]):
                _fill(_state["buttons"][i], v["title"], text)
            _fit()
            return


def _close_main(cancelled: bool):
    if _win is None or not _state["visible"]:
        return
    cb = _state["on_cancel"] if cancelled else None
    _state["on_pick"] = _state["on_cancel"] = None
    _state["closing"] = True  # не будить windowWillClose_ вторым отказом
    try:
        _win.orderOut_(None)
    finally:
        _state["closing"] = False
        _state["visible"] = False
    if cb:
        cb()


def show(variants, on_pick, on_cancel=None):
    """variants: [{"key", "title", "text"}]; text=None — «считаю…».
    on_pick(key, text) и on_cancel() зовутся на главном потоке."""
    AppHelper.callAfter(_show_main, variants, on_pick, on_cancel)


def update(key: str, text: str):
    AppHelper.callAfter(_update_main, key, text)


def close():
    AppHelper.callAfter(_close_main, True)


def is_visible() -> bool:
    return bool(_state["visible"])
