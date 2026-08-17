"""Окно захвата своей горячей клавиши: «нажми сочетание» + живой показ того,
что зажато. Само сочетание ловит слушатель клавиш в dictate.py и зовёт
update()/finish(); окно только рисует и даёт кнопку «Отмена».

Всё рисуется на главном потоке — из потока слушателя вызывать через
AppHelper (функции ниже сами это делают)."""
import threading

import AppKit
from Foundation import NSMakeRect, NSObject
from PyObjCTools import AppHelper

W = 460
_win = None
_ui = {}
_cb = {}


class _HotkeyCaptureHandler(NSObject):
    def cancel_(self, sender):
        fn = _cb.get("cancel")
        if fn:
            fn()


_handler = _HotkeyCaptureHandler.alloc().init()


def _label(text, size=13.0, bold=False, color=None, wrap=False, align=None):
    f = (AppKit.NSTextField.wrappingLabelWithString_(text) if wrap
         else AppKit.NSTextField.labelWithString_(text))
    f.setFont_(AppKit.NSFont.boldSystemFontOfSize_(size) if bold
               else AppKit.NSFont.systemFontOfSize_(size))
    if color is not None:
        f.setTextColor_(color)
    if align is not None:
        f.setAlignment_(align)
    f.setTranslatesAutoresizingMaskIntoConstraints_(False)
    return f


def _build():
    global _win
    style = AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable
    _win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(0, 0, W, 250), style, AppKit.NSBackingStoreBuffered, False)
    _win.setTitle_("Горячая клавиша диктовки")
    _win.setReleasedWhenClosed_(False)
    _win.setLevel_(AppKit.NSFloatingWindowLevel)  # меню-бар без Dock: иначе уедет под окна
    _win.center()
    content = _win.contentView()

    head = _label("Нажми сочетание клавиш", size=17.0, bold=True)
    hint = _label(
        "Подойдёт одиночный модификатор (Option, Command, Control, Shift, Fn/🌐), "
        "клавиша F1–F20 или любая клавиша с модификаторами (⌃ Space, ⌘⇧ D…). "
        "Одиночная буква или пробел не годятся — сработают при обычном наборе.",
        size=12.0, color=AppKit.NSColor.secondaryLabelColor(), wrap=True)
    box = AppKit.NSBox.alloc().initWithFrame_(NSMakeRect(0, 0, 10, 10))
    box.setBoxType_(AppKit.NSBoxCustom)
    box.setCornerRadius_(8.0)
    box.setBorderColor_(AppKit.NSColor.separatorColor())
    box.setFillColor_(AppKit.NSColor.controlBackgroundColor())
    box.setTranslatesAutoresizingMaskIntoConstraints_(False)
    held = _label("…", size=26.0, bold=True, align=AppKit.NSTextAlignmentCenter)
    box.addSubview_(held)
    status = _label("Ожидаю нажатия… Esc или «Отмена» — оставить как было.",
                    size=12.0, color=AppKit.NSColor.secondaryLabelColor(), wrap=True)
    cancel = AppKit.NSButton.buttonWithTitle_target_action_("Отмена", _handler, "cancel:")
    cancel.setKeyEquivalent_("")  # Esc и Return ловит слушатель, кнопке — только мышь
    cancel.setTranslatesAutoresizingMaskIntoConstraints_(False)

    for v in (head, hint, box, status, cancel):
        content.addSubview_(v)
    _ui.update(head=head, hint=hint, box=box, held=held, status=status, cancel=cancel)

    P = 20
    cs = [
        head.topAnchor().constraintEqualToAnchor_constant_(content.topAnchor(), P),
        head.leadingAnchor().constraintEqualToAnchor_constant_(content.leadingAnchor(), P),
        hint.topAnchor().constraintEqualToAnchor_constant_(head.bottomAnchor(), 6),
        hint.leadingAnchor().constraintEqualToAnchor_constant_(content.leadingAnchor(), P),
        hint.trailingAnchor().constraintEqualToAnchor_constant_(content.trailingAnchor(), -P),
        box.topAnchor().constraintEqualToAnchor_constant_(hint.bottomAnchor(), 14),
        box.leadingAnchor().constraintEqualToAnchor_constant_(content.leadingAnchor(), P),
        box.trailingAnchor().constraintEqualToAnchor_constant_(content.trailingAnchor(), -P),
        box.heightAnchor().constraintEqualToConstant_(64),
        held.centerXAnchor().constraintEqualToAnchor_(box.centerXAnchor()),
        held.centerYAnchor().constraintEqualToAnchor_(box.centerYAnchor()),
        held.leadingAnchor().constraintGreaterThanOrEqualToAnchor_constant_(box.leadingAnchor(), 8),
        status.topAnchor().constraintEqualToAnchor_constant_(box.bottomAnchor(), 12),
        status.leadingAnchor().constraintEqualToAnchor_constant_(content.leadingAnchor(), P),
        status.trailingAnchor().constraintEqualToAnchor_constant_(content.trailingAnchor(), -P),
        cancel.topAnchor().constraintEqualToAnchor_constant_(status.bottomAnchor(), 14),
        cancel.trailingAnchor().constraintEqualToAnchor_constant_(content.trailingAnchor(), -P),
        cancel.bottomAnchor().constraintEqualToAnchor_constant_(content.bottomAnchor(), -P),
    ]
    AppKit.NSLayoutConstraint.activateConstraints_(cs)


def _show_main(current_label):
    if _win is None:
        _build()
    _ui["held"].setStringValue_("…")
    _ui["status"].setStringValue_(
        f"Сейчас: {current_label}. Ожидаю нажатия… Esc или «Отмена» — оставить как было.")
    _ui["status"].setTextColor_(AppKit.NSColor.secondaryLabelColor())
    _ui["cancel"].setEnabled_(True)
    AppKit.NSApp.activateIgnoringOtherApps_(True)
    _win.makeKeyAndOrderFront_(None)


def _update_main(text, note=None):
    if _win is None:
        return
    _ui["held"].setStringValue_(text)
    if note is not None:
        _ui["status"].setStringValue_(note)
        _ui["status"].setTextColor_(AppKit.NSColor.systemOrangeColor())


def _finish_main(text, note):
    if _win is None:
        return
    _ui["held"].setStringValue_(text)
    _ui["status"].setStringValue_(note)
    _ui["status"].setTextColor_(AppKit.NSColor.systemGreenColor())
    _ui["cancel"].setEnabled_(False)


def _close_main():
    if _win is not None:
        _win.orderOut_(None)


def show(current_label: str, on_cancel):
    _cb["cancel"] = on_cancel
    AppHelper.callAfter(_show_main, current_label)


def update(text: str, note=None):
    """Показать, что зажато сейчас; note — предупреждение (например, «голая буква не годится»)."""
    AppHelper.callAfter(_update_main, text, note)


def finish(text: str, note: str, close_after: float = 1.2):
    """Сочетание принято: показать зелёный итог и через паузу закрыть окно."""
    AppHelper.callAfter(_finish_main, text, note)
    if close_after:
        threading.Timer(close_after, lambda: AppHelper.callAfter(_close_main)).start()


def close():
    AppHelper.callAfter(_close_main)


def is_visible() -> bool:
    return _win is not None and _win.isVisible()
