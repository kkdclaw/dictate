"""Окно «Состояние» — служба, разрешения, микрофон, модели, хоткей в одном месте.

Аналог панели SuperDictate: каждая строка — подпись, значение и (иногда) кнопка
действия. Содержимое приходит снимком из dictate.py (`snapshot()` -> секции),
обновляется таймером, пока окно открыто. Кнопки вызывают колбэки из словаря
`actions` по ключу. Всё на главном потоке (rumps-колбэк меню / таймер).

Формат снимка:
    [("Заголовок секции", [(label, value, button_title | None, action_key | None), ...]), ...]
"""
import time

import AppKit
from Foundation import NSMakeRect, NSObject

_win = None
_grid = None
_shape = None
_values = []      # NSTextField значений в порядке строк
_buttons = []     # NSButton в порядке строк (или None)
_actions = []     # action_key по индексу кнопки (tag)
_handlers = {}
_footer = None
_status = None
LABEL_W, VALUE_W, BUTTON_W = 190, 470, 190


class _Handler(NSObject):
    def press_(self, sender):
        key = _actions[sender.tag()] if 0 <= sender.tag() < len(_actions) else None
        fn = _handlers.get(key)
        if fn:
            try:
                fn()
            except Exception as e:  # noqa
                print(f"  действие «{key}» не выполнилось: {e}", flush=True)
            refresh()


_handler = _Handler.alloc().init()


def _label(text, bold=False, size=13.0, color=None):
    f = AppKit.NSTextField.labelWithString_(text)
    f.setFont_(AppKit.NSFont.boldSystemFontOfSize_(size) if bold
               else AppKit.NSFont.systemFontOfSize_(size))
    if color is not None:
        f.setTextColor_(color)
    return f


def _value(text):
    f = AppKit.NSTextField.wrappingLabelWithString_(text)
    f.setFont_(AppKit.NSFont.systemFontOfSize_(13.0))
    f.setPreferredMaxLayoutWidth_(VALUE_W - 8)
    f.setSelectable_(True)
    return f


def _build(sections):
    global _grid, _values, _buttons, _actions
    rows = []
    _values, _buttons, _actions = [], [], []
    for title, items in sections:
        head = _label(title, bold=True, size=14.0)
        rows.append([head, AppKit.NSGridCell.emptyContentView(),
                     AppKit.NSGridCell.emptyContentView()])
        for label, value, btn_title, action in items:
            lab = _label(label, color=AppKit.NSColor.secondaryLabelColor())
            val = _value(value)
            _values.append(val)
            if btn_title:
                b = AppKit.NSButton.buttonWithTitle_target_action_(btn_title, _handler, "press:")
                b.setTag_(len(_actions))
                b.setControlSize_(AppKit.NSControlSizeSmall)
                b.setFont_(AppKit.NSFont.systemFontOfSize_(11.5))
                _actions.append(action)
                _buttons.append(b)
                cell = b
            else:
                _buttons.append(None)
                cell = AppKit.NSGridCell.emptyContentView()
            rows.append([lab, val, cell])
    grid = AppKit.NSGridView.gridViewWithViews_(rows)
    grid.setRowSpacing_(6.0)
    grid.setColumnSpacing_(12.0)
    grid.setXPlacement_(AppKit.NSGridCellPlacementLeading)
    grid.setYPlacement_(AppKit.NSGridCellPlacementTop)
    grid.setRowAlignment_(AppKit.NSGridRowAlignmentFirstBaseline)
    grid.columnAtIndex_(0).setWidth_(LABEL_W)
    grid.columnAtIndex_(1).setWidth_(VALUE_W)
    grid.columnAtIndex_(2).setWidth_(BUTTON_W)
    grid.columnAtIndex_(2).setXPlacement_(AppKit.NSGridCellPlacementTrailing)
    # отступ перед каждым заголовком секции (кроме первого)
    r = 0
    for i, (title, items) in enumerate(sections):
        if i:
            grid.rowAtIndex_(r).setTopPadding_(14.0)
        r += 1 + len(items)
    grid.setTranslatesAutoresizingMaskIntoConstraints_(False)
    return grid


def _apply(sections):
    """Обновить значения без пересборки (форма — набор строк/кнопок — та же)."""
    i = 0
    for _, items in sections:
        for label, value, btn_title, action in items:
            if _values[i].stringValue() != value:
                _values[i].setStringValue_(value)
            i += 1


def _shape_of(sections):
    return tuple((t, tuple((l, b) for l, _, b, _ in items)) for t, items in sections)


def _layout(sections):
    global _grid, _shape, _footer
    content = _win.contentView()
    for sub in list(content.subviews()):
        sub.removeFromSuperview()
    _grid = _build(sections)
    _shape = _shape_of(sections)
    content.addSubview_(_grid)
    _footer = _label("", size=11.0, color=AppKit.NSColor.tertiaryLabelColor())
    _footer.setTranslatesAutoresizingMaskIntoConstraints_(False)
    content.addSubview_(_footer)
    m = 18.0
    AppKit.NSLayoutConstraint.activateConstraints_([
        _grid.topAnchor().constraintEqualToAnchor_constant_(content.topAnchor(), m),
        _grid.leadingAnchor().constraintEqualToAnchor_constant_(content.leadingAnchor(), m),
        _footer.topAnchor().constraintEqualToAnchor_constant_(_grid.bottomAnchor(), 12.0),
        _footer.leadingAnchor().constraintEqualToAnchor_constant_(content.leadingAnchor(), m),
        _footer.bottomAnchor().constraintEqualToAnchor_constant_(content.bottomAnchor(), -m),
        _grid.trailingAnchor().constraintEqualToAnchor_constant_(content.trailingAnchor(), -m),
    ])
    _fit()


def _fit():
    """Подогнать окно под содержимое, оставив верхний край на месте.

    Нужен не только при пересборке: строка «Модели загружаются…» или длинное
    имя микрофона переносится на две строки, и без пересчёта нижняя часть
    обрезается."""
    if _win is None:
        return
    content = _win.contentView()
    content.layoutSubtreeIfNeeded()
    size = content.fittingSize()
    old = _win.frame()
    new_h = max(size.height, 200)
    new_w = max(size.width, 600)
    if abs(new_h - old.size.height) < 1 and abs(new_w - old.size.width) < 1:
        return
    _win.setFrame_display_(NSMakeRect(old.origin.x, old.origin.y + old.size.height - new_h,
                                      new_w, new_h), True)


_snapshot = None


def show(snapshot, actions: dict, title="Dictate — состояние и разрешения",
         activate=True):
    """Открыть (или поднять) окно. snapshot() -> секции; actions: ключ -> callable."""
    global _win, _snapshot
    _snapshot = snapshot
    _handlers.clear()
    _handlers.update(actions)
    if _win is None:
        style = (AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable
                 | AppKit.NSWindowStyleMaskMiniaturizable)
        _win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 900, 600), style, AppKit.NSBackingStoreBuffered, False)
        _win.setTitle_(title)
        _win.setReleasedWhenClosed_(False)
        # поверх других окон: окно ведёт пользователя в Настройки, и прятать
        # его при уходе фокуса нельзя (мы — меню-бар без Dock, теряем фокус
        # при каждом клике мимо). Закрывается крестиком.
        _win.setLevel_(AppKit.NSFloatingWindowLevel)
        _win.center()
    refresh(force=True)
    if activate:
        _win.makeKeyAndOrderFront_(None)
        AppKit.NSApp.activateIgnoringOtherApps_(True)
    else:  # автопоказ при старте: не выдёргиваем пользователя из его окна
        _win.orderFrontRegardless()


def is_visible() -> bool:
    return _win is not None and _win.isVisible()


def refresh(force=False):
    if _win is None or _snapshot is None or (not force and not _win.isVisible()):
        return
    try:
        sections = _snapshot()
    except Exception as e:  # noqa
        print(f"  снимок состояния не собрался: {e}", flush=True)
        return
    if force or _grid is None or _shape_of(sections) != _shape:
        _layout(sections)
        _apply(sections)
    else:
        _apply(sections)
        _fit()  # текст мог стать выше/ниже — окно не должно резать содержимое
    if _footer is not None:
        _footer.setStringValue_("Обновлено " + time.strftime("%H:%M:%S")
                                + " · окно можно закрыть, служба продолжит работать")


def close():
    if _win is not None:
        _win.orderOut_(None)
