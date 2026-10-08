"""Окно «Модели» — все модели в одном месте вместо вложенного меню.

До 0.19.0 модели жили в меню-баре тремя уровнями подменю (роль → модель →
действие): чтобы увидеть, что скачано, приходилось лазить по выпадашкам, а
прогресс закачки перестраивал открытое меню под курсором. Теперь это окно, как
«Состояние»: секция на роль, строка на модель — отметка, имя, описание, размер
или прогресс, и кнопки действий прямо в строке.

Содержимое приходит снимком из dictate.py (`snapshot()`), кнопки отдают ключ
действия в `on_action(key)`. Всё на главном потоке (колбэк меню / таймер).

Формат снимка:
    {"sections": [{"title": str, "note": str, "rows": [row, ...]}, ...],
     "footer": str}
    row = {"key": str, "mark": str, "name": str, "desc": str, "status": str,
           "buttons": [(title, action_key[, destructive]), ...]}

Сетка пересобирается только когда меняется форма (набор строк и кнопок);
иначе обновляются тексты — так прогресс закачки не дёргает кнопку под мышью.
"""
import os
import time

import AppKit
from Foundation import NSMakeRange, NSMakeRect, NSObject

_win = None
_scroll = None
_doc = None
_grid = None
_shape = None
_status_fields = {}   # key -> NSTextField статуса (обновляется на месте)
_note_fields = {}     # title -> NSTextField подписи секции
_actions = []         # action_key по тегу кнопки
_on_action = None
_snapshot = None
_footer = None
MARK_W, NAME_W, STATUS_W, BTN_W = 26, 330, 300, 300
MARGIN = 18.0


class _ModelsHandler(NSObject):
    def press_(self, sender):
        key = _actions[sender.tag()] if 0 <= sender.tag() < len(_actions) else None
        self._run(key)

    def cache_(self, sender):  # кнопка нижней полосы живёт вне тегов: сетка пересобирается
        self._run("cache")

    @staticmethod
    def _run(key):
        if key and _on_action:
            try:
                _on_action(key)
            except Exception as e:  # noqa
                print(f"  действие «{key}» не выполнилось: {e}", flush=True)
        refresh()


_handler = _ModelsHandler.alloc().init()


class _ModelsFlippedView(AppKit.NSView):
    """Документ скролла с началом координат сверху — иначе содержимое липнет к низу."""

    def isFlipped(self):
        return True


def _label(text, bold=False, size=13.0, color=None, wrap=False, width=None):
    if wrap:
        f = AppKit.NSTextField.wrappingLabelWithString_(text)
        if width:
            f.setPreferredMaxLayoutWidth_(width - 8)
    else:
        f = AppKit.NSTextField.labelWithString_(text)
    f.setFont_(AppKit.NSFont.boldSystemFontOfSize_(size) if bold
               else AppKit.NSFont.systemFontOfSize_(size))
    if color is not None:
        f.setTextColor_(color)
    f.setSelectable_(True)
    return f


def _button(title, key, destructive=False):
    b = AppKit.NSButton.buttonWithTitle_target_action_(title, _handler, "press:")
    b.setTag_(len(_actions))
    b.setControlSize_(AppKit.NSControlSizeSmall)
    b.setFont_(AppKit.NSFont.systemFontOfSize_(11.5))
    if destructive:
        try:
            b.setHasDestructiveAction_(True)
        except Exception:
            pass
    _actions.append(key)
    return b


def _empty():
    return AppKit.NSGridCell.emptyContentView()


def _build(snap):
    """Собрать сетку: заголовок секции (на всю ширину), подпись, строки моделей."""
    global _actions, _status_fields, _note_fields
    _actions, _status_fields, _note_fields = [], {}, {}
    rows, merges, pads = [], [], []  # merges: индексы строк на всю ширину
    for si, sec in enumerate(snap["sections"]):
        if si:
            pads.append(len(rows))
        head = _label(sec["title"], bold=True, size=14.0)
        merges.append(len(rows))
        rows.append([head, _empty(), _empty(), _empty()])
        if sec.get("note"):
            note = _label(sec["note"], size=12.0, color=AppKit.NSColor.secondaryLabelColor(),
                          wrap=True, width=NAME_W + STATUS_W + BTN_W)
            _note_fields[sec["title"]] = note
            merges.append(len(rows))
            rows.append([note, _empty(), _empty(), _empty()])
        for row in sec["rows"]:
            mark = _label(row["mark"], size=14.0)
            name = _label(row["name"], bold=True, size=13.0, wrap=True, width=NAME_W)
            desc = _label(row["desc"], size=11.5, color=AppKit.NSColor.secondaryLabelColor(),
                          wrap=True, width=NAME_W)
            ident = AppKit.NSStackView.stackViewWithViews_([name, desc])
            ident.setOrientation_(AppKit.NSUserInterfaceLayoutOrientationVertical)
            ident.setAlignment_(AppKit.NSLayoutAttributeLeading)
            ident.setSpacing_(2.0)
            status = _label(row["status"], size=12.5, wrap=True, width=STATUS_W)
            _status_fields[row["key"]] = status
            btns = [_button(*b) for b in row.get("buttons", [])]
            if btns:
                bar = AppKit.NSStackView.stackViewWithViews_(btns)
                bar.setOrientation_(AppKit.NSUserInterfaceLayoutOrientationHorizontal)
                bar.setSpacing_(6.0)
                cell = bar
            else:
                cell = _empty()
            rows.append([mark, ident, status, cell])
    grid = AppKit.NSGridView.gridViewWithViews_(rows)
    grid.setRowSpacing_(8.0)
    grid.setColumnSpacing_(12.0)
    grid.setXPlacement_(AppKit.NSGridCellPlacementLeading)
    grid.setYPlacement_(AppKit.NSGridCellPlacementTop)
    grid.setRowAlignment_(AppKit.NSGridRowAlignmentFirstBaseline)
    grid.columnAtIndex_(0).setWidth_(MARK_W)
    grid.columnAtIndex_(1).setWidth_(NAME_W)
    grid.columnAtIndex_(2).setWidth_(STATUS_W)
    grid.columnAtIndex_(3).setWidth_(BTN_W)
    grid.columnAtIndex_(3).setXPlacement_(AppKit.NSGridCellPlacementTrailing)
    for r in merges:
        grid.mergeCellsInHorizontalRange_verticalRange_(NSMakeRange(0, 4), NSMakeRange(r, 1))
    for r in pads:
        grid.rowAtIndex_(r).setTopPadding_(16.0)
    grid.setTranslatesAutoresizingMaskIntoConstraints_(False)
    return grid


def _shape_of(snap):
    return tuple((s["title"], tuple((r["key"], r["mark"], tuple(b[0] for b in r.get("buttons", [])))
                                    for r in s["rows"]))
                 for s in snap["sections"])


def _apply(snap):
    """Обновить тексты без пересборки: форма та же, меняются статусы и подписи."""
    for sec in snap["sections"]:
        f = _note_fields.get(sec["title"])
        if f is not None and f.stringValue() != sec.get("note", ""):
            f.setStringValue_(sec.get("note", ""))
        for row in sec["rows"]:
            f = _status_fields.get(row["key"])
            if f is not None and f.stringValue() != row["status"]:
                f.setStringValue_(row["status"])
    if _footer is not None:
        _footer.setStringValue_(snap.get("footer", "") + " · обновлено "
                                + time.strftime("%H:%M:%S"))


def _layout(snap):
    global _grid, _shape
    for sub in list(_doc.subviews()):
        sub.removeFromSuperview()
    _grid = _build(snap)
    _shape = _shape_of(snap)
    _doc.addSubview_(_grid)
    AppKit.NSLayoutConstraint.activateConstraints_([
        _grid.topAnchor().constraintEqualToAnchor_constant_(_doc.topAnchor(), MARGIN),
        _grid.leadingAnchor().constraintEqualToAnchor_constant_(_doc.leadingAnchor(), MARGIN),
        _grid.trailingAnchor().constraintLessThanOrEqualToAnchor_constant_(
            _doc.trailingAnchor(), -MARGIN),
        _grid.bottomAnchor().constraintEqualToAnchor_constant_(_doc.bottomAnchor(), -MARGIN),
    ])
    _fit()


def _fit():
    """Подогнать окно под содержимое при пересборке, не выше видимой части экрана.

    Пользователь может растянуть окно сам — при обновлении текстов размер не
    трогаем, только когда меняется набор строк."""
    if _win is None:
        return
    _doc.layoutSubtreeIfNeeded()
    size = _grid.fittingSize()
    want_w = size.width + 2 * MARGIN + 20
    want_h = size.height + 2 * MARGIN + 44  # + нижняя полоса
    screen = _win.screen() or AppKit.NSScreen.mainScreen()
    if screen is not None:
        vis = screen.visibleFrame()
        want_h = min(want_h, vis.size.height - 60)
        want_w = min(want_w, vis.size.width - 60)
    old = _win.frame()
    new_h, new_w = max(want_h, 240), max(want_w, 700)
    if abs(new_h - old.size.height) < 1 and abs(new_w - old.size.width) < 1:
        return
    _win.setFrame_display_(NSMakeRect(old.origin.x, old.origin.y + old.size.height - new_h,
                                      new_w, new_h), True)


def _make_window(title):
    global _win, _scroll, _doc, _footer
    style = (AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable
             | AppKit.NSWindowStyleMaskMiniaturizable | AppKit.NSWindowStyleMaskResizable)
    _win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(0, 0, 1000, 640), style, AppKit.NSBackingStoreBuffered, False)
    _win.setTitle_(title)
    _win.setReleasedWhenClosed_(False)
    # поверх других окон: мы — меню-бар без Dock и теряем фокус при каждом
    # клике мимо; прятать окно при этом нельзя. Закрывается крестиком.
    _win.setLevel_(AppKit.NSFloatingWindowLevel)
    _win.setMinSize_(AppKit.NSMakeSize(700, 240))
    _win.center()
    content = _win.contentView()

    _scroll = AppKit.NSScrollView.alloc().initWithFrame_(content.bounds())
    _scroll.setHasVerticalScroller_(True)
    _scroll.setAutohidesScrollers_(True)
    _scroll.setDrawsBackground_(False)
    _scroll.setBorderType_(AppKit.NSNoBorder)
    _scroll.setTranslatesAutoresizingMaskIntoConstraints_(False)
    _doc = _ModelsFlippedView.alloc().initWithFrame_(content.bounds())
    _doc.setTranslatesAutoresizingMaskIntoConstraints_(False)
    _scroll.setDocumentView_(_doc)
    content.addSubview_(_scroll)

    _footer = _label("", size=11.0, color=AppKit.NSColor.tertiaryLabelColor(), wrap=True,
                     width=700)
    _footer.setTranslatesAutoresizingMaskIntoConstraints_(False)
    cache_btn = AppKit.NSButton.buttonWithTitle_target_action_("Открыть папку кэша", _handler,
                                                               "cache:")
    cache_btn.setControlSize_(AppKit.NSControlSizeSmall)
    cache_btn.setFont_(AppKit.NSFont.systemFontOfSize_(11.5))
    cache_btn.setTranslatesAutoresizingMaskIntoConstraints_(False)
    content.addSubview_(_footer)
    content.addSubview_(cache_btn)

    clip = _scroll.contentView()
    AppKit.NSLayoutConstraint.activateConstraints_([
        _scroll.topAnchor().constraintEqualToAnchor_(content.topAnchor()),
        _scroll.leadingAnchor().constraintEqualToAnchor_(content.leadingAnchor()),
        _scroll.trailingAnchor().constraintEqualToAnchor_(content.trailingAnchor()),
        _scroll.bottomAnchor().constraintEqualToAnchor_constant_(_footer.topAnchor(), -8.0),
        _doc.topAnchor().constraintEqualToAnchor_(clip.topAnchor()),
        _doc.leadingAnchor().constraintEqualToAnchor_(clip.leadingAnchor()),
        _doc.trailingAnchor().constraintEqualToAnchor_(clip.trailingAnchor()),
        _footer.leadingAnchor().constraintEqualToAnchor_constant_(content.leadingAnchor(), MARGIN),
        _footer.bottomAnchor().constraintEqualToAnchor_constant_(content.bottomAnchor(), -12.0),
        _footer.trailingAnchor().constraintLessThanOrEqualToAnchor_constant_(
            cache_btn.leadingAnchor(), -12.0),
        cache_btn.trailingAnchor().constraintEqualToAnchor_constant_(content.trailingAnchor(),
                                                                     -MARGIN),
        cache_btn.centerYAnchor().constraintEqualToAnchor_(_footer.centerYAnchor()),
    ])


def show(snapshot, on_action, title="Dictate — модели"):
    """Открыть (или поднять) окно. snapshot() -> снимок; on_action(key) — кнопки."""
    global _snapshot, _on_action
    _snapshot, _on_action = snapshot, on_action
    if _win is None:
        _make_window(title)
    refresh(force=True)
    _win.makeKeyAndOrderFront_(None)
    AppKit.NSApp.activateIgnoringOtherApps_(True)


def is_visible() -> bool:
    return _win is not None and _win.isVisible()


def refresh(force=False):
    if _win is None or _snapshot is None or (not force and not _win.isVisible()):
        return
    try:
        snap = _snapshot()
    except Exception as e:  # noqa
        print(f"  снимок моделей не собрался: {e}", flush=True)
        return
    if force or _grid is None or _shape_of(snap) != _shape:
        _layout(snap)
    _apply(snap)


def close():
    if _win is not None:
        _win.orderOut_(None)
