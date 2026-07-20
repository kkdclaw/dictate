"""Нативное окно с WKWebView для локального HTML — без браузера.

Окна создаются на главном потоке (вызывается из rumps-колбэков меню, которые
и так на главном потоке). Ссылки на окна держим, иначе Cocoa их соберёт.
Повторный показов той же вкладки переиспользует окно и обновляет содержимое.
"""
import os

import AppKit
import WebKit
from Foundation import NSURL

_windows = {}  # title -> (NSWindow, WKWebView)


def show(title: str, html_path: str, width: int = 1180, height: int = 820):
    existing = _windows.get(title)
    url = NSURL.fileURLWithPath_(os.path.abspath(html_path))
    req = AppKit.NSURLRequest.requestWithURL_(url)
    if existing is not None:
        win, web = existing
        web.loadRequest_(req)
        win.makeKeyAndOrderFront_(None)
        AppKit.NSApp.activateIgnoringOtherApps_(True)
        return

    rect = AppKit.NSMakeRect(0, 0, width, height)
    style = (AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable
             | AppKit.NSWindowStyleMaskResizable | AppKit.NSWindowStyleMaskMiniaturizable)
    win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        rect, style, AppKit.NSBackingStoreBuffered, False)
    win.setTitle_(title)
    win.setReleasedWhenClosed_(False)  # переиспользуем — не освобождать при закрытии
    win.center()

    web = WebKit.WKWebView.alloc().initWithFrame_(rect)
    web.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
    web.loadRequest_(req)
    win.contentView().addSubview_(web)

    win.makeKeyAndOrderFront_(None)
    AppKit.NSApp.activateIgnoringOtherApps_(True)
    _windows[title] = (win, web)
