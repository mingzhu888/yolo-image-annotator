# -*- coding: utf-8 -*-
"""云企智瞰标注工具桌面窗口入口。"""
import ctypes
import logging
import multiprocessing
import threading
import time
import traceback

import webview
from werkzeug.serving import make_server

import annotate_tool

APP_NAME = "云企智瞰标注工具"
_MUTEX = None


def _message(text, title=APP_NAME):
    ctypes.windll.user32.MessageBoxW(None, text, title, 0x10)


def _acquire_single_instance():
    global _MUTEX
    _MUTEX = ctypes.windll.kernel32.CreateMutexW(
        None, False, "Local\\YoloImageAnnotatorDesktop")
    if not _MUTEX:
        return False
    if ctypes.windll.kernel32.GetLastError() == 183:
        ctypes.windll.kernel32.CloseHandle(_MUTEX)
        _MUTEX = None
        _message("标注工具已经在运行。")
        return False
    return True


def main():
    multiprocessing.freeze_support()
    if not _acquire_single_instance():
        return 1

    data_dir = annotate_tool.runtime_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(data_dir / "desktop_error.log"),
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s\n%(message)s",
        encoding="utf-8",
    )

    server = make_server(
        "127.0.0.1", 0, annotate_tool.app, threaded=True)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        window = webview.create_window(
            APP_NAME,
            f"http://127.0.0.1:{port}/",
            width=1440,
            height=900,
            min_size=(1000, 680),
            background_color="#111114",
            text_select=True,
        )

        def focus_window():
            try:
                time.sleep(0.2)
                window.show()
                window.restore()
                window.on_top = True
                time.sleep(0.8)
                window.on_top = False
            except Exception:
                pass

        webview.start(func=focus_window, gui="edgechromium", debug=False)
        return 0
    except Exception:
        detail = traceback.format_exc()
        logging.error(detail)
        _message(
            "无法启动桌面窗口。\n\n"
            "请确认系统已安装 Microsoft Edge WebView2 Runtime。\n"
            "错误详情已写入：" + str(data_dir / "desktop_error.log"))
        return 1
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
