# -*- coding: utf-8 -*-
"""Windows 文件夹选择接口的最小回归测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import annotate_tool as at


def test_pick_folder_passes_initial_dir():
    original = at._pick_path
    seen = {}

    def fake_pick(is_dir, title, initial_dir=""):
        seen.update(is_dir=is_dir, title=title, initial_dir=initial_dir)
        return r"F:\images"

    try:
        at._pick_path = fake_pick
        d = at.app.test_client().post(
            "/api/pick_folder", json={"initial_dir": r"F:\images"}).get_json()
    finally:
        at._pick_path = original

    assert d == {"ok": True, "path": r"F:\images"}
    assert seen == {"is_dir": True, "title": "选择文件夹",
                    "initial_dir": r"F:\images"}


if __name__ == "__main__":
    test_pick_folder_passes_initial_dir()
    print("全部测试通过")
