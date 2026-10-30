# -*- coding: utf-8 -*-
"""网页内文件夹选择接口的回归测试。"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import annotate_tool as at


def test_browse_folders_lists_children_and_parent():
    with tempfile.TemporaryDirectory() as tmp:
        child = os.path.join(tmp, "images")
        os.makedirs(child)
        d = at.app.test_client().get(
            "/api/browse_folders", query_string={"path": tmp}).get_json()
        assert d["ok"] is True
        assert d["current"] == os.path.abspath(tmp)
        assert d["parent"]
        assert any(x["name"] == "images" and x["path"] == child
                   for x in d["dirs"])


def test_browse_folders_bad_path_falls_back_to_drives():
    d = at.app.test_client().get(
        "/api/browse_folders",
        query_string={"path": r"Z:\definitely-not-a-real-folder"}).get_json()
    assert d["ok"] is True
    assert d["current"] == ""
    assert d["roots"]
    assert d["msg"]


if __name__ == "__main__":
    test_browse_folders_lists_children_and_parent()
    test_browse_folders_bad_path_falls_back_to_drives()
    print("全部测试通过")
