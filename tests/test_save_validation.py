# -*- coding: utf-8 -*-
"""/api/save 标签校验的最小回归测试。"""
import json
import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

import annotate_tool as at


def _set_state(tmp):
    at.STATE["img_dir"] = tmp
    at.STATE["label_dir"] = tmp
    at.STATE["images"] = ["a.jpg"]
    at.STATE["classes"] = ["belt_off", "belt_on"]
    Image.new("RGB", (100, 80), "white").save(os.path.join(tmp, "a.jpg"))


def _save(tmp, boxes):
    return at.app.test_client().post(
        "/api/save/0", data=json.dumps({"boxes": boxes}),
        content_type="application/json").get_json()


def _label(tmp):
    p = os.path.join(tmp, "a.txt")
    return open(p, encoding="utf-8").read() if os.path.isfile(p) else None


def test_save_valid_box():
    with tempfile.TemporaryDirectory() as tmp:
        _set_state(tmp)
        d = _save(tmp, [{"cls": 1, "cx": 0.5, "cy": 0.5,
                         "w": 0.1, "h": 0.2}])
        assert d["ok"]
        assert d["count"] == 1
        assert _label(tmp) == "1 0.500000 0.500000 0.100000 0.200000\n"


def test_save_empty_boxes():
    with tempfile.TemporaryDirectory() as tmp:
        _set_state(tmp)
        d = _save(tmp, [])
        assert d["ok"] and d["count"] == 0 and d["done"] is False
        assert _label(tmp) == ""


def test_save_rejects_class_out_of_range():
    with tempfile.TemporaryDirectory() as tmp:
        _set_state(tmp)
        d = _save(tmp, [{"cls": 2, "cx": 0.5, "cy": 0.5,
                         "w": 0.1, "h": 0.2}])
        assert not d["ok"]
        assert "超出范围" in d["msg"]
        assert _label(tmp) is None  # 坏数据不落盘


def test_save_rejects_coord_out_of_range():
    with tempfile.TemporaryDirectory() as tmp:
        _set_state(tmp)
        d = _save(tmp, [{"cls": 0, "cx": 1.5, "cy": 0.5,
                         "w": 0.1, "h": 0.2}])
        assert not d["ok"]
        assert "cx=1.5 越界" in d["msg"]


def test_save_rejects_nan():
    with tempfile.TemporaryDirectory() as tmp:
        _set_state(tmp)
        d = _save(tmp, [{"cls": 0, "cx": float("nan"), "cy": 0.5,
                         "w": 0.1, "h": 0.2}])
        assert not d["ok"]


def test_save_rejects_bad_field():
    with tempfile.TemporaryDirectory() as tmp:
        _set_state(tmp)
        d = _save(tmp, [{"cls": "x", "cx": 0.5, "cy": 0.5,
                         "w": 0.1, "h": 0.2}])
        assert not d["ok"]
        assert "字段缺失或不是数字" in d["msg"]


if __name__ == "__main__":
    test_save_valid_box()
    test_save_empty_boxes()
    test_save_rejects_class_out_of_range()
    test_save_rejects_coord_out_of_range()
    test_save_rejects_nan()
    test_save_rejects_bad_field()
    print("全部测试通过")
