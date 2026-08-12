# -*- coding: utf-8 -*-
"""/api/predict 类别过滤的最小回归测试（不依赖真实模型/GPU）。"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

import annotate_tool as at


class _ListVal:
    def __init__(self, v):
        self.v = v

    def tolist(self):
        return self.v


class _Tensor:
    """模拟 YOLO 张量: t[0] 返回带 .tolist() 的对象。"""
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, i):
        return _ListVal(self._rows[i])


class _Box:
    def __init__(self, cls):
        self.cls = [cls]
        self.xywhn = _Tensor([[0.25, 0.25, 0.5, 0.5]])


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    names = {0: "belt_off", 1: "belt_on"}

    def predict(self, path, conf=0.25, verbose=False):
        return [_Result([_Box(0), _Box(1)])]


def _set_state(tmp):
    at.STATE["img_dir"] = tmp
    at.STATE["label_dir"] = tmp
    at.STATE["images"] = ["a.jpg"]
    at.STATE["model"] = _FakeModel()
    Image.new("RGB", (100, 80), "white").save(os.path.join(tmp, "a.jpg"))


def _predict(only_cls=None, cls_offset=0):
    payload = {"conf": 0.25, "cls_offset": cls_offset}
    if only_cls is not None:
        payload["only_cls"] = only_cls
    return at.app.test_client().post("/api/predict/0",
                                     json=payload).get_json()


def test_no_filter_keeps_all():
    with tempfile.TemporaryDirectory() as tmp:
        _set_state(tmp)
        d = _predict(cls_offset=1)
        assert d["ok"]
        assert len(d["boxes"]) == 2
        assert {b["cls"] for b in d["boxes"]} == {1, 2}


def test_filter_one_class():
    with tempfile.TemporaryDirectory() as tmp:
        _set_state(tmp)
        d = _predict(only_cls=[0], cls_offset=1)
        assert d["ok"]
        assert len(d["boxes"]) == 1
        assert d["boxes"][0]["cls"] == 1  # 模型类别0 + 偏移1


def test_filter_other_class():
    with tempfile.TemporaryDirectory() as tmp:
        _set_state(tmp)
        d = _predict(only_cls=[1])
        assert d["ok"]
        assert len(d["boxes"]) == 1
        assert d["boxes"][0]["cls"] == 1


def test_filter_empty_keeps_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        _set_state(tmp)
        d = _predict(only_cls=[])
        assert d["ok"]
        assert d["boxes"] == []


def test_filter_rejects_bad_values():
    with tempfile.TemporaryDirectory() as tmp:
        _set_state(tmp)
        # 非法内容退化为不过滤，与旧行为一致
        d = _predict(only_cls=["abc"])
        assert d["ok"]
        assert len(d["boxes"]) == 2


if __name__ == "__main__":
    test_no_filter_keeps_all()
    test_filter_one_class()
    test_filter_other_class()
    test_filter_empty_keeps_nothing()
    test_filter_rejects_bad_values()
    print("全部测试通过")
