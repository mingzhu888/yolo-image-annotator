# -*- coding: utf-8 -*-
"""/api/predict 多模型合并、类别过滤与 NMS 的最小回归测试(不依赖真实模型/GPU)。"""
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
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, i):
        return _ListVal(self._rows[i])


class _Box:
    def __init__(self, cls, conf=0.9, x=0.25, y=0.25, w=0.5, h=0.5):
        self.cls = [cls]
        self.conf = [conf]
        self.xywhn = _Tensor([[x, y, w, h]])


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    names = {0: "cls0", 1: "cls1"}

    def __init__(self, boxes):
        self._boxes = boxes

    def predict(self, path, conf=0.25, verbose=False):
        return [_Result([_Box(c, f) for c, f in self._boxes])]


class _BoxModel:
    names = {0: "cls0", 1: "cls1"}

    def __init__(self, boxes):
        # (cls, conf, x, y, w, h)
        self._boxes = boxes

    def predict(self, path, conf=0.25, verbose=False):
        return [_Result([_Box(c, f, x, y, w, h)
                         for c, f, x, y, w, h in self._boxes])]


def _entry(mid, boxes, only_cls=None, offset=0, conf=0.25):
    return {"id": mid, "name": f"m{mid}", "path": f"m{mid}.pt",
            "model": _FakeModel(boxes),
            "model_classes": ["cls0", "cls1"],
            "cls_offset": offset, "conf": conf, "only_cls": only_cls}


def _set_state(tmp, models):
    at.STATE["img_dir"] = tmp
    at.STATE["label_dir"] = tmp
    at.STATE["images"] = ["a.jpg"]
    at.STATE["classes"] = ["belt_off", "belt_on"]
    at.STATE["models"] = models
    at.STATE["auto"] = None
    Image.new("RGB", (100, 80), "white").save(os.path.join(tmp, "a.jpg"))


def _predict(iou=0.5):
    return at.app.test_client().post(
        "/api/predict/0", json={"iou": iou}).get_json()


def test_no_filter_keeps_all():
    with tempfile.TemporaryDirectory() as tmp:
        _set_state(tmp, [_entry(1, [(0, 0.9)], offset=1)])
        d = _predict()
        assert d["ok"]
        assert len(d["boxes"]) == 1
        assert d["boxes"][0]["cls"] == 1  # 模型类别0 + 偏移1


def test_filter_one_class():
    with tempfile.TemporaryDirectory() as tmp:
        _set_state(tmp, [_entry(1, [(0, 0.9), (1, 0.8)],
                                only_cls={0}, offset=1)])
        d = _predict()
        assert d["ok"]
        assert len(d["boxes"]) == 1
        assert d["boxes"][0]["cls"] == 1


def test_filter_empty_keeps_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        _set_state(tmp, [_entry(1, [(0, 0.9)], only_cls=set())])
        d = _predict()
        assert d["ok"]
        assert d["boxes"] == []


def test_nms_merges_duplicates_across_models():
    with tempfile.TemporaryDirectory() as tmp:
        # 两个模型都检出同一个目标(同类别、同位置),只保留置信度高的
        _set_state(tmp, [_entry(1, [(0, 0.9)]),
                         _entry(2, [(0, 0.6)])])
        d = _predict()
        assert d["ok"]
        assert len(d["boxes"]) == 1
        kept = {s["name"]: s["kept"] for s in d["stats"]}
        assert kept == {"m1": 1, "m2": 0}


def test_nms_keeps_different_classes():
    with tempfile.TemporaryDirectory() as tmp:
        # 同一个目标上的不同类别(如安全帽 + 反光衣)必须同时保留
        _set_state(tmp, [_entry(1, [(0, 0.9)]),
                         _entry(2, [(1, 0.9)])])
        d = _predict()
        assert d["ok"]
        assert len(d["boxes"]) == 2
        assert {b["cls"] for b in d["boxes"]} == {0, 1}


def test_nms_merges_contained_boxes():
    with tempfile.TemporaryDirectory() as tmp:
        # 大框完全套着小框(同类别,IoU<0.5),应视为同一目标只保留置信度高的
        m1 = _entry(1, [])
        m1["model"] = _BoxModel([(0, 0.9, 0.25, 0.25, 0.5, 0.5)])
        m2 = _entry(2, [])
        m2["model"] = _BoxModel([(0, 0.8, 0.25, 0.25, 0.3, 0.3)])
        _set_state(tmp, [m1, m2])
        d = _predict()
        assert d["ok"]
        assert len(d["boxes"]) == 1
        assert abs(d["boxes"][0]["w"] - 0.5) < 1e-6  # 保留置信度 0.9 的大框


def test_duplicate_path_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "model.pt")
        open(p, "w").close()
        at.STATE["img_dir"] = tmp
        at.STATE["label_dir"] = tmp
        at.STATE["images"] = ["a.jpg"]
        at.STATE["classes"] = ["a", "b"]
        at.STATE["models"] = [{"id": 1, "name": "m", "path": p}]
        at.STATE["auto"] = None
        r = at.app.test_client().post(
            "/api/load_model", json={"model_path": p}).get_json()
        assert not r["ok"]
        assert "已加载" in r["msg"]


if __name__ == "__main__":
    test_no_filter_keeps_all()
    test_filter_one_class()
    test_filter_empty_keeps_nothing()
    test_nms_merges_duplicates_across_models()
    test_nms_keeps_different_classes()
    test_nms_merges_contained_boxes()
    test_duplicate_path_rejected()
    print("全部测试通过")
