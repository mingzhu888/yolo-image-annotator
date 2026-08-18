# -*- coding: utf-8 -*-
"""/api/auto_annotate 自动化标注与模型增删的最小回归测试。"""
import os
import sys
import tempfile
import time
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
    def __init__(self, cls, conf=0.9):
        self.cls = [cls]
        self.conf = [conf]
        self.xywhn = _Tensor([[0.25, 0.25, 0.5, 0.5]])


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    names = {0: "cls0", 1: "cls1"}

    def __init__(self, boxes):
        self._boxes = boxes

    def predict(self, path, conf=0.25, verbose=False):
        return [_Result([_Box(c, f) for c, f in self._boxes])]


def _entry(mid, boxes, only_cls=None, offset=0):
    return {"id": mid, "name": f"m{mid}", "path": f"m{mid}.pt",
            "model": _FakeModel(boxes),
            "model_classes": ["cls0", "cls1"],
            "cls_offset": offset, "conf": 0.25, "only_cls": only_cls}


def _set_state(tmp, models, labeled_a=True):
    at.STATE["img_dir"] = tmp
    at.STATE["label_dir"] = tmp
    at.STATE["images"] = ["a.jpg", "b.jpg"]
    at.STATE["classes"] = ["belt_off", "belt_on"]
    at.STATE["exclusive_groups"] = []
    at.STATE["allow_multi_cls"] = False
    at.STATE["models"] = models
    at.STATE["auto"] = None
    for n in ("a.jpg", "b.jpg"):
        Image.new("RGB", (100, 80), "white").save(os.path.join(tmp, n))
    if labeled_a:
        with open(os.path.join(tmp, "a.txt"), "w", encoding="utf-8") as f:
            f.write("0 0.500000 0.500000 0.100000 0.100000\n")


def _wait_finished(client, timeout=20):
    deadline = time.time() + timeout
    st = {}
    while time.time() < deadline:
        st = client.get("/api/auto_annotate_status").get_json()
        if st.get("finished"):
            return st
        time.sleep(0.1)
    raise AssertionError(f"自动化标注超时未完成: {st}")


def test_auto_annotate_skips_labeled():
    with tempfile.TemporaryDirectory() as tmp:
        _set_state(tmp, [_entry(1, [(0, 0.9)])])
        c = at.app.test_client()
        r = c.post("/api/auto_annotate",
                   json={"skip_labeled": True, "iou": 0.5}).get_json()
        assert r["ok"]
        st = _wait_finished(c)
        assert st["saved"] == 1
        assert st["skipped"] == 1
        assert os.path.isfile(os.path.join(tmp, "b.txt"))
        with open(os.path.join(tmp, "a.txt"), encoding="utf-8") as f:
            assert f.read().startswith("0 0.500000")


def test_auto_annotate_overwrites_when_requested():
    with tempfile.TemporaryDirectory() as tmp:
        _set_state(tmp, [_entry(1, [(0, 0.9)])])
        c = at.app.test_client()
        r = c.post("/api/auto_annotate",
                   json={"skip_labeled": False, "iou": 0.5}).get_json()
        assert r["ok"]
        st = _wait_finished(c)
        assert st["saved"] == 2
        assert st["skipped"] == 0


def test_update_and_unload_model():
    with tempfile.TemporaryDirectory() as tmp:
        _set_state(tmp, [_entry(1, [(0, 0.9)])])
        c = at.app.test_client()
        r = c.post("/api/update_model",
                   json={"id": 1, "conf": 0.1, "only_cls": [0]}).get_json()
        assert r["ok"]
        assert r["models"][0]["conf"] == 0.1
        assert r["models"][0]["only_cls"] == [0]
        r = c.post("/api/unload_model", json={"id": 1}).get_json()
        assert r["ok"]
        assert r["models"] == []


if __name__ == "__main__":
    test_auto_annotate_skips_labeled()
    test_auto_annotate_overwrites_when_requested()
    test_update_and_unload_model()
    print("全部测试通过")
