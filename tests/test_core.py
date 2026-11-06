# -*- coding: utf-8 -*-
"""本地标注、模型加载与设备设置的离线回归测试。"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

import annotate_tool as at

CLASSES = ["person", "helmet", "no-helmet", "vest", "no-vest"]


def _set_state(tmp, files=("a.jpg", "b.jpg"), classes=CLASSES):
    at.STATE["img_dir"] = tmp
    at.STATE["label_dir"] = tmp
    at.STATE["images"] = list(files)
    at.STATE["classes"] = list(classes)
    at.STATE["models"] = []
    at.STATE["auto"] = None
    for name in files:
        Image.new("RGB", (100, 80), "white").save(
            os.path.join(tmp, name))


def test_label_mapping_prefers_longest_match():
    assert at._map_label_name("no-helmet", CLASSES) == 2
    assert at._map_label_name("no_helmet", CLASSES) == 2
    assert at._map_label_name("未戴安全帽 no-helmet", CLASSES) == 2
    assert at._map_label_name("helmet", CLASSES) == 1
    assert at._map_label_name("cat", CLASSES) is None
    assert at._map_label_name(None, CLASSES) is None
    assert at._map_label_name(99, CLASSES) is None


def test_files_endpoint_exposes_session():
    with tempfile.TemporaryDirectory() as tmp:
        _set_state(tmp)
        data = at.app.test_client().get("/api/files").get_json()
        assert data["ok"]
        assert data["img_dir"] == tmp
        assert data["classes"] == CLASSES
        assert [f["name"] for f in data["files"]] == ["a.jpg", "b.jpg"]


def test_config_page_has_no_vlm_and_uses_section_layout():
    html = at.app.test_client().get("/").get_data(as_text=True)
    assert 'class="modal config-modal"' in html
    assert "modelCards" in html
    assert "rknnModelSearch" in html
    for removed in ("btnVlmAuto", "btnVlmReview", "vlmMask", "revMask"):
        assert removed not in html


def test_v5_letterbox_geometry():
    img = Image.new("RGB", (800, 448), "white")
    canvas, ratio, px, py = at._v5_letterbox(img, size=640)
    assert canvas.size == (640, 640)
    assert abs(ratio - 0.8) < 1e-6
    assert px == 0
    assert py == (640 - int(round(448 * 0.8))) // 2


def test_v5_shim_requires_torch_gracefully():
    if at.nn is None:
        for fn in (at._install_yolov5_shim,
                   lambda: at._load_yolov5("whatever.pt")):
            try:
                fn()
                raise AssertionError("应该抛错")
            except RuntimeError as exc:
                assert "PyTorch" in str(exc)
    else:
        assert callable(at._install_yolov5_shim)


def test_load_local_model_prefers_yolov5():
    orig_v5, orig_v6 = at._load_yolov5, at._load_meituan_v6n
    try:
        at._load_yolov5 = lambda path: ("m", ["a", "b"])
        at._load_meituan_v6n = lambda path: (_ for _ in ()).throw(
            AssertionError("不该走到 v6n"))
        _, classes, backend = at._load_local_model("x.pt")
        assert backend == "yolov5" and classes == ["a", "b"]
    finally:
        at._load_yolov5, at._load_meituan_v6n = orig_v5, orig_v6


def test_load_local_model_reports_both_attempts():
    orig_v5, orig_v6 = at._load_yolov5, at._load_meituan_v6n
    try:
        at._load_yolov5 = lambda path: (_ for _ in ()).throw(
            RuntimeError("No module named 'models'"))
        at._load_meituan_v6n = lambda path: (_ for _ in ()).throw(
            RuntimeError("No module named 'yolov6'"))
        try:
            at._load_local_model("x.pt")
            raise AssertionError("应该抛错")
        except RuntimeError as exc:
            assert "YOLOv5" in str(exc)
            assert "v6n" in str(exc)
    finally:
        at._load_yolov5, at._load_meituan_v6n = orig_v5, orig_v6


def test_device_setting_and_endpoint():
    import shutil
    cfg = at.CONFIG_FILE
    backup = cfg + ".dev-bak"
    if os.path.isfile(cfg):
        shutil.copyfile(cfg, backup)
    try:
        client = at.app.test_client()
        data = client.get("/api/device").get_json()
        assert data["ok"] and data["want"] in ("auto", "cuda", "cpu")
        client.post("/api/config", json={"device": "cpu"})
        assert at.load_config().get("device") == "cpu"
        client.post("/api/config", json={"device": "GPU9999"})
        assert at.load_config().get("device") == "auto"
    finally:
        if os.path.isfile(backup):
            shutil.copyfile(backup, cfg)
            os.remove(backup)


def test_model_device_str_fallbacks():
    assert at._model_device_str({"model": object()}) == "?"
    assert isinstance(at._model_device_str({"model": None}), str)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print("ok  %s" % test.__name__)
    print("全部测试通过 (%d 项)" % len(tests))
