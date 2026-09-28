# -*- coding: utf-8 -*-
"""多模态（VLM）标注的离线回归测试。

不联网：fake 掉 _http_post_json / _infer_vlm，只验证解析、类别映射、
坐标换算、空结果与失败的落盘策略。
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

import annotate_tool as at

CLASSES = ["person", "helmet", "no-helmet", "vest", "no-vest"]


def _vcfg(**over):
    v = dict(at.VLM_DEFAULTS)
    v["api_key"] = "fake-key"
    v["base_url"] = "https://api.example.com/v1"
    v["model"] = "fake-vl"
    v["concurrency"] = 4
    v.update(over)
    return v


def _fake_response(boxes, w=100, h=80, extra=None):
    obj = {"image_width": w, "image_height": h, "boxes": boxes}
    if extra:
        obj.update(extra)
    return 200, {"choices": [{"message": {"content": json.dumps(obj)}}],
                 "usage": {"total_tokens": 1234}}


def _set_state(tmp, files=("a.jpg", "b.jpg"), classes=CLASSES):
    at.STATE["img_dir"] = tmp
    at.STATE["label_dir"] = tmp
    at.STATE["images"] = list(files)
    at.STATE["classes"] = list(classes)
    at.STATE["models"] = []
    at.STATE["auto"] = None
    for n in files:
        Image.new("RGB", (100, 80), "white").save(os.path.join(tmp, n))


def _wait_finished(client, timeout=20):
    deadline = time.time() + timeout
    st = {}
    while time.time() < deadline:
        st = client.get("/api/auto_annotate_status").get_json()
        if st.get("finished"):
            return st
        time.sleep(0.05)
    raise AssertionError("多模态标注超时未完成: %s" % st)


def test_label_mapping_prefers_longest_match():
    """no-helmet 不能被误判成 helmet（helmet 是它的子串）。"""
    assert at._vlm_map_label("no-helmet", CLASSES) == 2
    assert at._vlm_map_label("no_helmet", CLASSES) == 2
    assert at._vlm_map_label("未戴安全帽 no-helmet", CLASSES) == 2
    assert at._vlm_map_label("helmet", CLASSES) == 1
    assert at._vlm_map_label("person", CLASSES) == 0
    assert at._vlm_map_label("cat", CLASSES) is None
    assert at._vlm_map_label(None, CLASSES) is None
    assert at._vlm_map_label(3, CLASSES) == 3
    assert at._vlm_map_label(99, CLASSES) is None


def test_infer_parses_and_clamps():
    """正常返回：类别映射、坐标裁剪、退化框丢弃。"""
    orig_cfg, orig_post = at.vlm_config, at._http_post_json
    try:
        at.vlm_config = lambda: _vcfg()
        at._http_post_json = lambda *a, **k: _fake_response([
            {"label": "person", "x1": 10, "y1": 20, "x2": 50, "y2": 60},
            {"label": "no-helmet", "x1": 90, "y1": 70, "x2": 130, "y2": 100},
            {"label": "cat", "x1": 1, "y1": 1, "x2": 9, "y2": 9},
            {"label": "vest", "x1": 5, "y1": 5, "x2": 5, "y2": 9},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.jpg")
            Image.new("RGB", (100, 80), "white").save(p)
            boxes, meta = at._infer_vlm(p, CLASSES)
        assert meta["error"] is None
        assert meta["raw"] == 4
        assert meta["unmapped"] == 1      # cat
        assert meta["dropped"] == 1       # 宽度为 0
        assert len(boxes) == 2
        assert boxes[0]["cls"] == 0
        assert abs(boxes[0]["cx"] - 0.30) < 1e-6
        assert abs(boxes[0]["w"] - 0.40) < 1e-6
        assert boxes[1]["cls"] == 2
        # 越界的右下角被裁到图内
        assert boxes[1]["cx"] + boxes[1]["w"] / 2.0 <= 1.0 + 1e-9
    finally:
        at.vlm_config, at._http_post_json = orig_cfg, orig_post


def test_infer_rescales_when_model_reports_other_size():
    """模型按缩放后的图报坐标时，必须按比例换算回原图。"""
    orig_cfg, orig_post = at.vlm_config, at._http_post_json
    try:
        at.vlm_config = lambda: _vcfg()
        at._http_post_json = lambda *a, **k: _fake_response(
            [{"label": "person", "x1": 0, "y1": 0, "x2": 50, "y2": 40}],
            w=50, h=40)
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.jpg")
            Image.new("RGB", (100, 80), "white").save(p)
            boxes, _ = at._infer_vlm(p, CLASSES)
        assert len(boxes) == 1
        # 模型说图是 50x40，实际 100x80 -> 框应铺满整图
        assert abs(boxes[0]["cx"] - 0.5) < 1e-6
        assert abs(boxes[0]["w"] - 1.0) < 1e-6
        assert abs(boxes[0]["h"] - 1.0) < 1e-6
    finally:
        at.vlm_config, at._http_post_json = orig_cfg, orig_post


def test_infer_reports_http_error():
    orig_cfg, orig_post = at.vlm_config, at._http_post_json
    try:
        at.vlm_config = lambda: _vcfg()
        at._http_post_json = lambda *a, **k: (401, {"error": "bad key"})
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.jpg")
            Image.new("RGB", (100, 80), "white").save(p)
            boxes, meta = at._infer_vlm(p, CLASSES)
        assert boxes == []
        assert "401" in meta["error"]
    finally:
        at.vlm_config, at._http_post_json = orig_cfg, orig_post


def test_skip_empty_does_not_write_negative():
    """空结果不写标签：不能被当成负样本静默落盘。"""
    orig_cfg, orig_infer = at.vlm_config, at._infer_vlm
    try:
        at.vlm_config = lambda: _vcfg()
        at._infer_vlm = lambda *a, **k: ([], {"raw": 0, "unmapped": 0,
                                              "dropped": 0, "error": None})
        with tempfile.TemporaryDirectory() as tmp:
            _set_state(tmp)
            c = at.app.test_client()
            assert c.post("/api/auto_annotate",
                          json={"skip_labeled": False, "skip_empty": True,
                                "vlm": True}).get_json()["ok"]
            st = _wait_finished(c)
            assert st["empty"] == 2
            assert st["saved"] == 0
            assert not os.path.isfile(os.path.join(tmp, "a.txt"))
            assert not os.path.isfile(os.path.join(tmp, "b.txt"))
    finally:
        at.vlm_config, at._infer_vlm = orig_cfg, orig_infer


def test_empty_written_as_negative_when_not_skipping():
    """关掉该开关时保持原有语义：空结果写空 txt 作为负样本。"""
    orig_cfg, orig_infer = at.vlm_config, at._infer_vlm
    try:
        at.vlm_config = lambda: _vcfg()
        at._infer_vlm = lambda *a, **k: ([], {"raw": 0, "unmapped": 0,
                                              "dropped": 0, "error": None})
        with tempfile.TemporaryDirectory() as tmp:
            _set_state(tmp)
            c = at.app.test_client()
            assert c.post("/api/auto_annotate",
                          json={"skip_labeled": False, "skip_empty": False,
                                "vlm": True}).get_json()["ok"]
            st = _wait_finished(c)
            assert st["saved"] == 2
            assert st["empty"] == 0
            assert os.path.isfile(os.path.join(tmp, "a.txt"))
    finally:
        at.vlm_config, at._infer_vlm = orig_cfg, orig_infer


def test_api_failure_counts_as_failed_not_written():
    """接口报错时必须算失败，不能静默写成空标签。"""
    orig_cfg, orig_post = at.vlm_config, at._http_post_json
    try:
        at.vlm_config = lambda: _vcfg()
        at._http_post_json = lambda *a, **k: (500, {"error": "boom"})
        with tempfile.TemporaryDirectory() as tmp:
            _set_state(tmp)
            c = at.app.test_client()
            assert c.post("/api/auto_annotate",
                          json={"skip_labeled": False, "skip_empty": True,
                                "vlm": True}).get_json()["ok"]
            st = _wait_finished(c)
            assert st["failed"] == 2
            assert st["saved"] == 0
            assert not os.path.isfile(os.path.join(tmp, "a.txt"))
            assert any("500" in e for e in st["errors"])
    finally:
        at.vlm_config, at._http_post_json = orig_cfg, orig_post


def test_requires_api_key_and_classes():
    orig_cfg = at.vlm_config
    try:
        at.vlm_config = lambda: _vcfg(api_key="")
        with tempfile.TemporaryDirectory() as tmp:
            _set_state(tmp)
            c = at.app.test_client()
            r = c.post("/api/auto_annotate", json={"vlm": True}).get_json()
            assert not r["ok"] and "API Key" in r["msg"]
        at.vlm_config = lambda: _vcfg()
        with tempfile.TemporaryDirectory() as tmp:
            _set_state(tmp, classes=[])
            at.STATE["classes"] = []
            c = at.app.test_client()
            r = c.post("/api/auto_annotate", json={"vlm": True}).get_json()
            assert not r["ok"] and "类别" in r["msg"]
    finally:
        at.vlm_config = orig_cfg


def test_api_key_never_returned_to_browser():
    orig_vlm = at.vlm_config
    try:
        at.vlm_config = lambda: _vcfg(api_key="sk-secret")
        c = at.app.test_client()
        body = json.dumps(c.get("/api/vlm_config").get_json())
        assert "sk-secret" not in body
        assert c.get("/api/vlm_config").get_json()["vlm"]["has_key"] is True
        assert "sk-secret" not in json.dumps(c.get("/api/config").get_json())
    finally:
        at.vlm_config = orig_vlm


def test_vlm_test_endpoint_previews_without_writing():
    orig_cfg, orig_infer = at.vlm_config, at._infer_vlm
    try:
        at.vlm_config = lambda: _vcfg()
        at._infer_vlm = lambda *a, **k: (
            [{"cls": 2, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2,
              "conf": 0.6}],
            {"raw": 1, "unmapped": 0, "dropped": 0, "error": None})
        with tempfile.TemporaryDirectory() as tmp:
            _set_state(tmp)
            c = at.app.test_client()
            d = c.post("/api/vlm_test", json={"idx": 0}).get_json()
            assert d["ok"] and len(d["boxes"]) == 1
            assert d["boxes"][0]["label"] == "no-helmet"
            assert not os.path.isfile(os.path.join(tmp, "a.txt"))
    finally:
        at.vlm_config, at._infer_vlm = orig_cfg, orig_infer


def test_config_persists_across_restart():
    """模型信息（地址/模型名/Key/并发/思考/单价）存本机，重启后能回填。"""
    import importlib
    import shutil
    cfg = at.CONFIG_FILE
    bak = cfg + ".persist-bak"
    if os.path.isfile(cfg):
        shutil.copyfile(cfg, bak)
    try:
        if os.path.isfile(cfg):
            os.remove(cfg)
        mod = importlib.reload(at)
        c = mod.app.test_client()
        r = c.post("/api/vlm_config", json={
            "base_url": "https://api.example.com/v1",
            "model": "some-vl-model",
            "api_key": "sk-abc",
            "thinking": False,
            "concurrency": 7}).get_json()
        assert r["ok"]
        assert r["vlm"]["model"] == "some-vl-model"
        assert r["vlm"]["has_key"] is True
        assert r["vlm"]["api_key"] == ""

        # 模拟重启：重新加载模块，配置从磁盘重新读
        mod2 = importlib.reload(mod)
        c2 = mod2.app.test_client()
        v = c2.get("/api/config").get_json()["vlm"]
        assert v["base_url"] == "https://api.example.com/v1"
        assert v["model"] == "some-vl-model"
        assert v["concurrency"] == 7
        assert v["thinking"] is False
        assert v["has_key"] is True
        assert "api_key" not in v or v["api_key"] == ""
        # 磁盘上确实存了 key，服务端能继续用
        assert mod2.vlm_config()["api_key"] == "sk-abc"
    finally:
        if os.path.isfile(bak):
            shutil.copyfile(bak, cfg)
            os.remove(bak)
        elif os.path.isfile(cfg):
            os.remove(cfg)
        importlib.reload(at)


def test_generic_defaults_no_vendor():
    """默认值里不能绑死任何厂商。"""
    assert at.VLM_DEFAULTS["base_url"] == ""
    assert at.VLM_DEFAULTS["model"] == ""
    blob = json.dumps(at.VLM_DEFAULTS)
    for bad in ("deepseek", "openai.com", "gpt-"):
        assert bad not in blob


def test_missing_endpoint_or_model_reports_clearly():
    orig_cfg = at.vlm_config
    try:
        for miss, want in (({"base_url": ""}, "接口地址"),
                           ({"model": ""}, "模型名")):
            at.vlm_config = lambda m=miss: _vcfg(**m)
            with tempfile.TemporaryDirectory() as tmp:
                p = os.path.join(tmp, "a.jpg")
                Image.new("RGB", (40, 30), "white").save(p)
                boxes, meta = at._infer_vlm(p, CLASSES)
            assert boxes == []
            assert want in meta["error"]
    finally:
        at.vlm_config = orig_cfg


def test_class_hint_parsing_and_prompt():
    """类别名可以带定位提示；提示只进提示词，不进类别名。"""
    assert at._parse_class_spec("no-helmet") == ("no-helmet", None)
    assert at._parse_class_spec("no-helmet(头部)") == ("no-helmet", "头部")
    assert at._parse_class_spec("no-vest（躯干）") == ("no-vest", "躯干")
    assert at._parse_class_spec("  灭火器  ") == ("灭火器", None)
    # 名称或提示里再出现括号就整体当类别名，不猜
    assert at._parse_class_spec("头(部)(次要)") == ("头(部)(次要)", None)

    p = at._vlm_prompt(["helmet", "no-helmet"], {"no-helmet": "头部"})
    assert "no-helmet：头部" in p
    assert "定位提示" in p
    assert p.count("：头部") == 1        # 只给带提示的那个类别加
    # 通用规则必须在（不依赖任何具体类别）
    assert "不要框空地" in p
    assert "缺少某物" in p
    # 没有任何提示时不出现提示段
    assert "定位提示" not in at._vlm_prompt(["helmet", "vest"])


def test_open_strips_hints_but_persists_raw():
    """/api/open 要把提示从类别名里剥掉，同时把原始写法存进配置。"""
    import shutil
    cfg = at.CONFIG_FILE
    bak = cfg + ".hint-bak"
    if os.path.isfile(cfg):
        shutil.copyfile(cfg, bak)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _set_state(tmp)
            c = at.app.test_client()
            d = c.post("/api/open", json={
                "img_dir": tmp, "label_dir": tmp,
                "classes": ["no-helmet(头部)", "灭火器", "no-vest（躯干）"]}).get_json()
            assert d["ok"]
            assert d["classes"] == ["no-helmet", "灭火器", "no-vest"]
            assert at.STATE["class_hints"] == {"no-helmet": "头部",
                                               "no-vest": "躯干"}
            saved = json.load(open(cfg, encoding="utf-8"))
            assert "no-helmet(头部)" in saved["classes"]
            assert "no-vest（躯干）" in saved["classes"]
    finally:
        if os.path.isfile(bak):
            shutil.copyfile(bak, cfg)
            os.remove(bak)


def test_hint_reaches_the_request_payload():
    """定位提示必须真的出现在发给模型的提示词里。"""
    orig_cfg, orig_post = at.vlm_config, at._http_post_json
    seen = {}
    try:
        at.vlm_config = lambda: _vcfg()

        def fake_post(url, payload, headers, timeout, proxy=None):
            seen["text"] = payload["messages"][0]["content"][0]["text"]
            return _fake_response([{"label": "no-helmet", "x1": 1, "y1": 1,
                                    "x2": 9, "y2": 9}])

        at._http_post_json = fake_post
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.jpg")
            Image.new("RGB", (100, 80), "white").save(p)
            at._infer_vlm(p, ["helmet", "no-helmet"], hints={"no-helmet": "头部"})
        assert "no-helmet：头部" in seen["text"]
    finally:
        at.vlm_config, at._http_post_json = orig_cfg, orig_post


def test_extract_json_prefers_object_with_boxes():
    """模型先吐 {"type":"json_object"} 再吐真结果时，不能把真结果丢掉。"""
    raw = ('{"type": "json_object"}\n'
           '{"image_width": 800, "image_height": 448, '
           '"boxes": [{"label": "no-helmet", "x1": 1, "y1": 2, "x2": 3, "y2": 4}]}')
    obj = at._extract_json(raw)
    assert obj is not None and len(obj["boxes"]) == 1
    # 代码块包裹
    assert at._extract_json('```json\n{"boxes": []}\n```') == {"boxes": []}
    # 纯文本里夹一个对象
    assert at._extract_json('结果是 {"boxes": [], "image_width": 10} 完')["image_width"] == 10
    assert at._extract_json("没有任何 JSON") is None
    assert at._extract_json("") is None


def test_retries_transient_failures():
    """503 / 连接重置 / 空回复都是瞬时故障，必须退避重试而不是直接算失败。"""
    orig = (at.vlm_config, at._http_post_json, at.time.sleep)
    calls = {"n": 0}
    try:
        at.vlm_config = lambda: _vcfg(retries=3)
        at.time.sleep = lambda s: None

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return 503, {"error": "Service is too busy"}
            if calls["n"] == 2:
                return 0, {"error": "[WinError 10054] 远程主机强迫关闭了一个现有的连接"}
            return _fake_response([{"label": "person", "x1": 1, "y1": 1,
                                    "x2": 9, "y2": 9}])

        at._http_post_json = flaky
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.jpg")
            Image.new("RGB", (100, 80), "white").save(p)
            boxes, meta = at._infer_vlm(p, CLASSES)
        assert meta["error"] is None
        assert len(boxes) == 1
        assert calls["n"] == 3
    finally:
        at.vlm_config, at._http_post_json, at.time.sleep = orig


def test_empty_content_is_retried():
    """返回 200 但内容是空的，同样按瞬时故障重试。"""
    orig = (at.vlm_config, at._http_post_json, at.time.sleep)
    calls = {"n": 0}
    try:
        at.vlm_config = lambda: _vcfg(retries=3)
        at.time.sleep = lambda s: None

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return 200, {"choices": [{"message": {"content": ""}}]}
            return _fake_response([{"label": "vest", "x1": 1, "y1": 1,
                                    "x2": 9, "y2": 9}])

        at._http_post_json = flaky
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.jpg")
            Image.new("RGB", (100, 80), "white").save(p)
            boxes, meta = at._infer_vlm(p, CLASSES)
        assert meta["error"] is None and len(boxes) == 1
        assert calls["n"] == 2
    finally:
        at.vlm_config, at._http_post_json, at.time.sleep = orig


def test_non_retryable_error_fails_fast():
    """401 这类硬错误不该反复重试，浪费时间和钱。"""
    orig = (at.vlm_config, at._http_post_json, at.time.sleep)
    calls = {"n": 0}
    try:
        at.vlm_config = lambda: _vcfg(retries=3)
        at.time.sleep = lambda s: None

        def bad(*a, **k):
            calls["n"] += 1
            return 401, {"error": "bad key"}

        at._http_post_json = bad
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.jpg")
            Image.new("RGB", (100, 80), "white").save(p)
            boxes, meta = at._infer_vlm(p, CLASSES)
        assert boxes == [] and "401" in meta["error"]
        assert calls["n"] == 1
    finally:
        at.vlm_config, at._http_post_json, at.time.sleep = orig


def test_valid_empty_result_is_not_retried():
    """合法的「这张图没有目标」不能重试，否则白白多花钱。"""
    orig = (at.vlm_config, at._http_post_json, at.time.sleep)
    calls = {"n": 0}
    try:
        at.vlm_config = lambda: _vcfg(retries=3)
        at.time.sleep = lambda s: None

        def empty(*a, **k):
            calls["n"] += 1
            return _fake_response([])

        at._http_post_json = empty
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.jpg")
            Image.new("RGB", (100, 80), "white").save(p)
            boxes, meta = at._infer_vlm(p, CLASSES)
        assert boxes == [] and meta["error"] is None
        assert calls["n"] == 1
    finally:
        at.vlm_config, at._http_post_json, at.time.sleep = orig


def test_files_endpoint_exposes_session():
    """刷新页面要靠这个接口恢复现场。"""
    with tempfile.TemporaryDirectory() as tmp:
        _set_state(tmp)
        d = at.app.test_client().get("/api/files").get_json()
        assert d["ok"]
        assert d["img_dir"] == tmp
        assert d["classes"] == CLASSES
        assert [f["name"] for f in d["files"]] == ["a.jpg", "b.jpg"]


def test_render_review_image_draws_all_boxes():
    """复核图要把所有候选框画上去（模型按编号回答，不输出坐标）。"""
    img = Image.new("RGB", (400, 240), "white")
    boxes = [{"cls": 0, "cx": .2, "cy": .3, "w": .1, "h": .2},
             {"cls": 1, "cx": .7, "cy": .6, "w": .15, "h": .25}]
    vis = at._render_review_image(img, boxes)
    assert vis.size == img.size
    px = vis.load()
    reds = 0
    for y in range(0, vis.size[1], 2):
        for x in range(0, vis.size[0], 2):
            r, g, b = px[x, y][:3]
            if r > 200 and g < 100 and b < 100:
                reds += 1
    assert reds > 50, "应该画出了红框和编号"


def test_review_parses_verdicts():
    """模型返回 JSON 数组时，按 id 映射成 keep / reclass / drop。"""
    orig_cfg, orig_req = at.vlm_config, at._vlm_request
    try:
        at.vlm_config = lambda: _vcfg()
        boxes = [{"cls": 2, "cx": .3, "cy": .3, "w": .1, "h": .1},   # no-helmet
                 {"cls": 0, "cx": .6, "cy": .5, "w": .1, "h": .1},   # person
                 {"cls": 0, "cx": .5, "cy": .8, "w": .1, "h": .1}]   # person

        def fake_req(v, payload):
            assert isinstance(payload["messages"][0]["content"], list)
            n_img = sum(1 for c in payload["messages"][0]["content"]
                        if c.get("type") == "image_url")
            assert n_img == 1, "整图一次发出，框和编号画在图上"
            return ([{"id": 0, "label": "no-helmet"},
                     {"id": 1, "label": "helmet"},
                     {"id": 2, "label": "drop"}],
                    {"total_tokens": 12}, None)

        at._vlm_request = fake_req
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.jpg")
            Image.new("RGB", (200, 120), "white").save(p)
            verd, meta = at._vlm_review_image(p, boxes, CLASSES)
        assert meta["error"] is None
        assert verd[0] == ("keep", 2)
        assert verd[1] == ("reclass", 1)
        assert verd[2] == ("drop", None)
    finally:
        at.vlm_config, at._vlm_request = orig_cfg, orig_req


def test_review_drop_can_be_disallowed():
    """关掉删除权限时，模型说 drop 也只能保持原样。"""
    orig_cfg, orig_req = at.vlm_config, at._vlm_request
    try:
        at.vlm_config = lambda: _vcfg()
        at._vlm_request = lambda v, p: ([{"id": 0, "label": "drop"}], None, None)
        boxes = [{"cls": 0, "cx": .5, "cy": .5, "w": .2, "h": .3}]
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "a.jpg")
            Image.new("RGB", (200, 120), "white").save(p)
            verd, _ = at._vlm_review_image(p, boxes, CLASSES,
                                           drop_allowed=False)
        assert verd[0] == ("keep", None)
    finally:
        at.vlm_config, at._vlm_request = orig_cfg, orig_req


def test_apply_review_never_drops_unparsed():
    """没拿到结论的框必须原样保留——宁可不动，不可误删。"""
    boxes = [{"cls": 0, "cx": .2, "cy": .2, "w": .1, "h": .1},
             {"cls": 1, "cx": .5, "cy": .5, "w": .1, "h": .1},
             {"cls": 2, "cx": .8, "cy": .8, "w": .1, "h": .1}]
    out, changes = at._apply_review(boxes, {0: ("reclass", 3)}, CLASSES)
    assert len(out) == 3
    assert out[0]["cls"] == 3
    assert out[1]["cls"] == 1 and out[2]["cls"] == 2
    assert len(changes) == 1 and changes[0]["action"] == "reclass"


def test_review_worker_rewrites_labels():
    """显式要求写入时（dry_run=False），改类和删除都要落到磁盘上。"""
    orig_cfg, orig_review = at.vlm_config, at._vlm_review_image
    try:
        at.vlm_config = lambda: _vcfg()

        def fake_review(path, boxes, classes, vcfg=None, hints=None,
                        drop_allowed=True):
            return ({0: ("reclass", 3), 1: ("drop", None)}, {"error": None})

        at._vlm_review_image = fake_review
        with tempfile.TemporaryDirectory() as tmp:
            _set_state(tmp, files=("a.jpg",))
            with open(os.path.join(tmp, "a.txt"), "w", encoding="utf-8") as f:
                f.write("2 0.300000 0.300000 0.100000 0.100000\n"
                        "0 0.600000 0.500000 0.100000 0.100000\n")
            c = at.app.test_client()
            assert c.post("/api/vlm_review",
                          json={"dry_run": False}).get_json()["ok"]
            st = _wait_finished(c)
            assert st["mode"] == "review"
            assert st["reclassed"] == 1 and st["dropped"] == 1
            lines = [l for l in open(os.path.join(tmp, "a.txt"),
                                     encoding="utf-8").read().splitlines() if l.strip()]
            assert len(lines) == 1
            assert lines[0].startswith("3 ")     # 2 -> 3，另一个被删
    finally:
        at.vlm_config, at._vlm_review_image = orig_cfg, orig_review


def test_review_test_endpoint_does_not_write():
    orig_cfg, orig_review = at.vlm_config, at._vlm_review_image
    try:
        at.vlm_config = lambda: _vcfg()
        at._vlm_review_image = lambda *a, **k: (
            {0: ("reclass", 1)}, {"error": None, "unparsed": 0})
        with tempfile.TemporaryDirectory() as tmp:
            _set_state(tmp, files=("a.jpg",))
            lp = os.path.join(tmp, "a.txt")
            with open(lp, "w", encoding="utf-8") as f:
                f.write("0 0.500000 0.500000 0.100000 0.100000\n")
            before = open(lp, encoding="utf-8").read()
            d = at.app.test_client().post(
                "/api/vlm_review_test", json={"idx": 0}).get_json()
            assert d["ok"] and len(d["changes"]) == 1
            assert open(lp, encoding="utf-8").read() == before   # 预览不落盘
    finally:
        at.vlm_config, at._vlm_review_image = orig_cfg, orig_review


def test_review_dry_run_then_apply():
    """复核默认只出建议、不落盘；点「应用修改」才真正写。"""
    orig_cfg, orig_review = at.vlm_config, at._vlm_review_image
    try:
        at.vlm_config = lambda: _vcfg()
        at._vlm_review_image = lambda *a, **k: ({0: ("reclass", 3)},
                                                {"error": None})
        with tempfile.TemporaryDirectory() as tmp:
            _set_state(tmp, files=("a.jpg",))
            lp = os.path.join(tmp, "a.txt")
            with open(lp, "w", encoding="utf-8") as f:
                f.write("0 0.500000 0.500000 0.100000 0.100000\n")
            before = open(lp, encoding="utf-8").read()
            c = at.app.test_client()
            assert c.post("/api/vlm_review", json={}).get_json()["ok"]
            st = _wait_finished(c)
            assert st["mode"] == "review" and st["changed"] == 1
            assert open(lp, encoding="utf-8").read() == before, "预演阶段不能写盘"
            assert at.STATE["review_proposals"], "应该留下待应用的修改建议"
            d = c.post("/api/vlm_review_apply", json={}).get_json()
            assert d["ok"] and d["applied"] == 1
            assert open(lp, encoding="utf-8").read().startswith("3 ")
            assert not at.STATE["review_proposals"]
    finally:
        at.vlm_config, at._vlm_review_image = orig_cfg, orig_review


def test_v5_letterbox_geometry():
    """YOLOv5 的 letterbox：等比缩放 + 居中留边，输出固定尺寸。"""
    img = Image.new("RGB", (800, 448), "white")
    canvas, r, px, py = at._v5_letterbox(img, size=640)
    assert canvas.size == (640, 640)
    assert abs(r - 0.8) < 1e-6            # min(640/800, 640/448)
    assert px == 0
    assert py == (640 - int(round(448 * 0.8))) // 2
    # 竖向图应该是左右留边
    tall = Image.new("RGB", (448, 800), "white")
    c2, r2, px2, py2 = at._v5_letterbox(tall, size=640)
    assert c2.size == (640, 640) and py2 == 0 and px2 > 0


def test_v5_shim_requires_torch_gracefully():
    """没装 torch 时不能崩在导入阶段，调用时才给出可读提示。"""
    if at.nn is None:
        try:
            at._install_yolov5_shim()
            raise AssertionError("应该抛错")
        except RuntimeError as e:
            assert "PyTorch" in str(e)
        try:
            at._load_yolov5("whatever.pt")
            raise AssertionError("应该抛错")
        except RuntimeError as e:
            assert "PyTorch" in str(e)
    else:
        assert callable(at._install_yolov5_shim)


def test_load_local_model_reports_both_attempts():
    """两种非 ultralytics 格式都失败时，错误信息要能看出分别是什么原因。"""
    orig_v5, orig_v6 = at._load_yolov5, at._load_meituan_v6n
    try:
        def boom_v5(p):
            raise RuntimeError("No module named 'models'")

        def boom_v6(p):
            raise RuntimeError("No module named 'yolov6'")

        at._load_yolov5, at._load_meituan_v6n = boom_v5, boom_v6
        try:
            at._load_local_model("x.pt")
            raise AssertionError("应该抛错")
        except RuntimeError as e:
            msg = str(e)
            assert "YOLOv5" in msg and "models" in msg
            assert "v6n" in msg and "yolov6" in msg
    finally:
        at._load_yolov5, at._load_meituan_v6n = orig_v5, orig_v6


def test_load_local_model_prefers_yolov5():
    orig_v5, orig_v6 = at._load_yolov5, at._load_meituan_v6n
    try:
        at._load_yolov5 = lambda p: ("m", ["a", "b"])
        at._load_meituan_v6n = lambda p: (_ for _ in ()).throw(
            AssertionError("不该走到 v6n"))
        model, classes, backend = at._load_local_model("x.pt")
        assert backend == "yolov5" and classes == ["a", "b"]
    finally:
        at._load_yolov5, at._load_meituan_v6n = orig_v5, orig_v6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  %s" % fn.__name__)
    print("全部测试通过 (%d 项)" % len(fns))
