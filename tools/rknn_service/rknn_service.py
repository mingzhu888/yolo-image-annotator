# -*- coding: utf-8 -*-
"""RKNN 远程推理服务（跑在 RK3588 上）。

为什么需要它：.rknn 只能在 NPU 上执行，PC 端的模拟器明确不支持 load_rknn。
所以 PC 上的标注工具把图片发到这里，这里用真 NPU 推理后返回框。

只用标准库 + 设备上已有的 numpy/opencv/rknnlite，不需要 Flask。

接口：
  GET  /health        健康检查 + 已加载模型
  GET  /models        可用模型清单
  POST /infer         {"model": "...", "image": "<base64 jpeg/png>",
                       "conf": 0.1, "iou": 0.45,
                       "person_model": "person_infra",   # 可选，级联
                       "min_person_conf": 0.5, "crop_expand": 1.2}
"""
import argparse
import base64
import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from rknnlite.api import RKNNLite

DEFAULT_ANCHORS = [
    [10, 13], [16, 30], [33, 23],
    [30, 61], [62, 45], [59, 119],
    [116, 90], [156, 198], [373, 326],
]
MASKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

LOCK = threading.Lock()          # RKNNLite 不是线程安全的，串行推理
REGISTRY = {}                    # code -> manifest dict
LOADED = {}                      # code -> (RKNNLite, labels, input_size)


def scan_models(roots):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            mp = os.path.join(d, "manifest.json")
            w = os.path.join(d, "model.rknn")
            if not (os.path.isdir(d) and os.path.isfile(mp) and os.path.isfile(w)):
                continue
            try:
                m = json.load(open(mp, encoding="utf-8"))
            except Exception:
                continue
            m["_code"] = m.get("algorithm_code") or name
            m["_dir"] = d
            m["_weights"] = w
            if m.get("engine_format") != "rknn":
                continue
            REGISTRY[m["_code"]] = m


def get_model(code):
    if code in LOADED:
        return LOADED[code]
    m = REGISTRY.get(code)
    if not m:
        raise KeyError("未知模型: %s" % code)
    r = RKNNLite(verbose=False)
    if r.load_rknn(m["_weights"]) != 0:
        raise RuntimeError("load_rknn 失败: %s" % code)
    if r.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
        raise RuntimeError("init_runtime 失败: %s" % code)
    labels = m.get("labels") or []
    size = int(m.get("input_size") or 640)
    LOADED[code] = (r, labels, size)
    print("[load] %s  labels=%s  size=%d" % (code, labels, size), flush=True)
    return LOADED[code]


def letterbox(img, size):
    """与生产 letterbox_frame 完全一致：BGR 读入 -> resize+居中填充 -> 转 RGB。"""
    h, w = img.shape[:2]
    scale = min(size / float(h), size / float(w))
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dy, dx = (size - nh) // 2, (size - nw) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), scale, dx, dy


def nms(boxes, scores, iou_thr):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def decode(outputs, size, conf_thr, iou_thr, labels):
    """yolov5_anchor -> [{label, conf, x1,y1,x2,y2}]（letterbox 坐标系）"""
    anchors = np.array(DEFAULT_ANCHORS, dtype=np.float64)
    all_b, all_c, all_s = [], [], []
    for out, mask in zip(outputs[:3], MASKS):
        a = np.asarray(out)
        if a.ndim != 4:
            raise RuntimeError("输出维度异常: %s" % (a.shape,))
        c = a.shape[1] // 3
        data = a.reshape([3, c] + list(a.shape[-2:]))
        data = np.transpose(data, (2, 3, 0, 1))          # (h,w,3,5+nc)
        gh, gw = data.shape[:2]
        flat = data.reshape(-1, c)
        obj = flat[:, 4]
        cls = flat[:, 5:]
        cmax = cls.max(axis=-1)
        keep = np.where((obj >= conf_thr) & (cmax >= conf_thr))[0]
        if len(keep) == 0:
            continue
        ys, xs = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        grid = np.stack([xs, ys], -1).reshape(-1, 2)
        grid3 = np.repeat(grid, 3, axis=0)
        ap = np.tile(anchors[mask], (gh * gw, 1))
        stride = float(size) / gh
        sub = flat[keep]
        xy = (sub[:, :2] * 2.0 - 0.5 + grid3[keep]) * stride
        wh = np.square(sub[:, 2:4] * 2.0) * ap[keep]
        all_b.append(np.concatenate([xy - wh / 2, xy + wh / 2], 1))
        all_c.append(cls[keep].argmax(1))
        all_s.append(cmax[keep] * obj[keep])
    if not all_b:
        return []
    boxes = np.concatenate(all_b)
    classes = np.concatenate(all_c)
    scores = np.concatenate(all_s)
    res = []
    for cid in np.unique(classes):
        m = classes == cid
        for k in nms(boxes[m], scores[m], iou_thr):
            res.append((int(cid), float(scores[m][k]), boxes[m][k]))
    out = []
    for cid, s, b in res:
        lab = labels[cid] if cid < len(labels) else "class_%d" % cid
        out.append({"label": lab, "conf": s,
                    "x1": float(b[0]), "y1": float(b[1]),
                    "x2": float(b[2]), "y2": float(b[3])})
    return out


def infer_one(code, img, conf=None, iou=None):
    model, labels, size = get_model(code)
    m = REGISTRY[code]
    conf = float(conf if conf is not None else m.get("conf_thres", 0.25))
    iou = float(iou if iou is not None else m.get("iou_thres", 0.45))
    lb, scale, dx, dy = letterbox(img, size)
    t0 = time.time()
    with LOCK:
        outs = model.inference(inputs=[lb[None, ...]])
    ms = (time.time() - t0) * 1000
    boxes = decode(outs, size, conf, iou, labels)
    h, w = img.shape[:2]
    for b in boxes:            # letterbox -> 原图
        b["x1"] = max(0.0, min((b["x1"] - dx) / scale, w - 1.0))
        b["y1"] = max(0.0, min((b["y1"] - dy) / scale, h - 1.0))
        b["x2"] = max(0.0, min((b["x2"] - dx) / scale, w - 1.0))
        b["y2"] = max(0.0, min((b["y2"] - dy) / scale, h - 1.0))
    return boxes, ms


def infer_cascade(target, person_model, img, conf, iou,
                  min_person_conf, crop_expand):
    persons, ms1 = infer_one(person_model, img, conf=min_person_conf)
    h, w = img.shape[:2]
    out, ms2 = [], 0.0
    for p in persons[:8]:
        pw, ph = p["x2"] - p["x1"], p["y2"] - p["y1"]
        if pw < 8 or ph < 8:
            continue
        cx, cy = (p["x1"] + p["x2"]) / 2, (p["y1"] + p["y2"]) / 2
        hw, hh = pw * crop_expand / 2, ph * crop_expand / 2
        x1 = int(max(0, cx - hw)); y1 = int(max(0, cy - hh))
        x2 = int(min(w, cx + hw)); y2 = int(min(h, cy + hh))
        if x2 - x1 < 16 or y2 - y1 < 16:
            continue
        crop = img[y1:y2, x1:x2]
        boxes, ms = infer_one(target, crop, conf=conf, iou=iou)
        ms2 += ms
        for b in boxes:
            b["x1"] += x1; b["x2"] += x1
            b["y1"] += y1; b["y2"] += y1
            b["person"] = [p["x1"], p["y1"], p["x2"], p["y2"]]
            out.append(b)
    return out, ms1, ms2, len(persons)


class Handler(BaseHTTPRequestHandler):
    server_version = "rknn-svc/1.0"

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._json({"ok": True, "loaded": sorted(LOADED),
                               "models": len(REGISTRY)})
        if self.path.startswith("/models"):
            return self._json({"ok": True, "models": [
                {"code": k, "name": v.get("name"),
                 "labels": v.get("labels"), "input_size": v.get("input_size"),
                 "role": v.get("model_role")}
                for k, v in sorted(REGISTRY.items())]})
        return self._json({"ok": False, "error": "not found"}, 404)

    def do_POST(self):
        if not self.path.startswith("/infer"):
            return self._json({"ok": False, "error": "not found"}, 404)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            raw = base64.b64decode(req["image"])
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("图片解码失败")
            t0 = time.time()
            person_model = req.get("person_model")
            if person_model:
                boxes, ms1, ms2, np_ = infer_cascade(
                    req["model"], person_model, img,
                    req.get("conf"), req.get("iou"),
                    float(req.get("min_person_conf", 0.5)),
                    float(req.get("crop_expand", 1.2)))
                return self._json({"ok": True, "boxes": boxes,
                                   "ms": round((time.time() - t0) * 1000, 1),
                                   "ms_person": round(ms1, 1),
                                   "ms_target": round(ms2, 1),
                                   "persons": np_})
            boxes, ms = infer_one(req["model"], img,
                                  req.get("conf"), req.get("iou"))
            return self._json({"ok": True, "boxes": boxes,
                               "ms": round(ms, 1),
                               "size": [img.shape[1], img.shape[0]]})
        except Exception as e:
            return self._json({"ok": False, "error": str(e),
                               "trace": traceback.format_exc()[-800:]}, 500)

    def log_message(self, fmt, *args):
        print("[http] " + (fmt % args), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--models", default=(
        "/home/zql/deploy/yqzk-deploy/model_zoo,/home/zql/rknn_svc/models"))
    a = ap.parse_args()
    scan_models([p for p in a.models.split(",") if p])
    print("扫描到 %d 个 rknn 模型" % len(REGISTRY), flush=True)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print("监听 %s:%d" % (a.host, a.port), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
