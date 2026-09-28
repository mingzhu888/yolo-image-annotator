# -*- coding: utf-8 -*-
"""
YOLO 图片标注工具 (本地版) + 模型辅助标注
==========================================
- 现代深色界面, 左侧文件列表 (显示已标注/未标注状态)
- 鼠标拖拽画框 / 选中 / 缩放 / 删除 / 改类别
- 保存为 YOLO 格式 (class cx cy w h, 归一化)
- 可加载 YOLO 模型 (.pt) 进行自动预测辅助标注
- 模型含多个类别时,可勾选只标注其中一部分类别
- 自定义类别

依赖: flask, pillow, ultralytics (均已安装)
启动: 双击 start.bat 或运行 python annotate_tool.py
"""
import os
import random
import re
import sys
import json
import math
import socket
import subprocess
import threading
import time
import types
import webbrowser
from collections import defaultdict
from pathlib import Path
from flask import Flask, request, jsonify, send_file, Response
from PIL import Image

PORT = 5000


def _port_in_use(port):
    """端口是否已经有人在监听 (127.0.0.1)。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def _pids_listening_on(port):
    """返回正在 LISTENING 该端口的 PID 列表 (仅 Windows netstat)。"""
    pids = set()
    try:
        out = subprocess.check_output(["netstat", "-ano"], text=True,
                                      errors="ignore")
    except Exception:
        return []
    for line in out.splitlines():
        if "LISTENING" not in line:
            continue
        if f":{port} " not in line and not re.search(rf":{port}\b", line):
            continue
        m = re.search(r"(\d+)\s*$", line.strip())
        if m:
            pids.add(int(m.group(1)))
    return list(pids)


def _pid_commandline(pid):
    """尽量拿到进程命令行(小写);失败返回空字符串。"""
    try:
        out = subprocess.check_output(
            ["wmic", "process", "where", f"ProcessId={pid}",
             "get", "CommandLine", "/format:list"],
            text=True, errors="ignore", timeout=5)
        return out.lower()
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
            text=True, errors="ignore", timeout=5)
        return out.lower()
    except Exception:
        return ""


def free_port_if_stale(port):
    """启动前:若端口被旧实例(僵死或正常)占用,清掉它,保证本次是全新实例。
    只杀“命令行里包含 annotate_tool”的 python 进程,避免误伤其它服务。
    返回 True 表示端口现在可用。"""
    if not _port_in_use(port) and not _pids_listening_on(port):
        return True
    pids = _pids_listening_on(port)
    if not pids:
        # 端口像被占用但查不到监听 PID(多为 TIME_WAIT 残留),直接放行
        return True
    for pid in pids:
        if pid == os.getpid():
            continue
        try:
            info = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                text=True, errors="ignore").lower()
        except Exception:
            info = ""
        if "python" not in info:
            print(f"  [警告] 端口 {port} 被非 python 进程(PID {pid})占用,"
                  f"未自动清理。请手动关闭它后重试。")
            return False
        cmd = _pid_commandline(pid)
        if "annotate_tool" not in cmd:
            print(f"  [警告] 端口 {port} 的 python 进程(PID {pid})不是本工具"
                  f"({cmd.strip()[:80] or '无法读取命令行'}),未自动清理。")
            return False
        print(f"  检测到旧的标注工具实例(PID {pid})仍占用端口 {port},正在关闭它...")
        subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                       capture_output=True)
    # 等端口真正释放
    for _ in range(20):
        if not _port_in_use(port):
            break
        threading.Event().wait(0.25)
    print("  旧实例已清理,启动全新实例。")
    return True

app = Flask(__name__)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "annotate_config.json")

STATE = {
    "img_dir": None,
    "label_dir": None,
    "images": [],
    "classes": [],
    "class_hints": {},   # 可选：{类别名: 定位提示}，只用于多模态提示词
    "models": [],        # 已加载的模型列表 [{id,name,path,model,model_classes,cls_offset,conf,only_cls}]
    "auto": None,        # 自动化标注进度
}


def list_images(folder):
    p = Path(folder)
    if not p.exists():
        return []
    return sorted([f.name for f in p.iterdir() if f.suffix.lower() in IMG_EXTS])


def label_path_for(name):
    return os.path.join(STATE["label_dir"], Path(name).stem + ".txt")


def has_label(name):
    lp = label_path_for(name)
    if not os.path.isfile(lp):
        return False
    try:
        return os.path.getsize(lp) > 0
    except OSError:
        return False


def _read_label_boxes(lp):
    """读取 YOLO 标签文件,返回 [{cls,cx,cy,w,h}]。"""
    boxes = []
    if os.path.isfile(lp):
        for line in open(lp, "r", encoding="utf-8", errors="ignore"):
            line = line.strip()
            if not line:
                continue
            p = line.split()
            if len(p) < 5:
                continue
            try:
                boxes.append({"cls": int(float(p[0])),
                              "cx": float(p[1]), "cy": float(p[2]),
                              "w": float(p[3]), "h": float(p[4])})
            except ValueError:
                continue
    return boxes


def _iou_boxes(a, b):
    ax1, ay1 = a["cx"] - a["w"] / 2, a["cy"] - a["h"] / 2
    ax2, ay2 = a["cx"] + a["w"] / 2, a["cy"] + a["h"] / 2
    bx1, by1 = b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2
    bx2, by2 = b["cx"] + b["w"] / 2, b["cy"] + b["h"] / 2
    inter = (max(0, min(ax2, bx2) - max(ax1, bx1)) *
             max(0, min(ay2, by2) - max(ay1, by1)))
    union = ((ax2 - ax1) * (ay2 - ay1) +
             (bx2 - bx1) * (by2 - by1) - inter)
    return inter / union if union > 0 else 0


def _same_target(a, b, iou_thr):
    """判断两个同类框是否指向同一目标。
    IoU 超过阈值,或一个框几乎被另一个框包含(小框中心在大框内且面积占比足够大)
    都视为重复,避免不同模型框大小差异导致同一目标出多个框。"""
    if _iou_boxes(a, b) >= iou_thr:
        return True
    a_area = a["w"] * a["h"]
    b_area = b["w"] * b["h"]
    if a_area <= 0 or b_area <= 0:
        return False
    big, small = (a, b) if a_area >= b_area else (b, a)
    small_area = small["w"] * small["h"]
    big_area = big["w"] * big["h"]
    if small_area / big_area < 0.35:
        return False
    in_x = (big["cx"] - big["w"] / 2 <= small["cx"] <=
            big["cx"] + big["w"] / 2)
    in_y = (big["cy"] - big["h"] / 2 <= small["cy"] <=
            big["cy"] + big["h"] / 2)
    return in_x and in_y


def _validate_boxes(boxes, n_cls):
    """校验并格式化标签;返回 (lines, errors)。"""
    lines = []
    errors = []
    for i, b in enumerate(boxes, 1):
        try:
            cls = int(b["cls"])
            cx = float(b["cx"])
            cy = float(b["cy"])
            w = float(b["w"])
            h = float(b["h"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"第{i}个框字段缺失或不是数字")
            continue
        errs = []
        if n_cls and (cls < 0 or cls >= n_cls):
            errs.append(f"类别 {cls} 超出范围 (0~{n_cls - 1})")
        for k, v in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
            if not math.isfinite(v) or not (0 <= v <= 1):
                errs.append(f"{k}={v} 越界 (应为 0~1)")
        if w <= 0 or h <= 0:
            errs.append("宽高必须大于 0")
        if errs:
            errors.append(f"第{i}个框: " + "; ".join(errs))
            continue
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines, errors


def _write_label_file(lp, boxes):
    """把框列表写入 YOLO 标签文件。"""
    lines, errors = _validate_boxes(boxes, len(STATE["classes"]))
    if errors:
        raise ValueError("；".join(errors[:10]))
    with open(lp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")
    return lines


def load_config():
    if os.path.isfile(CONFIG_FILE):
        try:
            return json.load(open(CONFIG_FILE, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_config(cfg):
    try:
        json.dump(cfg, open(CONFIG_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception:
        pass


@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    if request.method == "POST":
        data = request.get_json() or {}
        cfg = load_config()
        if "device" in data:
            d = str(data.get("device") or "auto").strip().lower()
            cfg["device"] = d if d in ("auto", "cuda", "cpu") else "auto"
        save_config(cfg)
    cfg = dict(load_config())
    v = dict(cfg.get("vlm") or {})
    if v.get("api_key"):
        v["has_key"] = True
        v["api_key"] = ""      # 不回传明文 key 给浏览器
    cfg["vlm"] = v
    return jsonify(cfg)


@app.route("/api/device")
def api_device():
    """推理设备诊断：当前会用什么、显卡是否可用。"""
    info = {"ok": True, "want": (load_config().get("device") or "auto"),
            "current": torch_device(), "torch": None, "cuda": False, "gpu": None}
    if torch is not None:
        info["torch"] = torch.__version__
        try:
            info["cuda"] = bool(torch.cuda.is_available())
            if info["cuda"]:
                info["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return jsonify(info)


@app.route("/api/vlm_config", methods=["GET", "POST"])
def api_vlm_config():
    """读写多模态标注配置。api_key 只写不读，前端拿不到明文。"""
    if request.method == "GET":
        v = dict(vlm_config())
        if v.get("api_key"):
            v["has_key"] = True
            v["api_key"] = ""
        return jsonify({"ok": True, "vlm": v})
    data = request.get_json() or {}
    patch = {}
    for k in ("base_url", "model"):
        if data.get(k) is not None:
            patch[k] = str(data[k]).strip()
    if data.get("api_key"):
        patch["api_key"] = str(data["api_key"]).strip()
    if data.get("clear_key"):
        patch["api_key"] = ""
    if data.get("thinking") is not None:
        patch["thinking"] = bool(data["thinking"])
    if data.get("concurrency") is not None:
        try:
            patch["concurrency"] = max(1, min(32, int(data["concurrency"])))
        except (TypeError, ValueError):
            pass
    v = save_vlm_config(patch)
    out = dict(v)
    if out.get("api_key"):
        out["has_key"] = True
        out["api_key"] = ""
    return jsonify({"ok": True, "vlm": out})


@app.route("/api/vlm_test", methods=["POST"])
def api_vlm_test():
    """对指定图片跑一次多模态标注，只返回结果供预览，不写标签文件。"""
    if not STATE["img_dir"] or not STATE["images"]:
        return jsonify({"ok": False, "msg": "请先打开图片文件夹"})
    if not STATE["classes"]:
        return jsonify({"ok": False, "msg": "请先填写类别名"})
    data = request.get_json() or {}
    try:
        idx = int(data.get("idx", -1))
    except (TypeError, ValueError):
        idx = -1
    if idx < 0 or idx >= len(STATE["images"]):
        return jsonify({"ok": False, "msg": "图片下标越界"})
    iou = float(data.get("iou") or 0.5)
    path = os.path.join(STATE["img_dir"], STATE["images"][idx])
    boxes, stats = _predict_vlm(path, iou)
    if stats and stats[0].get("error"):
        return jsonify({"ok": False, "msg": stats[0]["error"]})
    return jsonify({"ok": True,
                    "boxes": [{"cls": b["cls"], "cx": b["cx"], "cy": b["cy"],
                               "w": b["w"], "h": b["h"],
                               "label": STATE["classes"][b["cls"]]} for b in boxes],
                    "stats": stats})


@app.route("/api/open", methods=["POST"])
def api_open():
    data = request.get_json()
    img_dir = data.get("img_dir", "").strip().strip('"')
    label_dir = data.get("label_dir", "").strip().strip('"')
    classes = data.get("classes", [])

    if not img_dir or not os.path.isdir(img_dir):
        return jsonify({"ok": False, "msg": f"图片文件夹不存在: {img_dir}"})
    if not label_dir:
        label_dir = img_dir
    os.makedirs(label_dir, exist_ok=True)

    imgs = list_images(img_dir)
    if not imgs:
        return jsonify({"ok": False, "msg": "该文件夹内没有图片"})

    STATE["img_dir"] = img_dir
    STATE["label_dir"] = label_dir
    STATE["images"] = imgs
    if classes:
        names, hints = [], {}
        for item in classes:
            nm, hint = _parse_class_spec(item)
            if not nm:
                continue
            names.append(nm)
            if hint:
                hints[nm] = hint
        STATE["classes"] = names
        STATE["class_hints"] = hints

    cfg = load_config()
    cfg.update({"img_dir": img_dir, "label_dir": label_dir,
                "classes": ",".join(classes) if classes
                else ",".join(STATE["classes"])})
    save_config(cfg)

    file_status = [{"name": n, "done": has_label(n)} for n in imgs]
    return jsonify({"ok": True, "count": len(imgs), "img_dir": img_dir,
                    "classes": STATE["classes"], "files": file_status})


@app.route("/api/files")
def api_files():
    files = [{"name": n, "done": has_label(n)} for n in STATE["images"]]
    return jsonify({"ok": True, "files": files,
                    "img_dir": STATE["img_dir"] or "",
                    "classes": STATE["classes"],
                    "auto": STATE.get("auto")})


def _pick_path(is_dir=False, title="选择"):
    """弹出原生选择窗口(本机工具专用),返回路径或 None。"""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        if is_dir:
            return filedialog.askdirectory(title=title, parent=root) or None
        return filedialog.askopenfilename(
            title=title, parent=root,
            filetypes=[("模型文件", "*.pt *.onnx *.ptq"),
                       ("所有文件", "*.*")]) or None
    finally:
        root.destroy()


@app.route("/api/pick_file", methods=["POST"])
def api_pick_file():
    path = _pick_path(False, "选择模型文件")
    if not path:
        return jsonify({"ok": False, "msg": "未选择文件或无法打开选择窗口"})
    return jsonify({"ok": True, "path": path})


@app.route("/api/pick_folder", methods=["POST"])
def api_pick_folder():
    path = _pick_path(True, "选择文件夹")
    if not path:
        return jsonify({"ok": False, "msg": "未选择文件夹或无法打开选择窗口"})
    return jsonify({"ok": True, "path": path})


def _model_device_str(m):
    """模型实际跑在哪个设备上（给界面显示）。"""
    try:
        return str(next(m["model"].parameters()).device)
    except Exception:
        try:
            return str(m["model"].device)
        except Exception:
            return "?"


def _model_info(m):
    return {"id": m["id"], "name": m["name"], "path": m["path"],
            "backend": m.get("backend"),
            "device": _model_device_str(m),
            "model_classes": m["model_classes"],
            "cls_offset": m["cls_offset"], "conf": m["conf"],
            "suggested_offset": _suggest_offset(m["model_classes"],
                                                STATE["classes"]),
            "auto_offset": m.get("auto_offset", False),
            "only_cls": sorted(m["only_cls"]) if m["only_cls"] is not None else None}


def _persist_model_meta():
    cfg = load_config()
    cfg["models"] = [{
        "path": m["path"],
        "backend": m.get("backend"),
        "cls_offset": m["cls_offset"],
        "conf": m["conf"],
        "only_cls": sorted(m["only_cls"]) if m["only_cls"] is not None else None,
    } for m in STATE["models"]]
    save_config(cfg)


@app.route("/api/models")
def api_models():
    return jsonify({"ok": True,
                    "models": [_model_info(m) for m in STATE["models"]]})


@app.route("/api/suggest_offset", methods=["POST"])
def api_suggest_offset():
    data = request.get_json() or {}
    try:
        mid = int(data.get("id", -1))
    except (TypeError, ValueError):
        mid = -1
    for m in STATE["models"]:
        if m["id"] == mid:
            sug = _suggest_offset(m["model_classes"], STATE["classes"])
            return jsonify({"ok": True, "suggested_offset": sug})
    return jsonify({"ok": False, "msg": "模型不存在"})


# --------------------------------------------------------------------------
# YOLOv5 (.pt) 支持
#
# YOLOv5 的 .pt 是「带类定义的 pickle」：里面直接引用了 yolov5 仓库自己的顶层
# 模块 models.*，缺了它就解不开，报 No module named 'models'。
# 这里在内存里造一套 models 兼容模块塞进 sys.modules，让 torch.load 能跑通，
# 再按官方 v7 公式自己做解码 + NMS。**不需要下载 yolov5 仓库**。
#
# 兼容层类名必须与官方一致（pickle 按「模块名.类名」查找）：
#   models.common     Conv / DWConv / Focus / Bottleneck / C3 / SPPF / Concat
#   models.yolo       Detect / Model / DetectionModel
#   models.experimental Ensemble
# --------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
except Exception:      # 没装 torch 时工具仍要能启动（只能手动标注）
    torch = None
    nn = None


if nn is not None:

    def _v5_autopad(k, p=None, d=1):
        if d > 1:
            k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p


    class _V5Conv(nn.Module):
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, d=1):
            super().__init__()
            self.conv = nn.Conv2d(c1, c2, k, s, _v5_autopad(k, p, d), groups=g,
                                  dilation=d, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.SiLU() if act is True else (
                act if isinstance(act, nn.Module) else nn.Identity())

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))


    class _V5DWConv(_V5Conv):
        def __init__(self, c1, c2, k=1, s=1, act=True):
            super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


    class _V5Focus(nn.Module):
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
            super().__init__()
            self.conv = _V5Conv(c1 * 4, c2, k, s, p, g, act)

        def forward(self, x):
            return self.conv(torch.cat(
                [x[..., ::2, ::2], x[..., 1::2, ::2],
                 x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


    class _V5Bottleneck(nn.Module):
        def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
            super().__init__()
            c_ = int(c2 * e)
            self.cv1 = _V5Conv(c1, c_, 1, 1)
            self.cv2 = _V5Conv(c_, c2, 3, 1, g=g)
            self.add = shortcut and c1 == c2

        def forward(self, x):
            return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


    class _V5C3(nn.Module):
        def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
            super().__init__()
            c_ = int(c2 * e)
            self.cv1 = _V5Conv(c1, c_, 1, 1)
            self.cv2 = _V5Conv(c1, c_, 1, 1)
            self.cv3 = _V5Conv(2 * c_, c2, 1)
            self.m = nn.Sequential(
                *(_V5Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

        def forward(self, x):
            return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


    class _V5SPPF(nn.Module):
        def __init__(self, c1, c2, k=5):
            super().__init__()
            c_ = c1 // 2
            self.cv1 = _V5Conv(c1, c_, 1, 1)
            self.cv2 = _V5Conv(c_ * 4, c2, 1, 1)
            self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        def forward(self, x):
            x = self.cv1(x)
            y1 = self.m(x)
            y2 = self.m(y1)
            y3 = self.m(y2)
            return self.cv2(torch.cat((x, y1, y2, y3), 1))


    class _V5Concat(nn.Module):
        def __init__(self, dimension=1):
            super().__init__()
            self.d = dimension

        def forward(self, x):
            return torch.cat(x, self.d)


    class _V5Detect(nn.Module):
        """YOLOv5 检测头。注意 v7 的网格带 -0.5 偏移，与下面 forward 配套；
        官方 v6 的写法（网格不偏移、forward 里 -0.5）数学上等价。"""
        stride = None
        dynamic = False
        export = False

        def __init__(self, nc=80, anchors=(), ch=()):
            super().__init__()
            self.nc = nc
            self.no = nc + 5
            self.nl = len(anchors)
            self.na = len(anchors[0]) // 2 if self.nl else 3
            self.grid = [torch.empty(0) for _ in range(self.nl)]
            self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]
            self.register_buffer(
                "anchors",
                torch.tensor(anchors).float().view(self.nl, -1, 2))
            self.m = nn.ModuleList(
                nn.Conv2d(x, self.no * self.na, 1) for x in ch)
            self.inplace = True

        def _make_grid(self, nx=20, ny=20, i=0):
            d = self.anchors[i].device
            t = self.anchors[i].dtype
            shape = 1, self.na, ny, nx, 2
            y = torch.arange(ny, device=d, dtype=t)
            x = torch.arange(nx, device=d, dtype=t)
            yv, xv = torch.meshgrid(y, x, indexing="ij")
            grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
            anchor_grid = (self.anchors[i] * self.stride[i]).view(
                (1, self.na, 1, 1, 2)).expand(shape)
            return grid, anchor_grid

        def forward(self, x):
            z = []
            for i in range(self.nl):
                x[i] = self.m[i](x[i])
                bs, _, ny, nx = x[i].shape
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(
                    0, 1, 3, 4, 2).contiguous()
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]
                wh = (wh * 2) ** 2 * self.anchor_grid[i]
                z.append(torch.cat((xy, wh, conf), 4).view(
                    bs, self.na * nx * ny, self.no))
            return torch.cat(z, 1)


    class _V5Model(nn.Module):
        """YOLOv5 主干容器：pickle 兼容 + 可真正前向。"""
        def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):
            super().__init__()
            self.model = nn.Sequential()
            self.save = []
            self.names = []

        def forward(self, x):
            y = []
            for m in self.model:
                if m.f != -1:
                    x = (y[m.f] if isinstance(m.f, int)
                         else [x if j == -1 else y[j] for j in m.f])
                x = m(x)
                y.append(x if m.i in self.save else None)
            return x


    class _V5Ensemble(nn.ModuleList):
        def __init__(self):
            super().__init__()

        def forward(self, x, augment=False):
            return torch.cat([m(x, augment)[0] for m in self], 1), None


def torch_device():
    """推理设备。配置 device = auto / cuda / cpu。
    auto: 有可用显卡就用显卡，否则 CPU。"""
    want = str(load_config().get("device") or "auto").strip().lower()
    if torch is None:
        return "cpu"
    if want.startswith("cpu"):
        return "cpu"
    if want.startswith(("cuda", "gpu")):
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _install_yolov5_shim():
    """把 models.* 兼容模块注册进 sys.modules（只做一次）。"""
    if nn is None:
        raise RuntimeError("加载模型需要 PyTorch，请先执行: "
                           "pip install -r requirements-full.txt")
    cur = sys.modules.get("models")
    if cur is not None and getattr(cur, "_v5shim", False):
        return
    common = types.ModuleType("models.common")
    yolo = types.ModuleType("models.yolo")
    experimental = types.ModuleType("models.experimental")
    pkg = types.ModuleType("models")
    pkg._v5shim = True
    mapping = {
        common: (("Conv", _V5Conv), ("DWConv", _V5DWConv), ("Focus", _V5Focus),
                 ("Bottleneck", _V5Bottleneck), ("C3", _V5C3),
                 ("SPPF", _V5SPPF), ("Concat", _V5Concat)),
        yolo: (("Detect", _V5Detect), ("Model", _V5Model),
               ("DetectionModel", _V5Model)),
        experimental: (("Ensemble", _V5Ensemble),),
    }
    for mod, items in mapping.items():
        for name, cls in items:
            setattr(mod, name, cls)
    pkg.common, pkg.yolo, pkg.experimental = common, yolo, experimental
    sys.modules["models"] = pkg
    sys.modules["models.common"] = common
    sys.modules["models.yolo"] = yolo
    sys.modules["models.experimental"] = experimental


def _load_yolov5(model_path):
    """加载 YOLOv5 .pt，返回 (模型, 类别名列表)。"""
    if nn is None:
        raise RuntimeError("加载模型需要 PyTorch，请先执行: "
                           "pip install -r requirements-full.txt")
    _install_yolov5_shim()
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or not (ck.get("ema") is not None
                                        or ck.get("model") is not None):
        raise RuntimeError("不是 YOLOv5 checkpoint（缺少 model/ema）")
    model = ck.get("ema") if ck.get("ema") is not None else ck.get("model")
    model = model.float().eval()
    dev = torch.device(torch_device())
    model = model.to(dev)
    names = getattr(ck.get("model"), "names", None) or getattr(model, "names", None)
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]
    if not names:
        nc = getattr(ck.get("model"), "nc", 0) or getattr(model, "nc", 0)
        names = ["class_%d" % i for i in range(int(nc))]
    return model, [str(n) for n in names]


def _v5_letterbox(img, size=640, color=114):
    w, h = img.size
    r = min(size / float(w), size / float(h))
    nw = max(1, int(round(w * r)))
    nh = max(1, int(round(h * r)))
    canvas = Image.new("RGB", (size, size), (color, color, color))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    return canvas, r, px, py


def _v5_to_tensor(canvas):
    """PIL -> NCHW float 张量。用 frombuffer 避免依赖 numpy。"""
    buf = bytearray(canvas.tobytes())
    t = torch.frombuffer(buf, dtype=torch.uint8)
    w, h = canvas.size
    t = t.view(h, w, 3).permute(2, 0, 1).contiguous()
    return t.float().div(255.0).unsqueeze(0)


def _v5_nms(boxes, scores, iou_thr):
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = torch.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = torch.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = torch.minimum(boxes[i, 3], boxes[rest, 3])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = ((boxes[rest, 2] - boxes[rest, 0])
             * (boxes[rest, 3] - boxes[rest, 1]))
        iou = inter / (a + b - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return torch.tensor(keep, dtype=torch.long)


def _infer_yolov5(model, img_path, conf, iou=0.45, size=640):
    """YOLOv5 推理，返回归一化 [{cls,cx,cy,w,h,conf}]，已做类内 NMS。"""
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    canvas, r, px, py = _v5_letterbox(img, size)
    x = _v5_to_tensor(canvas).to(next(model.parameters()).device)
    with torch.no_grad():
        out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    pred = out[0]
    pred[:, 5:] *= pred[:, 4:5]          # 置信度 = 目标性 × 类别概率
    score, cls = pred[:, 5:].max(1)
    keep = score > conf
    pred, score, cls = pred[keep], score[keep], cls[keep]
    if pred.numel() == 0:
        return []
    xy, wh = pred[:, :2], pred[:, 2:4]
    box = torch.cat((xy - wh / 2, xy + wh / 2), 1)
    parts = []
    for c in cls.unique():
        m = cls == c
        k = _v5_nms(box[m], score[m], iou)
        parts.append((box[m][k], score[m][k], c.repeat(k.numel())))
    box = torch.cat([p[0] for p in parts])
    score = torch.cat([p[1] for p in parts])
    cls = torch.cat([p[2] for p in parts])
    out_list = []
    for i in range(box.shape[0]):
        x1 = (box[i, 0].item() - px) / r
        y1 = (box[i, 1].item() - py) / r
        x2 = (box[i, 2].item() - px) / r
        y2 = (box[i, 3].item() - py) / r
        x1, y1 = min(max(x1, 0.0), W), min(max(y1, 0.0), H)
        x2, y2 = min(max(x2, 0.0), W), min(max(y2, 0.0), H)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        out_list.append({
            "cls": int(cls[i]), "conf": float(score[i]),
            "cx": ((x1 + x2) / 2.0) / W, "cy": ((y1 + y2) / 2.0) / H,
            "w": (x2 - x1) / W, "h": (y2 - y1) / H})
    return out_list


def _load_local_model(model_path):
    """非 ultralytics 格式的权重：先按 YOLOv5 checkpoint 试，再按美团 v6n 试。
    返回 (模型, 类别名列表, backend)。"""
    errors = []
    try:
        model, classes = _load_yolov5(model_path)
        return model, classes, "yolov5"
    except Exception as e:
        errors.append("YOLOv5: %s" % e)
    try:
        model = _load_meituan_v6n(model_path)
        return model, _meituan_class_names(model_path, model), "meituan_v6n"
    except Exception as e:
        errors.append("美团 v6n: %s" % e)
    raise RuntimeError("；".join(errors))


def _load_meituan_v6n(model_path):
    """加载美团 YOLOv6n checkpoint（非 ultralytics 格式）。
    YOLOv6 源码目录优先级: 环境变量 YOLOV6_REPO > 配置 v6n_repo > 程序目录下 YOLOv6。"""
    import torch
    repo = (os.environ.get("YOLOV6_REPO")
            or (load_config().get("v6n_repo") or "")
            or os.path.join(os.path.dirname(os.path.abspath(__file__)), "YOLOv6"))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from yolov6.models import effidehead
    from yolov6.layers.common import SimConv
    effidehead.Conv = SimConv
    ck = torch.load(model_path, map_location="cpu")
    model = ck.get("model") or ck.get("ema")
    if model is None:
        raise RuntimeError("checkpoint 里没有 model/ema")
    if hasattr(model, "module"):
        model = model.module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return model.float().to(device).eval()


def _read_yaml_names(path, nc):
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        names = data.get("names")
        if isinstance(names, dict):
            names = [names.get(i, "class_%d" % i) for i in range(nc)]
        elif isinstance(names, (list, tuple)):
            names = [str(n) for n in names[:nc]]
            while len(names) < nc:
                names.append("class_%d" % len(names))
        else:
            names = None
        return names
    except Exception:
        return None


def _meituan_class_names(model_path, model):
    nc = None
    for attr in ("detect", "model"):
        try:
            nc = int(getattr(model, attr).nc)
            break
        except Exception:
            continue
    if nc is None:
        nc = 0
    d = os.path.dirname(os.path.abspath(model_path))
    for _ in range(8):
        for fn in ("data_meituan.yaml", "data.yaml"):
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                names = _read_yaml_names(p, nc)
                if names:
                    return names
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return ["class_%d" % i for i in range(nc)]


def _infer_meituan(model, img_path, conf):
    """美团 v6n 推理，返回归一化 [{cls,cx,cy,w,h,conf}]，已做类内 NMS。"""
    import cv2
    import numpy as np
    import torch
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError("无法读取图片")
    h0, w0 = img.shape[:2]
    S = 640
    scale = min(S / h0, S / w0)
    nw, nh = int(w0 * scale), int(h0 * scale)
    resized = cv2.resize(img, (nw, nh))
    padded = np.full((S, S, 3), 114, dtype=np.uint8)
    dx, dy = (S - nw) // 2, (S - nh) // 2
    padded[dy:dy + nh, dx:dx + nw] = resized
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    x = np.ascontiguousarray(
        np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None])
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(torch.from_numpy(x).float().to(device))
    det = out[0][0].cpu().numpy()
    if det.shape[0] == 0:
        return []
    boxes = det[:, :4]
    scores = det[:, 5:]
    confs = scores.max(axis=1)
    clss = scores.argmax(axis=1)
    keep = confs >= conf
    boxes, confs, clss = boxes[keep], confs[keep], clss[keep]
    if len(boxes) == 0:
        return []
    cx = (boxes[:, 0] - dx) / scale / w0
    cy = (boxes[:, 1] - dy) / scale / h0
    ww = boxes[:, 2] / scale / w0
    hh = boxes[:, 3] / scale / h0
    raw = [{"cls": int(clss[i]), "cx": float(cx[i]), "cy": float(cy[i]),
            "w": float(ww[i]), "h": float(hh[i]), "conf": float(confs[i])}
           for i in range(len(boxes))]
    return _nms_boxes(raw, 0.5)


def _norm_cls_name(s):
    """类别名归一化:去掉空格/下划线/连字符并转小写。"""
    return re.sub(r"[\s_\-]+", "", str(s)).lower()


def _suggest_offset(model_classes, dataset_classes):
    """按类别名自动推断类别偏移。
    模型类别必须能在数据集类别里按顺序连续匹配;否则返回 None。"""
    if not model_classes:
        return None
    dmap = {_norm_cls_name(c): i for i, c in enumerate(dataset_classes)}
    hits = [dmap.get(_norm_cls_name(c)) for c in model_classes]
    if any(h is None for h in hits):
        return None
    base = hits[0]
    if hits == list(range(base, base + len(hits))):
        return base
    return None


# --------------------------------------------------------------------------
# 多模态（VLM）辅助标注
# 走 OpenAI 兼容的 chat/completions 接口，图片以 base64 内联，
# 要求模型按固定 JSON 结构返回像素坐标的边界框。
# --------------------------------------------------------------------------

VLM_DEFAULTS = {
    "base_url": "",
    "model": "",
    "api_key": "",
    "thinking": True,        # 关掉会显著退化：框偏移、还会静默返回空
    "concurrency": 4,
    "timeout": 180,
    "conf": 0.6,
    "retries": 3,            # 瞬时故障(503/连接重置/空回复)的重试次数
}


def vlm_config():
    """读取多模态配置。环境变量 VLM_API_KEY 优先于配置文件。"""
    v = dict(VLM_DEFAULTS)
    saved = load_config().get("vlm") or {}
    for k in VLM_DEFAULTS:
        if saved.get(k) is not None:
            v[k] = saved[k]
    env_key = os.environ.get("VLM_API_KEY")
    if env_key:
        v["api_key"] = env_key
    return v


def save_vlm_config(patch):
    cfg = load_config()
    v = dict(VLM_DEFAULTS)
    v.update(cfg.get("vlm") or {})
    v.update(patch or {})
    cfg["vlm"] = v
    save_config(cfg)
    return v


VLM_PROMPT = """你是图像标注助手。只输出 JSON，不要输出解释文字。

找出图中所有可以确定的目标，并为每个目标给出边界框。

类别只能是以下这些，label 必须原样使用我给出的类别名：
{classes}
{hints}

输出 JSON，格式必须是：
{{
  "image_width": <我发给你的这张图的宽度像素>,
  "image_height": <高度像素>,
  "boxes": [
    {{"label": "类别名", "x1": <左上角x>, "y1": <左上角y>, "x2": <右下角x>, "y2": <右下角y>}}
  ]
}}

要求：
1. 坐标必须是基于我发给你的这张图的像素坐标，不是归一化坐标，也不是 0-1000 的比例坐标。
2. 框必须贴住画面里真实存在的物体或人体部位，不要框空地、地面、墙面、天空、阴影这类背景。
3. 如果类别名表示"缺少某物"（例如 no-helmet），就框"本该有它的那个部位"，不要框整个场景。
4. 只标你确实看到、并且能明确对应到具体物体的目标；对不上就不要输出，不要猜、不要编造。
5. 没有任何目标时 boxes 返回空数组 []。
"""


def _vlm_prompt(classes, hints=None):
    lines = "\n".join("- %s" % c for c in classes)
    hints = hints or {}
    items = ["- %s：%s" % (c, hints[c]) for c in classes if hints.get(c)]
    hint_block = ""
    if items:
        hint_block = ("\n定位提示（只决定框在哪里，label 仍然只写类别名）：\n"
                      + "\n".join(items) + "\n")
    return VLM_PROMPT.format(classes=lines, hints=hint_block)


# 名称和提示都不允许再含括号；匹配不上就整个当类别名，避免误伤
_CLASS_HINT_RE = re.compile(r"^\s*([^（()）]+?)\s*[（(]\s*([^（()）]+?)\s*[)）]\s*$")


def _parse_class_spec(raw):
    """类别项支持写「名称(定位提示)」：提示只进提示词，不进类别名。
    例如 no-helmet(头部) -> 类别名 no-helmet，定位提示 头部。"""
    s = str(raw).strip()
    m = _CLASS_HINT_RE.match(s)
    if m and m.group(1).strip():
        return m.group(1).strip(), m.group(2).strip()
    return s, None


def _http_post_json(url, payload, headers, timeout, proxy=None):
    """用标准库发 POST，避免为一个可选功能引入 requests 依赖。"""
    import urllib.error
    import urllib.request
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    opener = (urllib.request.build_opener(urllib.request.ProxyHandler(proxy))
              if proxy else urllib.request.build_opener())
    try:
        with opener.open(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        return e.code, {"error": detail}
    except Exception as e:
        return 0, {"error": str(e)}


def _extract_json(text):
    """从模型回复里抠出 JSON（对象或数组）。
    模型偶尔会先吐一个 {"type":"json_object"} 再吐真结果，
    所以要扫出全部顶层对象，优先返回带 boxes 的那个。"""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        obj = json.loads(t)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass
    found = []
    for start, ch in enumerate(t):
        if ch != "{":
            continue
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(t[start:i + 1])
                        if isinstance(obj, dict):
                            found.append(obj)
                    except Exception:
                        pass
                    break
    for obj in found:
        if "boxes" in obj or "image_width" in obj:
            return obj
    if found:
        return found[0]
    # 再找一次顶层数组（部分模型直接返回 [...]）
    start = t.find("[")
    while start >= 0:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "[":
                depth += 1
            elif t[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        arr = json.loads(t[start:i + 1])
                        if isinstance(arr, list):
                            return arr
                    except Exception:
                        pass
                    break
        start = t.find("[", start + 1)
    return None


def _vlm_map_label(label, classes):
    """把模型返回的类别名映射到数据集类别下标。
    先精确匹配(忽略大小写/空格/下划线)，再退化为"最长被包含"匹配。
    匹配不上返回 None —— 宁可漏，也不要把 no-helmet 猜成 helmet。"""
    if label is None:
        return None
    if isinstance(label, int):
        return label if 0 <= label < len(classes) else None
    s = _norm_cls_name(label)
    if not s:
        return None
    dmap = {}
    for i, c in enumerate(classes):
        key = _norm_cls_name(c)
        if key and key not in dmap:
            dmap[key] = i
    if s in dmap:
        return dmap[s]
    best_i, best_len = None, 0
    for key, i in dmap.items():
        if len(key) >= 3 and key in s and len(key) > best_len:
            best_i, best_len = i, len(key)
    return best_i


def _vlm_request(v, payload):
    """发一次多模态请求，含瞬时故障退避重试。
    返回 (obj, usage, error)；obj 可能是 dict 或 list。
    服务端过载(503)、限流(429)、连接被重置、响应被截断、返回空内容都重试；
    合法的"没有目标"不重试，401 这类硬错误立刻失败。"""
    url = v["base_url"].rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer %s" % v["api_key"]}
    timeout = float(v["timeout"] or 180)
    RETRYABLE = (0, 408, 409, 425, 429, 500, 502, 503, 504)
    attempts = max(1, int(v.get("retries") or 1))
    obj, usage, last_err = None, None, None
    for attempt in range(attempts):
        if attempt:
            time.sleep(min(20.0, 2.0 * (2 ** (attempt - 1)))
                       + random.uniform(0, 1.5))
        status, data = _http_post_json(url, payload, headers, timeout)
        if status != 200:
            last_err = "HTTP %s: %s" % (status, str(data.get("error"))[:300])
            if status in RETRYABLE:
                continue
            break
        usage = data.get("usage")
        choices = data.get("choices") or []
        content = (choices[0].get("message", {}).get("content") or "") if choices else ""
        cand = _extract_json(content)
        if isinstance(cand, (dict, list)):
            obj, last_err = cand, None
            break
        last_err = "模型未返回可解析的 JSON: %s" % content[:200]
    return obj, usage, last_err


def _infer_vlm(img_path, classes, vcfg=None, hints=None):
    """调用多模态模型标注单张图。
    返回 (norm_boxes, meta)；norm_boxes 为归一化 [{cls,cx,cy,w,h,conf}]，
    meta 含 raw/unmapped/dropped/usage/error 等诊断信息。"""
    import base64
    v = vcfg or vlm_config()
    meta = {"raw": 0, "unmapped": 0, "dropped": 0, "usage": None, "error": None}
    if not v["api_key"]:
        meta["error"] = "未配置 API Key"
        return [], meta
    if not v["base_url"]:
        meta["error"] = "未配置接口地址"
        return [], meta
    if not v["model"]:
        meta["error"] = "未配置模型名"
        return [], meta
    if not classes:
        meta["error"] = "数据集类别为空，请先在配置里填写类别名"
        return [], meta

    with Image.open(img_path) as im:
        W, H = im.size
        fmt = (im.format or "JPEG").upper()
    mime = {"PNG": "image/png", "WEBP": "image/webp",
            "GIF": "image/gif", "BMP": "image/bmp"}.get(fmt, "image/jpeg")
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    payload = {
        "model": v["model"],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": _vlm_prompt(classes, hints)},
                {"type": "image_url",
                 "image_url": {"url": "data:%s;base64,%s" % (mime, b64),
                               "detail": "original"}},
            ],
        }],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "enabled" if v["thinking"] else "disabled"},
        "max_tokens": 4096,
    }
    obj, usage, last_err = _vlm_request(v, payload)
    meta["usage"] = usage
    if last_err or not isinstance(obj, dict):
        meta["error"] = last_err or "模型未返回可解析的 JSON"
        return [], meta

    # 模型自报尺寸与实际不符时按比例换算，避免整套框系统性偏移
    sx = sy = 1.0
    try:
        rw = float(obj.get("image_width") or 0)
        rh = float(obj.get("image_height") or 0)
        if rw > 0 and rh > 0:
            sx, sy = W / rw, H / rh
    except (TypeError, ValueError):
        pass

    out = []
    for b in (obj.get("boxes") or []):
        if not isinstance(b, dict):
            continue
        meta["raw"] += 1
        cls = _vlm_map_label(b.get("label"), classes)
        if cls is None:
            meta["unmapped"] += 1
            continue
        try:
            x1, y1 = float(b["x1"]) * sx, float(b["y1"]) * sy
            x2, y2 = float(b["x2"]) * sx, float(b["y2"]) * sy
        except (KeyError, TypeError, ValueError):
            meta["dropped"] += 1
            continue
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x1 = min(max(x1, 0.0), float(W))
        y1 = min(max(y1, 0.0), float(H))
        x2 = min(max(x2, 0.0), float(W))
        y2 = min(max(y2, 0.0), float(H))
        w, h = (x2 - x1) / W, (y2 - y1) / H
        if w <= 0 or h <= 0:
            meta["dropped"] += 1
            continue
        out.append({"cls": cls,
                    "cx": ((x1 + x2) / 2.0) / W,
                    "cy": ((y1 + y2) / 2.0) / H,
                    "w": w, "h": h,
                    "conf": float(v["conf"])})
    return out, meta


def _predict_vlm(path, iou_thr=0.5, classes=None, vcfg=None, hints=None):
    """多模态单图预测；返回 (boxes, stats)，结构与 _predict_all 一致。"""
    cls_list = classes if classes is not None else STATE["classes"]
    if hints is None:
        hints = STATE.get("class_hints") or {}
    v = vcfg or vlm_config()
    name = "多模态(%s)" % v["model"]
    try:
        preds, meta = _infer_vlm(path, cls_list, v, hints)
    except Exception as e:
        return [], [{"name": name, "raw": 0, "kept": 0, "error": str(e)}]
    err = meta.get("error")
    if err:
        return [], [{"name": name, "raw": 0, "kept": 0, "error": err}]
    kept = _nms_boxes(preds, iou_thr)
    note = None
    if meta["unmapped"]:
        note = "%d 个框类别名未匹配已丢弃" % meta["unmapped"]
    if meta["dropped"]:
        note = ((note + "，") if note else "") + "%d 个退化框已丢弃" % meta["dropped"]
    return kept, [{"name": name, "raw": meta["raw"], "kept": len(kept),
                   "error": None, "note": note, "usage": meta.get("usage")}]




VLM_REVIEW_PROMPT = """你在复核一批已有的标注框。图上已经用红框标出了每个候选框，
并在框的左上角标了编号（0, 1, 2 ...）。请判断每个编号的框里到底是什么。

类别只能是以下这些，label 必须原样使用我给出的类别名：
{classes}
{hints}
请判断每个候选区域到底是什么，输出 JSON 数组：
[
  {{"id": 0, "label": "类别名"}},
  {{"id": 1, "label": "drop"}}
]

要求：
1. id 必须和候选区域编号一一对应，一个都不能少、不能重复、不能新增。
2. 如果这个区域看不清、或不属于任何类别（框错了、框到背景了），label 填 "drop"。
3. 只需要判断类别，不要输出坐标。
4. 注意看编号旁边的区域，小目标要看仔细。
5. 不要解释，只输出 JSON 数组。
"""


def _review_prompt(classes, hints=None):
    lines = "\n".join("- %s" % c for c in classes)
    hints = hints or {}
    items = ["- %s：%s" % (c, hints[c]) for c in classes if hints.get(c)]
    block = ""
    if items:
        block = ("\n定位提示（只帮你判断，label 仍然只写类别名）：\n"
                 + "\n".join(items) + "\n")
    return VLM_REVIEW_PROMPT.format(classes=lines, hints=block)


def _review_font(size):
    from PIL import ImageFont
    for name in ("msyh.ttc", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _render_review_image(img, boxes):
    """把候选框和编号画到原图上。让模型看着带编号的原图回答，
    既保住完整分辨率（小目标不被裁糊），又不需要它输出坐标。"""
    from PIL import ImageDraw
    vis = img.copy()
    d = ImageDraw.Draw(vis)
    W, H = vis.size
    lw = max(2, int(min(W, H) * 0.005))
    fs = max(14, int(min(W, H) * 0.05))
    font = _review_font(fs)
    for i, b in enumerate(boxes):
        x1 = (b["cx"] - b["w"] / 2.0) * W
        y1 = (b["cy"] - b["h"] / 2.0) * H
        x2 = (b["cx"] + b["w"] / 2.0) * W
        y2 = (b["cy"] + b["h"] / 2.0) * H
        d.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=lw)
        tag = str(i)
        tw = int(fs * 0.62) * len(tag) + 8
        tx = min(max(x1, 0), max(0, W - tw))
        ty = max(0, y1 - fs - 6)
        d.rectangle([tx, ty, tx + tw, ty + fs + 6], fill=(255, 40, 40))
        d.text((tx + 4, ty + 2), tag, fill=(255, 255, 255), font=font)
    return vis


def _img_to_data_url(img, fmt="JPEG", quality=88):
    import base64
    import io as _io
    buf = _io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=quality)
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return "data:%s;base64,%s" % (
        mime, base64.b64encode(buf.getvalue()).decode("ascii"))


def _vlm_review_image(img_path, boxes, classes, vcfg=None, hints=None,
                      drop_allowed=True):
    """复核一张图里已有的框。模型只判断每个裁剪块属于哪个类别，或该删。
    返回 (verdicts, meta)；verdicts: {下标: ("keep"|"reclass"|"drop", 类别下标或 None)}"""
    v = vcfg or vlm_config()
    meta = {"raw": 0, "unparsed": 0, "usage": None, "error": None}
    if not v["api_key"] or not v["base_url"] or not v["model"]:
        meta["error"] = "多模态接口未配置完整（地址/模型名/Key）"
        return {}, meta
    if not boxes:
        return {}, meta

    with Image.open(img_path) as im:
        im = im.convert("RGB")
        vis = _render_review_image(im, boxes)
        content = [
            {"type": "text", "text": _review_prompt(classes, hints)},
            {"type": "image_url",
             "image_url": {"url": _img_to_data_url(vis), "detail": "original"}},
        ]
    kept_idx = list(range(len(boxes)))

    payload = {
        "model": v["model"],
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "enabled" if v["thinking"] else "disabled"},
        "max_tokens": 4096,
    }
    obj, usage, err = _vlm_request(v, payload)
    meta["usage"] = usage
    if err:
        meta["error"] = err
        return {}, meta

    items = obj
    if isinstance(obj, dict):
        for k in ("results", "boxes", "items", "verdicts"):
            if isinstance(obj.get(k), list):
                items = obj[k]
                break
        else:
            items = []
    if not isinstance(items, list):
        meta["error"] = "复核结果不是 JSON 数组"
        return {}, meta

    verdicts = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        meta["raw"] += 1
        try:
            idx = int(it.get("id"))
        except (TypeError, ValueError):
            continue
        if idx not in kept_idx:
            continue
        lab = it.get("label")
        if isinstance(lab, str) and _norm_cls_name(lab) in ("drop", "delete",
                                                           "none", "删除"):
            verdicts[idx] = ("drop" if drop_allowed else "keep", None)
            continue
        cls = _vlm_map_label(lab, classes)
        if cls is None:
            verdicts[idx] = ("unknown", None)
            continue
        verdicts[idx] = ("keep" if boxes[idx]["cls"] == cls else "reclass", cls)
    meta["unparsed"] = len(kept_idx) - len(verdicts)
    return verdicts, meta


def _apply_review(boxes, verdicts, classes):
    """把复核结论套回框列表。没拿到结论的框一律保持原样（宁可不动，不可误删）。"""
    out, changes = [], []
    for i, b in enumerate(boxes):
        act, cls = verdicts.get(i, ("unknown", None))
        if act == "drop":
            changes.append({"i": i, "action": "drop",
                            "from": classes[b["cls"]]})
            continue
        if act == "reclass" and cls is not None:
            changes.append({"i": i, "action": "reclass",
                            "from": classes[b["cls"]], "to": classes[cls]})
            nb = dict(b)
            nb["cls"] = cls
            out.append(nb)
            continue
        out.append(b)
    return out, changes


@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    data = request.get_json() or {}
    model_path = data.get("model_path", "").strip().strip('"')
    if not model_path or not os.path.isfile(model_path):
        return jsonify({"ok": False, "msg": f"模型文件不存在: {model_path}"})
    norm_path = os.path.normcase(os.path.abspath(model_path))
    for m in STATE["models"]:
        if os.path.normcase(os.path.abspath(m["path"])) == norm_path:
            return jsonify({"ok": False,
                            "msg": f"该模型已加载: {m['name']}，请勿重复添加"})
    try:
        from ultralytics import YOLO
    except ImportError:
        YOLO = None      # 没有 ultralytics 也能加载 YOLOv5 / 美团 v6n 权重
    try:
        model = model_classes = backend = None
        if YOLO is not None:
            try:
                model = YOLO(model_path)
                names = model.names
                if isinstance(names, dict):
                    model_classes = [names[i] for i in sorted(names.keys())]
                else:
                    model_classes = list(names)
                try:      # 按配置把 ultralytics 模型放到指定设备
                    model.to(torch_device())
                except Exception:
                    pass
                backend = "ultralytics"
            except Exception:
                backend = None
        if backend is None:
            model, model_classes, backend = _load_local_model(model_path)
        mid = 1
        if STATE["models"]:
            mid = max(m["id"] for m in STATE["models"]) + 1
        raw_offset = data.get("cls_offset")
        if raw_offset is None:
            suggested = _suggest_offset(model_classes, STATE["classes"])
            cls_offset = suggested if suggested is not None else 0
            auto_offset = suggested is not None
        else:
            cls_offset = int(raw_offset or 0)
            auto_offset = False
        entry = {
            "id": mid,
            "name": os.path.basename(model_path),
            "path": model_path,
            "model": model,
            "backend": backend,
            "model_classes": model_classes,
            "cls_offset": cls_offset,
            "auto_offset": auto_offset,
            "conf": float(data.get("conf", 0.25) or 0.25),
            "only_cls": _parse_only_cls(data.get("only_cls")),
        }
        STATE["models"].append(entry)
        _persist_model_meta()
        return jsonify({"ok": True,
                        "models": [_model_info(m) for m in STATE["models"]]})
    except Exception as e:
        return jsonify({"ok": False, "msg": f"加载失败: {e}"})


@app.route("/api/update_model", methods=["POST"])
def api_update_model():
    data = request.get_json() or {}
    try:
        mid = int(data.get("id", -1))
    except (TypeError, ValueError):
        mid = -1
    for m in STATE["models"]:
        if m["id"] == mid:
            if "cls_offset" in data:
                m["cls_offset"] = int(data["cls_offset"] or 0)
            if "conf" in data:
                m["conf"] = float(data["conf"] or 0.25)
            if "only_cls" in data:
                m["only_cls"] = _parse_only_cls(data["only_cls"])
            _persist_model_meta()
            return jsonify({"ok": True,
                            "models": [_model_info(x) for x in STATE["models"]]})
    return jsonify({"ok": False, "msg": "模型不存在"})


@app.route("/api/unload_model", methods=["POST"])
def api_unload_model():
    data = request.get_json() or {}
    try:
        mid = int(data.get("id", -1))
    except (TypeError, ValueError):
        mid = -1
    STATE["models"] = [m for m in STATE["models"] if m["id"] != mid]
    _persist_model_meta()
    return jsonify({"ok": True,
                    "models": [_model_info(m) for m in STATE["models"]]})


@app.route("/api/image/<int:idx>")
def api_image(idx):
    if idx < 0 or idx >= len(STATE["images"]):
        return "", 404
    return send_file(os.path.join(STATE["img_dir"], STATE["images"][idx]))


@app.route("/api/meta/<int:idx>")
def api_meta(idx):
    if idx < 0 or idx >= len(STATE["images"]):
        return jsonify({"ok": False})
    name = STATE["images"][idx]
    path = os.path.join(STATE["img_dir"], name)
    with Image.open(path) as im:
        w, h = im.size
    boxes = _read_label_boxes(label_path_for(name))
    return jsonify({"ok": True, "name": name, "width": w, "height": h,
                    "boxes": boxes, "total": len(STATE["images"]), "idx": idx})


@app.route("/api/save/<int:idx>", methods=["POST"])
def api_save(idx):
    if idx < 0 or idx >= len(STATE["images"]):
        return jsonify({"ok": False})
    boxes = request.get_json().get("boxes", [])
    name = STATE["images"][idx]
    lp = label_path_for(name)
    lines, errors = _validate_boxes(boxes, len(STATE["classes"]))
    if errors:
        shown = errors[:10]
        msg = "；".join(shown) + ("…" if len(errors) > 10 else "")
        return jsonify({"ok": False, "msg": msg, "errors": errors})
    _write_label_file(lp, boxes)
    return jsonify({"ok": True, "count": len(lines), "done": len(lines) > 0})


def _parse_only_cls(raw):
    """解析前端传来的模型类别过滤列表。
    None = 不过滤(保留全部类别)；[] = 全部过滤(不输出任何框)。"""
    if raw is None:
        return None
    if isinstance(raw, (str, int, float)):
        raw = [raw]
    try:
        return set(int(x) for x in raw)
    except (TypeError, ValueError):
        return None


@app.route("/api/predict/<int:idx>", methods=["POST"])
def api_predict(idx):
    if not STATE["models"]:
        return jsonify({"ok": False, "msg": "未加载任何模型"})
    if idx < 0 or idx >= len(STATE["images"]):
        return jsonify({"ok": False})
    data = request.get_json() or {}
    iou_thr = float(data.get("iou", 0.5) or 0.5)
    path = os.path.join(STATE["img_dir"], STATE["images"][idx])
    boxes, stats = _predict_all(path, iou_thr)
    return jsonify({"ok": True, "boxes": boxes, "stats": stats})


def _predict_all(path, iou_thr=0.5):
    """用所有已加载模型预测,跨模型做类别内 NMS 去重。
    返回 (boxes, stats); boxes 不含 conf。"""
    n_cls = len(STATE["classes"])
    collected = []
    stats = []
    for m in STATE["models"]:
        raw = 0
        try:
            if m.get("backend") in ("meituan_v6n", "yolov5"):
                preds = (_infer_meituan(m["model"], path, m["conf"])
                         if m.get("backend") == "meituan_v6n"
                         else _infer_yolov5(m["model"], path, m["conf"]))
                raw = len(preds)
                for b in preds:
                    model_cls = b["cls"]
                    if m["only_cls"] is not None and model_cls not in m["only_cls"]:
                        continue
                    cls = model_cls + m["cls_offset"]
                    if n_cls:
                        cls = max(0, min(n_cls - 1, cls))
                    collected.append({"cls": cls, "cx": b["cx"], "cy": b["cy"],
                                      "w": b["w"], "h": b["h"],
                                      "conf": b["conf"], "source": m["id"]})
            else:
                res = m["model"].predict(path, conf=m["conf"], verbose=False)[0]
                if res.boxes is not None:
                    for b in res.boxes:
                        model_cls = int(b.cls[0])
                        if m["only_cls"] is not None and model_cls not in m["only_cls"]:
                            continue
                        raw += 1
                        try:
                            xc, yc, ww, hh = b.xywhn[0].tolist()
                            conf = float(b.conf[0]) if b.conf is not None else 1.0
                        except Exception:
                            continue
                        cls = model_cls + m["cls_offset"]
                        if n_cls:
                            cls = max(0, min(n_cls - 1, cls))
                        collected.append({"cls": cls, "cx": xc, "cy": yc,
                                          "w": ww, "h": hh,
                                          "conf": conf,
                                          "source": m["id"]})
        except Exception as e:
            stats.append({"name": m["name"], "raw": 0, "kept": 0,
                          "error": str(e)})
            continue
        stats.append({"name": m["name"], "raw": raw, "kept": 0,
                      "error": None})
    merged = _nms_boxes(collected, iou_thr)
    kept_by_source = defaultdict(int)
    for b in merged:
        kept_by_source[b["source"]] += 1
    for m, s in zip(STATE["models"], stats):
        s["kept"] = kept_by_source.get(m["id"], 0)
    out = [{"cls": b["cls"], "cx": b["cx"], "cy": b["cy"],
            "w": b["w"], "h": b["h"]} for b in merged]
    return out, stats


def _nms_boxes(boxes, iou_thr):
    """按 (模型来源, 类别) 分组做 NMS，避免不同模型的同类框互相吞掉。"""
    kept = []
    by_key = defaultdict(list)
    for b in boxes:
        by_key[(b.get("source"), b["cls"])].append(b)
    for key, items in by_key.items():
        items.sort(key=lambda x: x.get("conf", 0), reverse=True)
        for b in items:
            if all(not _same_target(b, k, iou_thr) for k in kept
                   if k.get("source") == key[0] and k["cls"] == key[1]):
                kept.append(b)
    return kept


def _dedupe_with_existing(existing, preds, iou_thr):
    """预测框与已有框做同类去重后合并。"""
    out = [dict(b) for b in existing]
    for p in preds:
        if any(_same_target(p, b, iou_thr) for b in out
               if b["cls"] == p["cls"]):
            continue
        out.append(p)
    return out


@app.route("/api/auto_annotate", methods=["POST"])
def api_auto_annotate():
    data = request.get_json() or {}
    use_vlm = bool(data.get("vlm"))
    if use_vlm:
        if not STATE["classes"]:
            return jsonify({"ok": False, "msg": "请先填写类别名"})
        if not vlm_config()["api_key"]:
            return jsonify({"ok": False, "msg": "未配置多模态 API Key"})
    elif not STATE["models"]:
        return jsonify({"ok": False, "msg": "未加载任何模型"})
    if not STATE["img_dir"]:
        return jsonify({"ok": False, "msg": "请先打开图片文件夹"})
    if not STATE["images"]:
        return jsonify({"ok": False, "msg": "没有图片"})
    st = STATE.get("auto")
    if st and st.get("running"):
        return jsonify({"ok": False, "msg": "自动化标注已在运行中"})
    skip_labeled = bool(data.get("skip_labeled", True))
    skip_empty = bool(data.get("skip_empty", False)) and use_vlm
    iou_thr = float(data.get("iou", 0.5) or 0.5)
    threading.Thread(target=_auto_worker,
                     args=(skip_labeled, iou_thr, use_vlm, skip_empty),
                     daemon=True).start()
    return jsonify({"ok": True})


def _auto_worker(skip_labeled, iou_thr, use_vlm=False, skip_empty=False):
    total = len(STATE["images"])
    st = {"running": True, "finished": False, "cancel": False,
          "total": total, "done": 0, "saved": 0, "skipped": 0,
          "failed": 0, "empty": 0, "mode": "vlm" if use_vlm else "models",
          "errors": [], "current": "", "percent": 0}
    STATE["auto"] = st
    lock = threading.Lock()
    vcfg = vlm_config() if use_vlm else None
    # ultralytics 模型不是线程安全的，只有纯多模态这一路才并发
    workers = max(1, int(vcfg["concurrency"])) if use_vlm else 1

    def handle(name):
        with lock:
            if st["cancel"]:
                return
            st["current"] = name
        lp = label_path_for(name)
        try:
            if skip_labeled and has_label(name):
                with lock:
                    st["skipped"] += 1
                return
            path = os.path.join(STATE["img_dir"], name)
            if use_vlm:
                preds, stats = _predict_vlm(path, iou_thr, vcfg=vcfg)
                if stats and stats[0].get("error"):
                    raise RuntimeError(stats[0]["error"])
            else:
                preds, _ = _predict_all(path, iou_thr)
            existing = _read_label_boxes(lp)
            merged = _dedupe_with_existing(existing, preds, iou_thr)
            # 多模态会静默漏标。空结果若当负样本写下去,
            # "没检出" 和 "确实没有目标" 就分不开了, 复查时无从发现。
            if skip_empty and not merged:
                with lock:
                    st["empty"] += 1
                return
            _write_label_file(lp, merged)
            with lock:
                st["saved"] += 1
        except Exception as e:
            with lock:
                st["failed"] += 1
                if len(st["errors"]) < 20:
                    st["errors"].append("%s: %s" % (name, e))
        finally:
            with lock:
                st["done"] += 1
                st["percent"] = round(st["done"] / total * 100, 1) if total else 100

    try:
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(handle, list(STATE["images"])))
        else:
            for name in list(STATE["images"]):
                handle(name)
        st["running"] = False
        st["finished"] = True
    except Exception as e:
        st["running"] = False
        st["finished"] = True
        st["errors"].append(str(e))


@app.route("/api/auto_annotate_status")
def api_auto_status():
    return jsonify(STATE.get("auto") or {
        "running": False, "finished": False, "cancel": False,
        "total": 0, "done": 0, "saved": 0, "skipped": 0,
        "failed": 0, "empty": 0, "mode": "models",
        "errors": [], "current": "", "percent": 0})


@app.route("/api/auto_cancel", methods=["POST"])
def api_auto_cancel():
    st = STATE.get("auto")
    if st:
        st["cancel"] = True
    return jsonify({"ok": True})


def _review_worker(vcfg, opts):
    """批量复核：对每张已有标签的图，把它的框逐个交给多模态改类别或删除。"""
    total = len(STATE["images"])
    st = {"running": True, "finished": False, "cancel": False,
          "total": total, "done": 0, "saved": 0, "skipped": 0,
          "failed": 0, "empty": 0, "unchanged": 0, "mode": "review",
          "changed": 0, "reclassed": 0, "dropped": 0,
          "changes": [], "errors": [], "current": "", "percent": 0}
    STATE["auto"] = st
    lock = threading.Lock()
    workers = max(1, int(vcfg["concurrency"]))
    hints = STATE.get("class_hints") or {}
    classes = STATE["classes"]
    drop_allowed = bool(opts.get("drop_allowed", True))
    dry_run = bool(opts.get("dry_run", True))
    proposals = {}
    STATE["review_proposals"] = proposals

    def handle(name):
        with lock:
            if st["cancel"]:
                return
            st["current"] = name
        lp = label_path_for(name)
        try:
            boxes = _read_label_boxes(lp)
            if not boxes:
                with lock:
                    st["skipped"] += 1
                return
            path = os.path.join(STATE["img_dir"], name)
            verdicts, meta = _vlm_review_image(
                path, boxes, classes, vcfg, hints, drop_allowed)
            if meta.get("error"):
                raise RuntimeError(meta["error"])
            new_boxes, changes = _apply_review(boxes, verdicts, classes)
            with lock:
                if changes:
                    if not dry_run:
                        _write_label_file(lp, new_boxes)
                    else:
                        proposals[name] = new_boxes
                    st["saved"] += 1
                    st["changed"] += len(changes)
                    st["reclassed"] += sum(1 for c in changes
                                           if c["action"] == "reclass")
                    st["dropped"] += sum(1 for c in changes
                                         if c["action"] == "drop")
                    for c in changes:
                        if len(st["changes"]) < 300:
                            c2 = dict(c)
                            c2["file"] = name
                            st["changes"].append(c2)
                else:
                    st["unchanged"] += 1
        except Exception as e:
            with lock:
                st["failed"] += 1
                if len(st["errors"]) < 20:
                    st["errors"].append("%s: %s" % (name, e))
        finally:
            with lock:
                st["done"] += 1
                st["percent"] = round(st["done"] / total * 100, 1) if total else 100

    try:
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(handle, list(STATE["images"])))
        else:
            for name in list(STATE["images"]):
                handle(name)
        st["running"] = False
        st["finished"] = True
    except Exception as e:
        st["running"] = False
        st["finished"] = True
        st["errors"].append(str(e))


@app.route("/api/vlm_review", methods=["POST"])
def api_vlm_review():
    """启动批量多模态复核（只改类别或删除，不新增框）。"""
    if not STATE["img_dir"] or not STATE["images"]:
        return jsonify({"ok": False, "msg": "请先打开图片文件夹"})
    if not STATE["classes"]:
        return jsonify({"ok": False, "msg": "请先填写类别名"})
    v = vlm_config()
    if not (v["api_key"] and v["base_url"] and v["model"]):
        return jsonify({"ok": False, "msg": "多模态接口未配置完整（地址/模型名/Key）"})
    st = STATE.get("auto")
    if st and st.get("running"):
        return jsonify({"ok": False, "msg": "已有任务在运行中"})
    data = request.get_json() or {}
    opts = {"drop_allowed": bool(data.get("drop_allowed", True)),
            "dry_run": bool(data.get("dry_run", True))}
    threading.Thread(target=_review_worker, args=(v, opts), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/vlm_review_apply", methods=["POST"])
def api_vlm_review_apply():
    """把上一次复核产生的修改建议真正写进标签文件。"""
    props = STATE.get("review_proposals") or {}
    st = STATE.get("auto")
    if st and st.get("running"):
        return jsonify({"ok": False, "msg": "复核还在运行，等它跑完再应用"})
    if not props:
        return jsonify({"ok": False, "msg": "没有待应用的修改建议"})
    n = 0
    errs = []
    for name, boxes in list(props.items()):
        try:
            _write_label_file(label_path_for(name), boxes)
            n += 1
        except Exception as e:
            errs.append("%s: %s" % (name, e))
    STATE["review_proposals"] = {}
    return jsonify({"ok": True, "applied": n, "errors": errs[:5]})


@app.route("/api/vlm_review_test", methods=["POST"])
def api_vlm_review_test():
    """对当前图做一次复核预览：返回将要发生的修改，不写盘。"""
    if not STATE["img_dir"] or not STATE["images"]:
        return jsonify({"ok": False, "msg": "请先打开图片文件夹"})
    data = request.get_json() or {}
    try:
        idx = int(data.get("idx", -1))
    except (TypeError, ValueError):
        idx = -1
    if idx < 0 or idx >= len(STATE["images"]):
        return jsonify({"ok": False, "msg": "图片下标越界"})
    name = STATE["images"][idx]
    boxes = _read_label_boxes(label_path_for(name))
    if not boxes:
        return jsonify({"ok": False, "msg": "当前图没有已标注的框"})
    v = vlm_config()
    if not (v["api_key"] and v["base_url"] and v["model"]):
        return jsonify({"ok": False, "msg": "多模态接口未配置完整"})
    path = os.path.join(STATE["img_dir"], name)
    verdicts, meta = _vlm_review_image(
        path, boxes, STATE["classes"], v,
        STATE.get("class_hints") or {},
        bool(data.get("drop_allowed", True)))
    if meta.get("error"):
        return jsonify({"ok": False, "msg": meta["error"]})
    new_boxes, changes = _apply_review(boxes, verdicts, STATE["classes"])
    return jsonify({"ok": True, "changes": changes,
                    "kept": len(new_boxes), "total": len(boxes),
                    "unparsed": meta.get("unparsed", 0)})


@app.route("/api/split", methods=["POST"])
def api_split():
    """把当前 img_dir / label_dir 里的图片+标签随机划分到
    输出目录下的 images/train,val 和 labels/train,val。
    可选 also 生成 data.yaml。"""
    import random
    import shutil
    data = request.get_json() or {}
    out_dir = data.get("out_dir", "").strip().strip('"')
    val_ratio = float(data.get("val_ratio", 0.2))
    seed = int(data.get("seed", 0))
    move = bool(data.get("move", False))          # True=移动, False=复制
    write_yaml = bool(data.get("write_yaml", True))

    if not STATE["img_dir"]:
        return jsonify({"ok": False, "msg": "请先打开图片文件夹"})
    if not out_dir:
        return jsonify({"ok": False, "msg": "请填写输出目录"})
    if val_ratio <= 0 or val_ratio >= 1:
        return jsonify({"ok": False, "msg": "验证集比例应在 0~1 之间"})

    imgs = list_images(STATE["img_dir"])
    if not imgs:
        return jsonify({"ok": False, "msg": "没有可划分的图片"})

    # 建目录
    dirs = {}
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        d = os.path.join(out_dir, *sub.split("/"))
        os.makedirs(d, exist_ok=True)
        dirs[sub] = d

    random.seed(seed)
    shuffled = imgs[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    val_set = set(shuffled[:n_val])

    op = shutil.move if move else shutil.copy2
    counts = {"train": 0, "val": 0, "train_empty": 0, "val_empty": 0}

    for name in imgs:
        split = "val" if name in val_set else "train"
        # 图片
        src_img = os.path.join(STATE["img_dir"], name)
        op(src_img, os.path.join(dirs[f"images/{split}"], name))
        # 标签 (没有则生成空 txt, 作为负样本)
        stem = Path(name).stem
        src_lbl = os.path.join(STATE["label_dir"], stem + ".txt")
        dst_lbl = os.path.join(dirs[f"labels/{split}"], stem + ".txt")
        if os.path.isfile(src_lbl):
            op(src_lbl, dst_lbl)
            if os.path.getsize(dst_lbl) == 0:
                counts[f"{split}_empty"] += 1
        else:
            open(dst_lbl, "w").close()
            counts[f"{split}_empty"] += 1
        counts[split] += 1

    yaml_path = None
    if write_yaml:
        classes = STATE["classes"]
        yaml_path = os.path.join(out_dir, "data.yaml")
        lines = [
            f"path: {out_dir.replace(os.sep, '/')}",
            "train: images/train",
            "val: images/val",
            "",
            f"nc: {len(classes)}",
            "names:",
        ]
        for i, c in enumerate(classes):
            lines.append(f"  {i}: {c}")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    return jsonify({"ok": True, "counts": counts, "out_dir": out_dir,
                    "yaml": yaml_path, "moved": move})


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>YOLO 标注工具</title>
<style>
  :root{
    --bg:#0d0d0f; --surface:#151518; --surface-2:#1c1c20; --surface-3:#26262b;
    --border:rgba(255,255,255,.08); --border-strong:rgba(255,255,255,.14);
    --text:#f5f5f7; --text-2:#a1a1a6; --text-3:#6e6e73;
    --accent:#0a84ff; --accent-weak:rgba(10,132,255,.16);
    --green:#30d158; --red:#ff453a; --orange:#ff9f0a;
    --canvas:#111114; --inset:rgba(255,255,255,.04);
    --ease-out:cubic-bezier(.23,1,.32,1);
    --radius-s:6px; --radius-m:8px; --radius-l:12px; --radius-xl:16px;
    --shadow-sm:0 1px 2px rgba(0,0,0,.35);
    --shadow-lg:0 18px 50px rgba(0,0,0,.5);
    --topbar-h:48px; --statusbar-h:28px;
  }
  html{color-scheme:dark;}
  [data-theme="light"]{
    --bg:#f2f2f7; --surface:#ffffff; --surface-2:#fafafc; --surface-3:#ececf1;
    --border:rgba(0,0,0,.08); --border-strong:rgba(0,0,0,.14);
    --text:#1d1d1f; --text-2:#6e6e73; --text-3:#98989d;
    --accent:#007aff; --accent-weak:rgba(0,122,255,.12);
    --canvas:#e5e5ea; --inset:rgba(0,0,0,.04);
    --shadow-sm:0 1px 2px rgba(0,0,0,.08);
    --shadow-lg:0 18px 50px rgba(0,0,0,.2);
    color-scheme:light;
  }
  *{box-sizing:border-box;}
  body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","PingFang SC","Microsoft YaHei",system-ui,sans-serif;background:var(--bg);color:var(--text);overflow:hidden;-webkit-font-smoothing:antialiased;}
  button{font-family:inherit;cursor:pointer;border:none;background:none;color:inherit;}
  input[type=text],input[type=number],select{background:var(--surface-2);color:var(--text);border:1px solid var(--border-strong);padding:0 10px;height:32px;border-radius:var(--radius-m);font-size:13px;outline:none;transition:border-color .15s ease, box-shadow .15s ease;}
  input:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-weak);}
  :focus-visible{outline:2px solid var(--accent);outline-offset:2px;}
  ::-webkit-scrollbar{width:10px;height:10px;}
  ::-webkit-scrollbar-thumb{background:var(--border-strong);border-radius:99px;border:3px solid transparent;background-clip:content-box;}
  ::-webkit-scrollbar-track{background:transparent;}

  /* 顶栏 */
  #topbar{height:var(--topbar-h);display:flex;align-items:center;gap:6px;padding:0 12px;background:var(--surface);border-bottom:1px solid var(--border);}
  #topbar .logo{font-weight:600;font-size:13px;margin-right:4px;display:flex;align-items:center;gap:8px;color:var(--text-2);flex:0 0 auto;white-space:nowrap;}
  #topbar .logo .dot{width:9px;height:9px;border-radius:50%;background:var(--accent);}
  /* 文件夹路径是唯一的「弹性」元素：窗口窄时它先被省略，
     按钮用 nowrap + flex:0 0 auto 保证绝不被压到换行（中文没有空格，
     一旦允许收缩就会被压成一个字宽，文字竖排，非常难看）。 */
  #curFolder{font-size:12px;color:var(--text-3);flex:0 1 auto;min-width:60px;max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  .tbtn{height:32px;padding:0 11px;font-size:13px;font-weight:500;background:var(--surface-2);color:var(--text);border:1px solid var(--border);border-radius:var(--radius-m);display:inline-flex;align-items:center;gap:6px;white-space:nowrap;flex:0 0 auto;transition:background-color .15s ease,border-color .15s ease,transform .1s var(--ease-out);}
  .tbtn:active{transform:scale(.97);}
  @media (hover:hover) and (pointer:fine){.tbtn:hover{background:var(--surface-3);}}
  .tbtn.primary{background:var(--accent);border-color:transparent;color:#fff;}
  @media (hover:hover) and (pointer:fine){.tbtn.primary:hover{background:#0a93ff;}}
  @media (hover:hover) and (pointer:fine){[data-theme="light"] .tbtn.primary:hover{background:#0071e3;}}
  .tbtn.ghost{background:transparent;border-color:transparent;}
  @media (hover:hover) and (pointer:fine){.tbtn.ghost:hover{background:var(--surface-2);}}
  .tbtn:disabled{opacity:.4;cursor:default;transform:none;}
  #navInfo{font-size:12px;color:var(--text-3);flex:0 0 auto;min-width:96px;text-align:center;font-variant-numeric:tabular-nums;white-space:nowrap;}
  .spacer{flex:1 1 auto;min-width:8px;}
  .tsep{flex:0 0 auto;width:1px;height:18px;background:var(--border);margin:0 3px;}

  /* 布局 */
  #layout{display:flex;height:calc(100vh - var(--topbar-h) - var(--statusbar-h));}
  #filePanel{width:220px;background:var(--surface);border-right:1px solid var(--border);display:flex;flex-direction:column;}
  #filePanel .hd{padding:12px 10px 6px;font-size:11px;font-weight:600;color:var(--text-3);letter-spacing:.05em;text-transform:uppercase;display:flex;justify-content:space-between;}
  #fileList{flex:1;overflow-y:auto;padding:2px 6px 8px;}
  .fitem{height:32px;padding:0 10px;margin:1px 0;font-size:12px;display:flex;align-items:center;gap:8px;cursor:pointer;border-radius:var(--radius-m);white-space:nowrap;overflow:hidden;transition:background-color .12s ease;}
  @media (hover:hover) and (pointer:fine){.fitem:hover{background:var(--surface-2);}}
  .fitem.active{background:var(--accent-weak);}
  .fitem .badge{width:7px;height:7px;border-radius:50%;flex-shrink:0;background:var(--border-strong);}
  .fitem.done .badge{background:var(--green);}
  .fitem .nm{overflow:hidden;text-overflow:ellipsis;color:var(--text-2);}
  .fitem.active .nm{color:var(--text);}

  #center{flex:1;display:flex;flex-direction:column;background:var(--canvas);}
  #canvasWrap{flex:1;overflow:hidden;display:flex;align-items:center;justify-content:center;position:relative;}
  canvas{cursor:crosshair;border-radius:var(--radius-l);box-shadow:var(--shadow-sm);}
  #zoomHint{position:absolute;bottom:12px;left:14px;font-size:11px;color:rgba(255,255,255,.75);background:rgba(0,0,0,.55);padding:4px 10px;border-radius:99px;pointer-events:none;}

  #rightPanel{width:260px;background:var(--surface);border-left:1px solid var(--border);display:flex;flex-direction:column;}
  .section{padding:14px 12px;border-bottom:1px solid var(--border);}
  .section h4{margin:0 0 10px;font-size:11px;font-weight:600;color:var(--text-3);text-transform:uppercase;letter-spacing:.05em;}
  .cls-btn{display:flex;align-items:center;gap:8px;width:100%;text-align:left;height:34px;margin:3px 0;padding:0 10px;background:var(--surface-2);color:var(--text);font-size:13px;border:1px solid var(--border);border-radius:var(--radius-m);transition:background-color .12s ease,border-color .12s ease,transform .1s var(--ease-out);}
  .cls-btn:active{transform:scale(.98);}
  @media (hover:hover) and (pointer:fine){.cls-btn:hover{background:var(--surface-3);}}
  .cls-btn.active{border-color:var(--accent);background:var(--accent-weak);}
  .cls-btn .sw{width:12px;height:12px;border-radius:4px;flex-shrink:0;}
  .cls-btn .key{margin-left:auto;font-size:11px;color:var(--text-3);background:var(--surface-3);padding:1px 6px;border-radius:6px;}
  .model-card{border:1px solid var(--border);border-radius:var(--radius-l);padding:12px;margin-top:10px;background:var(--inset);}
  .model-card .mc-top{display:flex;gap:6px;align-items:center;}
  .model-card .mc-top input[type=text]{flex:1;}
  .model-card .mc-opts{display:flex;gap:10px;margin-top:8px;}
  .model-card .mc-opts>div{flex:1;}
  .model-card .mc-opts label{display:block;font-size:11px;color:var(--text-3);margin-bottom:4px;}
  .model-card .mc-opts input{width:100%;}
  .model-card .mc-status{font-size:11px;color:var(--text-2);margin-top:6px;word-break:break-all;line-height:1.4;}
  .model-card .mc-cls{max-height:140px;overflow-y:auto;border:1px solid var(--border);border-radius:var(--radius-m);padding:6px;margin-top:8px;background:var(--surface);}
  .model-card .mc-cls label{display:flex;align-items:center;gap:8px;padding:4px 6px;font-size:12px;cursor:pointer;border-radius:var(--radius-s);}
  @media (hover:hover) and (pointer:fine){.model-card .mc-cls label:hover{background:var(--surface-2);}}
  #boxList{max-height:240px;overflow-y:auto;padding-right:2px;}
  .box-item{display:flex;align-items:center;gap:8px;height:30px;padding:0 8px;margin:3px 0;background:var(--surface-2);border:1px solid var(--border);border-radius:var(--radius-m);font-size:12px;transition:background-color .12s ease,border-color .12s ease;}
  @media (hover:hover) and (pointer:fine){.box-item:hover{background:var(--surface-3);}}
  .box-item.sel{border-color:var(--accent);}
  .box-item select{flex:1;height:24px;padding:0 6px;background:var(--surface);border-color:var(--border);border-radius:var(--radius-s);font-size:12px;}
  .box-item .del{color:var(--text-3);cursor:pointer;padding:0 6px;font-size:14px;line-height:1;}
  @media (hover:hover) and (pointer:fine){.box-item .del:hover{color:var(--red);}}

  /* 状态条 */
  #statusbar{height:var(--statusbar-h);background:var(--surface);border-top:1px solid var(--border);display:flex;align-items:center;padding:0 14px;font-size:12px;gap:16px;color:var(--text-2);}
  #statusbar .prog{margin-left:auto;color:var(--text-3);}
  #dirtyTag{color:var(--orange);}

  /* 弹窗 */
  #modalMask,#splitMask,#autoMask{position:fixed;inset:0;background:rgba(0,0,0,.5);align-items:center;justify-content:center;z-index:100;}
  .mask-open{animation:maskIn .18s var(--ease-out);}
  .mask-close{animation:maskOut .14s var(--ease-out) forwards;}
  .modal{width:560px;max-height:calc(100vh - 48px);overflow-y:auto;overscroll-behavior:contain;background:var(--surface);border:1px solid var(--border-strong);border-radius:var(--radius-xl);padding:20px;box-shadow:var(--shadow-lg);}
  .modal-in{animation:modalIn .2s var(--ease-out);}
  .modal-out{animation:modalOut .14s var(--ease-out) forwards;}
  .modal h2{margin:0 0 2px;font-size:16px;font-weight:600;letter-spacing:-.01em;}
  .modal .sub{color:var(--text-2);font-size:12px;margin-bottom:14px;}
  .modal .field{margin:10px 0;}
  .modal label{display:block;font-size:12px;color:var(--text-2);margin-bottom:4px;}
  .modal input{width:100%;}
  .modal .hint{font-size:11px;color:var(--text-3);margin-top:4px;line-height:1.5;}
  .modal .actions{margin:16px -20px -20px;padding:12px 20px 20px;display:flex;gap:8px;justify-content:flex-end;background:var(--surface);border-top:1px solid var(--border);position:sticky;bottom:0;z-index:2;}
  .grp{border:1px solid var(--border);border-radius:var(--radius-l);padding:12px;margin-top:12px;background:var(--inset);}
  .grp .gt{font-size:12px;color:var(--text);margin-bottom:8px;font-weight:600;}
  .inline{display:flex;gap:10px;} .inline>div{flex:1;}

  @keyframes maskIn{from{opacity:0}to{opacity:1}}
  @keyframes maskOut{from{opacity:1}to{opacity:0}}
  @keyframes modalIn{from{opacity:0;transform:scale(.97)}to{opacity:1;transform:none}}
  @keyframes modalOut{from{opacity:1;transform:none}to{opacity:0;transform:scale(.985)}}

  @media (prefers-reduced-motion: reduce){
    .mask-open,.mask-close,.modal-in,.modal-out{animation:none;}
    .tbtn,.cls-btn,.fitem,.box-item,input{transition:none;}
  }
</style>
</head>
<body>

<div id="topbar">
  <div class="logo"><span class="dot"></span>YOLO 标注工具 <span style="font-size:11px;color:var(--text-3)">v1.0</span></div>
  <span id="curFolder" style="font-size:12px;color:var(--text-3);max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title=""></span>
  <button class="tbtn" onclick="openModal()">配置</button>
  <button class="tbtn ghost" id="themeBtn" onclick="toggleTheme()" title="切换深色 / 浅色">浅色</button>
  <button class="tbtn" id="btnSplit" onclick="openSplit()" disabled>划分数据集</button>
  <div class="spacer"></div>
  <button class="tbtn" id="btnPrev" onclick="go(-1)" disabled>上一张</button>
  <span id="navInfo">未打开文件夹</span>
  <button class="tbtn" id="btnNext" onclick="go(1)" disabled>下一张</button>
  <div class="spacer"></div>
  <button class="tbtn primary" id="btnPredict" onclick="predict()" disabled>模型预测</button>
  <button class="tbtn" id="btnAuto" onclick="openAuto()" disabled>自动化标注</button>
  <button class="tbtn" id="btnVlmAuto" onclick="openVlmAuto()" disabled>多模态标注</button>
  <button class="tbtn" id="btnVlmReview" onclick="openVlmReview()" disabled>多模态复核</button>
  <div class="tsep"></div>
  <button class="tbtn" id="btnClear" onclick="clearBoxes()" disabled>清空</button>
  <button class="tbtn primary" id="btnSave" onclick="saveLabels()" disabled>保存</button>
</div>

<div id="layout">
  <div id="filePanel">
    <div class="hd"><span>文件列表</span><span id="fileCount">0</span></div>
    <div id="fileList"></div>
  </div>

  <div id="center">
    <div id="canvasWrap">
      <canvas id="cv"></canvas>
      <div id="zoomHint">滚轮缩放 · 拖拽画框 · 右键平移</div>
    </div>
  </div>

  <div id="rightPanel">
    <div class="section">
      <h4>类别 (画框时使用)</h4>
      <div id="clsBtns"></div>
    </div>
    <div class="section" style="flex:1;overflow:hidden;display:flex;flex-direction:column;">
      <h4>本图标注框 (<span id="boxCount">0</span>)</h4>
      <div id="boxList"></div>
    </div>
  </div>
</div>

<div id="statusbar">
  <span id="dirtyTag" style="display:none;color:#f59e0b;">● 未保存</span>
  <span id="statusText">就绪</span>
  <span class="prog" id="progText"></span>
</div>

<!-- 配置弹窗 -->
<div id="modalMask">
  <div class="modal">
    <h2>配置</h2>
    <div class="sub">设置图片目录、类别，以及可选的辅助标注模型</div>

    <div class="grp">
      <div class="gt">① 数据</div>
      <div class="field">
        <label>图片文件夹路径</label>
        <div style="display:flex;gap:6px;">
          <input id="imgDir" type="text" placeholder="如 F:\data\images\train">
          <button class="tbtn" type="button" onclick="pickImgDir()" style="flex-shrink:0;">选择</button>
        </div>
      </div>
      <div class="field">
        <label>标签文件夹路径 (留空 = 与图片同目录)</label>
        <div style="display:flex;gap:6px;">
          <input id="labelDir" type="text" placeholder="如 F:\data\labels\train">
          <button class="tbtn" type="button" onclick="pickLabelDir()" style="flex-shrink:0;">选择</button>
        </div>
      </div>
      <div class="field">
        <label>类别 (英文，逗号分隔，顺序即为类别ID)</label>
        <input id="classesIn" type="text" placeholder="如 acetylene_cylinder,oxygen_cylinder">
        <div class="hint">第1个=0, 第2个=1 ... 建议用英文，避免训练时中文路径问题</div>
      </div>
    </div>

    <div class="grp">
      <div class="gt">② 模型辅助标注 (可加载多个，预测时合并)</div>
      <div id="modelCards"></div>
      <button class="tbtn" style="margin-top:8px" onclick="addModelCard()">+ 添加模型</button>
      <div class="inline">
        <div class="field">
          <label>全局去重 IoU 阈值</label>
          <input id="iouThr" type="number" step="0.05" min="0" max="1" value="0.5">
        </div>
        <div class="field">
          <label>推理设备</label>
          <select id="devSel" onchange="saveDevice()">
            <option value="auto">自动（有显卡就用显卡）</option>
            <option value="cuda">显卡（最快）</option>
            <option value="cpu">CPU（把显卡让给训练）</option>
          </select>
        </div>
      </div>
      <div class="hint" id="devHint" style="margin-top:6px"></div>
      <div class="hint">加载模型时会按类别名称自动匹配类别偏移（忽略大小写/下划线）；匹配不上可点卡片里的“自动匹配偏移”。同类别重叠框只保留置信度高的，不同类别（如安全帽 + 反光衣）同时保留。</div>
    </div>

    <div class="grp">
      <div class="gt">③ 多模态标注 (OpenAI 兼容接口)</div>
      <div class="inline">
        <div class="field" style="flex:2">
          <label>接口地址</label>
          <input id="vlmBase" type="text" placeholder="如 https://api.openai.com/v1">
        </div>
        <div class="field" style="flex:2">
          <label>模型名</label>
          <input id="vlmModel" type="text" placeholder="如 gpt-5-mini / qwen-vl-max">
        </div>
      </div>
      <div class="inline">
        <div class="field" style="flex:2">
          <label>API Key <span id="vlmKeyState" style="font-weight:400"></span></label>
          <input id="vlmKey" type="password" placeholder="sk-...">
        </div>
        <div class="field" style="flex:1">
          <label>并发数</label>
          <input id="vlmConc" type="number" min="1" max="32" value="4">
        </div>
      </div>
      <div class="inline" style="align-items:center">
        <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
          <input id="vlmThink" type="checkbox" style="width:auto" checked> 思考模式
        </label>
      </div>
      <div class="inline" style="align-items:center">
        <button class="tbtn" onclick="saveVlmCfg(true)">保存</button>
      </div>
    </div>

    <div class="actions">
      <button class="tbtn" onclick="closeModal()">取消</button>
      <button class="tbtn primary" onclick="openFolder()">打开文件夹</button>
    </div>
  </div>
</div>

<!-- 划分数据集弹窗 -->
<div id="splitMask" style="position:fixed;inset:0;background:rgba(0,0,0,.65);display:none;align-items:center;justify-content:center;z-index:100;">
  <div class="modal">
    <h2>划分数据集</h2>
    <div class="sub">把当前已标注的图片+标签随机分成训练集 / 验证集，并生成标准结构和 data.yaml</div>

    <div class="grp">
      <div class="gt">划分设置</div>
      <div class="field">
        <label>输出目录 (会在此目录下生成 images/ 和 labels/)</label>
        <input id="splitOut" type="text" placeholder="如 D:\datasets\my_dataset_split">
      </div>
      <div class="inline">
        <div class="field">
          <label>验证集比例</label>
          <input id="splitRatio" type="number" step="0.05" min="0.05" max="0.5" value="0.2">
        </div>
        <div class="field">
          <label>随机种子</label>
          <input id="splitSeed" type="number" value="0">
        </div>
      </div>
      <div class="field">
        <label style="display:flex;align-items:center;gap:8px;cursor:pointer">
          <input id="splitMove" type="checkbox" style="width:auto"> 移动文件 (不勾选=复制，保留原图)
        </label>
      </div>
      <div class="field">
        <label style="display:flex;align-items:center;gap:8px;cursor:pointer">
          <input id="splitYaml" type="checkbox" style="width:auto" checked> 同时生成 data.yaml
        </label>
      </div>
      <div class="hint">没有标注框的图片会生成空 txt，作为负样本(反例)一并划分。</div>
    </div>

    <div class="actions">
      <button class="tbtn" onclick="closeOverlay('splitMask')">取消</button>
      <button class="tbtn primary" id="splitGo" onclick="doSplit()">开始划分</button>
    </div>
    <div id="splitMsg" style="font-size:12px;margin-top:10px"></div>
  </div>
</div>

<!-- 自动化标注弹窗 -->
<div id="autoMask" style="position:fixed;inset:0;background:rgba(0,0,0,.65);display:none;align-items:center;justify-content:center;z-index:100;">
  <div class="modal" style="width:480px;">
    <h2>自动化标注</h2>
    <div class="sub">批量运行已加载的模型预测，自动去重后直接保存标注</div>
    <div class="field">
      <label style="display:flex;align-items:center;gap:8px;cursor:pointer">
        <input id="autoSkip" type="checkbox" style="width:auto" checked> 仅标注未标注的图片
      </label>
    </div>
    <div class="field">
      <label>去重 IoU 阈值</label>
      <input id="autoIou" type="number" step="0.05" min="0" max="1" value="0.5">
    </div>
    <div style="background:var(--bg);border-radius:8px;height:16px;overflow:hidden;margin:14px 0;">
      <div id="autoBar" style="height:100%;width:0%;background:var(--accent);transition:width .3s;"></div>
    </div>
    <div id="autoMsg" style="font-size:12px;color:var(--text-2);min-height:60px;white-space:pre-wrap;"></div>
    <div class="actions">
      <button class="tbtn" id="autoCancelBtn" onclick="cancelAuto()" disabled>取消</button>
      <button class="tbtn primary" id="autoStartBtn" onclick="startAuto()">开始自动化标注</button>
    </div>
  </div>
</div>

<!-- 多模态标注弹窗 -->
<div id="vlmMask" style="position:fixed;inset:0;background:rgba(0,0,0,.65);display:none;align-items:center;justify-content:center;z-index:100;">
  <div class="modal" style="width:540px;">
    <h2>多模态自动标注</h2>
    <div class="sub"><b id="vlmNameTag"></b></div>
    <div class="field">
      <label style="display:flex;align-items:center;gap:8px;cursor:pointer">
        <input id="vlmSkip" type="checkbox" style="width:auto" checked> 仅标注未标注的图片
      </label>
    </div>
    <div class="field">
      <label style="display:flex;align-items:center;gap:8px;cursor:pointer">
        <input id="vlmSkipEmpty" type="checkbox" style="width:auto" checked> 空结果不写标签（留待复查）
      </label>
    </div>
    <div class="field">
      <label>去重 IoU 阈值</label>
      <input id="vlmIou" type="number" step="0.05" min="0" max="1" value="0.5">
    </div>
    <div style="background:var(--bg);border-radius:8px;height:16px;overflow:hidden;margin:12px 0;">
      <div id="vlmBar" style="height:100%;width:0%;background:var(--accent);transition:width .3s;"></div>
    </div>
    <div id="vlmMsg" style="font-size:12px;color:var(--text-2);min-height:60px;white-space:pre-wrap;"></div>
    <div class="actions">
      <button class="tbtn" id="vlmTestBtn" onclick="testVlm()">用当前图试一张</button>
      <button class="tbtn" id="vlmCancelBtn" onclick="cancelVlm()" disabled>取消</button>
      <button class="tbtn primary" id="vlmStartBtn" onclick="startVlm()">开始多模态标注</button>
    </div>
  </div>
</div>

<!-- 多模态复核弹窗 -->
<div id="revMask" style="position:fixed;inset:0;background:rgba(0,0,0,.65);display:none;align-items:center;justify-content:center;z-index:100;">
  <div class="modal" style="width:620px;">
    <h2>多模态复核</h2>
    <div class="sub">先出修改建议，确认后再应用；只改类别或删除，不新增框</div>
    <div class="field">
      <label style="display:flex;align-items:center;gap:8px;cursor:pointer">
        <input id="revDrop" type="checkbox" style="width:auto" checked> 允许删除（模型判定不属于任何类别的框会被删掉）
      </label>
    </div>
    <div style="background:var(--bg);border-radius:8px;height:16px;overflow:hidden;margin:12px 0;">
      <div id="revBar" style="height:100%;width:0%;background:var(--accent);transition:width .3s;"></div>
    </div>
    <div id="revMsg" style="font-size:12px;color:var(--text-2);min-height:70px;white-space:pre-wrap;"></div>
    <div class="actions">
      <button class="tbtn" id="revTestBtn" onclick="testVlmReview()">用当前图试一次</button>
      <button class="tbtn" id="revCancelBtn" onclick="cancelReview()" disabled>取消</button>
      <button class="tbtn" id="revApplyBtn" onclick="applyVlmReview()" disabled>应用修改</button>
      <button class="tbtn primary" id="revStartBtn" onclick="startVlmReview()">开始复核</button>
    </div>
  </div>
</div>

<script>
let idx=0,total=0,imgW=0,imgH=0,scale=1,baseScale=1,offX=0,offY=0;
let boxes=[],classes=[],curCls=0,selIdx=-1,files=[];
let drawing=false,sx=0,sy=0,ex=0,ey=0;
let panning=false,panX=0,panY=0;
let modelSlots=[];
let modelKeySeq=1;
let autoTimer=null;
let autoActive=false;
let vlmTimer=null;
let vlmActive=false;
let predicting=false;
const overlayTimers={};
let dirty=false;
let undoStack=[],redoStack=[];
let moving=false,resizing=false,resizeHandle=null;
let moveStartX=0,moveStartY=0,origBox=null;
const MAX_UNDO=50;
const palette=["#ef4444","#22c55e","#3b82f6","#f59e0b","#a855f7","#06b6d4","#ec4899","#84cc16","#f97316","#14b8a6"];
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const img=new Image();

function st(t){document.getElementById('statusText').textContent=t;}
(function initTheme(){
  const saved=localStorage.getItem('annotator-theme')||'dark';
  document.documentElement.dataset.theme=saved;
  const b=document.getElementById('themeBtn');
  if(b)b.textContent=saved==='light'?'深色':'浅色';
})();
function toggleTheme(){
  const cur=document.documentElement.dataset.theme==='light'?'dark':'light';
  document.documentElement.dataset.theme=cur;
  localStorage.setItem('annotator-theme',cur);
  const b=document.getElementById('themeBtn');
  if(b)b.textContent=cur==='light'?'深色':'浅色';
}
function openOverlay(id){
  const el=document.getElementById(id);
  if(overlayTimers[id]){clearTimeout(overlayTimers[id]);overlayTimers[id]=null;}
  el.classList.remove('mask-close');
  el.classList.add('mask-open');
  el.style.display='flex';
  const m=el.querySelector('.modal');
  if(m){m.classList.remove('modal-out');m.classList.add('modal-in');}
}
function closeOverlay(id){
  const el=document.getElementById(id);
  if(overlayTimers[id])clearTimeout(overlayTimers[id]);
  el.classList.remove('mask-open');
  el.classList.add('mask-close');
  const m=el.querySelector('.modal');
  if(m){m.classList.remove('modal-in');m.classList.add('modal-out');}
  overlayTimers[id]=setTimeout(()=>{
    el.style.display='none';
    el.classList.remove('mask-open','mask-close');
    if(m)m.classList.remove('modal-in','modal-out');
    overlayTimers[id]=null;
  },180);
}
function openModal(){openOverlay('modalMask');}
function closeModal(){closeOverlay('modalMask');}
function pickFile(cb){
  fetch('/api/pick_file',{method:'POST'}).then(r=>r.json()).then(d=>{
    if(d.ok&&d.path)cb(d.path);
    else if(d&&d.msg)alert(d.msg);
  });
}
function pickFolder(cb){
  fetch('/api/pick_folder',{method:'POST'}).then(r=>r.json()).then(d=>{
    if(d.ok&&d.path)cb(d.path);
    else if(d&&d.msg)alert(d.msg);
  });
}
function pickImgDir(){pickFolder(p=>{document.getElementById('imgDir').value=p;});}
function pickLabelDir(){pickFolder(p=>{document.getElementById('labelDir').value=p;});}

function snap(){
  undoStack.push(JSON.parse(JSON.stringify(boxes)));
  if(undoStack.length>MAX_UNDO)undoStack.shift();
  redoStack=[];
}
function undo(){
  if(!undoStack.length)return;
  redoStack.push(JSON.parse(JSON.stringify(boxes)));
  boxes=undoStack.pop();
  selIdx=Math.min(selIdx,boxes.length-1);
  dirty=true;redraw();saveLabels(true);
}
function redo(){
  if(!redoStack.length)return;
  undoStack.push(JSON.parse(JSON.stringify(boxes)));
  boxes=redoStack.pop();
  selIdx=Math.min(selIdx,boxes.length-1);
  dirty=true;redraw();saveLabels(true);
}
function markDirty(){
  dirty=true;
  const dt=document.getElementById('dirtyTag');
  if(dt)dt.style.display='inline';
}

// 载入上次配置
fetch('/api/config').then(r=>r.json()).then(c=>{
  if(c.img_dir)document.getElementById('imgDir').value=c.img_dir;
  if(c.label_dir)document.getElementById('labelDir').value=c.label_dir;
  if(c.classes)document.getElementById('classesIn').value=c.classes;
  fillVlmFields(c.vlm||{});
  refreshDeviceInfo();
  initModelSlots(c.models||[]);
  // 服务端可能还开着文件夹、甚至还有批量任务在跑，刷新页面不该把这些丢掉
  fetch('/api/files').then(r=>r.json()).then(d=>{
    if(d.ok&&d.files&&d.files.length&&d.classes&&d.classes.length){
      restoreSession(d);
    }else{
      openModal();
    }
  }).catch(()=>openModal());
});

function restoreSession(d){
  classes=d.classes.slice();
  files=d.files;total=files.length;idx=0;
  const cf=document.getElementById('curFolder');
  if(cf){cf.textContent=d.img_dir||'';cf.title=d.img_dir||'';}
  ['btnPrev','btnNext','btnSave','btnClear','btnPredict','btnAuto','btnSplit','btnVlmAuto','btnVlmReview']
    .forEach(id=>{const el=document.getElementById(id);if(el)el.disabled=false;});
  buildClsBtns();buildFileList();load(0);
  const a=d.auto||{};
  if(a.running)resumeBatch(a);
}

function resumeBatch(a){
  if(a.mode==='vlm'){
    vlmActive=true;
    document.getElementById('vlmStartBtn').disabled=true;
    const cb=document.getElementById('vlmCancelBtn');
    cb.disabled=false;cb.textContent='取消';cb.onclick=cancelVlm;
    openOverlay('vlmMask');
    pollVlm();
  }else{
    autoActive=true;
    document.getElementById('autoStartBtn').disabled=true;
    const cb=document.getElementById('autoCancelBtn');
    cb.disabled=false;cb.textContent='取消';cb.onclick=cancelAuto;
    openOverlay('autoMask');
    pollAuto();
  }
}

function refreshDeviceInfo(){
  fetch('/api/device').then(r=>r.json()).then(d=>{
    if(!d.ok)return;
    const sel=document.getElementById('devSel');
    if(sel&&d.want)sel.value=d.want;
    const h=document.getElementById('devHint');
    if(!h)return;
    if(!d.torch){
      h.textContent='未安装 PyTorch，无法加载模型（只做手动标注不需要）';
    }else if(!d.cuda){
      h.textContent='PyTorch '+d.torch+'：未检测到可用显卡，只能用 CPU 推理';
    }else{
      h.textContent='PyTorch '+d.torch+' · '+d.gpu+' · 当前推理设备 '+
                    d.current+(d.current==='cpu'?'（已让出显卡）':'');
    }
  }).catch(()=>{});
}
function saveDevice(){
  const v=document.getElementById('devSel').value;
  fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({device:v})}).then(()=>{
      refreshDeviceInfo();
      st('推理设备已设为「'+v+'」，重新加载模型后生效');
    });
}

function fillVlmFields(v){
  if(v.model)document.getElementById('vlmModel').value=v.model;
  if(v.base_url)document.getElementById('vlmBase').value=v.base_url;
  if(v.concurrency)document.getElementById('vlmConc').value=v.concurrency;
  if(v.thinking!==undefined)document.getElementById('vlmThink').checked=!!v.thinking;
  const st=document.getElementById('vlmKeyState');
  const inp=document.getElementById('vlmKey');
  if(v.has_key){
    st.textContent='(已保存)';
    inp.placeholder='•••••••• 已保存';
  }else{
    st.textContent='';
    inp.placeholder='sk-...';
  }
}

function saveVlmCfg(loud){
  const body={
    model:document.getElementById('vlmModel').value.trim(),
    base_url:document.getElementById('vlmBase').value.trim(),
    concurrency:parseInt(document.getElementById('vlmConc').value)||4,
    thinking:document.getElementById('vlmThink').checked,
  };
  const keyEl=document.getElementById('vlmKey');
  if(keyEl.value.trim())body.api_key=keyEl.value.trim();
  return fetch('/api/vlm_config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)}).then(r=>r.json()).then(d=>{
    if(d.ok){
      fillVlmFields(d.vlm);
      keyEl.value='';
      if(loud)st('已保存');
    }else if(loud){
      st('保存失败');
    }
    return d;
  });
}

function openFolder(){
  const img_dir=document.getElementById('imgDir').value;
  const label_dir=document.getElementById('labelDir').value;
  classes=document.getElementById('classesIn').value.split(',').map(s=>s.trim()).filter(s=>s);
  if(!classes.length){alert('请至少填写一个类别');return;}
  const keyEl=document.getElementById('vlmKey');
  const p=keyEl.value.trim()?saveVlmCfg(false):Promise.resolve();
  p.then(()=>fetch('/api/open',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({img_dir,label_dir,classes})}).then(r=>r.json()).then(d=>{
    if(!d.ok){alert(d.msg);return;}
    total=d.count;idx=0;files=d.files;
    // 服务端会把「类别名(提示)」里的提示剥掉，这里用回干净的类别名
    if(d.classes&&d.classes.length)classes=d.classes.slice();
    const cf=document.getElementById('curFolder');
    if(cf){cf.textContent=d.img_dir||'';cf.title=d.img_dir||'';}
    closeModal();
    ['btnPrev','btnNext','btnSave','btnClear','btnPredict','btnAuto','btnSplit','btnVlmAuto'].forEach(id=>document.getElementById(id).disabled=false);
    buildClsBtns();buildFileList();load(0);
  }));
}

function newModelSlot(){
  return {key:modelKeySeq++,id:null,path:'',cls_offset:0,conf:0.25,
          only_cls:null,model_classes:[]};
}
function initModelSlots(saved){
  modelSlots=[];
  (saved||[]).forEach(m=>{
    modelSlots.push({key:modelKeySeq++,id:null,path:m.path||'',
      cls_offset:m.cls_offset||0,conf:m.conf||0.5,
      only_cls:m.only_cls?m.only_cls.slice():null,model_classes:[]});
  });
  if(!modelSlots.length)modelSlots.push(newModelSlot());
  renderModelCards();
  fetch('/api/models').then(r=>r.json()).then(d=>{
    if(d.ok&&d.models)applyServerModels(d.models,true);
  });
}
function applyServerModels(list,quiet){
  const paths=list.map(m=>m.path);
  modelSlots.forEach(s=>{
    if(s.path&&!paths.includes(s.path)){
      s.id=null;s.model_classes=[];s.only_cls=null;
    }
  });
  list.forEach(m=>{
    let s=modelSlots.find(x=>x.path===m.path);
    if(!s){
      s=newModelSlot();s.path=m.path;modelSlots.push(s);
    }
  s.id=m.id;s.model_classes=m.model_classes;s.device=m.device;
    s.cls_offset=m.cls_offset;s.conf=m.conf;s.only_cls=m.only_cls;
  });
  renderModelCards();
}
function addModelCard(){modelSlots.push(newModelSlot());renderModelCards();}
function renderModelCards(){
  const box=document.getElementById('modelCards');
  box.innerHTML='';
  modelSlots.forEach(s=>box.appendChild(buildModelCard(s)));
}
function buildModelCard(s){
  const card=document.createElement('div');
  card.className='model-card';card.dataset.key=s.key;
  const top=document.createElement('div');top.className='mc-top';
  const path=document.createElement('input');path.type='text';
  path.value=s.path;path.placeholder='如 ...\\weights\\best.pt';
  if(s.id)path.readOnly=true;
  path.addEventListener('input',()=>{s.path=path.value.trim();});
  top.appendChild(path);
  const pickBtn=document.createElement('button');
  pickBtn.className='tbtn';pickBtn.textContent='选择';pickBtn.title='打开文件窗口选择 .pt 模型';
  pickBtn.disabled=!!s.id;
  pickBtn.onclick=()=>pickFile(p=>{s.path=p;path.value=p;loadModelSlot(s.key);});
  top.appendChild(pickBtn);
  const loadBtn=document.createElement('button');
  loadBtn.className='tbtn';loadBtn.textContent=s.id?'已加载':'加载模型';
  loadBtn.disabled=!!s.id;
  loadBtn.onclick=()=>loadModelSlot(s.key);
  top.appendChild(loadBtn);
  const delBtn=document.createElement('button');
  delBtn.className='tbtn';delBtn.textContent='×';delBtn.title='移除该模型';
  delBtn.onclick=()=>removeModelSlot(s.key);
  top.appendChild(delBtn);
  card.appendChild(top);
  const opts=document.createElement('div');opts.className='mc-opts';
  const offWrap=document.createElement('div');
  const offLb=document.createElement('label');offLb.textContent='类别偏移';
  const off=document.createElement('input');off.type='number';off.value=s.cls_offset;
  off.onchange=()=>{s.cls_offset=parseInt(off.value)||0;if(s.id)updateModelSlot(s.key,{cls_offset:s.cls_offset});};
  offWrap.appendChild(offLb);offWrap.appendChild(off);
  const cfWrap=document.createElement('div');
  const cfLb=document.createElement('label');cfLb.textContent='置信度阈值';
  const cf=document.createElement('input');cf.type='number';cf.step='0.05';cf.value=s.conf;
  cf.onchange=()=>{s.conf=parseFloat(cf.value)||0.25;if(s.id)updateModelSlot(s.key,{conf:s.conf});};
  cfWrap.appendChild(cfLb);cfWrap.appendChild(cf);
  opts.appendChild(offWrap);opts.appendChild(cfWrap);
  if(s.device){
    const devWrap=document.createElement('div');
    const devLb=document.createElement('label');devLb.textContent='运行设备';
    const devTag=document.createElement('div');
    devTag.style.cssText='height:32px;display:flex;align-items:center;font-size:12px;'+
      'color:var(--text-2);font-variant-numeric:tabular-nums;';
    devTag.textContent=s.device;
    devWrap.appendChild(devLb);devWrap.appendChild(devTag);
    opts.appendChild(devWrap);
  }
  card.appendChild(opts);
  if(s.id&&classes.length){
    const autoBtn=document.createElement('button');
    autoBtn.className='tbtn';
    autoBtn.style.cssText='margin-top:8px;height:28px;padding:0 10px;font-size:12px;';
    autoBtn.textContent='自动匹配偏移';
    autoBtn.title='按模型类别名与数据集类别名重新推断偏移';
    autoBtn.onclick=()=>applySuggested(s.key);
    card.appendChild(autoBtn);
  }
  const status=document.createElement('div');status.className='mc-status';
  if(s.id){
    if(classes.length){
      const mapTxt=s.model_classes.map((c,i)=>{
        const aid=i+s.cls_offset;
        return c+' → '+(classes[aid]||('ID '+aid));
      }).join('，');
      status.textContent='偏移 '+s.cls_offset+(s.auto_offset?'（自动匹配）':'')+' | '+mapTxt;
    }else{
      status.textContent='模型类别: '+s.model_classes.join(', ');
    }
  }
  card.appendChild(status);
  if(s.id){
    const clsBox=document.createElement('div');clsBox.className='mc-cls';
    s.model_classes.forEach((name,i)=>{
      const lab=document.createElement('label');
      const cb=document.createElement('input');cb.type='checkbox';
      cb.checked=!(s.only_cls)||s.only_cls.includes(i);cb.value=i;
      cb.onchange=()=>{collectSlotCls(s);updateModelSlot(s.key,{only_cls:s.only_cls});};
      const sp=document.createElement('span');sp.textContent=i+': '+name;
      lab.appendChild(cb);lab.appendChild(sp);clsBox.appendChild(lab);
    });
    const btnRow=document.createElement('div');
    btnRow.style.cssText='margin-top:6px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;';
    const allBtn=document.createElement('button');allBtn.className='tbtn';
    allBtn.style.cssText='padding:3px 10px;font-size:11px;';allBtn.textContent='全选';
    allBtn.onclick=()=>{clsBox.querySelectorAll('input[type=checkbox]').forEach(c=>c.checked=true);collectSlotCls(s);updateModelSlot(s.key,{only_cls:s.only_cls});};
    const noneBtn=document.createElement('button');noneBtn.className='tbtn';
    noneBtn.style.cssText='padding:3px 10px;font-size:11px;';noneBtn.textContent='全不选';
    noneBtn.onclick=()=>{clsBox.querySelectorAll('input[type=checkbox]').forEach(c=>c.checked=false);collectSlotCls(s);updateModelSlot(s.key,{only_cls:s.only_cls});};
    const hint=document.createElement('span');hint.className='hint';
    hint.textContent='未勾选的类别预测时会被过滤；全不勾选 = 不输出任何框';
    btnRow.appendChild(allBtn);btnRow.appendChild(noneBtn);btnRow.appendChild(hint);
    card.appendChild(clsBox);card.appendChild(btnRow);
  }
  return card;
}
function collectSlotCls(s){
  const card=document.querySelector('.model-card[data-key="'+s.key+'"]');
  const cbs=card?card.querySelectorAll('.mc-cls input[type=checkbox]'):[];
  const sel=[];
  cbs.forEach(cb=>{if(cb.checked)sel.push(parseInt(cb.value));});
  s.only_cls=sel;
}
function loadModelSlot(key){
  const s=modelSlots.find(x=>x.key===key);
  if(!s||!s.path.trim()){alert('请先填写模型路径');return;}
  fetch('/api/load_model',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({model_path:s.path.trim(),cls_offset:null,conf:s.conf})}).then(r=>r.json()).then(d=>{
    if(!d.ok){alert('加载失败: '+(d.msg||'未知错误'));return;}
    applyServerModels(d.models);
  });
}
function applySuggested(key){
  const s=modelSlots.find(x=>x.key===key);
  if(!s||!s.id)return;
  fetch('/api/suggest_offset',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({id:s.id})}).then(r=>r.json()).then(d=>{
    if(!d.ok){alert(d.msg||'模型不存在');return;}
    if(d.suggested_offset===null||d.suggested_offset===undefined){
      alert('未能自动匹配：请检查模型类别名与数据集类别名是否一致');
      return;
    }
    s.cls_offset=d.suggested_offset;
    updateModelSlot(s.key,{cls_offset:s.cls_offset});
  });
}
function updateModelSlot(key,patch){
  const s=modelSlots.find(x=>x.key===key);
  if(!s||!s.id)return;
  fetch('/api/update_model',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(Object.assign({id:s.id},patch))}).then(r=>r.json()).then(d=>{
    if(d.ok)applyServerModels(d.models);
  });
}
function removeModelSlot(key){
  const s=modelSlots.find(x=>x.key===key);
  if(!s)return;
  if(s.id){
    fetch('/api/unload_model',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({id:s.id})}).then(r=>r.json()).then(d=>{
      if(d.ok)applyServerModels(d.models);
    });
  }
  modelSlots=modelSlots.filter(x=>x.key!==key);
  renderModelCards();
}

function buildClsBtns(){
  const box=document.getElementById('clsBtns');box.innerHTML='';
  classes.forEach((c,i)=>{
    const b=document.createElement('button');
    b.className='cls-btn'+(i===curCls?' active':'');
    const sw=document.createElement('span');
    sw.className='sw';sw.style.background=palette[i%palette.length];
    const tx=document.createElement('span');
    tx.textContent=c;
    b.appendChild(sw);b.appendChild(tx);
    if(i<10){
      const key=document.createElement('span');
      key.className='key';key.textContent=i;
      b.appendChild(key);
    }
    b.onclick=()=>{curCls=i;buildClsBtns();};
    box.appendChild(b);
  });
}

function buildFileList(){
  const box=document.getElementById('fileList');box.innerHTML='';
  document.getElementById('fileCount').textContent=total;
  files.forEach((f,i)=>{
    const d=document.createElement('div');
    d.className='fitem'+(i===idx?' active':'')+(f.done?' done':'');
    const badge=document.createElement('span');
    badge.className='badge';
    const nm=document.createElement('span');
    nm.className='nm';
    nm.textContent=(i+1)+'. '+f.name;
    d.appendChild(badge);d.appendChild(nm);
    d.onclick=()=>{saveLabels(true);load(i);};
    d.id='f'+i;
    box.appendChild(d);
  });
  updateProg();
}

function updateProg(){
  const done=files.filter(f=>f.done).length;
  document.getElementById('progText').textContent=`已标注 ${done} / ${total}`;
}

function load(i){
  if(i<0||i>=total)return;
  idx=i;selIdx=-1;dirty=false;
  const dt=document.getElementById('dirtyTag');if(dt)dt.style.display='none';
  document.querySelectorAll('.fitem').forEach(e=>e.classList.remove('active'));
  const fi=document.getElementById('f'+i);if(fi){fi.classList.add('active');fi.scrollIntoView({block:'nearest'});}
  fetch('/api/meta/'+idx).then(r=>r.json()).then(d=>{
    if(!d.ok)return;
    imgW=d.width;imgH=d.height;boxes=d.boxes;
    document.getElementById('navInfo').textContent=`${idx+1} / ${total}`;
    st(d.name+'  ('+imgW+'×'+imgH+')');
    img.onload=()=>{fit();};
    img.src='/api/image/'+idx+'?t='+Date.now();
  });
}

function fit(){
  const wrap=document.getElementById('canvasWrap');
  const maxW=wrap.clientWidth-30,maxH=wrap.clientHeight-30;
  baseScale=Math.min(maxW/imgW,maxH/imgH);
  scale=baseScale;offX=0;offY=0;
  applySize();
}
function applySize(){
  cv.width=imgW*scale;cv.height=imgH*scale;
  cv.style.transform=`translate(${offX}px,${offY}px)`;
  redraw();
}

function redraw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.drawImage(img,0,0,cv.width,cv.height);
  boxes.forEach((b,i)=>{
    const x=(b.cx-b.w/2)*cv.width,y=(b.cy-b.h/2)*cv.height,w=b.w*cv.width,h=b.h*cv.height;
    ctx.lineWidth=(i===selIdx)?3:2;
    ctx.strokeStyle=palette[b.cls%palette.length];
    ctx.strokeRect(x,y,w,h);
    const label=b.label||classes[b.cls]||b.cls;
    ctx.font='13px sans-serif';
    const tw=ctx.measureText(label).width;
    ctx.fillStyle=palette[b.cls%palette.length];
    ctx.fillRect(x,y-17,tw+8,17);
    ctx.fillStyle='#fff';ctx.fillText(label,x+4,y-4);
  });
  if(drawing){ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.setLineDash([5,3]);ctx.strokeRect(sx,sy,ex-sx,ey-sy);ctx.setLineDash([]);}
  if(selIdx>=0)drawHandles();
  buildBoxList();
}

function buildBoxList(){
  const box=document.getElementById('boxList');box.innerHTML='';
  document.getElementById('boxCount').textContent=boxes.length;
  boxes.forEach((b,i)=>{
    const d=document.createElement('div');d.className='box-item'+(i===selIdx?' sel':'');
    const sw=document.createElement('span');sw.style.cssText=`width:12px;height:12px;border-radius:3px;background:${palette[b.cls%palette.length]}`;
    const sel=document.createElement('select');
    classes.forEach((c,ci)=>{const o=document.createElement('option');o.value=ci;o.text=c;if(ci===b.cls)o.selected=true;sel.appendChild(o);});
    sel.onchange=()=>{snap();b.cls=parseInt(sel.value);markDirty();redraw();};
    const del=document.createElement('span');del.className='del';del.textContent='×';
    del.onclick=(e)=>{e.stopPropagation();snap();boxes.splice(i,1);selIdx=-1;markDirty();redraw();};
    d.appendChild(sw);d.appendChild(sel);d.appendChild(del);
    d.onclick=(e)=>{if(e.target!==del&&e.target!==sel){selIdx=i;redraw();}};
    box.appendChild(d);
  });
}

function getPos(e){const r=cv.getBoundingClientRect();return{x:e.clientX-r.left,y:e.clientY-r.top};}

function drawHandles(){
  const b=boxes[selIdx];
  if(!b)return;
  const x=(b.cx-b.w/2)*cv.width,y=(b.cy-b.h/2)*cv.height,w=b.w*cv.width,h=b.h*cv.height;
  const H=5;
  const pts=[[x,y],[x+w/2,y],[x+w,y],[x+w,y+h/2],[x+w,y+h],[x+w/2,y+h],[x,y+h],[x,y+h/2]];
  ctx.fillStyle='#fff';ctx.strokeStyle='#111';ctx.lineWidth=1;
  pts.forEach(p=>{ctx.fillRect(p[0]-H,p[1]-H,H*2,H*2);ctx.strokeRect(p[0]-H,p[1]-H,H*2,H*2);});
}

function handleAt(p){
  if(selIdx<0)return null;
  const b=boxes[selIdx];
  const x=(b.cx-b.w/2)*cv.width,y=(b.cy-b.h/2)*cv.height,w=b.w*cv.width,h=b.h*cv.height;
  const H=8;
  const pts={nw:[x,y],n:[x+w/2,y],ne:[x+w,y],e:[x+w,y+h/2],se:[x+w,y+h],s:[x+w/2,y+h],sw:[x,y+h],w:[x,y+h/2]};
  for(const k in pts){
    if(Math.abs(p.x-pts[k][0])<=H&&Math.abs(p.y-pts[k][1])<=H)return k;
  }
  return null;
}

function boxAt(p){
  for(let i=boxes.length-1;i>=0;i--){
    const b=boxes[i],x=(b.cx-b.w/2)*cv.width,y=(b.cy-b.h/2)*cv.height,w=b.w*cv.width,h=b.h*cv.height;
    if(p.x>=x&&p.x<=x+w&&p.y>=y&&p.y<=y+h)return i;
  }
  return -1;
}

function doMove(e){
  const b=boxes[selIdx];
  if(!b)return;
  const p=getPos(e);
  const dx=(p.x-moveStartX)/cv.width,dy=(p.y-moveStartY)/cv.height;
  const halfW=b.w/2,halfH=b.h/2;
  b.cx=Math.max(halfW,Math.min(1-halfW,origBox.cx+dx));
  b.cy=Math.max(halfH,Math.min(1-halfH,origBox.cy+dy));
  redraw();
}

function doResize(e){
  const b=boxes[selIdx];
  if(!b)return;
  const p=getPos(e);
  let x1=(b.cx-b.w/2)*cv.width,y1=(b.cy-b.h/2)*cv.height;
  let x2=(b.cx+b.w/2)*cv.width,y2=(b.cy+b.h/2)*cv.height;
  if(resizeHandle.includes('w'))x1=p.x;
  if(resizeHandle.includes('e'))x2=p.x;
  if(resizeHandle.includes('n'))y1=p.y;
  if(resizeHandle.includes('s'))y2=p.y;
  let lx=Math.min(x1,x2),rx=Math.max(x1,x2);
  let ty=Math.min(y1,y2),by=Math.max(y1,y2);
  lx=Math.max(0,lx);ty=Math.max(0,ty);
  rx=Math.min(cv.width,rx);by=Math.min(cv.height,by);
  if(rx-lx<10||by-ty<10)return;
  b.cx=(lx+rx)/2/cv.width;b.cy=(ty+by)/2/cv.height;
  b.w=(rx-lx)/cv.width;b.h=(by-ty)/cv.height;
  redraw();
}

cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('mousedown',e=>{
  if(e.button===2){panning=true;panX=e.clientX-offX;panY=e.clientY-offY;return;}
  const p=getPos(e);
  const h=handleAt(p);
  if(h){
    resizing=true;resizeHandle=h;
    const b=boxes[selIdx];
    snap();markDirty();redraw();
    return;
  }
  const bi=boxAt(p);
  if(bi>=0){
    selIdx=bi;
    moving=true;
    moveStartX=p.x;moveStartY=p.y;
    origBox={cx:boxes[bi].cx,cy:boxes[bi].cy};
    snap();markDirty();redraw();
    return;
  }
  drawing=true;sx=ex=p.x;sy=ey=p.y;
});
window.addEventListener('mousemove',e=>{
  if(panning){offX=e.clientX-panX;offY=e.clientY-panY;cv.style.transform=`translate(${offX}px,${offY}px)`;return;}
  if(moving){doMove(e);return;}
  if(resizing){doResize(e);return;}
  if(!drawing)return;const p=getPos(e);ex=p.x;ey=p.y;redraw();
});
window.addEventListener('mouseup',e=>{
  if(panning){panning=false;return;}
  if(resizing){resizing=false;saveLabels(true);return;}
  if(moving){moving=false;saveLabels(true);return;}
  if(!drawing)return;drawing=false;
  const x1=Math.min(sx,ex),y1=Math.min(sy,ey),x2=Math.max(sx,ex),y2=Math.max(sy,ey);
  if(x2-x1<5||y2-y1<5){redraw();return;}
  snap();
  boxes.push({cls:curCls,cx:((x1+x2)/2)/cv.width,cy:((y1+y2)/2)/cv.height,w:(x2-x1)/cv.width,h:(y2-y1)/cv.height});
  selIdx=boxes.length-1;markDirty();redraw();
});
cv.addEventListener('dblclick',()=>{if(selIdx>=0){snap();boxes.splice(selIdx,1);selIdx=-1;markDirty();redraw();}});
document.getElementById('canvasWrap').addEventListener('wheel',e=>{
  e.preventDefault();
  const f=e.deltaY<0?1.1:0.9;
  scale=Math.max(baseScale*0.5,Math.min(baseScale*8,scale*f));
  applySize();
},{passive:false});

function go(d){saveLabels(true);load(idx+d);}
function saveLabels(silent){
  if(total===0)return;
  fetch('/api/save/'+idx,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({boxes})}).then(r=>r.json()).then(d=>{
    if(d.ok){
      dirty=false;
      const dt=document.getElementById('dirtyTag');if(dt)dt.style.display='none';
      files[idx].done=d.done;const fi=document.getElementById('f'+idx);if(fi)fi.classList.toggle('done',d.done);updateProg();
      if(!silent)st('已保存 '+d.count+' 个框');
    }else{
      st('保存失败: '+(d.msg||'未知错误'));
      const dt=document.getElementById('dirtyTag');if(dt)dt.style.display='inline';
    }
  });
}
function iou(a,b){
  const ax1=a.cx-a.w/2,ay1=a.cy-a.h/2,ax2=a.cx+a.w/2,ay2=a.cy+a.h/2;
  const bx1=b.cx-b.w/2,by1=b.cy-b.h/2,bx2=b.cx+b.w/2,by2=b.cy+b.h/2;
  const ix1=Math.max(ax1,bx1),iy1=Math.max(ay1,by1),ix2=Math.min(ax2,bx2),iy2=Math.min(ay2,by2);
  const iw=Math.max(0,ix2-ix1),ih=Math.max(0,iy2-iy1),inter=iw*ih;
  const union=((ax2-ax1)*(ay2-ay1))+((bx2-bx1)*(by2-by1))-inter;
  return union>0?inter/union:0;
}
function sameTarget(a,b,iouThr){
  if(iou(a,b)>=iouThr)return true;
  const aa=a.w*a.h,ba=b.w*b.h;
  if(!(aa>0&&ba>0))return false;
  const big=aa>=ba?a:b;
  const small=big===a?b:a;
  const sArea=small.w*small.h,bigArea=big.w*big.h;
  if(sArea/bigArea<0.35)return false;
  return (big.cx-big.w/2<=small.cx&&small.cx<=big.cx+big.w/2&&
          big.cy-big.h/2<=small.cy&&small.cy<=big.cy+big.h/2);
}
function dedupeAppend(newBoxes,iouThr){
  iouThr=iouThr||0.5;
  const out=[];
  newBoxes.forEach(nb=>{
    let dup=false;
    for(const ob of boxes){
      if(ob.cls===nb.cls&&sameTarget(ob,nb,iouThr)){dup=true;break;}
    }
    if(!dup){for(const ob of out){
      if(ob.cls===nb.cls&&sameTarget(ob,nb,iouThr)){dup=true;break;}
    }}
    if(!dup)out.push(nb);
  });
  return out;
}
function predict(){
  if(predicting)return;
  if(!modelSlots.some(s=>s.id)){st('请先在配置中加载模型');return;}
  predicting=true;
  const btn=document.getElementById('btnPredict');if(btn)btn.disabled=true;
  const iou=parseFloat(document.getElementById('iouThr').value)||0.5;
  st('正在预测...');
  fetch('/api/predict/'+idx,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({iou})}).then(r=>r.json()).then(d=>{
    predicting=false;
    if(btn)btn.disabled=false;
    if(!d.ok){st('预测失败: '+(d.msg||''));return;}
    // 过滤模型可能输出的退化框(宽高<=0/非数字),避免出现"看不见、点不中、存不了"的框
    const validBoxes=d.boxes.filter(b=>b&&isFinite(b.cx)&&isFinite(b.cy)&&b.w>0&&b.h>0);
    const badCount=d.boxes.length-validBoxes.length;
    const added=dedupeAppend(validBoxes,iou);
    let warn='';
    if(classes.length){
      let clsBad=0;
      added.forEach(b=>{
        if(b.cls<0||b.cls>=classes.length){
          clsBad++;
          b.cls=Math.max(0,Math.min(classes.length-1,b.cls));
        }
      });
      if(clsBad)warn='，其中 '+clsBad+' 个框类别超出范围已自动归到合法类别，请检查类别偏移';
    }
    if(added.length){snap();boxes.push(...added);markDirty();}
    const stats=(d.stats||[]).map(s=>s.error? s.name+': 失败('+s.error+')'
                                              : s.name+': '+s.raw+'→'+s.kept).join(' / ');
    st('预测合并: '+stats+' 新增 '+added.length+' 个 (过滤退化框 '+badCount+' 个)'+warn);
    redraw();
  }).catch(()=>{
    predicting=false;
    if(btn)btn.disabled=false;
    st('预测失败: 请求异常');
  });
}
function clearBoxes(){if(!boxes.length)return;snap();boxes=[];selIdx=-1;markDirty();redraw();}

function openSplit(){
  saveLabels(true);
  const out=document.getElementById('splitOut');
  if(!out.value){
    // 默认在图片父目录旁建一个 _split
    out.value='';
  }
  document.getElementById('splitMsg').textContent='';
  openOverlay('splitMask');
}
function doSplit(){
  const out_dir=document.getElementById('splitOut').value.trim();
  if(!out_dir){alert('请填写输出目录');return;}
  const val_ratio=parseFloat(document.getElementById('splitRatio').value)||0.2;
  const seed=parseInt(document.getElementById('splitSeed').value)||0;
  const move=document.getElementById('splitMove').checked;
  const write_yaml=document.getElementById('splitYaml').checked;
  const msg=document.getElementById('splitMsg');
  const btn=document.getElementById('splitGo');
  btn.disabled=true;msg.style.color='#8b93a4';msg.textContent='划分中，请稍候...';
  fetch('/api/split',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({out_dir,val_ratio,seed,move,write_yaml})}).then(r=>r.json()).then(d=>{
    btn.disabled=false;
    if(!d.ok){msg.style.color='#ef4444';msg.textContent='✗ '+d.msg;return;}
    const c=d.counts;
    msg.style.color='#22c55e';
    msg.innerHTML='划分完成！<br>'+
      `训练集 ${c.train} 张 (含反例 ${c.train_empty})<br>`+
      `验证集 ${c.val} 张 (含反例 ${c.val_empty})<br>`+
      `输出: ${d.out_dir}`+(d.yaml?`<br>已生成: ${d.yaml}`:'');
    st('数据集划分完成');
  }).catch(e=>{btn.disabled=false;msg.style.color='#ef4444';msg.textContent='✗ '+e;});
}

function openAuto(){
  if(!modelSlots.some(s=>s.id)){alert('请先在配置中加载至少一个模型');return;}
  autoActive=true;
  document.getElementById('autoIou').value=document.getElementById('iouThr').value;
  document.getElementById('autoBar').style.width='0%';
  document.getElementById('autoMsg').textContent='';
  document.getElementById('autoStartBtn').disabled=false;
  const cb=document.getElementById('autoCancelBtn');
  cb.disabled=false;cb.textContent='取消';cb.onclick=cancelAuto;
  openOverlay('autoMask');
}
function startAuto(){
  saveLabels(true);
  const skip=document.getElementById('autoSkip').checked;
  const iou=parseFloat(document.getElementById('autoIou').value)||0.5;
  const btn=document.getElementById('autoStartBtn');
  btn.disabled=true;
  document.getElementById('autoCancelBtn').disabled=false;
  document.getElementById('autoMsg').textContent='准备中...';
  fetch('/api/auto_annotate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({skip_labeled:skip,iou})}).then(r=>r.json()).then(d=>{
    if(!d.ok){alert(d.msg);btn.disabled=false;document.getElementById('autoCancelBtn').disabled=true;return;}
    pollAuto();
  });
}
function pollAuto(){
  if(!autoActive)return;
  fetch('/api/auto_annotate_status').then(r=>r.json()).then(d=>{
    document.getElementById('autoBar').style.width=(d.percent||0)+'%';
    const errs=d.errors||[];
    let msg='进度: '+d.done+' / '+d.total+' ('+(d.percent||0)+'%)\n'+
            '当前: '+(d.current||'')+'\n'+
            '已标注: '+d.saved+'  跳过: '+d.skipped;
    if(errs.length)msg+='\n错误 '+errs.length+' 个: '+errs.slice(0,3).join(' | ');
    document.getElementById('autoMsg').textContent=msg;
    if(d.running){
      autoTimer=setTimeout(pollAuto,700);
    }else{
      document.getElementById('autoStartBtn').disabled=false;
      if(d.finished){
        const cb=document.getElementById('autoCancelBtn');
        cb.disabled=false;cb.textContent='关闭';
        cb.onclick=()=>{closeOverlay('autoMask');autoActive=false;};
        st('自动化标注完成: 新增 '+d.saved+' 张, 跳过 '+d.skipped+' 张');
        refreshAfterAuto();
      }
    }
  });
}
function cancelAuto(){
  fetch('/api/auto_cancel',{method:'POST'});
  autoActive=false;
  closeOverlay('autoMask');
}
function refreshAfterAuto(){
  fetch('/api/files').then(r=>r.json()).then(d=>{
    if(d.ok){files=d.files;buildFileList();}
  });
  load(idx);
}

// ---------- 多模态标注 ----------
let vlmTestSec=0;
function openVlmAuto(){
  if(!classes.length){alert('请先打开文件夹并填写类别');return;}
  const m=document.getElementById('vlmModel').value.trim();
  const b=document.getElementById('vlmBase').value.trim();
  if(!m||!b){alert('请先在配置里填写接口地址和模型名');return;}
  document.getElementById('vlmNameTag').textContent=m;
  document.getElementById('vlmIou').value=document.getElementById('iouThr').value;
  document.getElementById('vlmBar').style.width='0%';
  document.getElementById('vlmMsg').textContent='';
  document.getElementById('vlmStartBtn').disabled=false;
  const cb=document.getElementById('vlmCancelBtn');
  cb.disabled=false;cb.textContent='取消';cb.onclick=cancelVlm;
  vlmActive=true;
  openOverlay('vlmMask');
}

function testVlm(){
  saveVlmCfg(false).then(()=>{
    const btn=document.getElementById('vlmTestBtn');
    btn.disabled=true;
    document.getElementById('vlmMsg').textContent='试标中...';
    const iou=parseFloat(document.getElementById('vlmIou').value)||0.5;
    const t0=Date.now();
    fetch('/api/vlm_test',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({idx,iou})}).then(r=>r.json()).then(d=>{
      btn.disabled=false;
      if(!d.ok){document.getElementById('vlmMsg').textContent='试标失败: '+(d.msg||'');return;}
      vlmTestSec=(Date.now()-t0)/1000;
      const s0=(d.stats||[])[0]||{};
      const valid=d.boxes.filter(b=>b&&isFinite(b.cx)&&isFinite(b.cy)&&b.w>0&&b.h>0);
      const added=dedupeAppend(valid,iou);
      if(added.length){snap();boxes.push(...added);markDirty();redraw();}
      let m='试标: '+(s0.raw||0)+'→'+(s0.kept||0)+'，新增 '+added.length+
            '，'+vlmTestSec.toFixed(1)+'s';
      if(s0.note)m+='\n'+s0.note;
      document.getElementById('vlmMsg').textContent=m;
    }).catch(()=>{
      btn.disabled=false;
      document.getElementById('vlmMsg').textContent='试标失败: 请求异常';
    });
  });
}
function startVlm(){
  saveLabels(true);
  saveVlmCfg(false).then(()=>{
    const skip=document.getElementById('vlmSkip').checked;
    const skipEmpty=document.getElementById('vlmSkipEmpty').checked;
    const iou=parseFloat(document.getElementById('vlmIou').value)||0.5;
    const btn=document.getElementById('vlmStartBtn');
    btn.disabled=true;
    document.getElementById('vlmCancelBtn').disabled=false;
    document.getElementById('vlmMsg').textContent='准备中...';
    fetch('/api/auto_annotate',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({skip_labeled:skip,skip_empty:skipEmpty,iou,vlm:true})
    }).then(r=>r.json()).then(d=>{
      if(!d.ok){document.getElementById('vlmMsg').textContent=d.msg||'启动失败';
        btn.disabled=false;document.getElementById('vlmCancelBtn').disabled=true;return;}
      pollVlm();
    });
  });
}
function pollVlm(){
  if(!vlmActive)return;
  fetch('/api/auto_annotate_status').then(r=>r.json()).then(d=>{
    document.getElementById('vlmBar').style.width=(d.percent||0)+'%';
    const errs=d.errors||[];
    let msg='进度: '+d.done+' / '+d.total+' ('+(d.percent||0)+'%)\n'+
            '当前: '+(d.current||'')+'\n'+
            '已标注: '+d.saved+'　跳过: '+d.skipped+
            (d.empty?('　空结果未写(待复查): '+d.empty):'')+
            (d.failed?('　失败: '+d.failed):'');
    if(errs.length)msg+='\n错误 '+errs.length+' 个: '+errs.slice(0,3).join(' | ');
    document.getElementById('vlmMsg').textContent=msg;
    if(d.running){
      vlmTimer=setTimeout(pollVlm,900);
    }else{
      document.getElementById('vlmStartBtn').disabled=false;
      if(d.finished){
        const cb=document.getElementById('vlmCancelBtn');
        cb.disabled=false;cb.textContent='关闭';cb.onclick=()=>{closeOverlay('vlmMask');vlmActive=false;};
        st('多模态标注完成: 标注 '+d.saved+' 张'
           +(d.empty?('，空结果未写 '+d.empty+' 张'):'')
           +(d.failed?('，失败 '+d.failed+' 张'):'')+'，请逐张复核');
        refreshAfterAuto();
      }
    }
  });
}
function cancelVlm(){
  fetch('/api/auto_cancel',{method:'POST'});
  vlmActive=false;
  clearTimeout(vlmTimer);
  closeOverlay('vlmMask');
}

// ---------- 多模态复核 ----------
let revTimer=null, revActive=false;
function openVlmReview(){
  if(!classes.length){alert('请先打开文件夹并填写类别');return;}
  const m=document.getElementById('vlmModel').value.trim();
  const b=document.getElementById('vlmBase').value.trim();
  if(!m||!b){alert('请先在配置里填写接口地址和模型名');return;}
  document.getElementById('revBar').style.width='0%';
  document.getElementById('revMsg').textContent='';
  document.getElementById('revStartBtn').disabled=false;
  document.getElementById('revApplyBtn').disabled=true;
  const cb=document.getElementById('revCancelBtn');
  cb.disabled=false;cb.textContent='取消';cb.onclick=cancelReview;
  revActive=true;
  openOverlay('revMask');
}
function fmtChanges(cs){
  return (cs||[]).map(c=>c.action==='drop'
      ? ('  删除 '+c.from)
      : ('  '+c.from+' → '+c.to)).join('\n');
}
function testVlmReview(){
  saveVlmCfg(false).then(()=>{
    const btn=document.getElementById('revTestBtn');
    btn.disabled=true;
    document.getElementById('revMsg').textContent='复核当前图...';
    fetch('/api/vlm_review_test',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({idx,drop_allowed:document.getElementById('revDrop').checked})
    }).then(r=>r.json()).then(d=>{
      btn.disabled=false;
      if(!d.ok){document.getElementById('revMsg').textContent='失败: '+(d.msg||'');return;}
      let s='本图 '+d.total+' 个框，复核后保留 '+d.kept+' 个';
      if(!d.changes.length)s+='，无改动';
      if(d.unparsed)s+='（'+d.unparsed+' 个未给结论，已保持原样）';
      document.getElementById('revMsg').textContent=s+'\n'+fmtChanges(d.changes);
    }).catch(()=>{btn.disabled=false;
      document.getElementById('revMsg').textContent='失败: 请求异常';});
  });
}
function startVlmReview(){
  saveLabels(true);
  saveVlmCfg(false).then(()=>{
    const btn=document.getElementById('revStartBtn');
    btn.disabled=true;
    document.getElementById('revCancelBtn').disabled=false;
    document.getElementById('revMsg').textContent='准备中...';
    fetch('/api/vlm_review',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({drop_allowed:document.getElementById('revDrop').checked,
                           dry_run:true})
    }).then(r=>r.json()).then(d=>{
      if(!d.ok){document.getElementById('revMsg').textContent=d.msg||'启动失败';
        btn.disabled=false;
        document.getElementById('revCancelBtn').disabled=true;return;}
      pollReview();
    });
  });
}
function pollReview(){
  if(!revActive)return;
  fetch('/api/auto_annotate_status').then(r=>r.json()).then(d=>{
    document.getElementById('revBar').style.width=(d.percent||0)+'%';
    let msg=d.done+' / '+d.total+' ('+(d.percent||0)+'%)　'+(d.current||'')+
            '\n改类 '+(d.reclassed||0)+'　删除 '+(d.dropped||0)+
            '　无改动 '+(d.unchanged||0)+(d.skipped?('　跳过 '+d.skipped):'')+
            (d.failed?('　失败 '+d.failed):'');
    const cs=d.changes||[];
    if(cs.length)msg+='\n最近改动:\n'+fmtChanges(cs.slice(-8));
    const errs=d.errors||[];
    if(errs.length)msg+='\n'+errs.slice(0,2).join(' | ');
    document.getElementById('revMsg').textContent=msg;
    if(d.running){
      revTimer=setTimeout(pollReview,900);
    }else{
      document.getElementById('revStartBtn').disabled=false;
      if(d.finished){
        if(d.changed)document.getElementById('revApplyBtn').disabled=false;
        const cb=document.getElementById('revCancelBtn');
        cb.disabled=false;cb.textContent='关闭';
        cb.onclick=()=>{closeOverlay('revMask');revActive=false;};
        st('复核完成: 建议改类 '+(d.reclassed||0)+'，建议删除 '+(d.dropped||0)+
           '，未改动 '+(d.unchanged||0)+'（尚未写盘）');
      }
    }
  });
}
function applyVlmReview(){
  const btn=document.getElementById('revApplyBtn');
  btn.disabled=true;
  fetch('/api/vlm_review_apply',{method:'POST'}).then(r=>r.json()).then(d=>{
    if(!d.ok){alert(d.msg||'应用失败');return;}
    st('已应用修改: '+d.applied+' 张');
    refreshAfterAuto();
  });
}
function cancelReview(){
  fetch('/api/auto_cancel',{method:'POST'});
  revActive=false;
  clearTimeout(revTimer);
  closeOverlay('revMask');
}

document.addEventListener('keydown',e=>{
  if(document.getElementById('modalMask').style.display==='flex')return;
  if(document.getElementById('splitMask').style.display==='flex')return;
  if(document.getElementById('autoMask').style.display==='flex')return;
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  if(e.ctrlKey||e.metaKey){
    if(e.key.toLowerCase()==='z'){e.preventDefault();if(e.shiftKey)redo();else undo();}
    else if(e.key.toLowerCase()==='y'){e.preventDefault();redo();}
    return;
  }
  const k=e.key.toLowerCase();
  if(k==='a')go(-1);
  else if(k==='d')go(1);
  else if(k==='s'){e.preventDefault();saveLabels();}
  else if(k==='e')predict();
  else if(e.key==='Delete'){if(selIdx>=0){snap();boxes.splice(selIdx,1);selIdx=-1;markDirty();redraw();}}
  else if(e.key>='0'&&e.key<='9'){const n=parseInt(e.key);if(n<classes.length){if(selIdx>=0){snap();boxes[selIdx].cls=n;markDirty();}else{curCls=n;buildClsBtns();}redraw();}}
});
setInterval(()=>{if(dirty&&total>0)saveLabels(true);},10000);
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});
window.addEventListener('blur',()=>{if(dirty&&total>0)saveLabels(true);});
window.addEventListener('resize',()=>{if(imgW)fit();});
</script>
</body>
</html>
"""


def open_browser():
    webbrowser.open(f"http://127.0.0.1:{PORT}")


if __name__ == "__main__":
    print("=" * 52)
    print("  YOLO 标注工具启动中...")
    print("=" * 52)
    # 关键:先清理占用端口的旧实例,否则"重启"只会连回旧服务器,
    # 显示的还是上一次的文件夹。
    if not free_port_if_stale(PORT):
        print(f"  端口 {PORT} 被占用且无法自动清理,已退出。")
        input("  按回车键关闭...")
        sys.exit(1)
    print(f"  浏览器将自动打开 http://127.0.0.1:{PORT}")
    print("  关闭此窗口即可退出工具")
    print("=" * 52)
    threading.Timer(1.5, open_browser).start()
    app.run(host="127.0.0.1", port=PORT, debug=False)
