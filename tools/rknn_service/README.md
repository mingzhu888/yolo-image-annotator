# RKNN 远程推理服务

让本机的 YOLO 标注工具能够使用 **RKNN 模型**（`.rknn`）做自动标注。

## 为什么需要它

`.rknn` 是 Rockchip NPU 的编译产物，只能在 **RK3588 这类设备的 NPU** 上执行：

- `rknn-toolkit2` 官方只提供 manylinux wheel，**没有 Windows 版**
- PC 上的模拟器**明确不支持**运行已编译的 `.rknn`
  （实测报错：`RKNN model that loaded by 'load_rknn' not support inference on the simulator`）
- 所以即使原始 `.pt/.onnx` 丢了，只要还有 `.rknn`，把它放到 3588 上仍然可以拿来标注

## 架构

```
PC（Windows，标注工具）                 RK3588（设备端）
+----------------------+  HTTP    +-----------------------+
| rknn://模型代码       | -------> | 本服务（真实 NPU）     |
| 127.0.0.1:8077       | <------- | rknn-toolkit-lite2    |
+----------------------+  boxes   +-----------------------+
        ^ 经 SSH 隧道映射，无需在设备上开放端口
```

## 设备端部署

设备需要已安装 `rknn-toolkit-lite2`（通常出厂自带），依赖 `numpy`、`opencv-python`。

```bash
mkdir -p ~/rknn_svc && cd ~/rknn_svc
nohup python3 rknn_service.py --port 8077 \
      --models /path/to/model_zoo > svc.log 2>&1 &
```

服务会自动扫描 `--models` 目录下**同时包含 `manifest.json` 和 `model.rknn`** 的子目录，
按 manifest 里的 `labels` / `input_size` / `conf_thres` 建立模型注册表。

## PC 端

不修改设备防火墙，用 SSH 隧道把设备端口映射到本机：

```bash
ssh -N -L 8077:127.0.0.1:8077 user@<device-ip>
```

然后在标注工具里：

1. 配置页填「RKNN 推理服务地址」= `http://127.0.0.1:8077`
2. 模型路径填 `rknn://模型代码`，例如 `rknn://person_infra`
3. 两级级联（先找人、再在人像裁剪上跑第二级）：
   `rknn://yqzk_smoke?person=person_infra`

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET | `/models` | 可用模型清单（代码、类别、输入尺寸） |
| POST | `/infer` | 推理 |

`POST /infer` 请求：

```json
{
  "model": "yqzk_smoke",
  "image": "<base64 jpeg/png>",
  "conf": 0.1,
  "person_model": "person_infra",
  "min_person_conf": 0.5,
  "crop_expand": 1.2
}
```

返回原图像素坐标的框：

```json
{"ok": true, "ms": 25.5,
 "boxes": [{"label": "hand_with_smoke", "conf": 0.786,
            "x1": 468, "y1": 430, "x2": 486, "y2": 462}]}
```

## 已知坑

- **颜色通道**：必须 `BGR -> RGB` 后再送 NPU，否则检测不到（本服务已处理）。
- **manifest 与模型可能不同步**：实测某设备上 `zql_smoke` 的 manifest 只写了 3 个类别，
  而模型实际输出 4 类，导致 `other` 被标成 `class_3`。可在服务目录下另放一份正确的 manifest。
- **裁剪类模型不能整帧用**：manifest 里 `skip_full_frame_inference: true` 表示它是二级模型，
  必须配合第一级人体检测（用 `person_model` 参数走级联）。
