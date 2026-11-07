# AI 接续开发文档：云企智瞰标注工具

> 基线版本：`c77c105`（`main`，2026-11-07）
>
> 远程仓库：`https://github.com/mingzhu888/yolo-image-annotator.git`
>
> 本文给下一个接手的 AI/开发者使用。先读完本文，再查看 `README.md` 和源码。
> 文档中的“已验证”表示在当前代码和本机环境实际执行过；“已知风险”表示从源码
> 可以确认、但还没有完整产品级验证；“待决策”表示需要结合用户需求判断，不能擅自改。

## 1. 项目目标与边界

这是一个面向个人和小团队的 Windows 本地 YOLO 图片标注工具，当前产品名为
**云企智瞰标注工具**。

核心目标：

- 在本机选择图片目录和标签目录，标注结果写入 YOLO 格式 `.txt`
- 支持手动画框、移动、缩放、删除、改类别、撤销/重做、自动保存
- 用本地 `.pt` 模型辅助单图预测和整目录自动标注
- 支持 YOLOv5 `.pt`、Ultralytics `.pt`、美团 YOLOv6n checkpoint
- 通过 RK3588 设备上的远程服务使用 `.rknn` 模型
- 一键划分 `train/val` 并生成 `data.yaml`
- 提供带安装向导的 Windows EXE，最终用户不需要配置 Python 环境

当前明确不在产品范围内的内容：

- 多模态/VLM 自动标注 UI 已经从产品中移除
- 当前没有云端标注入口、API Key 输入框或 VLM 按钮
- 产品不是 Web SaaS，默认只在本机运行
- `.rknn` 不在 Windows 上直接执行；必须借助 RK3588 设备

用户未来仍可能重新考虑“检测器出框 + 多模态复核”或 grounding/SAM 方案，但必须以
明确的新需求为准。不要把旧调研文档直接当成当前产品规格。

## 2. 当前状态快照

当前 Git 基线：

- 分支：`main`
- 提交：`c77c105 feat: 添加桌面版品牌图标与 Windows 安装程序`
- 远端：`origin/main` 已同步
- 基线提交时 Git 工作区干净

当前主要产物：

- 桌面 EXE：`dist/云企智瞰标注工具/云企智瞰标注工具.exe`
- 安装包：`installer/output/云企智瞰标注工具_安装程序_1.2.1.exe`
- 品牌 PNG：`assets/yunqi-zhikan-annotator.png`
- Windows ICO：`assets/yunqi-zhikan-annotator.ico`

最后一次打包验证：

- EXE 使用 PyInstaller `onedir` 构建成功
- 安装包由 Inno Setup 7 构建成功
- 安装包约 2,622,888,062 字节
- 安装程序退出码 `0`
- 安装程序正确识别已有 WebView2，没有再次执行 WebView2 bootstrapper
- 安装后的窗口能启动，窗口标题为 `云企智瞰标注工具`
- 安装包和 EXE 当前未做代码签名，其他电脑可能出现 SmartScreen 提示

功能回归：

- `tests/test_core.py`：9 项通过
- `tests/test_auto_annotate.py`：3 项通过
- `tests/test_predict_filter.py`：10 项通过
- `tests/test_save_validation.py`：6 项通过
- `tests/test_folder_picker.py`：1 项通过
- 合计 29 项通过

这些测试主要覆盖纯逻辑、Flask test client 和假模型。它们不能替代真实 `.pt`、
RKNN 设备和真实数据集验收。

## 3. 快速开始

当前开发机使用的主 Python 环境：

```text
F:\miniconda\envs\yolo_4060\python.exe
```

启动源码版浏览器模式：

```powershell
cd F:\Codex\开源工具
F:\miniconda\envs\yolo_4060\python.exe annotate_tool.py
```

启动源码版桌面窗口：

```powershell
cd F:\Codex\开源工具
F:\miniconda\envs\yolo_4060\python.exe desktop_app.py
```

安装开发依赖：

```powershell
F:\miniconda\envs\yolo_4060\python.exe -m pip install -r requirements-full.txt
F:\miniconda\envs\yolo_4060\python.exe -m pip install -r requirements-build.txt
```

运行测试：

```powershell
F:\miniconda\envs\yolo_4060\python.exe tests\test_core.py
F:\miniconda\envs\yolo_4060\python.exe tests\test_auto_annotate.py
F:\miniconda\envs\yolo_4060\python.exe tests\test_predict_filter.py
F:\miniconda\envs\yolo_4060\python.exe tests\test_save_validation.py
F:\miniconda\envs\yolo_4060\python.exe tests\test_folder_picker.py
```

当前环境中没有安装 `pytest`，测试文件都提供了 `__main__` 入口，所以直接以脚本
方式运行是当前可靠的回归方式。将来若引入 pytest，应把它加入独立的开发依赖文件，
不要把测试依赖打进正式 EXE。

## 4. 总体架构

```text
desktop_app.py
  ├─ Single-instance mutex
  ├─ Flask/Werkzeug HTTP server on 127.0.0.1:<random port>
  └─ pywebview + WebView2 desktop window
          │
          ▼
annotate_tool.py
  ├─ Flask API
  ├─ Embedded HTML/CSS/JavaScript
  ├─ In-memory STATE
  ├─ YOLO label read/write/validation
  ├─ Local model adapters
  ├─ Batch annotation worker
  └─ Dataset split
          │
          ├─ local .pt ──> torch / ultralytics
          │
          └─ rknn://... ──HTTP──> RK3588 rknn_service.py
```

### 4.1 运行入口

`annotate_tool.py`

- 浏览器开发模式入口
- 固定端口 `5000`
- 启动后自动打开系统浏览器
- 同时包含 Flask 后端和全部内嵌前端

`desktop_app.py`

- 正式桌面入口
- 用 `make_server("127.0.0.1", 0, ...)` 随机选择空闲端口
- 用 pywebview 的 `edgechromium` 后端显示窗口
- 不打开外部浏览器，不显示控制台窗口
- 单实例互斥锁名为 `Local\YoloImageAnnotatorDesktop`
- 启动异常写入本地 `desktop_error.log`

正式打包入口是 `desktop_app.py`，不是 `annotate_tool.py`。

## 5. 数据与配置路径

### 5.1 配置目录

源码运行时：

```text
<repo>/annotate_config.json
```

打包运行或安装版运行时：

```text
%LOCALAPPDATA%\YunqiZhikanAnnotator\annotate_config.json
%LOCALAPPDATA%\YunqiZhikanAnnotator\desktop_error.log
```

可以通过环境变量覆盖：

```text
YOLO_ANNOTATOR_DATA_DIR=<自定义目录>
```

关键实现：`annotate_tool.runtime_dir()`。

### 5.2 用户数据

用户数据和配置分开：

- 图片目录和标签目录由用户在界面中选择
- 程序不会默认复制用户图片
- 标签文件与图片同 stem，例如 `abc.jpg` 对应 `abc.txt`
- `annotate_config.json` 只存路径、类别和模型元信息，不存图片

### 5.3 主要配置字段

```json
{
  "img_dir": "F:\\dataset\\images",
  "label_dir": "F:\\dataset\\labels",
  "classes": "person,helmet,no-helmet",
  "device": "auto",
  "rknn_url": "http://127.0.0.1:8077",
  "models": [
    {
      "path": "F:\\models\\best.pt",
      "backend": "ultralytics",
      "cls_offset": 0,
      "conf": 0.25,
      "only_cls": [0, 1]
    }
  ]
}
```

可能存在的兼容字段：

- `v6n_repo`：美团 YOLOv6 源码路径
- `rknn_min_person_conf`：RKNN 两级级联的第一级人体置信度
- `rknn_crop_expand`：人体裁剪扩边比例
- `rknn_timeout`：RKNN HTTP 超时
- `vlm`：旧多模态功能遗留字段，当前 UI 不使用

不要提交真实 API Key、账号密码、设备口令或用户数据集路径到 Git。

### 5.4 配置持久化行为

- 打开数据目录时保存 `img_dir`、`label_dir` 和 `classes`
- 切换 GPU/CPU 后保存 `device`
- 保存 RKNN 服务地址
- 模型加载、参数修改、卸载后保存模型元信息
- 下次启动会恢复上次目录和模型卡片参数
- 模型卡片恢复后仍需要用户重新点击“加载模型”；不会自动把大模型加载进显存

`save_config()` 当前会吞掉写入异常，因此配置保存失败不一定有 UI 提示。

## 6. YOLO 标签规则

内部框结构：

```json
{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.2}
```

写入格式：

```text
0 0.500000 0.500000 0.100000 0.200000
```

规则：

- `cls` 必须是整数
- `cx/cy/w/h` 必须是有限数字且在 0~1
- `w/h` 必须大于 0
- 越界类别直接拒绝，不允许 clamp 到最后一类
- 覆盖已有标签前，如果同目录还没有 `.bak`，会备份一次

重要行为：

- 空 `.txt` 在 YOLO 训练中表示“负样本”
- 当前 `has_label()` 把空文件视为“未标注”
- 因此空标签会显示为未完成，也可能被批量标注再次处理

如果产品需要区分“明确确认为负样本”和“从未标注”，必须增加独立状态，不能继续
只看文件是否存在且非空。

## 7. 手动标注流程

前端内嵌在 `annotate_tool.py` 的 `HTML_PAGE` 中。

主要能力：

- 左键拖动画框
- 点选框后移动
- 拖动 8 个手柄缩放
- 双击框或 `Delete` 删除
- 数字键切换/修改类别
- `Ctrl+Z`、`Ctrl+Shift+Z`、`Ctrl+Y`
- 滚轮缩放，右键拖动平移
- 每 10 秒自动保存
- 切图、窗口失焦、关闭页面前触发保存
- 文件列表显示完成状态
- 深色/浅色主题存于 WebView `localStorage`

注意：

- 服务端 `STATE` 是全局内存状态，没有按会话隔离
- 同一进程只适合单用户使用
- 不要在没有认证的情况下把 Flask 服务暴露到局域网

## 8. 模型后端

统一的预测数据契约是：

```json
{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "conf": 0.9}
```

`cls` 最终应是数据集类别 ID：

- 小于 0 或大于等于数据集类别数时丢弃
- 不允许夹到最后一类

### 8.1 `ultralytics`

- 先用 `ultralytics.YOLO(path)` 加载
- 读取 `model.names`
- 按配置把模型移动到 `cuda:0` 或 `cpu`
- 预测时读取 `res.boxes.xywhn`、`conf` 和 `cls`

### 8.2 `yolov5`

- 不依赖外部 YOLOv5 仓库
- `_install_yolov5_shim()` 动态注册 `models.*` 兼容模块
- 自定义 letterbox、YOLOv5 解码和类内 NMS
- 必须保持权重 pickle 中需要的模块名与官方一致

改动这里时要重点运行：

- `tests/test_core.py::test_v5_letterbox_geometry`
- 真实 YOLOv5 v5/v6/v7 权重验收

### 8.3 `meituan_v6n`

- 从 checkpoint 的 `model` 或 `ema` 取模型
- 需要 YOLOv6 源码
- 源码路径优先级：
  1. 环境变量 `YOLOV6_REPO`
  2. 配置 `v6n_repo`
  3. 程序旁边的 `YOLOv6` 目录

### 8.4 `rknn_remote`

模型路径格式：

```text
rknn://<model-code>
rknn://<target-model>?person=<person-model>
```

第二个格式用于两级级联。

### 8.5 类别偏移与过滤

- 类别偏移公式：`数据集类别 ID = 模型类别 ID + cls_offset`
- `only_cls` 使用模型自身类别 ID，在加偏移之前过滤
- `None` 表示不过滤
- 空集合表示全部过滤
- 类别名自动匹配必须连续匹配；不能连续匹配时返回 `None`

### 8.6 当前 NMS 语义存在规格不一致

`_nms_boxes()` 当前按 `(模型来源, 类别)` 分组。因此：

- 同一模型内的同类重叠框会去重
- 不同模型检出的同类框会同时保留
- 不同类别的框始终保留

`tests/test_predict_filter.py` 明确锁定了“不同模型的同类框都保留”这一行为。

但 README/旧说明曾写过“跨模型 NMS 去重”，这与当前测试和实现不一致。

批量写盘前又调用了 `_dedupe_with_existing()`，它会按类别将预测框与已有结果去重。
所以单图“模型预测”和批量“自动标注”的实际合并语义并不完全相同。

下一个开发者应先向用户确认想要哪一种：

- A：每个模型结果都保留，供人工对照
- B：跨模型同类只保留最高置信度
- C：单图预览保留，写盘时统一融合

确认后同时改实现、测试和 README。

## 9. RKNN 远程标注

### 9.1 为什么必须远程

`.rknn` 是 Rockchip NPU 编译产物。PC 上的 `rknn-toolkit2` 模拟器不能对
`load_rknn()` 后的模型做真实推理。当前受支持的方式是：

```text
Windows 标注工具
  127.0.0.1:8077
       │ SSH 本地端口转发
       ▼
RK3588 上的 rknn_service.py
       │
       ▼
真实 NPU 推理
```

### 9.2 设备端服务

文件：

- `tools/rknn_service/rknn_service.py`
- `tools/rknn_service/README.md`
- `tools/rknn_service/start_service.sh`
- `tools/rknn_service/rknn-svc.service`

服务扫描同时包含以下文件的子目录：

```text
<model-dir>/manifest.json
<model-dir>/model.rknn
```

`manifest.json` 至少应提供：

- `algorithm_code`
- `engine_format: "rknn"`
- `labels`
- `input_size`
- 可选 `conf_thres`、`iou_thres`、`model_role`

接口：

- `GET /health`
- `GET /models`
- `POST /infer`

### 9.3 Windows 端流程

1. 在 RK3588 启动服务
2. 在 Windows 建立 SSH 隧道：

   ```powershell
   ssh -N -L 8077:127.0.0.1:8077 <user>@<device-ip>
   ```

3. 配置页填写 `http://127.0.0.1:8077`
4. 选择模型并加载
5. 返回主界面执行“模型预测”或“自动化标注”

不要把设备账号密码写入仓库或本文。每次需要凭据时向用户获取，并避免在输出中回显。

### 9.4 RKNN 已知坑

- 输入必须从 OpenCV BGR 转 RGB
- manifest 的 `labels` 必须与模型实际输出类别数量和顺序一致
- 裁剪模型不能整帧直接推理，必须使用 `person` 级联
- 远距离小目标在人体裁剪或预处理后仍可能丢失
- `rknn_service.py` 的解码目前按 YOLOv5 anchor 输出实现，不能假定所有 RKNN 模型
  都能直接使用
- 服务默认监听 `0.0.0.0` 且没有认证；安全部署应监听 loopback 并通过 SSH 隧道访问
- 默认模型目录包含特定机器的绝对路径，迁移前必须改参数

### 9.5 RKNN 类别偏移的潜在 bug

`_infer_rknn_remote()` 已经把模型返回的 `label` 通过 `_map_label_name()` 映射成
数据集类别 ID。随后 `_predict_all()` 又会对 `rknn_remote` 应用 `cls_offset`。

如果 `cls_offset` 非 0，就可能发生“双偏移”。

下一次修改 RKNN 相关代码时，应先增加一个针对“类别名已映射 + 非零偏移”的回归测试，
再决定统一契约：

- 所有 backend 都返回模型自身类别 ID，由上层统一加偏移；或
- RKNN backend 返回最终数据集类别 ID，并对 RKNN 跳过 offset

不能只改一边就让测试通过。

## 10. 单图和批量标注

### 10.1 单图预测

`POST /api/predict/<idx>`：

1. 遍历 `STATE["models"]`
2. 按 backend 推理
3. 应用类别过滤和偏移
4. 丢弃越界类别
5. 按 `(source, cls)` 去重
6. 返回未保存的框列表

用户需要人工确认后再保存。

### 10.2 自动标注

`POST /api/auto_annotate` 启动后台线程。

状态字段：

- `running`
- `finished`
- `cancel`
- `total`
- `done`
- `saved`
- `skipped`
- `failed`
- `empty`
- `current`
- `percent`
- `errors`

行为：

- `skip_labeled=true` 时跳过已有非空标签
- 每个文件调用 `_predict_all()`
- 读取已有框并调用 `_dedupe_with_existing()`
- 写盘前再次做标签校验
- 最多在 UI 状态中保留前 20 条错误
- 取消只在每个文件开始前检查

当前需要关注的问题：

- `empty` 字段存在，但 worker 没有在“预测为空”时递增它
- 没有预测时仍可能写出空 `.txt`，这表示负样本，不一定是用户期望
- 批量任务是内存线程，没有持久任务队列；应用退出后不会恢复
- 即使 `skip_labeled=false`，当前也是与已有框合并，不是无条件清空后覆盖

如果用户要求“员工模式”或大规模稳定运行，应把批量任务改成有明确状态、日志和
可恢复机制的后台任务。

## 11. 数据集划分

`POST /api/split` 输出：

```text
out/
├─ images/
│  ├─ train/
│  └─ val/
├─ labels/
│  ├─ train/
│  └─ val/
└─ data.yaml
```

规则：

- 按 `val_ratio` 随机切分
- 使用固定 seed 保证同输入可复现
- 默认复制，可选移动
- 没有标签文件的图片会生成空 `.txt` 作为负样本
- `data.yaml` 写入绝对输出路径

潜在问题：

- 选择“移动”后，`STATE["img_dir"]` 和 `STATE["label_dir"]` 仍指向旧目录
- 移动完成后当前界面可能继续引用已经不存在或变空的源目录
- 如果输出目录位于源目录内部，下一次 `list_images()` 可能把子目录情况复杂化；
  当前只扫描源目录第一层文件，不会递归

修改切分功能时必须增加只读/移动/负样本/同 stem 冲突的测试。

## 12. HTTP API

服务只应绑定 `127.0.0.1`。

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/` | 返回内嵌前端 |
| GET/POST | `/api/config` | 读取/更新设备与 RKNN 地址 |
| GET | `/api/device` | PyTorch/CUDA 诊断 |
| GET | `/api/rknn/models` | 代理远程 RKNN 模型清单 |
| POST | `/api/open` | 打开图片目录并保存配置 |
| GET | `/api/files` | 当前文件与完成状态 |
| POST | `/api/pick_file` | 调 Windows 文件选择器 |
| POST | `/api/pick_folder` | 调 Windows 文件夹选择器 |
| GET | `/api/models` | 当前已加载模型 |
| POST | `/api/suggest_offset` | 自动推断类别偏移 |
| POST | `/api/load_model` | 加载 `.pt` 或 `rknn://` |
| POST | `/api/update_model` | 更新偏移/阈值/类别过滤 |
| POST | `/api/unload_model` | 卸载模型 |
| GET | `/api/image/<idx>` | 返回当前图片 |
| GET | `/api/meta/<idx>` | 返回图片尺寸和已有框 |
| POST | `/api/save/<idx>` | 校验并保存标签 |
| POST | `/api/predict/<idx>` | 单图多模型预测 |
| POST | `/api/auto_annotate` | 启动批量任务 |
| GET | `/api/auto_annotate_status` | 查询批量进度 |
| POST | `/api/auto_cancel` | 请求取消批量任务 |
| POST | `/api/split` | 划分数据集 |

这些接口没有认证和 CSRF 防护，设计前提是“仅本机单用户”。

## 13. 前端结构

前端不是独立工程，全部嵌在 `annotate_tool.py` 的 `HTML_PAGE` 字符串中。

优点：

- 单文件部署简单
- 不需要 npm 或前端构建

缺点：

- Python 和 HTML/CSS/JS 强耦合
- 语法错误只能在运行时发现
- 测试 DOM 只能通过字符串断言或 Flask test client
- 可访问性、组件化和状态管理难以演进

如果准备长期维护，建议下一步拆成：

```text
annotator/
  server.py
  config.py
  labels.py
  models/
    ultralytics_backend.py
    yolov5_backend.py
    meituan_backend.py
    rknn_backend.py
  batch.py
  split.py
  web/
    index.html
    app.js
    app.css
```

拆分时要先补 Flask API 集成测试，避免一次大重构。

## 14. 桌面窗口

`desktop_app.py` 的关键点：

- pywebview + WebView2，不依赖系统默认浏览器
- Flask 使用随机端口，默认只监听 `127.0.0.1`
- 使用 Windows 命名 mutex 防重复实例
- 窗口 1440×900，最小 1000×680
- 窗口显示后会短暂置顶并恢复
- 出错日志写入数据目录

已知限制：

- 第二次双击只弹“已经在运行”，不会把已有窗口带到前台
- 缺少 WebView2 时必须依靠安装程序或用户手动安装
- pywebview 和浏览器启动入口并存，容易测试错入口

## 15. 打包与安装程序

### 15.1 EXE

文件：

- `build_exe.bat`
- `YoloAnnotator.spec`
- `requirements-build.txt`

构建命令：

```powershell
cd F:\Codex\开源工具
cmd.exe /c build_exe.bat
```

当前构建脚本硬编码 Python 路径：

```text
F:\miniconda\envs\yolo_4060\python.exe
```

换机器前必须修改脚本或改成环境变量/命令行参数。

Spec 会收集：

- 全部 Ultralytics 子模块
- pywebview、pythonnet、clr_loader
- Torch/CUDA 原生库

因此产物很大，并且可能包含 IPython、jedi、测试包等不必要内容。优化安装体积时，
应重点调整 `hiddenimports`、`excludes` 和 PyInstaller hook，而不是只换压缩算法。

### 15.2 安装包

文件：

- `installer/YoloAnnotator.iss`
- `build_installer.bat`
- `installer/redist/MicrosoftEdgeWebview2Setup.exe`
- `installer/redist/vc_redist.x64.exe`

`redist` 和 `installer/output` 被 `.gitignore` 排除，所以克隆仓库后必须自行准备两个
运行时安装器。

当前 Inno Setup 路径：

```text
%LOCALAPPDATA%\Programs\Inno Setup 7\ISCC.exe
```

关键安装行为：

- `PrivilegesRequired=lowest`
- 默认目录 `%LOCALAPPDATA%\Programs\云企智瞰标注工具`
- 中文安装向导
- 可选桌面快捷方式
- 开始菜单快捷方式和卸载入口
- 自动检测 VC++ 和 WebView2

### 15.3 WebView2 检测

正确的 WebView2 产品 GUID：

```text
{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}
```

旧安装包曾使用错误 GUID，导致机器已经安装 WebView2 时仍重复运行 bootstrapper，
返回 `-2147219416` / `0x80040828`。这个问题已在 `1.2.1` 修复。

不要把这个 GUID 再改回旧值。

### 15.4 未解决发布问题

- EXE 和安装包未签名
- 其他电脑可能触发 SmartScreen
- 需要正式代码签名证书才能消除陌生发布者提示
- 当前安装包约 2.44 GB
- WebView2 bootstrapper 是联网安装器；目标机器无网络时需要额外准备离线运行时
- 构建不是完全可复现：依赖版本未完全锁定，构建路径和 Inno Setup 路径是硬编码

发布前必须重新计算安装包 SHA256，不要复制旧哈希当成当前产物。

## 16. 当前技术债与风险

### P0：发布前优先确认

1. **RKNN 可能双应用类别偏移**
   - 见 9.5
   - 会直接把类别标错

2. **多模型 NMS 的产品语义不一致**
   - 单图预览、批量写盘和 README 描述不完全一致
   - 见 8.6

3. **安装包未签名**
   - 面向员工分发时会影响信任和安装体验

4. **真实模型验收不足**
   - 当前自动测试没有真实 `.pt`、真实 RKNN 和真实业务数据

### P1：下一步整理

5. **旧 VLM 死代码仍存在**
   - UI、API 路由和测试已移除
   - 后端仍保留 `VLM_DEFAULTS`、`vlm_config`、`_vlm_request`、`_infer_vlm`、
     `_vlm_review_image`、`_apply_review` 等函数
   - 删除时保留 `_norm_cls_name`、`_map_label_name`、`_parse_class_spec`、
     `_suggest_offset`
   - `_parse_class_spec` 当前有重复定义，清理时要保留兼容行为

6. **空标签语义不完整**
   - 空 txt 是负样本，但 UI 当成未标注

7. **批量任务状态只存在内存**
   - 应用退出后任务丢失
   - 没有持久日志和恢复

8. **配置写入异常被静默吞掉**
   - 无法可靠提示磁盘权限或空间问题

9. **切分后“移动模式”仍保留旧 STATE 路径**
   - 可能导致后续保存/预测失败

10. **RKNN 设备服务安全边界不足**
    - 默认 `0.0.0.0`、无认证

### P2：可维护性

11. `annotate_tool.py` 约 146 KB，后端、前端和业务逻辑全部混在一个文件
12. 安装包包含大量不必要的第三方文件
13. 构建脚本硬编码本机路径
14. 目录内还有 `_rknn_test`、`backup`、生成的 `annotate_config.json` 等本地状态
15. 第二次启动应用不能聚焦已有窗口

## 17. 旧多模态调研文档的使用边界

`docs/vlm-annotation-research.md` 是历史调研记录，包含有价值的社区方案、坐标体系、
SAM/grounding 资料和实测记录，但**不是当前产品规范**。

阅读时注意：

- 当前 UI 已没有多模态入口
- 文中有“28.8% 废框”被勘误为 3.8% 的说明
- 后面部分仍偶有沿用 28.8% 的旧表述
- 多模态复核实测准确度不足
- 不要把文档中存在过的路由/按钮当成当前实现

如果未来恢复多模态功能，建议重新写一份新的产品规格和 ADR，不要直接回滚旧 UI。

## 18. 推荐的下一阶段任务

### 任务 A：修正并锁定 RKNN 类别契约

优先级最高。

完成标准：

- 明确 RKNN backend 返回的是模型类别还是数据集类别
- 加测试覆盖 `label` 映射和非零 `cls_offset`
- 用至少一个真实 RKNN 模型验证类别名、头部、手部等映射
- 更新 RKNN README 和主 README

### 任务 B：统一多模型融合语义

完成标准：

- 明确跨模型同类框是否去重
- 单图预测和批量写盘采用同一套规则，或明确说明两者不同
- 更新 `_nms_boxes()`、`_dedupe_with_existing()` 和测试
- README 不再使用与实现矛盾的描述

### 任务 C：完善负样本和批量状态

完成标准：

- 区分“未标注”“空标签负样本”“有目标”
- 自动标注无预测时由用户配置是否写空标签
- `empty` 计数准确
- UI 显示失败、跳过、空结果和错误明细

### 任务 D：删除 VLM 死代码

完成标准：

- 删除未调用的 VLM 函数和配置读写
- 保留类别名映射和旧类别提示语法兼容
- 所有测试通过
- 全局搜索不再出现可执行的 VLM 路由或前端按钮

### 任务 E：发布工程化

完成标准：

- 构建脚本不再依赖固定 `F:` 路径
- 准备可重现的 redist 获取脚本
- 对 EXE 和安装包签名
- 在干净 Windows 虚拟机上从安装到启动完整验收
- 生成校验值和发布说明

### 任务 F：体积优化

可考虑：

- CPU-only 安装包
- CUDA 运行库按需安装
- 更精确的 PyInstaller excludes
- 排除 IPython、jedi、tests、示例资源和未使用 backend

验收时必须确认 `.pt`、YOLOv5、RKNN 和 WebView2 仍正常。

## 19. 调试手册

### 桌面窗口打不开

1. 查看：

   ```text
   %LOCALAPPDATA%\YunqiZhikanAnnotator\desktop_error.log
   ```

2. 检查 WebView2 Runtime
3. 检查 VC++ x64 运行库
4. 直接运行 `dist\云企智瞰标注工具\云企智瞰标注工具.exe`
5. 用源码运行 `desktop_app.py` 获取更直接异常

### 配置丢失

1. 确认是否运行了打包版
2. 检查 `%LOCALAPPDATA%\YunqiZhikanAnnotator\annotate_config.json`
3. 检查是否设置了 `YOLO_ANNOTATOR_DATA_DIR`
4. 检查目录写权限

### 本地模型加载失败

1. 检查 `torch`、`ultralytics`
2. 检查 `.pt` 是否来自 YOLOv5 而不是其它自定义 pickle
3. 查看 `/api/device`
4. 临时切换到 CPU，排除 CUDA/显存问题
5. 检查模型类别名与数据集类别名

### RKNN 模型不可达

1. 在设备上请求 `/health`
2. 检查 `/models`
3. 检查 SSH 隧道是否监听本机 `8077`
4. 检查 `manifest.json` 和 `model.rknn`
5. 检查模型是否需要 `person=` 级联

### 类别明显错位

依次检查：

1. 数据集类别顺序
2. 模型 `names`
3. `cls_offset`
4. `only_cls`
5. RKNN 是否发生双偏移
6. 是否存在旧 `.bak` 或旧标签文件

### 自动化标注卡住

1. 查询 `/api/auto_annotate_status`
2. 看 `current` 是哪张图
3. 看 `errors`
4. 检查该图的 backend 是否阻塞
5. RKNN 场景检查 HTTP 超时
6. 注意任务在应用退出后不会恢复

## 20. 接手后的操作顺序

建议下一个 AI 按以下顺序工作：

1. `git status --short`
2. `git log -5 --oneline`
3. 阅读本文件
4. 运行 5 个测试文件
5. 确认用户这次要改的是 UI、模型、RKNN、数据还是打包
6. 只修改相关模块
7. 用真实数据或真实 backend 验证关键行为
8. 更新 README 和本文件中的状态
9. 扫描 secret
10. 构建 EXE
11. 如有安装器改动，构建并实际安装
12. 最后再声明完成

## 21. 不应擅自做的事情

- 不要重新加入多模态 UI，除非用户明确要求
- 不要提交 API Key、SSH 密码、设备账号或私有数据集路径
- 不要把 Flask 服务绑定到 `0.0.0.0`
- 不要删除用户数据或标签
- 不要用 clamp 修复越界类别
- 不要为了隐藏报错而吞掉模型/配置异常
- 不要只改 README 来“修复”实现与测试的矛盾
- 不要在未注册真实模型验证的情况下声称 RKNN 或模型标注“完全可用”
- 不要把 `dist`、`build`、`installer/output`、`installer/redist` 提交到 Git
- 不要假定所有 `.rknn` 都是同一种输出布局

## 22. 给用户的交付口径

当前可以确认的能力：

- 本机桌面窗口可启动
- 手动标注、保存、自动标注框架、切分框架和本地模型适配器已实现
- 安装包可安装
- 自动测试通过

需要谨慎表述的能力：

- 某个具体业务模型是否准确，必须用真实数据实测
- RKNN 模型是否兼容，取决于输出布局、manifest 和级联配置
- 安装包尚未签名
- 当前构建环境和路径绑定特定开发机

当用户问“为什么”时，先给出可复现证据，再改代码；不要只凭文件名或旧对话猜测。
