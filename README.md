# YOLO 标注工具（YOLO Image Annotator）

一个运行在本机的 YOLO 检测图片标注工具：Flask 后端 + 浏览器前端，深色中文界面，无需前端构建。支持手动标注、模型辅助标注、类别过滤、预测去重、数据集自动划分与图片去重，适合个人和小团队快速完成标注并进入训练循环。

## 特色功能

### 手动标注

- 鼠标拖拽画框，画完可继续调整：点选框后拖动移动，拖动 8 个白色手柄缩放
- 双击框或按 `Delete` 删除；右侧列表改类别、删除
- 数字键 `0-9` 快速切换 / 修改类别
- 滚轮缩放、右键拖动平移
- 撤销 / 重做（`Ctrl+Z` / `Ctrl+Shift+Z` / `Ctrl+Y`），最多 50 步
- 文件列表实时显示已标注 / 未标注状态与整体进度

### 模型辅助标注（主动学习工作流）

推荐工作流：**先手动标注一部分 → 训练一个基础模型 → 用模型辅助标注剩余图片 → 修正后继续训练**，越标越多、模型越准、标注速度越快。

1. 先手动标注一部分图片（100~300 张），尽量覆盖不同场景、角度和光照
2. 用 ultralytics 训练一个基础模型：

   ```bash
   pip install ultralytics
   yolo detect train data=data.yaml model=yolov8s.pt epochs=100 imgsz=640 batch=16
   ```

3. 回到本工具，加载训练好的 `best.pt`，对剩余图片一键预测
4. 人工确认并修正预测框（移动 / 缩放 / 删改），保存后继续迭代训练

辅助标注细节：

- 置信度阈值可调（默认 0.25）
- 类别偏移：模型类别 ID + 偏移 = 标注类别 ID
- 类别过滤：模型有多个类别时，可勾选只输出需要的类别
- 预测去重：与已有框同类别且 IoU ≥ 0.5 自动去重，并过滤退化框
- 未安装模型依赖时，加载模型会给出明确的安装提示

### 数据集自动划分

一键把当前图片 + 标签随机划分为 `train / val`，并生成标准 `data.yaml`：

```
输出目录/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

- 支持验证集比例、随机种子、复制或移动、是否生成 `data.yaml`
- 没有标注框的图片会生成空 txt 作为负样本一起划分
- 划分完直接用生成的 `data.yaml` 训练即可

### 数据安全与工程细节

- 自动保存：每 10 秒自动保存；切图自动保存；窗口失焦保存；关闭页面前有未保存提示
- 标签校验：类别越界、坐标超出 0~1、NaN 等坏数据拒绝写入并提示，不会污染数据集
- 端口安全：启动时只清理本工具自己的旧实例，不误杀其他程序
- 图片去重工具：字节级重复隔离（预演 + 执行）、近重复检测

## 安装到本机

### 环境要求

- Windows 10 / 11 64 位（macOS / Linux 也可用，见“手动安装”）
- Python 3.8 及以上（建议 3.9 ~ 3.12，64 位）
- 可选：NVIDIA GPU + CUDA 加速模型预测；没有 GPU 也能用 CPU 正常标注

检查 Python 是否安装：

```bash
python --version
```

### 方式一：Windows 一键安装（推荐）

1. 下载本仓库 ZIP 并解压，或执行 `git clone https://github.com/mingzhu888/yolo-image-annotator.git`
2. 双击 `install_deps.bat`
   - 自动创建 `.venv` 虚拟环境
   - 自动安装完整依赖（含 ultralytics / PyTorch，首次约 5~15 分钟，取决于网速）
3. 双击 `start.bat` 启动
   - 自动使用虚拟环境运行，浏览器会自动打开 `http://127.0.0.1:5000`
4. 关闭黑色终端窗口即退出工具

### 方式二：手动安装（Windows / macOS / Linux）

```bash
# 1. 进入项目目录
cd yolo-image-annotator

# 2. 创建并激活虚拟环境
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. 安装依赖
# 完整功能（含模型辅助标注，推荐）：
pip install -r requirements-full.txt

# 或仅手动标注（轻量，不含模型辅助）：
pip install -r requirements.txt

# 4. 启动
python annotate_tool.py
```

浏览器会自动打开 `http://127.0.0.1:5000`，关闭终端窗口即退出。

### 没有 GPU / 想省空间：只装 CPU 版 PyTorch

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-full.txt
```

### 验证安装是否成功

```bash
python -c "import flask, PIL; print('core ok')"
python -c "from ultralytics import YOLO; print('model ok')"
```

两条都输出 `ok` 说明安装成功。第一次启动会自动在项目目录生成 `annotate_config.json`，无需手动创建。

### 更新到新版

```bash
git pull
pip install -r requirements-full.txt
```

## 使用说明

### 1. 打开数据

1. 点“⚙ 配置”
2. 填写图片文件夹路径
3. 填写标签文件夹路径（留空 = 与图片同目录）
4. 填写类别（英文逗号分隔，顺序即类别 ID，如 `Belt_off,Belt_on`）
5. 点“打开文件夹”

### 2. 手动标注

- 左键拖拽画框
- 点选框后：拖动移动、拖白色手柄缩放
- 右侧“本图标注框”可改类别 / 删除
- 按 `S` 保存，或直接切换图片（自动保存）

### 3. 模型辅助标注

1. 配置 → 模型权重路径填 `.pt` 文件（如 `weights/best.pt`）→ 点“加载模型”
2. 加载后勾选“只标注以下模型类别”（默认全选；全部不勾 = 不输出任何框）
3. 设置置信度阈值、类别偏移
4. 点“✨ 模型预测”，人工确认 / 修正后保存

### 4. 划分数据集

1. 点“📂 划分数据集”
2. 填输出目录、验证集比例、随机种子
3. 选择复制（默认）或移动；是否生成 `data.yaml`
4. 点“开始划分”，完成后直接用生成的 `data.yaml` 训练

## 快捷键

| 操作 | 按键 |
| --- | --- |
| 上一张 / 下一张 | `A` / `D` |
| 保存 | `S` |
| 模型预测 | `E` |
| 删除选中框 | `Delete` 或双击框 |
| 切换 / 修改类别 | `0-9` |
| 撤销 | `Ctrl+Z` |
| 重做 | `Ctrl+Shift+Z` 或 `Ctrl+Y` |
| 缩放 | 鼠标滚轮 |
| 平移 | 右键拖动 |
| 画框 | 左键拖拽 |
| 移动框 | 点选后左键拖动 |
| 缩放框 | 拖动白色手柄 |

## 图片去重工具

字节级去重（默认预演，`--apply` 才真正移动；重复文件移到隔离目录而不是删除）：

```bash
python _dedup.py --img-dir 图片目录 --label-dir 标注目录 --apply
```

近重复检测（只读，不删除任何文件）：

```bash
python _dup_check.py --img-dir 图片目录 --report 报告输出路径.txt
```

## 目录结构

```
yolo-image-annotator/
├── annotate_tool.py        # 主程序（Flask + 内嵌前端）
├── requirements.txt        # 轻量依赖（手动标注）
├── requirements-full.txt   # 完整依赖（含模型辅助标注）
├── install_deps.bat        # Windows 一键安装依赖
├── start.bat               # Windows 一键启动
├── _dedup.py               # 字节级去重工具
├── _dup_check.py           # 近重复检测工具
├── tests/                  # 回归测试
├── README.md
└── LICENSE
```

## 测试

```bash
python tests/test_predict_filter.py
python tests/test_save_validation.py
```

## 常见问题

**Q：启动提示端口被占用？**

工具启动时会自动清理自己之前的旧实例；如果端口被其他程序占用会明确提示，关闭占用程序后重试即可。

**Q：加载模型提示“未安装模型辅助依赖”？**

执行 `pip install -r requirements-full.txt`。

**Q：预测没有框？**

检查类别过滤是否勾选了需要的类别、置信度阈值是否过高（可降到 0.1~0.25）、模型是否加载成功。

**Q：保存提示类别越界？**

检查类别顺序是否与标注类别一致、类别偏移是否配置正确。

**Q：PyTorch 下载太慢？**

参考“只装 CPU 版 PyTorch”的安装命令，或使用代理 / 国内镜像源。

## Roadmap

- COCO / VOC 格式导入导出
- YOLO-seg 多边形标注
- 视频抽帧标注
- train / val / test 三段划分与 k-fold
- 整目录批量预标注

## License

MIT © 2026 mingzhu888
