# 多模态自动标注：社区方案调研与本机实测

> 调研目的：本工具的多模态自动标注质量不理想，找出社区/工业界已验证的更好做法。
> 文中每条结论都标注了来源。**「实测」= 在本机 465 张监控图上跑出来的数字；「文档原文」= 官方仓库/文档里逐字抄录；「推断」= 我的判断，未经实测。**

## 一、结论：三条路，按投入产出比排序

### A. 检测器出框 + 多模态判裁剪

让已有的 YOLO 模型出 person / head 的框——**定位这件事检测器是可靠的**——然后把裁剪出来的小图交给多模态，只回答"这里戴没戴"。多模态不再负责"找在哪"，只负责"是什么"。

> **注意**：本条初版声称"直接消灭 no-vest 28.8% 的废框"，该数字已被证伪（见第二节勘误，真实为 3.8%）。
> 这条路的价值需要重新评估 —— 它仍然能治"状态类指代不明"这个结构问题，但**不再是基于高废框率提出的**。
> 另据第四章的本地 A/B 实测，在当前数据上它的收益尚未得到验证。

- 改动量：小。不需要下载任何新模型，改的是喂给多模态的图——从整张图换成小裁块
- 同类先例：本机 65 机上 `zql_garbage` 找候选、`zql_garbage_classify` 判裁剪，就是同一套结构，是你自己验证过的
- 顺带修好：之前发现的 helmet 误报。框的位置本来就对，错的是类别判断
- 风险：依赖人/头检测器在每个场景都稳；裁剪会丢掉上下文（比如"这个人是不是在施工区"）

### B. 用本地 grounding 模型替掉 VLM 出框

用 Florence-2 / Grounding DINO 这类**开放词表检测器**直接产框，不加 SAM 就用它的框，加了 SAM 2 就是社区标准的 Grounded-SAM 两段链，框会明显更紧。

- 改动量：中。要装环境、下权重
- **但对当前这批数据投入产出比最低**：对 helmet 只有 1.6% 的改善空间，对 no-vest 完全无效——因为"没穿反光衣"本身不是一个可以在画面里指认的物体，grounding 模型同样找不到它
- 什么时候值得：等你未来要标的**全是真实存在的物体**（灭火器、烟头、车辆、person）时，这条路才是对的

### C. 分块推理 / SAHI

把整图切成 2×2 或 3×3 分别送。治的是**漏检**，不治废框。

- 你的 465 张里有 17 个空标签文件，这是 C 的地盘
- 代价：调用次数翻几倍，费用和时间同比上升

## 二、实测：几何废框率很低（含一次度量错误的更正）

> **勘误**：本节初版给出的"no-vest 28.8% 废框、合计 13.8%"是**错误的**。
> 原判定用了归一化坐标计算 h/w，没有还原图像宽高比。800×448 的图里，一个正常站立的人
> 归一化 h/w 就有 4.3，会被误判成"过于细长"。改用像素口径重算后，真实数值如下表。

判定标准（像素口径）：`像素 h/w > 4 或 < 0.25`（过于细长），或 `面积 > 整图 35%`。
参考值：站立人体像素 h/w 约 2.5~3.5，人头约 1.0~1.3。

对本机 379 个标签文件、746 个框统计：

| 类别 | 框数 | 废框 | 占比 |
|---|---|---|---|
| helmet（真实物体） | 182 | 0 | **0.0%** |
| no-helmet（状态） | 263 | 3 | **1.1%** |
| vest（真实物体） | 61 | 0 | **0.0%** |
| no-vest（状态） | 240 | 9 | **3.8%** |
| **合计** | **746** | **12** | **1.6%** |

**状态类仍然比物体类差（3.8% vs 0%），但差距远没有初版说的那么大。**
更重要的是：几何指标本身**测不出"标得对不对"**，它只能测"框的形状像不像个正常物体"。
要判断标注质量，需要人工看图或真值，不能只看这个数。

**同一批数据、同一个模型，真实物体 1.6% 崩，状态类 28.8% 崩——差 18 倍。**

进一步抓模型原始返回确认过：不是本工具的坐标换算放大了框。模型自己就说 `no-vest: x1=0, y1=120, x2=800, y2=470`（横跨整幅画面），本工具只是照它说的画；自报尺寸偏差只带来 ±5% 的微调。

**结论：模型定位真实物体没问题，它栽在"定位一个画面里不存在的东西"上。** no-vest 在画面中没有指代物，模型只能瞎框。

## 三、社区是怎么做的

### 1. 主流标注工具把「出框」和「聊天式 VLM」当成两件事

X-AnyLabeling（社区最大的开源 AI 标注工具）README 的模型表原文如下，这一条我们在官方仓库逐字核对过：

```
| 🧩 Segment Anything | SAM 1/2/3, SAM-HQ, SAM-Med2D, EdgeSAM, EfficientViT-SAM, MobileSAM |
| 🗣️ Vision Foundation Models | Rex-Omni, Florence2 |
| 👁️ Vision Language Models | Qwen3-VL, Gemini, ChatGPT, GLM |
| 📍 Grounding | Grounding DINO, YOLO-World, YOLOE, SAM 3, LocateAnything |
```

来源：[github.com/CVHub520/X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)

**出框归 "Grounding" 那一格，聊天式 VLM 在另一格。** 该工具的 Chatbot 功能（[docs/en/chatbot.md](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/chatbot.md)）做的是视觉问答与批量处理，导出格式是 ShareGPT（用于 LLaMA-Factory 多模态微调），不是 YOLO 标签。

### 2. 自动标注的标准管线是「基准模型出标注 → 蒸馏成小模型」

Autodistill 官方 README 对流程的定义（原文）：

```
To use autodistill, you input unlabeled data into a Base Model which uses an
Ontology to label a Dataset that is used to train a Target Model...

Base Model - A Base Model is a large foundation model that knows a lot about a lot...
Target Model - A Target Model is a supervised model that consumes a Dataset and
outputs a distilled model...
```

官方示例命令：

```bash
autodistill images --base="grounding_dino" --target="yolov8" --ontology '{"prompt": "label"}'
```

来源：[github.com/autodistill/autodistill](https://github.com/autodistill/autodistill)

**注意 base 用的是 `grounding_dino`，不是任何一个聊天式 VLM。** 这个思路和本工具已有的"用 YOLO 模型辅助标注"是一致的，只是社区把基准模型从 YOLO 换成了开放词表检测器。

### 3. 「框松」的标准药方是 grounding + 分割两段

`autodistill-grounded-sam-2` 的 README 原文：

> This repository contains the code implementing Grounded SAM 2 using **Florence-2 as a grounding model** and **Segment Anything 2 as a segmentation model**.

来源：[github.com/autodistill/autodistill-grounded-sam-2](https://github.com/autodistill/autodistill-grounded-sam-2)

即：先粗定位，再让分割模型贴着物体边缘收一遍。这正是"框松"的解药。

### 4. Florence-2 是社区常用的 grounding 底座

- 论文：[arxiv.org/abs/2311.06242](https://arxiv.org/abs/2311.06242)
- 权重：[huggingface.co/microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large)、[Florence-2-large-ft](https://huggingface.co/microsoft/Florence-2-large-ft)

它的任务表里有两个任务正好对应本工具的两个需求：`Open vocabulary detection`（开放词表出框）和 `Region to category`（给一个框，判断它是什么）。**后者就是方案 A 的"判裁剪"环节。**

模型只有 0.23B / 0.77B 两个规格，本机 RTX 4060 Laptop（8GB 显存）跑得动。*（推断，未实测）*

### 5. SAHI：切片推理治小目标

来源：[github.com/obss/sahi](https://github.com/obss/sahi)

专门做切片推理（Slicing Aided Hyper Inference）的框架，把小目标在大图里被重采样糊掉的问题，用分块 + 重叠 + 合并的方式绕开。对应方案 C。

## 四、怎么落到本工具上

| 方案 | 要改的地方 | 现有可复用的部分 |
|---|---|---|
| A 检测器出框 + VLM 判裁剪 | `_infer_vlm()` 的输入从整图换成检测框裁块；新增"复核"通道 | 类别偏移、`only_cls`、跨模型 NMS、`_dedupe_with_existing`、写盘校验、并发 worker 全部复用；多模态配置与重试逻辑也复用 |
| B 换 grounding 模型 | `_predict_all()` 里新增一个 `backend`，与现有 `ultralytics` / `meituan_v6n` 并列 | 同上，接口契约不变（返回归一化 `{cls,cx,cy,w,h,conf}`） |
| C 分块 | `_infer_vlm()` 内先切片、推理后按偏移拼回；复用现有 NMS 去重 | 同上 |

三个方案互不冲突，可以叠加。**A 是唯一能命中 28.8% 那个主要矛盾的一条。**

## 四之二、纯多模态方案：确实存在，但要换一类模型

前面第一节把结论说成"要用检测器出框"，这偏了。真正的分界线不是「有没有检测器」，而是**这个多模态模型有没有做过 grounding 训练**。

### 1. grounding 训练过的 VLM（一个模型端到端出框，不需要检测器）

**Qwen3-VL / Qwen2.5-VL** —— 官方 README 原文：

> **Advanced Spatial Perception**: Judges object positions, viewpoints, and occlusions; provides stronger **2D grounding** and enables 3D grounding for spatial reasoning and embodied AI.

官方 cookbook 中有一节专门讲定位：

> [Precise Object Grounding Across Formats](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb) —— Using relative position coordinates, it supports both **boxes and points**, allowing for diverse combinations of positioning and labeling tasks.

来源：[github.com/QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)

**坐标体系的坑（重要）** —— 官方 cookbook 原文：

> Coordinate System: Qwen3-VL's default coordinate system has been changed from the absolute coordinates used in Qwen2.5-VL to **relative coordinates ranging from 0 to 1000**.

也就是说：**Qwen2.5-VL 用像素绝对坐标，Qwen3-VL 用 0~1000 相对坐标**。本工具当前的提示词里写着"不是归一化坐标，也不是 0-1000 的比例坐标"——换用 Qwen3-VL 时必须同时改提示词，否则所有框都会错位。

**Florence-2**（微软）—— 见第三节第 4 点，任务表里有 `Open vocabulary detection`，Autodistill 直接把它当作 base model 产标注。规格 0.23B / 0.77B。

**Molmo**（Allen AI）—— 训练数据包含 [PixMo-Points](https://huggingface.co/datasets/allenai/pixmo-points)（"images paired with referring expressions and annotated points"），专门训练"指"这个动作。来源：[github.com/allenai/molmo](https://github.com/allenai/molmo)

### 2. 通用聊天式 VLM（本工具当前用的 DeepSeek-V4.1-Flash 属于这类）

坐标是"用文字生成的"，没有专门的 grounding 训练。第二节实测出来的 13.8% 废框、非确定性、空结果，都是这个差别的后果。

### 3. 本地可跑性（已核实）

Ollama 官方模型库有 `qwen3-vl`，可选规格与体积：

| 标签 | 体积 |
|---|---|
| `qwen3-vl:2b` | 1.9 GB |
| `qwen3-vl:4b` | 3.3 GB |
| `qwen3-vl:latest` | 6.1 GB |

来源：[ollama.com/library/qwen3-vl](https://ollama.com/library/qwen3-vl)。三档都在本机 RTX 4060 Laptop（8GB 显存）可承载范围内。Ollama 暴露 OpenAI 兼容接口，本工具只需改「接口地址」与「模型名」两个输入框，**不需要改代码**。

### 4. 但有个关键的未知

第二节那 28.8% 的 `no-vest` 废框，根源是"画面里没有可指认的对象"。grounding 模型训练的是**真实存在的东西**，所以：

- `helmet` / `vest` 这类真实物体类：换 grounding 模型**预期会有明显改善**（推断）
- `no-helmet` / `no-vest` 这类缺失状态类：**换模型能改善多少，未知**。有一种可能是 grounding 模型倾向于"指不到就不给框"，那会比现在瞎框更好；也可能行为一样。

**这一条必须实测，不能靠推断。**

## 五、没验证的部分与风险

### 双通道实测结果（已实现并实跑）

已在本工具中实现「检测器出框 + 多模态只做复核」的双通道，并用真实 API 跑通。
复核的设计是：把候选框和编号画在**原图**上发给模型，只要求它回答每个编号是什么类别
（不需要输出坐标），然后改类别或删除，**不新增框**。

**实测结论：当前模型（DeepSeek-V4.1-Flash）做这件事的准确度不合格。**

样本 4 张图 10 个框，模型提出 8 处修改，其中约 3 处正确、5 处错误
（例如把 `helmet` 改成 `person`）。另外模型经常只对部分框给结论
（某次 4 个框只答了 2 个）。失败原因：这些框只有约 10×10 像素，
无论是裁剪放大还是带编号的原图，模型都难以判准。

**因此该功能以「先出建议、人工确认后再应用」的模式合入**，绝不静默改写标签：

- 复核默认只产生**修改建议**，不写盘
- 界面上列出建议明细，点「应用修改」才真正写入
- 模型没给结论的框一律保持原样（宁可不动，不可误删）
- 允许删除默认开启，可关闭

**待验证**：换更强的模型（如 DashScope 上的 `qwen3-vl-235b-a22b-instruct`，
已验证该域名可达且接口 OpenAI 兼容）是否能把这个任务的准确度提到可用水平。
在那之前，不建议对全量标签使用复核。

1. **上面三条社区方案，一条都没有在本机数据上实跑过**，结论来自官方文档与项目设计意图 + 本机标签文件的统计。数字部分是实测的，方案效果是推断的。
2. 方案 A 的收益上限取决于人/头检测器的召回率。检测器漏掉的人，裁剪环节也没有机会。
3. 状态类标注本身仍是模糊的：给"没穿反光衣"标一个框，到底该框躯干、框半身、还是框整人？**这是标注规范问题，不是模型问题**，需要先定下来，否则换任何模型都不一致。
4. 本机目录里存在 17 个空 txt。若「空结果不写标签」是开启状态，它们不应存在，需确认该选项是否被取消勾选——空文件在 YOLO 里等于"这张图没有目标"的负样本，会把模型的漏检固化成错误标签。

## 六、来源清单

## 七、GitHub 上的现成框架（补充调研）

用 GitHub 官方搜索 API 按多组关键词检索并按 star 排序，找到下面这些**已把「AI 预标注 → 人工修正 → 导出 → 训练」做成闭环**的项目。

### 最接近「直接可用」的两个

**VisioFirm — 438★** ｜ [github.com/OschAI/VisioFirm](https://github.com/OschAI/VisioFirm) ｜ Apache 2.0

> AI-Driven Pre-Annotation: Automatically detect and segment objects using **YOLOv10, SAM2, and Grounding DINO**—saving up to 80% of manual effort.

本地 Web 界面，支持普通框 / 旋转框 / 多边形分割，导出 YOLO / COCO，SQLite 本地存储，浏览器内 SAM2 点选分割。许可宽松。

**VLM-AutoYOLO — 243★** ｜ [github.com/Somnusochi/VLM-AutoYOLO](https://github.com/Somnusochi/VLM-AutoYOLO) ｜ AGPL v3

README 顶部的流水线定义：

```
🖼️ image/video → 🔍 VLM / SAM3 detection → 🎯 SAM2/SAM3 mask → ✏️ refine → 📦 export → 🚀 YOLO → ✅ model
```

"Images or videos in → YOLO model out"。视觉定位用 **NVIDIA LocateAnything-3B（Qwen2.5-3B + MoonViT）**，收边用 SAM 2.1 / SAM3，带人工修正画布、NMS、多格式导出、一键训练 YOLOv8/v11/v26 的训练队列。是这个方向上最完整的开源闭环。

**两条必须注意的限制：**

1. 它的输入是**开放词表物体描述**（README 示例：`fire, smoke`、`red car`）——**全部是真实物体，没有一个"缺失状态"类**。再次印证：这类框架解决的是物体检测，状态/合规判断不在其中。
2. 其 README 声明 LocateAnything-3B 为 **non-commercial use only**（链接指向该模型在 HuggingFace 的 LICENSE）。**商用需换许可宽松的替身**（Florence-2 为 MIT，Qwen 系列为 Apache 2.0）。*（该 LICENSE 页面本次未能直接打开核实，此为仓库 README 的声明。）*

### 其他相关项目

| 项目 | stars | 出框用什么 | 备注 |
|---|---|---|---|
| [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) | 10.5k | Grounding 那一格 | 桌面标注工具，模型分档见第三节 |
| [Autodistill](https://github.com/autodistill/autodistill) | 2.8k | Grounding DINO / Florence-2 / GroundedSAM2 | 库形态，base → target |
| [AlvaroCavalcante/auto_annotate](https://github.com/AlvaroCavalcante/auto_annotate) | 163 | 传统检测器 | "Labeling is boring" |
| [Fudan-ProjectTitan/OpenAnnotate3D](https://github.com/Fudan-ProjectTitan/OpenAnnotate3D) | 110 | 开放词表 | 面向 3D/多模态，学术项目 |
| [yangdi-cv/SAM3-Automatic-Annotator-UI](https://github.com/yangdi-cv/SAM3-Automatic-Annotator-UI) | 29 | SAM3 | 零样本全自动 |
| [962753401-cloud/qwen3-vl-annotation](https://github.com/962753401-cloud/qwen3-vl-annotation) | 7 | Qwen3-VL | 中文 Web 标注平台，导出 COCO |
| [huazi32/images-qwen3vl-annotation](https://github.com/huazi32/images-qwen3vl-annotation) | 1 | Qwen3-VL | 中文，Qwen3VL → YOLO 闭环 |

### 交叉验证的结论

**上面每一个项目，出框那一环用的都是 grounding / 检测模型（Grounding DINO、LocateAnything、YOLOv10、SAM），没有任何一个用通用聊天式 VLM 出框。** 这不是巧合，是被同一类问题逼出来的相同选择。

因此「缺框架」这个判断成立，但要补一句：**现成框架的编排层可以直接借鉴，而它们同时也会把出框模型换掉。** 两者是绑在一起的。

## 八、来源清单（第二部分）

| 来源 | 用途 |
|---|---|
| [VisioFirm](https://github.com/OschAI/VisioFirm) | 预标注模型组合、许可、工作流 |
| [VLM-AutoYOLO](https://github.com/Somnusochi/VLM-AutoYOLO) | 端到端闭环流水线、模型选择、许可限制 |
| [NVIDIA LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) | 视觉定位模型（许可页面本次未能打开） |
| [Fudan-ProjectTitan/OpenAnnotate3D](https://github.com/Fudan-ProjectTitan/OpenAnnotate3D) | 开放词表自动标注（多模态） |
| GitHub 官方搜索 API（多组关键词） | 用于穷举候选项目，非结论来源 |

| 来源 | 用途 |
|---|---|
| [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) | 模型分类表（Grounding 与 VLM 分格）；已逐字核对 |
| [X-AnyLabeling Chatbot 文档](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/chatbot.md) | Chatbot 的定位与 ShareGPT 导出格式 |
| [Autodistill](https://github.com/autodistill/autodistill) | Base Model → Dataset → Target Model 流程；已逐字核对 |
| [autodistill-grounded-sam-2](https://github.com/autodistill/autodistill-grounded-sam-2) | Florence-2 grounding + SAM 2 分割的两段链 |
| [Florence-2 论文](https://arxiv.org/abs/2311.06242) / [HF 模型卡](https://huggingface.co/microsoft/Florence-2-large) | 任务能力与模型规格 |
| [SAHI](https://github.com/obss/sahi) | 切片推理框架 |
| 本机 465 张监控图、379 个标签文件、746 个框 | 第二、五节的实测数字 |

*调研协助：Codex 后台调研 agent；关键两条证据（X-AnyLabeling 与 Autodistill）由主 agent 亲自复核。*
