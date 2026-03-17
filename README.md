# GLUS: Grounded Language Understanding for Video Segmentation

> **数据流文档 / Data Flow Documentation**

---

## 目录 / Table of Contents

1. [项目概述 / Project Overview](#项目概述--project-overview)
2. [系统架构 / System Architecture](#系统架构--system-architecture)
3. [训练数据流 / Training Data Flow](#训练数据流--training-data-flow)
4. [推理数据流 / Inference Data Flow](#推理数据流--inference-data-flow)
5. [核心组件 / Core Components](#核心组件--core-components)
6. [数据集结构 / Dataset Structure](#数据集结构--dataset-structure)
7. [配置与脚本 / Configuration & Scripts](#配置与脚本--configuration--scripts)

---

## 项目概述 / Project Overview

**GLUS** 是一个用于**指代视频目标分割（Referring Video Object Segmentation, RVOS）**的多模态 AI 系统。它接收视频帧序列和自然语言描述作为输入，输出每一帧中对应目标的二值分割掩码。

**GLUS** is a multimodal AI system for **Referring Video Object Segmentation (RVOS)**. It takes a sequence of video frames and a natural language description as input, and outputs per-frame binary segmentation masks for the described object.

**核心能力 / Core Capabilities:**
- 多模态理解：融合视觉信息和自然语言描述 / Multimodal understanding: fusing vision and language
- 时序感知：通过记忆库（Memory Bank）跨帧保持一致性 / Temporal awareness via Memory Bank across frames
- 双向处理：推理时同时进行前向和后向传播 / Bidirectional inference (forward + backward passes)
- 轨迹感知：通过物体质心历史辅助定位 / Trajectory-aware localization using centroid history

---

## 系统架构 / System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         GLUS 系统架构                                     │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   视频帧 (Video Frames)           文字描述 (Text Description)             │
│        │                                    │                              │
│        ▼                                    ▼                              │
│  ┌─────────────┐                   ┌──────────────────┐                   │
│  │  CLIP 视觉  │                   │  LLaVA-Llama LLM │                   │
│  │  编码器     │──────────────────▶│  语言模型        │                   │
│  │ (224×224)  │   视觉 Token       │  (Tokenizer +    │                   │
│  └─────────────┘                   │   Transformer)   │                   │
│                                    └────────┬─────────┘                   │
│                                             │ 隐藏状态 (4096-dim)         │
│                                             ▼                              │
│                                    ┌──────────────────┐                   │
│                                    │  文本投影层       │                   │
│                                    │  4096 → 256 dim  │                   │
│                                    └────────┬─────────┘                   │
│                                             │ [SEG] Token 嵌入 (256-dim)  │
│        ┌────────────────────────────────────┘                              │
│        ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                         SAM2 视觉模型                                │  │
│  │                                                                      │  │
│  │  视频帧 (1024×1024)                                                  │  │
│  │       │                                                              │  │
│  │       ▼                                                              │  │
│  │  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐ │  │
│  │  │ SAM2 主干网  │────▶│  记忆注意力  │────▶│    掩码解码器        │ │  │
│  │  │ (多尺度特征) │     │  Memory Attn │     │    Mask Decoder      │ │  │
│  │  └──────────────┘     └──────────────┘     └──────────┬───────────┘ │  │
│  │                              ▲                         │              │  │
│  │                        ┌─────────────┐                │              │  │
│  │                        │  记忆库     │◀───────────────┘              │  │
│  │                        │  Memory Bank│   (存储前帧掩码特征)           │  │
│  │                        └─────────────┘                               │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│                         分割掩码 (Segmentation Masks)                      │
│                         [每帧 PNG，uint8，0/255]                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 训练数据流 / Training Data Flow

### 完整训练流程 / Complete Training Pipeline

```
原始数据 (Raw Data)
       │
       │  多数据集
       │  MEVIS / Ref-YouTube-VOS / DAVIS17 / ReVOS
       ▼
┌─────────────────────────────────────────────────────────┐
│  数据预处理 (Data Preprocessing)                         │
│                                                          │
│  1. 帧采样策略                                           │
│     ├─ 上下文帧 (Context Frames): 均匀采样于整段视频    │
│     └─ 问题帧 (Question Frames): 连续帧序列             │
│                                                          │
│  2. 图像处理                                             │
│     ├─ CLIP 路径: 缩放至 224×224，归一化                │
│     └─ SAM  路径: 缩放至 1024×1024                      │
│                                                          │
│  3. 文本处理                                             │
│     └─ 描述文本 → Tokenize → 带特殊 Token 的 Prompt     │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  编码阶段 (Encoding Stage)                               │
│                                                          │
│  CLIP 视觉编码器                                         │
│       └─ 224×224 图像 → 视觉 Token (256-dim)            │
│                                                          │
│  LLaVA-Llama LLM                                        │
│       ├─ 输入: 文本 Token + 视觉 Token                   │
│       ├─ 处理: Transformer 多头注意力                    │
│       └─ 输出: 隐藏状态 (4096-dim)                      │
│                                                          │
│  投影层 (Projection Layer)                               │
│       └─ 4096-dim → FC → ReLU → FC → 256-dim            │
│          (生成 [SEG] Token 嵌入)                         │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  SAM2 分割阶段 (SAM2 Segmentation Stage)                 │
│                                                          │
│  1. SAM2 主干网 (1024×1024 输入)                         │
│     └─ 提取多尺度特征: 256×256 / 128×128 / 64×64        │
│                                                          │
│  2. 记忆注意力 (Memory Attention)                        │
│     ├─ 读取记忆库中最多 7 帧的历史掩码特征               │
│     └─ 与当前帧特征融合                                  │
│                                                          │
│  3. 掩码解码器 (Mask Decoder)                            │
│     ├─ 输入: 图像嵌入 + [SEG] 提示嵌入 + 高分辨率特征   │
│     ├─ 交叉注意力: 文本提示 ↔ 图像特征                  │
│     └─ 输出: 低分辨率掩码 → 上采样至原始分辨率          │
│                                                          │
│  4. 记忆编码 (Memory Encoding)                           │
│     └─ 将预测掩码特征写入记忆库                          │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  损失计算 (Loss Computation)                             │
│                                                          │
│  CE Loss       (权重 1.0): 语言生成监督                  │
│  Sigmoid BCE   (权重 2.0): 像素级掩码二元交叉熵           │
│  DICE Loss     (权重 0.5): 区域重叠率损失                │
│  Contrastive   (权重 0.1): 多帧一致性损失 (可选)         │
│                                                          │
│  Total = CE + BCE + DICE + Contrastive                   │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
              反向传播 → 参数更新 (DeepSpeed Stage 2)
```

### 训练参数 / Training Parameters

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epochs` | 10 | 训练轮数 |
| `steps_per_epoch` | 500 | 每轮步数 |
| `batch_size` | 2 | 每 GPU 批大小 |
| `grad_accumulation_steps` | 10 | 梯度累积步数 |
| `lr` | 3e-4 | 学习率 |
| `context_frame_num` | 4 | 上下文帧数 |
| `question_frame_num` | 4 | 问题帧数（连续帧） |
| `precision` | bf16 | 精度 (fp32/fp16/bf16) |

---

## 推理数据流 / Inference Data Flow

### 完整推理流程 / Complete Inference Pipeline

```
输入 (Input)
  ├─ 视频帧路径列表 (Video frame paths)
  ├─ 目标对象描述 (Target object description)
  └─ 可选: 关键帧索引 (Optional: keyframe index)
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  加载元数据 (Load Metadata)                                      │
│  从 JSON 读取: 帧列表、目标描述、掩码文件名                       │
│  支持数据集: mevis / refyoutube_vos / revos / davis17            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  帧采样 (Frame Sampling)                                         │
│                                                                  │
│  以视频中间帧为基准, 将视频分为前半段和后半段                    │
│  context_frames: 在整段视频中均匀采样                             │
└───────────────┬──────────────────────────┬──────────────────────┘
                │                          │
                ▼                          ▼
   ┌──────────────────────────┐     ┌──────────────────────────┐
   │   前向推理 (Forward)      │     │   后向推理 (Backward)     │
   │   从中间帧 → 末帧         │     │   从中间帧 → 首帧         │
   └─────────────┬────────────┘     └────────────┬─────────────┘
            │                              │
            ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  逐帧推理循环 (Per-Frame Inference Loop)                          │
│                                                                  │
│  while 还有未处理的帧:                                           │
│      │                                                           │
│      ├─ 1. 构建输入窗口                                          │
│      │     context_frames (上下文帧, 固定) +                     │
│      │     question_frames[i : i+question_frame_num] (滑窗)      │
│      │                                                           │
│      ├─ 2. 构建 Prompt                                           │
│      │     首帧: CONTEXT_INFO + QUESTION                         │
│      │     后续帧: QUESTION only                                   │
│      │                                                           │
│      ├─ 3. 模型前向传播 (model.evaluate)                         │
│      │     ├─ CLIP: 提取视觉特征                                  │
│      │     ├─ LLM: 生成 [SEG] Token 嵌入                        │
│      │     ├─ SAM2: 结合记忆库生成掩码                            │
│      │     └─ 轨迹注入 (Trajectory Injection):                   │
│      │           将前几帧的质心坐标注入嵌入                        │
│      │                                                           │
│      ├─ 4. 提取当前帧质心并加入轨迹历史                           │
│      │                                                           │
│      └─ 5. 保存掩码 PNG                                          │
│            路径: {save_path}/{video}/{exp_id}/{frame_id:05d}.png  │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  输出 (Output)                                                   │
│  每帧二值分割掩码 (Per-frame Binary Segmentation Masks)          │
│  格式: PNG, uint8, 0=背景 / 255=目标                             │
└─────────────────────────────────────────────────────────────────┘
```

### 双向推理示意 / Bidirectional Inference Illustration

```
视频帧序列 (Video Frame Sequence):
  [F0] [F1] [F2] ... [Fk] ... [Fn-1] [Fn]
                      ▲
                  第一个标注帧 (First annotated frame)

前向推理 (Forward):  Fk ──▶ Fk+1 ──▶ ... ──▶ Fn
后向推理 (Backward): Fk ──▶ Fk-1 ──▶ ... ──▶ F0

记忆库 (Memory Bank): 每处理一帧，将该帧的掩码特征存入记忆库
                      下一帧通过 Memory Attention 读取前 7 帧历史
```

---

## 核心组件 / Core Components

### 文件说明 / File Descriptions

| 文件 | 作用 |
|------|------|
| `model/GLUS.py` | 核心模型定义：`GlusMetaModel`, `GlusModel`, `GLUSForCausalLM` |
| `train_ds.py` | 训练入口：分布式训练、数据加载、验证循环 |
| `inference_iter.py` | 推理入口：双向迭代推理、掩码保存 |
| `dataset/dataset.py` | 训练数据集：`HybridDataset`, `ValDataset`, `collate_fn` |
| `dataset/refer_video_seg_dataset.py` | 多数据集加载器：帧采样、掩码读取 |
| `dataset/mevis.py` | MEVIS 数据集加载 |
| `dataset/refyoutube_vos.py` | Ref-YouTube-VOS 数据集加载 |
| `dataset/davis17.py` | DAVIS17 数据集加载 |
| `dataset/revos.py` | ReVOS 数据集加载 |
| `utils/utils.py` | 工具函数：Token 常量、Prompt 模板、IoU 计算 |
| `utils/traj.py` | 轨迹工具：从掩码提取质心坐标 |
| `utils/contrastive_loss.py` | 对比损失：多帧一致性 |
| `utils/eval_mevis.py` | MEVIS 评估指标（J&F） |
| `model/llava/` | LLaVA 语言模型和 CLIP 视觉塔 |

### 组件交互关系 / Component Interaction

```
train_ds.py
  ├─→ GLUSForCausalLM          (model/GLUS.py)
  │     ├─→ GlusModel
  │     │     ├─→ LlavaLlamaModel   (视觉-语言主干)
  │     │     │     ├─→ CLIP 视觉塔  (model/llava/)
  │     │     │     └─→ LLaVA Llama LM
  │     │     └─→ SAM2 视觉模型     (掩码编解码)
  │     ├─→ 掩码投影层 (Mask Projection)
  │     ├─→ 轨迹 MLP (Trajectory MLP)
  │     └─→ 记忆库 (Memory Bank)
  │
  ├─→ HybridDataset            (dataset/dataset.py)
  │     ├─→ ReferVideoSegDataset  (dataset/refer_video_seg_dataset.py)
  │     │     ├─→ mevis.py
  │     │     ├─→ refyoutube_vos.py
  │     │     ├─→ davis17.py
  │     │     └─→ revos.py
  │     └─→ collate_fn()
  │
  └─→ DeepSpeed               (分布式训练框架)

inference_iter.py
  ├─→ GLUSForCausalLM.evaluate()
  │     ├─→ 记忆库管理
  │     ├─→ 逐帧掩码生成
  │     └─→ 轨迹追踪
  └─→ SAM2Transforms          (图像预处理)
```

### 模型层次结构 / Model Layer Stack

```
输入层 (Input):
  Video frames (N × H × W × 3) + Text description

Layer 1 - 视觉编码器 (Visual Encoder):
  CLIP-ViT-Large (frozen)
  224×224 → 256-dim visual tokens

Layer 2 - 语言模型 (Language Model):
  LLaVA-Llama (7B)
  [Image tokens + Text tokens] → 4096-dim hidden states

Layer 3 - 多模态投影 (Multimodal Projection):
  FC(4096→4096) → ReLU → FC(4096→256) → Dropout
  输出: [SEG] token embedding (256-dim)

Layer 4 - SAM2 视觉分割 (Visual Segmentation):
  ├─ 主干网: HViT, 多尺度特征 256/128/64 分辨率
  ├─ 提示编码器: [SEG] 嵌入 → sparse + dense embeddings
  ├─ 掩码解码器: 交叉注意力 → 低分辨率掩码
  └─ 记忆注意力: 融合前 7 帧历史特征

Layer 5 - 后处理 (Post-processing):
  Bilinear upsample → Sigmoid → Binary threshold (0.5)
  输出: per-frame masks (H × W, uint8)
```

---

## 数据集结构 / Dataset Structure

```
data/refer_seg/
├─ mevis/
│  ├─ train/
│  │  ├─ JPEGImages/
│  │  │  └─ {video_id}/
│  │  │       └─ {frame_id}.jpg
│  │  ├─ meta_expressions.json   # 视频和文字描述元数据
│  │  └─ mask_dict.json          # RLE 编码的真值掩码
│  └─ valid_u/                   # 验证集 (union)
│
├─ refyoutube_vos/
│  ├─ train/
│  │  ├─ JPEGImages/{video_id}/{frame_id}.jpg
│  │  └─ meta.json
│  └─ valid/
│
├─ davis17/
├─ revos/
└─ lvvis/

Key_frame/                       # 用于从关键帧继续推理
├─ mevis_v.json                  # MEVIS valid 关键帧索引
├─ mevis_u.json                  # MEVIS union 关键帧索引
└─ refyoutube.json               # Ref-YouTube-VOS 关键帧索引
```

**元数据 JSON 结构 / Metadata JSON Structure (MEVIS):**

```json
{
  "videos": {
    "video_id": {
      "length": 50,
      "frames": ["00000", "00001", "..."],
      "expressions": {
        "0": {
          "exp": "the dog on the left",
          "obj_id": ["1"],
          "anno_id": "anno_123",
          "file_names": ["00000.jpg", "00001.jpg", "..."]
        }
      }
    }
  }
}
```

**掩码字典 JSON 结构 / Mask Dict JSON Structure:**

```json
{
  "str_id_video_0": {
    "obj_id_1": {
      "size": [480, 360],
      "counts": "YV;h1..."
    }
  }
}
```
> 掩码以 RLE（Run-Length Encoding）格式编码 / Masks are RLE-encoded

---

## 配置与脚本 / Configuration & Scripts

```
scripts/
├─ train_glus_a.sh       # 训练 GLUS-A 变体
├─ train_glus_s.sh       # 训练 GLUS-S 变体
├─ train_glus_final.sh   # 最终训练配置
├─ train_traj.sh         # 启用轨迹感知的训练
├─ inference.sh          # 标准推理
├─ inference_kf.sh       # 基于关键帧的推理（从指定帧继续）
└─ eval_mevis.sh         # MEVIS 数据集评估
```

### 快速开始 / Quick Start

**训练 / Training:**
```bash
bash scripts/train_glus_final.sh
```

**推理 / Inference:**
```bash
bash scripts/inference.sh
```

**从关键帧继续推理 / Resume from Keyframe:**
```bash
bash scripts/inference_kf.sh
```

**评估 / Evaluation:**
```bash
bash scripts/eval_mevis.sh
```

---

## 数据流总结 / Data Flow Summary

```
原始视频帧 + 文字描述
        │
        ▼
  ┌─────────────┐
  │  数据加载    │  (dataset/*)
  │  帧采样      │  context frames + question frames
  │  图像预处理  │  CLIP (224×224) + SAM (1024×1024)
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  视觉编码   │  CLIP-ViT → 视觉 Token
  │  语言编码   │  Tokenizer → 文本 Token
  │  LLM 融合  │  LLaVA-Llama → 隐藏状态 → [SEG] 嵌入
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  SAM2 分割  │  图像嵌入 + [SEG] 提示 + 记忆库 → 掩码
  │  记忆更新   │  当前帧掩码特征写入记忆库
  │  轨迹追踪   │  质心历史辅助下一帧定位
  └──────┬──────┘
         │
         ▼
  二值分割掩码 (PNG)
  每帧: 0=背景, 255=目标
```
