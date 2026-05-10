# Pocket-Dentist

**面向牙科图像理解的紧凑型视觉语言模型基准测试**

[![Paper](https://img.shields.io/badge/NeurIPS%202026-Evaluations%20%26%20Datasets-blue)](https://anonymous.4open.science/r/pocket-dentist-DD77)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

> **论文**: Pocket-Dentist: Benchmarking Compact Vision-Language Models for Dental Image Understanding
>
> **会议**: NeurIPS 2026 Evaluations & Datasets Track（审稿中）

---

## 目录

1. [简介](#概览) — Pocket-Dentist 是什么？
2. [开发指南](#快速开始) — 环境搭建、推理评估与训练
3. [附录：定性案例分析](#定性分析qualitative-analysis) — 6 种临床任务类型的补充分析

---

## 概览

Pocket-Dentist 是一个**大规模多模态基准测试和部署感知评估管线**，用于评估牙科视觉语言模型（VLM）。它整合并标准化了 7 个异构牙科数据集，构建统一的视觉-语言基准，支持在多种成像模态、临床任务类型、适配策略和部署约束下对 VLM 进行系统评估。

### 核心亮点

- **7 个牙科数据集**，涵盖全景 X 光片、口内照片、根尖片和头颅侧位片
- **71,000+ 张图像**，来自 **6,000+ 患者**，覆盖 **6 种任务类型**和 **14 个评估指标**
- **14 个 VLM** 在 zero-shot、few-shot（1-shot、2-shot）和 LoRA 微调设定下评估
- **核心发现**：在统一的低成本 LoRA 适配预算下，紧凑型 VLM——尤其是 **Qwen3-VL-4B**——在大多数主要任务指标上可匹敌甚至超越更大的开源模型（7B–32B）

### 基准数据集

| 数据集 | 模态 | 任务类型 | 测试集大小 | 主要指标 |
|--------|------|---------|-----------|---------|
| **COde** | 口内照片 + 全景 X 光 | 分类、报告生成 | 1,200 | Weighted F1 / BERTScore F1 |
| **MetaDent** | 口内照片 | VQA、分类、描述生成 | 2,301 | Accuracy / Weighted F1 / BERTScore F1 |
| **BRAR** | 全景 X 光片 | 分类（Grade 1/2/3） | 149 | Macro F1 |
| **Aariz** | 头颅侧位片 | VQA、CVM 分类 | 630 / 126 | Accuracy |
| **DenPAR** | 根尖片 | 牙齿结构、位置、计数 | 200 × 3 | Accuracy / Weighted F1 / MAE ↓ |
| **DentalCaries** | 口内照片 | 龋齿检测、牙列分类 | 628 / 226 | Accuracy / Weighted F1 |
| **DR** | 全景 X 光 | 多标签分类 | 73 | Weighted F1 |

### 评估模型

| 层级 | 模型 |
|------|------|
| **大模型（≥ 7B）** | Lingshu-32B, MedMO-8B-Next, Qwen2.5-VL-7B, Gemini-2.0-Flash, Gemini-2.5-Flash |
| **紧凑型模型（≤ 4B）** | Qwen3-VL-4B, Qwen3.5-4B, gemma-4-E4B-it, gemma-4-E2B-it, SmolVLM2-2.2B, InternVL3.5-2B, InternVL3.5-1B, medgemma-4b-it, paligemma2-3b-mix-448 |

---

## 快速开始

环境搭建、运行评估、SFT 训练和数据格式的详细说明请参见 **[Development Guide](Development.md)**。

### 快速上手

```bash
# 安装依赖
pip install -r requirements.txt

# 在 MetaDent 上运行 zero-shot 评估
bash scripts/run_metadent.sh --models Qwen3-VL-4B-Instruct --tasks baseline

# 运行 LoRA 微调
bash scripts/run_metadent_sft.sh --models Qwen3-VL-4B-Instruct
```

### 硬件要求

| 环境 | 用途 | 最低 GPU | 最高 GPU |
|------|------|---------|---------|
| `NeurlPS2026-benchmark` | vLLM 推理 + 评估 | A100 40GB（1–4B 模型） | H100 96GB（32B 模型） |
| `NeurlPS2026-train` | LoRA SFT 训练 | A100 40GB（1–4B 模型） | H100 96GB（32B 模型） |

---

## 定性分析（Qualitative Analysis）

为展示紧凑型模型适配的实际效果，我们从所有 6 种任务类型中各选取一个代表性测试集案例，展示 LoRA 适配后的 **Qwen3-VL-4B**（4B 参数）预测正确，而参数量大 8 倍的 **Lingshu-32B**（32B 参数）在相同 LoRA 预算下预测错误的情况。

所有案例均来自 SFT（LoRA）设定下的测试集预测结果。

---

### 1. VQA（视觉问答）

**案例 ID: `22966`** · **数据集: MetaDent**

![VQA — 侧位口内照](assets/images/1_vqa_primary_22966.jpg)

**问题**: 侧位口内照中是否所有下颌前牙均可见。
**选项**: A: True, B: False

**✅ Ground Truth**:
```json
{"answer": "B", "reason": "Because the photograph is a lateral view taken from the left side, only the left portion of the mandibular anterior region is captured; the right-side anterior teeth are not visible. Therefore the statement is false."}
```

**❌ Lingshu-32B (SFT)**:
```json
{"answer": "A", "reason": "The overall description states that the image shows the maxillary and mandibular anterior teeth, indicating that all mandibular anterior teeth are present."}
```
> ⚠️ Lingshu 未能识别侧位视图的视角限制，错误判断所有下颌前牙均可见。

**✅ Qwen3-VL-4B (SFT)**:
```json
{"answer": "B", "reason": "The description notes that the mandibular anterior teeth are not fully visible, indicating that not all of them are captured in the image."}
```

---

### 2. 分类（骨吸收分级）

**案例 ID: `patient_image_000223_4ebeaf32`** · **数据集: BRAR**

![分类 — BRAR 全景片 Grade 2](assets/images/2_cls_primary_000223.jpg)

**问题**: 根据此全景 X 光片，评估牙周骨吸收严重程度（1、2 或 3 级）。

**✅ Ground Truth**: `{"grade": 2}`

**❌ Lingshu-32B (SFT)**: `{"grade": 3}`
> ⚠️ **升级误判**（Grade 2 → 3）：将中度骨吸收错误判断为重度，可能导致过度治疗。

**✅ Qwen3-VL-4B (SFT)**: `{"grade": 2}`

---

### 3. 描述生成（临床图像描述）

**案例 ID: `23182`** · **数据集: MetaDent**

![描述生成 — 左侧口内照](assets/images/3_cap_primary_23182.jpg)

**问题**: 观察该临床图像，生成一段生动的自然语言描述。

**✅ Ground Truth**（摘录）:
```
In this intraoral photograph the viewer is looking at the labial-buccal aspects of the left side
of both the maxillary and mandibular arches while the patient is in centric occlusion. A clear,
transparent lateral retractor is positioned on the left, gently pulling the cheek and lip tissue
away to expose the teeth and gingiva...
```

**❌ Lingshu-32B (SFT)**（摘录）:
```
The photograph presents a clear view of the maxillary dentition taken from the right side, with
the patient's mouth opened wide enough to expose the full complement of upper teeth. A metal lip
retractor is positioned on the right side...
```
> ⚠️ **多重错误**: ① 左侧→右侧 ② 上下颌→仅上颌 ③ 透明塑料→金属 retractor ④ 咬合状态错误

**✅ Qwen3-VL-4B (SFT)**（摘录）:
```
The photograph presents a clear, side-view intraoral image of the patient's maxillary and
mandibular arches, captured with the patient's mouth slightly open and the lips retracted
using a mouth prop...
```

---

### 4. 报告生成（结构化诊断报告）

**案例 ID: `code_rpt_00529`** · **数据集: COde**

![报告生成 — 口内照](assets/images/4_rpt_primary_00529.jpg)

**问题**: 根据临床图像和患者信息，生成完整的诊断报告。

**患者信息**: 65 岁女性，主诉下左后牙复诊。

**✅ Ground Truth**（摘录）:
```
Patient Record: Follow-up consultation. Diagnosis: Pulpitis (K04.000).
Examination: Tooth 35 shows no percussion pain or gingival inflammation and is temporarily sealed.
Diagnosis: Pulpitis in tooth 35 (K04.000).
```

**❌ Lingshu-32B (SFT)**（摘录）:
```
Patient Record: Follow-up consultation. Diagnosis: Pulpitis (K04.005).
Examination: Tooth 37 shows no percussion pain or gingival inflammation, with a temporary crown in place.
Diagnosis: Pulpitis in tooth 37 (K04.005).
```
> ⚠️ **牙位误识**: #35→#37 | **修复类型错误**: 临时封闭→临时冠 | **ICD 编码错误**: K04.000→K04.005

**✅ Qwen3-VL-4B (SFT)**（摘录）:
```
Patient Record: Follow-up consultation. Diagnosis: Pulpitis (K04.005).
Examination: Tooth 36 is temporarily sealed, with no percussion pain or gingival inflammation.
Diagnosis: Pulpitis in tooth 36 (K04.005).
```

---

### 5. 检测（龋齿检测）

**案例 ID: `caries_test_00071`** · **数据集: DentalCaries**

![检测 — 侧方照，可见龋齿](assets/images/5_det_primary_00071.jpg)

**问题**: 图中是否可见龋齿？（Yes/No）

**✅ Ground Truth**: `Yes`

**❌ Lingshu-32B (SFT)**: `No`
> ⚠️ **假阴性（漏诊）**: 图中可见明显龋齿病变，但 Lingshu 未能检测到。

**✅ Qwen3-VL-4B (SFT)**: `Yes`

---

### 6. 计数（牙齿计数）

**案例 ID: `denpar_count_178`** · **数据集: DenPAR**

![计数 — 根尖片，2 颗牙齿](assets/images/6_cnt_primary_178.jpg)

**问题**: 计算此根尖片中可见牙齿的总数。

**✅ Ground Truth**: `2`

**❌ Lingshu-32B (SFT)**: `6`
> ⚠️ **严重过度计数**（2→6）：将仅有 2 颗牙齿的根尖片误计为 6 颗，可能将邻近骨组织或伪影误计为牙齿。

**✅ Qwen3-VL-4B (SFT)**: `2`

---

## 引用

```bibtex
@inproceedings{pocket-dentist-2026,
  title     = {Pocket-Dentist: Benchmarking Compact Vision-Language Models for Dental Image Understanding},
  author    = {Anonymous},
  booktitle = {NeurIPS 2026 Evaluations \& Datasets Track},
  year      = {2026},
  note      = {Under review}
}
```

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可协议。
