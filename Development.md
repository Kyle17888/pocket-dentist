# Development Guide

## 项目架构

```
Neurips2026-DentistVLM/
├── configs/                          # 配置文件
│   ├── metadent/                     # MetaDent 数据集配置
│   │   ├── config.yaml               # 数据路径 (image_dir, test_file)
│   │   ├── few-shots.yaml            # few-shot 示例
│   │   ├── slms/                     # 小模型配置 (4B 以下)
│   │   ├── llms/                     # 大模型配置 (7B+)
│   │   └── llms-api/                 # API 模型配置 (GPT, Gemini)
│   └── brar/                         # BRAR 数据集配置
│       ├── config.yaml
│       ├── few-shots.yaml
│       ├── slms/
│       └── llms/
│
├── dataset/                          # 本地调试数据 (.gitignore)
│   ├── MetaDent/
│   │   ├── input/                    # 原始数据 + JSONL
│   │   └── output/                   # images/ + 处理后数据
│   └── BRAR/
│
├── results/                          # 运行时输出 (.gitignore)
│   ├── metadent/<run_tag>/<model>/   # MetaDent 推理 + 评估结果
│   └── brar/<run_tag>/<model>/       # BRAR 推理 + 评估结果
│
├── src/                              # 推理 + 评估 (Python 包)
│   ├── main.py                       # 主入口
│   ├── start_vllm.py                 # vLLM 服务启动脚本
│   ├── unified_predictor.py          # 统一预测器
│   ├── prediction_runner.py          # 预测调度
│   ├── evaluation_runner.py          # 评估调度 (Evaluator Registry)
│   ├── models/                       # 模型加载 (API / Local)
│   ├── tasks/                        # 任务评估器
│   │   ├── vqa/evaluator.py          # VQA 评估 (Accuracy)
│   │   ├── classification/evaluator.py  # 分类评估 (P/R/F1/EM)
│   │   ├── captioning/evaluator.py   # 描述评估 (BERTScore + LLM-as-judge)
│   │   └── brar/evaluator.py         # BRAR 评估 (3-class)
│   └── utils/                        # 工具函数
│
├── training/                         # 训练流程 (独立于推理)
│   ├── data_process/                 # 数据处理脚本
│   │   └── metadent/                 # MetaDent 数据集处理
│   │       ├── 01_build_jsonl.py         # 原始数据 → 基础 JSONL (全量)
│   │       └── 02_balance_for_sft.py     # 基础 JSONL → SFT 平衡采样
│   └── sft/                          # SFT LoRA 微调 (训练后自动合并)
│       ├── sft-qwen3.py              # 各模型架构训练脚本
│       ├── sft-gemma4.py
│       ├── configs/brar/             # 按数据集分组的训练配置
│       ├── configs/metadent/         # MetaDent 训练配置
│       └── run_all_sft.sh            # 批量训练
│
├── scripts/                          # 共享工具脚本
│   └── vllm_utils.sh                # vLLM 生命周期管理 (start/stop/wait)
│
├── run_metadent.sh                   # MetaDent 本地模型入口 (默认 auto vLLM, --vllm_server 手动模式)
├── run_metadent_api.sh               # MetaDent API 模型入口
├── run_brar.sh                       # BRAR 本地模型入口 (默认 auto vLLM, --vllm_server 手动模式)
└── run_brar_api.sh                   # BRAR API 模型入口
```

---

## 数据流

```
test.jsonl (统一 messages 格式)
       │
       ▼
unified_predictor.py  ─→  predictions.jsonl
       │
       ▼
evaluation_runner.py  ─→  按 task 字段分流到各 evaluator
       │
       ▼
results/<dataset>/<run_tag>/<model>/
  ├── predictions.jsonl
  ├── failures.jsonl
  ├── vqa/metrics.json
  ├── classification/metrics.json
  └── captioning/BertScore/summary.json
```

---

## 命名规范

### NeSI 目录结构

```
# 数据集
datasets/2_Neurips2026/{Dataset}/          # e.g. MetaDent/, BRAR/

# SFT 模型
models/Neurlps2026-SFT/{Dataset}/{slms,llms}/{model}/
  ├── source/                              # LoRA adapter + 训练产物
  └── merged/                              # 合并后的完整模型 (vLLM 加载)
```

### WandB 命名

| 字段 | 格式 | 示例 |
|---|---|---|
| `project` | `Neurlps2026-SFT-{Dataset}` | `Neurlps2026-SFT-MetaDent` |
| `name` | `{model_dir}-{Dataset}` | `InternVL3_5-2B-HF-MetaDent` |
| `group` | `Dataset-{Dataset}` | `Dataset-MetaDent` |

- `{model_dir}` 统一使用 **configs 目录名**（与 `configs/{dataset}/{slms,llms}/` 下的目录名一致）
- 新增数据集时，按此模式创建对应的 WandB 项目

### Config 目录名 ↔ 模型映射

| 目录名 | HuggingFace 模型 |
|---|---|
| `InternVL3_5-2B-HF` | `OpenGVLab/InternVL3_5-2B-HF` |
| `Qwen3-VL-4B-Instruct` | `Qwen/Qwen3-VL-4B-Instruct` |
| `gemma-4-E2B-it` | `google/gemma-4-E2B-it` |
| `medgemma-4b-it` | `google/medgemma-4b-it` |
| `paligemma2-3b-mix-448` | `google/paligemma2-3b-mix-448` |
| ... | 完整列表见 `configs/{dataset}/models.txt` |

---

## 快速开始

### 1. 运行 BRAR 评估

#### 本地模型 (vLLM)

```bash
# 默认自动管理 vLLM (推荐)
bash run_brar.sh --models Qwen3-VL-4B-Instruct --tasks baseline

# 多个 tasks
bash run_brar.sh --models Qwen3-VL-4B-Instruct --tasks "baseline,1shot,2shot,sft"

# 多个 models (逗号分隔, 串行执行)
bash run_brar.sh --models "Qwen3-VL-4B-Instruct,InternVL3_5-2B-HF" --tasks baseline

# 手动模式: 指定已运行的 vLLM 地址
python src/start_vllm.py --config configs/brar/slms/Qwen3-VL-4B-Instruct/vllm.yaml
bash run_brar.sh --models Qwen3-VL-4B-Instruct --tasks baseline --vllm_server http://localhost:9000/v1
```

#### API 模型 (GPT / Gemini)

```bash
# GPT-4o-mini
bash run_brar_api.sh \
    --model gpt-4o-mini \
    --tasks "baseline,1shot,2shot" \
    --api_base https://api.openai.com/v1 \
    --api_key $OPENAI_API_KEY

# Gemini 2.5 Flash
bash run_brar_api.sh \
    --model gemini-2.5-flash \
    --tasks "baseline,1shot" \
    --api_base https://generativelanguage.googleapis.com/v1beta/openai/ \
    --api_key $GEMINI_API_KEY
```

### 2. 运行 MetaDent 评估

#### 本地模型 (vLLM)

```bash
# 单个模型
bash run_metadent.sh --models Qwen3-VL-4B-Instruct --tasks baseline
bash run_metadent.sh --models Qwen3-VL-4B-Instruct --tasks "baseline,1shot,2shot,sft"

# 多个模型 (逗号分隔)
bash run_metadent.sh --models "Qwen3-VL-4B-Instruct,InternVL3_5-2B-HF" --tasks baseline

# 按 GPU tier 批量跑 (推荐用于 NeSI SLURM)
bash run_metadent.sh --tiers t1 --tasks "baseline,1shot,2shot,sft"   # 1-2B  (A100 21GB)
bash run_metadent.sh --tiers t2 --tasks "baseline,1shot,2shot,sft"   # 3-4B  (A100 40GB)
bash run_metadent.sh --tiers t3 --tasks "baseline,1shot,2shot,sft"   # 7-8B  (A100 80GB)
bash run_metadent.sh --tiers t4 --tasks "baseline,1shot,2shot,sft"   # 32B   (H100 80GB)
bash run_metadent.sh --tiers t1,t2 --tasks "baseline,sft"            # 组合多个 tier
bash run_metadent.sh --tiers all                                     # 全部

# 从某个模型恢复 (跳过已完成的)
bash run_metadent.sh --tiers t2 --resume-from gemma-4-E4B-it
```

#### API 模型

```bash
bash run_metadent_api.sh \
    --model gpt-4o-mini \
    --tasks baseline \
    --api_base https://api.openai.com/v1 \
    --api_key $OPENAI_API_KEY
```

#### Captioning LLM-as-Judge 评估 (自动触发)

Captioning 评估会自动检测项目根目录的 `.env` 文件。配置了 LLM Judge 后无需手动操作：

```bash
# 1. 配置 .env (仅需一次)
cp .env.example .env
# 编辑 .env 填入: LLM_JUDGE_MODEL, LLM_JUDGE_API_KEY, LLM_JUDGE_API_BASE

# 2. 正常运行评估, LLM Judge 自动触发
bash run_metadent.sh --models Qwen3-VL-4B-Instruct --tasks baseline
```

如果未配置 `.env`，评估仅运行 BERTScore（离线指标），不会报错。

---

## CLI 参数说明

### 核心参数

| 参数 | 必需 | 说明 | 示例 |
|---|---|---|---|
| `--dataset` | ✅ | 数据集名称 | `metadent`, `brar` |
| `--task` | ✅ | 主任务类型 | `prediction`, `evaluation` |
| `--model_name` | ✅ | 模型名称 | `Qwen3-VL-4B-Instruct`, `gpt-4o-mini` |
| `--run_tag` | | 运行标签 (默认 `baseline`) | `baseline`, `1shot`, `2shot`, `sft` |
| `--subtask` | | 子任务过滤 (默认 `all`) | `vqa`, `classification`, `captioning` |

### 模型参数

| 参数 | 说明 |
|---|---|
| `--client_type` | `api` (远程 API) 或 `local` (本地权重) |
| `--api_base_url` | OpenAI 兼容 API 地址 |
| `--api_key` | API 密钥 |
| `--workers` | 并发线程数 (API 建议 1，本地建议 16) |

### Few-shot 参数

| 参数 | 说明 |
|---|---|
| `--few_shot_config` | few-shot YAML 配置文件路径 |
| `--num_shots` | 覆盖 few-shot 数量 (0 = 禁用) |

### 评估参数

| 参数 | 说明 |
|---|---|
| `--enable_llm_judge` | (已弃用) 现在通过 `.env` 自动检测 |
| `--evaluator_model_name` | (已弃用) 使用 `.env` 中的 `LLM_JUDGE_MODEL` |
| `--chunk` | 启用 BERTScore 分块计算 (避免 GPU OOM) |
| `--chunk_size` | BERTScore 分块大小 (默认 512) |

---

## 输出目录结构

所有输出存放在 `results/<dataset>/<run_tag>/<model>/`：

```
results/metadent/baseline/Qwen3-VL-4B-Instruct/
├── predictions.jsonl           # 模型推理结果 (逐行 JSON)
├── failures.jsonl              # 推理失败记录
├── vqa/                        # VQA 评估
│   ├── metrics.json            # 各 source 的 accuracy
│   └── per_sample.json         # 逐样本结果
├── classification/             # 分类评估
│   ├── metrics.json            # P/R/F1/EM 汇总
│   ├── per_sample.json         # 逐样本结果
│   ├── classwise_metrics.csv   # 逐类别 P/R/F1
│   └── overall_metrics.csv     # Macro/Micro 汇总
└── captioning/                 # 描述评估
    ├── BertScore/
    │   ├── per_sample.json     # 逐样本 BERTScore
    │   └── summary.json        # 分 source 汇总
    └── confusion_matrix/       # LLM-as-judge 混淆矩阵
        ├── per_sample.json
        └── summary.json
```

BRAR 输出结构类似但更简单：
```
results/brar/baseline/Qwen3-VL-4B-Instruct/
├── predictions.jsonl
├── failures.jsonl
└── brar_classification/        # BRAR 评估
    ├── metrics.json            # accuracy, F1, precision, recall
    ├── confusion_matrix.csv    # 3×3 混淆矩阵
    └── per_class_metrics.csv   # 逐等级 P/R/F1
```

---

## 支持的模型

### 小模型 (≤ 4B, 本地 vLLM)

| 模型 | 配置目录 |
|---|---|
| Qwen3-VL-4B-Instruct | `configs/*/slms/Qwen3-VL-4B-Instruct/` |
| InternVL2_5-4B | `configs/*/slms/InternVL2_5-4B/` |
| InternVL3_5-1B-HF | `configs/*/slms/InternVL3_5-1B-HF/` |
| InternVL3_5-2B-HF | `configs/*/slms/InternVL3_5-2B-HF/` |
| SmolVLM2-2.2B-Instruct | `configs/*/slms/SmolVLM2-2.2B-Instruct/` |
| Qwen3.5-4B | `configs/*/slms/Qwen3.5-4B/` |
| gemma-4-E2B-it | `configs/*/slms/gemma-4-E2B-it/` |
| gemma-4-E4B-it | `configs/*/slms/gemma-4-E4B-it/` |
| medgemma-4b-it | `configs/*/slms/medgemma-4b-it/` |
| dentalgemma-1.5-4b-it | `configs/*/slms/dentalgemma-1.5-4b-it/` |
| paligemma2-3b-mix-448 | `configs/*/slms/paligemma2-3b-mix-448/` |

### 大模型 (7B+, 本地 vLLM)

| 模型 | 配置目录 |
|---|---|
| Qwen2.5-VL-7B-Instruct | `configs/*/llms/Qwen2.5-VL-7B-Instruct/` |
| MedMO-8B-Next | `configs/*/llms/MedMO-8B-Next/` |
| Lingshu-32B | `configs/*/llms/Lingshu-32B/` |

### API 模型

| 模型 | 配置目录 |
|---|---|
| gpt-4o-mini | `configs/metadent/llms-api/gpt-4o-mini/` |
| gemini-2.5-flash | `configs/metadent/llms-api/gemini-2.5-flash/` |

---

## Benchmark Audit & Verification

在 NeSI 上运行以下脚本检查 benchmark 完整性：

```bash
# 1️⃣ 检查推理结果完整性 (predictions + metrics)
bash scripts/audit_results.sh                  # 全量扫描
bash scripts/audit_results.sh --issues-only    # 只显示问题
bash scripts/audit_results.sh --dataset aariz  # 单个数据集
bash scripts/audit_results.sh --setting sft    # 单个设置

# 2️⃣ 检查 SFT 训练完成状态
bash scripts/audit_sft_status.sh               # 7 datasets × 12 models

# 3️⃣ 检查 LoRA 合并状态
bash scripts/audit_sft_merge.sh                # merged/ 目录是否有权重文件

# 4️⃣ 查看剩余待办任务
cat scripts/rerun_tasks.sh                     # 完整的待办清单和命令
```

### Audit Scripts

| 脚本 | 功能 | 何时使用 |
|------|------|---------|
| `scripts/audit_results.sh` | 扫描 `results/datasets/` 检查 predictions + metrics | 每次推理完成后 |
| `scripts/audit_sft_status.sh` | 检查 SFT 训练是否完成 (adapter + merged) | 训练后确认 |
| `scripts/audit_sft_merge.sh` | 检查 LoRA → merged 合并状态 | 合并前后 |
| `scripts/rerun_tasks.sh` | 剩余任务清单和执行命令 | 参考执行 |

---

## 论文表格生成 & 数据校验

论文中的 benchmark 表格 (`tables_all.tex`) 由自动化脚本从原始 `metrics.json` 生成，确保数据一致性。

### 数据流

```
results/datasets/<dataset>/<setting>/<model>/<task>/metrics.json  (原始数据源)
         │
         ▼
results/datasets/summary_overleaf_table.py --generate --verify
         │
         ├──→ results/datasets/overleaf_table_metrics.json     (3位小数真值)
         └──→ neurips-paper/overleaf/v0.1.0/tables_all.tex     (2位小数显示)
```

### 使用方式

```bash
# 只提取 JSON (3位小数精度)
python3 results/datasets/summary_overleaf_table.py

# 提取 + 生成 LaTeX 表格
python3 results/datasets/summary_overleaf_table.py --generate

# 提取 + 验证当前 LaTeX 与源数据一致性
python3 results/datasets/summary_overleaf_table.py --verify

# 完整流程: 提取 + 生成 + 验证 (推荐)
python3 results/datasets/summary_overleaf_table.py --generate --verify
```

### 加粗/下划线规则

- **排序**使用 3 位小数精度 (来自 `overleaf_table_metrics.json`)
- **显示**使用 2 位小数 (LaTeX 表格中)
- **加粗** = 每列最佳值 (所有并列都加粗)
- **下划线** = 每列次佳值 (仅第一个出现的模型标记)

---

## 扩展新数据集

当需要新增数据集 `DatasetX` 时：

```bash
# 1. 准备 JSONL 数据（统一 messages 格式）
dataset/DatasetX/output/test.jsonl

# 2. 创建配置
configs/datasetx/config.yaml    # data.image_dir + data.test_file

# 3. 实现 evaluator
src/tasks/task_y/evaluator.py   # 实现 evaluate(predictions, output_dir, args, ...)

# 4. 注册到 EVALUATOR_REGISTRY（src/evaluation_runner.py）
EVALUATOR_REGISTRY["task_y"] = ("src.tasks.task_y.evaluator", "evaluate", False)

# 5. 创建 shell 脚本（可复制 run_brar.sh 模板，改 DATASET 变量）
run_datasetx.sh
```

**Predictor 代码零修改**，完全复用 `unified_predictor.py`。
*(注意：对于像分类、计数等返回原始字符串而不是 JSON 对象的任务，在 `unified_predictor.py` 中的 `_RAW_OUTPUT_TASKS` 注册该任务，此时 `output_type` 会自动设为 `str`，底层 Model 会直接返回模型原始文本，跳过 JSON 解析过程)*

---

## JSONL 数据格式规范

所有数据集的 `test.jsonl` 必须遵循以下格式：

```json
{
    "id": "<unique_id>",
    "task": "vqa | classification | captioning | brar_classification",
    "split": "test",
    "source": "DS1",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "<relative_image_path>"},
                {"type": "text", "text": "<完整 prompt>"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<ground truth JSON string>"}
            ]
        }
    ]
}
```

- `task`: 决定使用哪个 evaluator
- `source`: 用于分层评估 (e.g., DS1/DS2/DS3)
- `image`: 相对于 `config.yaml` 中 `data.image_dir` 的路径
- `assistant.content.text`: 模型应输出的 ground truth

---

## 训练 (SFT + LoRA 合并)

训练相关脚本位于 `training/` 目录，独立于推理管线。

### SFT LoRA 微调

训练完成后会 **自动合并 LoRA 权重到 base model**，无需手动操作。

```bash
# 激活训练环境
conda activate NeurlPS2026-train

# 单模型训练 (训练 + 自动合并)
python training/sft/sft-qwen3.py --config training/sft/configs/metadent/slms/Qwen3-VL-4B-Instruct.yaml

# 批量训练 — 推荐使用 run_metadent_sft.sh
bash run_metadent_sft.sh --models InternVL3_5-1B-HF                    # 单个模型
bash run_metadent_sft.sh --models "InternVL3_5-1B-HF,gemma-4-E2B-it"   # 多个模型
bash run_metadent_sft.sh --tiers t1                                     # 按 GPU tier
bash run_metadent_sft.sh --tiers t1,t2                                  # 组合 tier
bash run_metadent_sft.sh --tiers all                                    # 全部模型
bash run_metadent_sft.sh --tiers t2 --resume-from gemma-4-E4B-it        # 从某个模型恢复
```

训练输出：
```
.../Qwen3-VL-4B-Instruct/
├── source/     ← LoRA adapter (训练产物，保留备份)
└── merged/     ← 合并后的完整模型 (直接用于 vLLM 推理)
```

每个模型架构有独立的训练脚本：

| 脚本 | 支持的模型 |
|---|---|
| `sft-qwen3.py` | Qwen3-VL-4B-Instruct |
| `sft-qwen3.5.py` | Qwen3.5-4B |
| `sft-qwen2.5.py` | Qwen2.5-VL-7B-Instruct |
| `sft-gemma4.py` | gemma-4-E2B-it, gemma-4-E4B-it |
| `sft-medgemma.py` | medgemma-4b-it |
| `sft-dentalgemma.py` | dentalgemma-1.5-4b-it |
| `sft-internvl.py` | InternVL2_5-4B |
| `sft-internvl3.5-2b-hf.py` | InternVL3_5-1B-HF, InternVL3_5-2B-HF |
| `sft-smolvlm2.py` | SmolVLM2-2.2B-Instruct |
| `sft-paligemma2.py` | paligemma2-3b-mix-448 |
| `sft-medmo.py` | MedMO-8B-Next |
| `sft-lingshu.py` | Lingshu-32B |

合并后的模型通过 vLLM 启动即可用于推理。

### 手动合并 LoRA 权重 (Manual Merge)

训练完成后通常会自动合并 LoRA 权重。如果自动合并失败（OOM），或者你想通过 WandB `eval_loss` 曲线选择最优 checkpoint 手动合并，使用 `training/model_merge/merge_lora.py`。

#### 检查合并状态

```bash
# 扫描所有数据集/模型，查看哪些已合并、哪些需要合并
bash scripts/audit_sft_merge.sh
```

#### 批量合并（推荐）

```bash
# 申请 CPU 大内存节点（不需要 GPU）
srun --account=uoa04670 --job-name=merge-lora \
  --partition=milan --cpus-per-task=8 --mem=128G --time=1:00:00 --pty bash

# 激活训练环境
source ~/01_kyle/NeurlPS2026-train.sh

# 自动扫描并合并所有未合并的模型
python training/model_merge/merge_lora.py --batch

# 先预览，不执行
python training/model_merge/merge_lora.py --batch --dry-run

# 只合并特定数据集
python training/model_merge/merge_lora.py --batch --datasets "BRAR,MetaDent"

# 只合并特定模型
python training/model_merge/merge_lora.py --batch --models "Lingshu-32B"

# 组合过滤
python training/model_merge/merge_lora.py --batch --datasets "COde" --models "Lingshu-32B"
```

#### 单个合并（指定 checkpoint）

当你想合并特定 checkpoint（比如 WandB 上 eval_loss 最低的那一步）：

1. **修改配置** `training/model_merge/merge_config.yaml`：
   ```yaml
   # 指向你想要合并的 checkpoint
   lora_path: ".../Neurlps2026-SFT/DenPAR/llms/Lingshu-32B/source/checkpoint-150"
   # 合并后的保存路径
   merged_output_dir: ".../Neurlps2026-SFT/DenPAR/llms/Lingshu-32B/merged"
   # base_model_path: "" (留空自动从 adapter_config.json 读取)
   ```

2. **执行合并**：
   ```bash
   python training/model_merge/merge_lora.py --config training/model_merge/merge_config.yaml
   ```

#### NeSI 资源需求（CPU only）

| 模型大小 | 内存 | 耗时 |
|----------|------|------|
| 1-4B | 32GB | ~2 分钟 |
| 7-8B | 64GB | ~5 分钟 |
| 32B (Lingshu) | 128GB | ~15 分钟 |

