# Development Guide

Detailed guide for setting up, running, and extending the Pocket-Dentist benchmark pipeline.

---

## Project Structure

```
pocket-dentist/
├── configs/                          # Configuration files
│   ├── <dataset>/                    # Per-dataset configs (metadent, brar, dr, aariz, code, denpar, dentalcaries)
│   │   ├── config.yaml               # Data paths (image_dir, test_file)
│   │   ├── few-shots.yaml            # Few-shot exemplars
│   │   ├── slms/                     # Compact model configs (≤ 4B)
│   │   └── llms/                     # Large model configs (7B+)
│
├── src/                              # Inference + evaluation (Python package)
│   ├── main.py                       # Main entry point
│   ├── start_vllm.py                 # vLLM server launcher
│   ├── unified_predictor.py          # Unified predictor (dataset-agnostic)
│   ├── prediction_runner.py          # Prediction dispatcher
│   ├── evaluation_runner.py          # Evaluation dispatcher (Evaluator Registry)
│   ├── models/                       # Model loading (API / Local)
│   ├── tasks/                        # Task evaluators
│   │   ├── vqa/evaluator.py          # VQA evaluation (Accuracy)
│   │   ├── classification/evaluator.py  # Classification (P/R/F1/EM)
│   │   ├── captioning/evaluator.py   # Captioning (BERTScore + LLM-as-judge)
│   │   └── brar/evaluator.py         # BRAR evaluation (3-class)
│   └── utils/                        # Utility functions
│
├── training/                         # Training pipeline (independent from inference)
│   ├── data_process/                 # Data preprocessing scripts
│   ├── model_merge/                  # LoRA weight merging utility
│   └── sft/                          # SFT LoRA fine-tuning
│       ├── sft-qwen3.py              # Per-architecture training scripts
│       ├── sft-gemma4.py
│       └── configs/                  # Per-dataset training configs
│
├── scripts/                          # Shell scripts
│   ├── run_<dataset>.sh              # Local model evaluation entry points
│   ├── run_<dataset>_api.sh          # API model evaluation entry points
│   ├── run_<dataset>_sft.sh          # SFT training entry points
│   ├── vllm_utils.sh                 # vLLM lifecycle management (start/stop/wait)
│   └── audit_*.sh                    # Benchmark audit utilities
│
├── results/                          # Evaluation outputs
│   └── datasets/<dataset>/<setting>/<model>/
│
├── assets/images/                    # Qualitative analysis images
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Hardware Requirements

| Environment | Purpose | Min GPU (1–4B models) | Max GPU (32B models) |
|-------------|---------|----------------------|---------------------|
| `NeurlPS2026-benchmark` | vLLM inference + evaluation | 1× NVIDIA A100 40GB | 1× NVIDIA H100 96GB |
| `NeurlPS2026-train` | LoRA SFT training | 1× NVIDIA A100 40GB | 1× NVIDIA H100 96GB |

**GPU tier mapping** (used by `--tiers` flag):

| Tier | Model Size | Recommended GPU |
|------|-----------|----------------|
| `t1` | 1–2B | A100 21GB |
| `t2` | 3–4B | A100 40GB |
| `t3` | 7–8B | A100 80GB |
| `t4` | 32B | H100 96GB |

---

## Environment Setup

Two separate conda environments are used:

### Inference Environment

```bash
# Create environment
conda create -n NeurlPS2026-benchmark python=3.11 -y
conda activate NeurlPS2026-benchmark

# Install vLLM (includes torch + transformers)
pip install vllm>=0.8.5

# Install project dependencies
pip install -r requirements.txt
```

### Training Environment

```bash
# Create environment
conda create -n NeurlPS2026-train python=3.11 -y
conda activate NeurlPS2026-train

# Install PyTorch
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126

# Install training dependencies
pip install transformers==5.7.0 peft>=0.15.0 accelerate>=1.5.0
pip install flash-attn --no-build-isolation

# Install project dependencies
pip install -r requirements.txt
```

### LLM-as-Judge Configuration (Optional)

Captioning evaluation supports LLM-as-Judge scoring. To enable it:

```bash
cp .env.example .env
# Edit .env and fill in: LLM_JUDGE_MODEL, LLM_JUDGE_API_KEY, LLM_JUDGE_API_BASE
```

If `.env` is not configured, evaluation runs BERTScore only (offline metrics).

---

## Running Evaluation (Inference)

### Local Models (vLLM)

```bash
# Single model, single task
bash scripts/run_metadent.sh --models Qwen3-VL-4B-Instruct --tasks baseline

# Single model, multiple tasks
bash scripts/run_metadent.sh --models Qwen3-VL-4B-Instruct --tasks "baseline,1shot,2shot,sft"

# Multiple models (comma-separated, run sequentially)
bash scripts/run_metadent.sh --models "Qwen3-VL-4B-Instruct,InternVL3_5-2B-HF" --tasks baseline

# By GPU tier (recommended for batch runs)
bash scripts/run_metadent.sh --tiers t1 --tasks "baseline,1shot,2shot,sft"   # 1-2B
bash scripts/run_metadent.sh --tiers t2 --tasks "baseline,1shot,2shot,sft"   # 3-4B
bash scripts/run_metadent.sh --tiers t3 --tasks "baseline,1shot,2shot,sft"   # 7-8B
bash scripts/run_metadent.sh --tiers t4 --tasks "baseline,1shot,2shot,sft"   # 32B
bash scripts/run_metadent.sh --tiers all                                      # all models

# Resume from a specific model
bash scripts/run_metadent.sh --tiers t2 --resume-from gemma-4-E4B-it

# Manual vLLM mode (connect to an already-running vLLM server)
python src/start_vllm.py --config configs/brar/slms/Qwen3-VL-4B-Instruct/vllm.yaml
bash scripts/run_brar.sh --models Qwen3-VL-4B-Instruct --tasks baseline --vllm_server http://localhost:9000/v1
```

Replace `metadent` with any dataset: `brar`, `dr`, `aariz`, `code`, `denpar`, `dentalcaries`.

### API Models (GPT / Gemini)

```bash
# GPT-4o-mini
bash scripts/run_brar_api.sh \
    --model gpt-4o-mini \
    --tasks "baseline,1shot,2shot" \
    --api_base https://api.openai.com/v1 \
    --api_key $OPENAI_API_KEY

# Gemini 2.5 Flash
bash scripts/run_brar_api.sh \
    --model gemini-2.5-flash \
    --tasks "baseline,1shot" \
    --api_base https://generativelanguage.googleapis.com/v1beta/openai/ \
    --api_key $GEMINI_API_KEY
```

---

## Running SFT Training

LoRA fine-tuning with automatic weight merging:

```bash
conda activate NeurlPS2026-train

# Single model
bash scripts/run_metadent_sft.sh --models Qwen3-VL-4B-Instruct

# Multiple models
bash scripts/run_metadent_sft.sh --models "InternVL3_5-1B-HF,gemma-4-E2B-it"

# By GPU tier
bash scripts/run_metadent_sft.sh --tiers t1          # 1-2B models
bash scripts/run_metadent_sft.sh --tiers all          # all models

# Resume from a specific model
bash scripts/run_metadent_sft.sh --tiers t2 --resume-from gemma-4-E4B-it
```

Training output:
```
<MODEL_ROOT>/<Dataset>/<tier>/<Model>/
├── source/     ← LoRA adapter (training artifacts)
└── merged/     ← Merged full model (used for vLLM inference)
```

### Manual LoRA Merge

If auto-merge fails (e.g., OOM) or you want to select a specific checkpoint:

```bash
# Batch merge all pending models
python training/model_merge/merge_lora.py --batch

# Preview only (dry run)
python training/model_merge/merge_lora.py --batch --dry-run

# Filter by dataset or model
python training/model_merge/merge_lora.py --batch --datasets "BRAR,MetaDent"
python training/model_merge/merge_lora.py --batch --models "Lingshu-32B"
```

### Per-Architecture Training Scripts

| Script | Supported Models |
|--------|-----------------|
| `sft-qwen3.py` | Qwen3-VL-4B-Instruct |
| `sft-qwen3.5.py` | Qwen3.5-4B |
| `sft-qwen2.5.py` | Qwen2.5-VL-7B-Instruct |
| `sft-gemma4.py` | gemma-4-E2B-it, gemma-4-E4B-it |
| `sft-medgemma.py` | medgemma-4b-it |
| `sft-internvl3.5-2b-hf.py` | InternVL3_5-1B-HF, InternVL3_5-2B-HF |
| `sft-smolvlm2.py` | SmolVLM2-2.2B-Instruct |
| `sft-paligemma2.py` | paligemma2-3b-mix-448 |
| `sft-medmo.py` | MedMO-8B-Next |
| `sft-lingshu.py` | Lingshu-32B |

---

## Supported Models

### Compact VLMs (≤ 4B, Local vLLM)

| Model | Config Directory |
|-------|-----------------|
| Qwen3-VL-4B-Instruct | `configs/*/slms/Qwen3-VL-4B-Instruct/` |
| Qwen3.5-4B | `configs/*/slms/Qwen3.5-4B/` |
| gemma-4-E4B-it | `configs/*/slms/gemma-4-E4B-it/` |
| gemma-4-E2B-it | `configs/*/slms/gemma-4-E2B-it/` |
| medgemma-4b-it | `configs/*/slms/medgemma-4b-it/` |
| paligemma2-3b-mix-448 | `configs/*/slms/paligemma2-3b-mix-448/` |
| SmolVLM2-2.2B-Instruct | `configs/*/slms/SmolVLM2-2.2B-Instruct/` |
| InternVL3.5-2B | `configs/*/slms/InternVL3_5-2B-HF/` |
| InternVL3.5-1B | `configs/*/slms/InternVL3_5-1B-HF/` |

### Large VLMs (7B+, Local vLLM)

| Model | Config Directory |
|-------|-----------------|
| Qwen2.5-VL-7B-Instruct | `configs/*/llms/Qwen2.5-VL-7B-Instruct/` |
| MedMO-8B-Next | `configs/*/llms/MedMO-8B-Next/` |
| Lingshu-32B | `configs/*/llms/Lingshu-32B/` |

### API Models

| Model | Config Directory |
|-------|-----------------|
| Gemini-2.5-Flash | `configs/metadent/llms-api/gemini-2.5-flash/` |
| Gemini-2.0-Flash | `configs/metadent/llms-api/gemini-2.0-flash/` |

---

## Data Format

All datasets use a unified JSONL `messages` format:

```json
{
    "id": "<unique_id>",
    "task": "vqa | classification | captioning | brar_classification | ...",
    "split": "test",
    "source": "DS1",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "<relative_image_path>"},
                {"type": "text", "text": "<full prompt>"}
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

| Field | Description |
|-------|-------------|
| `task` | Determines which evaluator is used |
| `source` | Used for stratified evaluation (e.g., DS1/DS2/DS3) |
| `image` | Relative path to `config.yaml → data.image_dir` |
| `assistant.content.text` | Ground truth that the model should produce |

---

## Output Directory Structure

All outputs are stored in `results/<dataset>/<run_tag>/<model>/`:

```
results/metadent/baseline/Qwen3-VL-4B-Instruct/
├── predictions.jsonl           # Model predictions (line-delimited JSON)
├── failures.jsonl              # Inference failures
├── vqa/                        # VQA evaluation
│   ├── metrics.json            # Per-source accuracy
│   └── per_sample.json         # Per-sample results
├── classification/             # Classification evaluation
│   ├── metrics.json            # P/R/F1/EM summary
│   ├── per_sample.json
│   ├── classwise_metrics.csv   # Per-class P/R/F1
│   └── overall_metrics.csv     # Macro/Micro summary
└── captioning/                 # Captioning evaluation
    ├── BertScore/
    │   ├── per_sample.json
    │   └── summary.json
    └── confusion_matrix/       # LLM-as-judge results
        ├── per_sample.json
        └── summary.json
```

---

## Extending to New Datasets

To add a new dataset `DatasetX`:

```bash
# 1. Prepare JSONL data (unified messages format)
dataset/DatasetX/output/test.jsonl

# 2. Create configuration
configs/datasetx/config.yaml    # data.image_dir + data.test_file

# 3. Implement evaluator
src/tasks/task_y/evaluator.py   # implement evaluate(predictions, output_dir, args, ...)

# 4. Register in EVALUATOR_REGISTRY (src/evaluation_runner.py)
EVALUATOR_REGISTRY["task_y"] = ("src.tasks.task_y.evaluator", "evaluate", False)

# 5. Create shell script (copy from any existing run_*.sh template)
scripts/run_datasetx.sh
```

The predictor code requires **zero modifications** — it fully reuses `unified_predictor.py`.

> **Note**: For tasks returning raw strings instead of JSON objects (e.g., classification, counting), register the task in `_RAW_OUTPUT_TASKS` in `unified_predictor.py`. This sets `output_type` to `str` and bypasses JSON parsing.

---

## CLI Reference

### Core Arguments

| Argument | Required | Description | Example |
|----------|----------|-------------|---------|
| `--dataset` | ✅ | Dataset name | `metadent`, `brar` |
| `--task` | ✅ | Main task type | `prediction`, `evaluation` |
| `--model_name` | ✅ | Model name | `Qwen3-VL-4B-Instruct` |
| `--run_tag` | | Run tag (default: `baseline`) | `baseline`, `1shot`, `2shot`, `sft` |
| `--subtask` | | Subtask filter (default: `all`) | `vqa`, `classification`, `captioning` |

### Model Arguments

| Argument | Description |
|----------|-------------|
| `--client_type` | `api` (remote API) or `local` (local weights) |
| `--api_base_url` | OpenAI-compatible API endpoint |
| `--api_key` | API key |
| `--workers` | Concurrent threads (API: 1, local: 16) |

### Few-Shot Arguments

| Argument | Description |
|----------|-------------|
| `--few_shot_config` | Few-shot YAML config file path |
| `--num_shots` | Override few-shot count (0 = disabled) |

### Evaluation Arguments

| Argument | Description |
|----------|-------------|
| `--chunk` | Enable BERTScore chunk computation (avoids GPU OOM) |
| `--chunk_size` | BERTScore chunk size (default: 512) |
