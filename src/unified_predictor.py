#!/usr/bin/env python3
from __future__ import annotations
"""
Unified Predictor — Data-driven inference for all datasets and tasks.

Reads a test.jsonl file where each sample follows the standardized messages format:
  {
    "id": "...",
    "task": "vqa | classification | captioning | brar_classification",
    "source": "DS1",
    "messages": [
      {"role": "user", "content": [{"type": "image", ...}, {"type": "text", ...}]},
      {"role": "assistant", "content": [{"type": "text", "text": "<ground_truth>"}]}
    ]
  }

Supports:
  - All datasets (MetaDent, BRAR, future datasets)
  - All task types via the same code path
  - Few-shot injection per task type
  - Breakpoint resume (skip completed IDs)
  - Concurrent inference (ThreadPoolExecutor)
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.utils.few_shot import load_few_shot_config, build_few_shot_messages


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

# Default task type per dataset — used when test.jsonl entries lack a "task" field.
# Datasets with a single task type map to that task; multi-task datasets (e.g. metadent,
# code, aariz, denpar, dentalcaries) should always include "task" in their JSONL.
_DATASET_DEFAULT_TASK = {
    "brar": "brar_classification",
    "dr": "dr_classification",
}


def load_test_data(config: dict, dataset: str = "") -> list[dict]:
    """
    Load test.jsonl and extract structured fields for each sample.

    Args:
        config:  Parsed YAML config with data.test_file and data.image_dir.
        dataset: Dataset name (e.g. 'brar'). Used to infer a default task when
                 the JSONL entries don't include a "task" field.

    Returns:
        List of dicts: {id, task, source, image_path, prompt_text, ground_truth}
    """
    test_file = config["data"]["test_file"]
    default_task = _DATASET_DEFAULT_TASK.get(dataset, "unknown")
    image_dir = config["data"]["image_dir"]

    samples = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            user_msg = item["messages"][0]
            asst_msg = item["messages"][1]

            # Extract image paths and prompt text from user message
            image_rels = []
            prompt_text = None
            for part in user_msg["content"]:
                if part["type"] == "image":
                    image_rels.append(part["image"])
                elif part["type"] == "text":
                    prompt_text = part["text"]

            # Extract ground truth from assistant message
            ground_truth = ""
            for part in asst_msg["content"]:
                if part["type"] == "text":
                    ground_truth = part["text"]

            # Resolve image paths (multi-image support)
            image_paths = [os.path.join(image_dir, rel) for rel in image_rels]
            # Backward compat: image_path = first image (single-image tasks)
            image_path = image_paths[0] if image_paths else None

            samples.append({
                "id": item["id"],
                "task": item.get("task", default_task),
                "source": item.get("source", ""),
                "image_path": image_path,
                "image_paths": image_paths,
                "prompt_text": prompt_text,
                "ground_truth": ground_truth,
            })

    return samples


# ──────────────────────────────────────────────────────────────
# Output type inference
# ──────────────────────────────────────────────────────────────

# Tasks where the model should return a raw string (post-processing done by evaluator)
_RAW_OUTPUT_TASKS = {"brar_classification", "dr_classification", "code_classification", "code_report", "aariz_cvm", "aariz_vqa", "denpar_count", "denpar_arch", "denpar_site", "caries_detect", "caries_cls"}

# Tasks where the model should return a list
_LIST_OUTPUT_TASKS = {"classification"}

def _get_output_type(task: str):
    """Determine the output_type kwarg for model.generate_from_image_and_text()."""
    if task in _RAW_OUTPUT_TASKS:
        return str
    if task in _LIST_OUTPUT_TASKS:
        return list
    # Default: dict (vqa, captioning, etc.)
    return dict


# ──────────────────────────────────────────────────────────────
# Few-shot message builder (per task type)
# ──────────────────────────────────────────────────────────────

# Mapping from JSONL task field to few-shot config section name
_TASK_TO_FEW_SHOT_SECTION = {
    "vqa": "vqa",
    "classification": "classification",
    "captioning": "captioning",
    "brar_classification": "classification",  # BRAR reuses classification few-shots
    "code_classification": "code_classification",
    "code_report": "code_report",
    "aariz_cvm": "aariz_cvm",
    "aariz_vqa": "aariz_vqa",
    "denpar_count": "denpar_count",
    "denpar_arch": "denpar_arch",
    "denpar_site": "denpar_site",
    "dr_classification": "dr_classification",
    "caries_detect": "caries_detect",
    "caries_cls": "caries_cls",
}

# Short prompts used as the "user text" in few-shot example turns
_TASK_SHORT_PROMPTS = {
    "vqa": "Please select only one correct answer based on the visual evidence from the image.",
    "classification": "Please perform multi-class category extraction based on this dental clinical image.",
    "captioning": "Observe the clinical image and generate a vivid natural language description.",
    "brar_classification": "Grade the periodontal bone resorption severity (1, 2, or 3) based on this panoramic radiograph.",
    "code_classification": "Identify the primary oro-dental anomaly from the clinical images.",
    "code_report": "Generate a complete diagnostic report based on the clinical images and patient information.",
    "aariz_cvm": "Classify the CVM stage from this lateral cephalometric radiograph.",
    "aariz_vqa": "Answer the clinical question based on this lateral cephalometric radiograph.",
    "denpar_count": "Count the visible teeth in this periapical radiograph.",
    "denpar_arch": "Determine if this periapical radiograph is from the upper or lower jaw.",
    "denpar_site": "Determine the anatomical region of this periapical radiograph.",
    "dr_classification": "Identify which dental finding categories are present in this panoramic radiograph.",
    "caries_detect": "Determine whether dental caries is visible in this intraoral photograph.",
    "caries_cls": "Classify the type of teeth affected by the caries in this intraoral photograph.",
}


def _build_few_shot_for_task(task: str, few_shot_cfg: dict, model_name: str) -> list:
    """Build few-shot messages for a specific task type."""
    if not few_shot_cfg or few_shot_cfg.get("num_shots", 0) == 0:
        return []

    section = _TASK_TO_FEW_SHOT_SECTION.get(task, task)
    short_prompt = _TASK_SHORT_PROMPTS.get(task, "")
    return build_few_shot_messages(section, short_prompt, few_shot_cfg, model_name=model_name)


# ──────────────────────────────────────────────────────────────
# Single-sample prediction
# ──────────────────────────────────────────────────────────────

def predict_single(sample: dict, model, few_shot_messages: list, output_type) -> dict:
    """
    Run inference on a single sample.

    Supports both single-image (image_path) and multi-image (image_paths) samples.

    Returns:
        {id, task, source, ground_truth, prediction, raw_output, failed}
    """
    try:
        image_paths = sample.get("image_paths", [])

        if len(image_paths) > 1:
            # Multi-image: use generate_from_images_and_text
            result = model.generate_from_images_and_text(
                image_paths=image_paths,
                prompt=sample["prompt_text"],
                output_type=output_type,
                few_shot_messages=few_shot_messages,
            )
        else:
            # Single-image: use existing method (backward compat)
            result = model.generate_from_image_and_text(
                image_path=sample["image_path"],
                prompt=sample["prompt_text"],
                output_type=output_type,
                few_shot_messages=few_shot_messages,
            )
        raw_output = json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result

        return {
            "id": sample["id"],
            "task": sample["task"],
            "source": sample["source"],
            "ground_truth": sample["ground_truth"],
            "prediction": raw_output,
            "failed": False,
        }

    except Exception as e:
        return {
            "id": sample["id"],
            "task": sample["task"],
            "source": sample["source"],
            "ground_truth": sample["ground_truth"],
            "prediction": str(e),
            "failed": True,
        }


# ──────────────────────────────────────────────────────────────
# Breakpoint resume
# ──────────────────────────────────────────────────────────────

def load_completed_ids(pred_file: str) -> set:
    """Load IDs of already-completed predictions for resume support."""
    completed = set()
    if os.path.exists(pred_file):
        with open(pred_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    completed.add(str(obj["id"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


# ──────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────

def run_unified_prediction(model, yaml_cfg: dict, args):
    """
    Unified prediction entry point, called by prediction_runner.

    Args:
        model:     Initialized APIModel or BaseModel instance.
        yaml_cfg:  Global config (configs/<dataset>/config.yaml).
        args:      Parsed argparse.Namespace with dataset, run_tag, workers, etc.
    """
    # Resolve output directory: results/<dataset>/<run_tag>/<model>/
    output_dir = os.path.join(
        args.save_root_dir,
        args.run_tag,
        args.model_name.split("/")[-1],
    )
    os.makedirs(output_dir, exist_ok=True)

    pred_file = os.path.join(output_dir, "predictions.jsonl")
    fail_file = os.path.join(output_dir, "failures.jsonl")

    print(f"{'=' * 60}")
    print(f"Unified Prediction")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Model:      {args.model_name}")
    print(f"  Run Tag:    {args.run_tag}")
    print(f"  Workers:    {args.workers}")
    print(f"  Output:     {output_dir}")
    print(f"{'=' * 60}")

    # 1. Load few-shot config
    num_shots = args.num_shots if getattr(args, "num_shots", None) is not None else 0
    few_shot_cfg = load_few_shot_config(
        getattr(args, "few_shot_config", None),
        num_shots_override=num_shots,
    )
    if num_shots > 0 and few_shot_cfg:
        print(f"  Few-shot:   {num_shots} shots configured")
    else:
        print(f"  Few-shot:   disabled")

    # 2. Load test data
    test_data = load_test_data(yaml_cfg, dataset=args.dataset)

    # 3. Filter by subtask if specified
    if getattr(args, "subtask", "all") != "all":
        test_data = [s for s in test_data if s["task"] == args.subtask]

    print(f"  Test data:  {len(test_data)} samples")

    # 4. Breakpoint resume: skip completed IDs
    completed = load_completed_ids(pred_file)
    pending = [s for s in test_data if str(s["id"]) not in completed]

    if completed:
        print(f"  Completed:  {len(completed)} (resuming)")
    print(f"  Pending:    {len(pending)}")

    if not pending:
        print("All samples already completed.")
        return

    # 5. Pre-build few-shot messages per task type (cache to avoid rebuilding)
    few_shot_cache = {}
    task_types = set(s["task"] for s in pending)
    for task_type in task_types:
        few_shot_cache[task_type] = _build_few_shot_for_task(
            task_type, few_shot_cfg, args.model_name
        )

    # 6. Run prediction with concurrent workers
    success_count = 0
    fail_count = 0

    with open(pred_file, "a", encoding="utf-8") as f_out, \
         open(fail_file, "a", encoding="utf-8") as f_fail:

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for s in pending:
                output_type = _get_output_type(s["task"])
                fs_msgs = few_shot_cache.get(s["task"], [])
                future = executor.submit(predict_single, s, model, fs_msgs, output_type)
                futures[future] = s

            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Predicting", unit="sample", dynamic_ncols=True,
            ):
                result = future.result()
                line = json.dumps(result, ensure_ascii=False) + "\n"

                if result.get("failed"):
                    f_fail.write(line)
                    f_fail.flush()
                    fail_count += 1
                else:
                    f_out.write(line)
                    f_out.flush()
                    success_count += 1

    # 7. Print summary
    print(f"\n{'=' * 60}")
    print(f"Prediction Complete")
    print(f"  Success:    {success_count}")
    print(f"  Failed:     {fail_count}")
    print(f"  Output:     {pred_file}")
    print(f"{'=' * 60}")
