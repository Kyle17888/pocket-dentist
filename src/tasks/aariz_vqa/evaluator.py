"""
Aariz Cephalometric VQA Evaluation — Multi-question classification metrics.

Evaluates 5 question types derived from cephalometric landmark measurements:
  - skeletal_class: Class I / Class II / Class III
  - growth_pattern: Hypodivergent / Normodivergent / Hyperdivergent
  - maxilla_position: Prognathic / Normal / Retrognathic
  - mandible_position: Prognathic / Normal / Retrognathic
  - incisor_inclination: Proclined / Normal / Retroclined

Computes per-question-type accuracy and overall weighted accuracy.
"""

import json
import os
import re
from collections import defaultdict
from decimal import Decimal

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.utils.file_io import save_csv_data, save_json_data


# Valid answers for each question type
VALID_ANSWERS = {
    "skeletal_class": ["Class I", "Class II", "Class III"],
    "growth_pattern": ["Hypodivergent", "Normodivergent", "Hyperdivergent"],
    "maxilla_position": ["Prognathic", "Normal", "Retrognathic"],
    "mandible_position": ["Prognathic", "Normal", "Retrognathic"],
    "incisor_inclination": ["Proclined", "Normal", "Retroclined"],
}

# Fuzzy matching aliases
_ALIASES = {
    "Class I": ["class i", "class 1", "skeletal class i"],
    "Class II": ["class ii", "class 2", "skeletal class ii"],
    "Class III": ["class iii", "class 3", "skeletal class iii"],
    "Hypodivergent": ["hypodivergent", "horizontal growth", "low angle"],
    "Normodivergent": ["normodivergent", "normal growth", "average"],
    "Hyperdivergent": ["hyperdivergent", "vertical growth", "high angle"],
    "Prognathic": ["prognathic", "protrusive", "anterior"],
    "Normal": ["normal"],
    "Retrognathic": ["retrognathic", "retrusive", "posterior", "recessive"],
    "Proclined": ["proclined", "protruded", "flared"],
    "Retroclined": ["retroclined", "retruded", "upright"],
}

_ALIAS_MAP = {}
for cls_name, aliases in _ALIASES.items():
    _ALIAS_MAP[cls_name.lower()] = cls_name
    for alias in aliases:
        _ALIAS_MAP[alias.lower()] = cls_name


def extract_answer(raw_prediction: str, question_type: str) -> str | None:
    """Extract the predicted answer from model output."""
    if not raw_prediction or not isinstance(raw_prediction, str):
        return None

    text = raw_prediction.strip().rstrip(".")
    text_lower = text.lower().strip()

    # Direct match
    if text_lower in _ALIAS_MAP:
        return _ALIAS_MAP[text_lower]

    # Try finding valid answers in text
    valid = VALID_ANSWERS.get(question_type, [])
    for answer in valid:
        if answer.lower() in text_lower:
            return answer

    # Try aliases
    for alias, answer in _ALIAS_MAP.items():
        if alias in text_lower and answer in valid:
            return answer

    return None


def evaluate(predictions: list[dict], output_dir: str, args, model=None, yaml_cfg=None):
    """
    Evaluate Aariz VQA predictions across 5 question types.

    Args:
        predictions: List of {id, task, source, ground_truth, prediction, failed, question_type}
        output_dir:  Directory to save evaluation results
        args:        Parsed argparse.Namespace
        model:       Not used
        yaml_cfg:    Not used
    """
    # Group by question type
    by_type = defaultdict(lambda: {"y_true": [], "y_pred": [], "samples": {}})
    unmapped_total = 0

    for p in tqdm(predictions, desc=f"[Aariz VQA] {args.model_name}", dynamic_ncols=True):
        if p.get("failed"):
            continue

        sample_id = p["id"]
        gt_text = p.get("ground_truth", "").strip()
        pred_text = p.get("prediction", "")

        # Determine question type from sample ID
        q_type = None
        for qt in VALID_ANSWERS.keys():
            if qt in sample_id:
                q_type = qt
                break

        if q_type is None:
            # Try from the question_type field if present
            q_type = p.get("question_type", "unknown")

        pred_answer = extract_answer(pred_text, q_type)

        if gt_text not in VALID_ANSWERS.get(q_type, [gt_text]):
            continue

        if pred_answer is None:
            unmapped_total += 1
            pred_answer = "UNMAPPED"

        correct = 1 if gt_text == pred_answer else 0
        by_type[q_type]["y_true"].append(gt_text)
        by_type[q_type]["y_pred"].append(pred_answer)
        by_type[q_type]["samples"][sample_id] = {
            "gt": gt_text,
            "pred": pred_answer,
            "correct": correct,
            "raw_prediction": pred_text[:200],
        }

    if not by_type:
        print("⚠️  No valid Aariz VQA predictions found.")
        return

    # Compute per-question-type metrics
    point = "0.000"
    type_results = []
    all_y_true = []
    all_y_pred = []
    all_per_sample = {}

    for q_type in sorted(VALID_ANSWERS.keys()):
        if q_type not in by_type:
            continue
        data = by_type[q_type]
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_per_sample.update(data["samples"])

        acc = accuracy_score(y_true, y_pred)
        valid_labels = VALID_ANSWERS[q_type]
        w_f1 = f1_score(y_true, y_pred, labels=valid_labels, average="weighted", zero_division=0)

        type_results.append({
            "Question Type": q_type,
            "Accuracy": float(Decimal(str(acc)).quantize(Decimal(point))),
            "F1 (weighted)": float(Decimal(str(w_f1)).quantize(Decimal(point))),
            "Samples": len(y_true),
        })

    type_df = pd.DataFrame(type_results)

    # Overall metrics
    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_f1 = f1_score(all_y_true, all_y_pred, average="weighted", zero_division=0)

    summary = {
        "accuracy": float(Decimal(str(overall_acc)).quantize(Decimal(point))),
        "f1_weighted": float(Decimal(str(overall_f1)).quantize(Decimal(point))),
        "total_samples": len(all_y_true),
        "unmapped_predictions": unmapped_total,
        "per_question_type": type_results,
    }

    save_json_data(all_per_sample, output_dir, "per_sample.json",
                   title=f"Aariz VQA Per-Sample - {args.model_name}")
    save_json_data(summary, output_dir, "metrics.json",
                   title=f"Aariz VQA Summary - {args.model_name}")
    save_csv_data(type_df, output_dir, "per_question_type.csv",
                  title=f"Aariz VQA Per-Type - {args.model_name}")

    print(f"\n  Aariz Cephalometric VQA Results:")
    print(f"    Overall Accuracy: {summary['accuracy']:.3f}")
    print(f"    Overall F1:       {summary['f1_weighted']:.3f}")
    print(f"    Samples:          {summary['total_samples']}")
    print(f"\n    Per Question Type:")
    for r in type_results:
        print(f"      {r['Question Type']:25s}  Acc={r['Accuracy']:.3f}  F1={r['F1 (weighted)']:.3f}  (n={r['Samples']})")
    if unmapped_total > 0:
        print(f"    ⚠️  Unmapped predictions: {unmapped_total}")
