"""
Aariz CVM Stage Classification Evaluation — 6-class single-label metrics.

Reads unified predictions and computes:
  - Per-class Precision, Recall, F1
  - Overall Accuracy, Weighted F1, Macro F1
  - Confusion matrix (6×6)

CVM stages: CVM-S1 through CVM-S6.
"""

import json
import os
import re
from decimal import Decimal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from src.utils.file_io import save_csv_data, save_json_data


# The 6 CVM stages
CVM_CLASSES = ["CVM-S1", "CVM-S2", "CVM-S3", "CVM-S4", "CVM-S5", "CVM-S6"]

# Fuzzy matching aliases
_ALIASES = {
    "CVM-S1": ["cvm-s1", "cs1", "cvms1", "stage 1", "s1", "cvm s1"],
    "CVM-S2": ["cvm-s2", "cs2", "cvms2", "stage 2", "s2", "cvm s2"],
    "CVM-S3": ["cvm-s3", "cs3", "cvms3", "stage 3", "s3", "cvm s3"],
    "CVM-S4": ["cvm-s4", "cs4", "cvms4", "stage 4", "s4", "cvm s4"],
    "CVM-S5": ["cvm-s5", "cs5", "cvms5", "stage 5", "s5", "cvm s5"],
    "CVM-S6": ["cvm-s6", "cs6", "cvms6", "stage 6", "s6", "cvm s6"],
}

# Build reverse lookup
_ALIAS_MAP = {}
for cls_name, aliases in _ALIASES.items():
    _ALIAS_MAP[cls_name.lower()] = cls_name
    for alias in aliases:
        _ALIAS_MAP[alias.lower()] = cls_name


def extract_predicted_class(raw_prediction: str) -> str | None:
    """Extract the predicted CVM stage from model output using fuzzy matching."""
    if not raw_prediction or not isinstance(raw_prediction, str):
        return None

    text = raw_prediction.strip().rstrip(".")

    # Try direct match first (case-insensitive)
    text_lower = text.lower().strip()
    if text_lower in _ALIAS_MAP:
        return _ALIAS_MAP[text_lower]

    # Try regex: find CVM-S\d pattern
    match = re.search(r"CVM[-\s]?S(\d)", text, re.IGNORECASE)
    if match:
        stage_num = match.group(1)
        stage = f"CVM-S{stage_num}"
        if stage in CVM_CLASSES:
            return stage

    # Try CS\d pattern
    match = re.search(r"\bCS(\d)\b", text, re.IGNORECASE)
    if match:
        stage_num = match.group(1)
        stage = f"CVM-S{stage_num}"
        if stage in CVM_CLASSES:
            return stage

    # Try finding any class name in the text
    for cls_name in CVM_CLASSES:
        if cls_name.lower() in text_lower:
            return cls_name

    # Try aliases in text
    for alias, cls_name in _ALIAS_MAP.items():
        if alias in text_lower:
            return cls_name

    # Handle "Not assessable"
    if "not assessable" in text_lower or "insufficient" in text_lower:
        return "Not assessable"

    return None


def evaluate(predictions: list[dict], output_dir: str, args, model=None, yaml_cfg=None):
    """
    Evaluate Aariz CVM classification predictions (6-class single-label).

    Args:
        predictions: List of {id, task, source, ground_truth, prediction, failed}
        output_dir:  Directory to save evaluation results
        args:        Parsed argparse.Namespace
        model:       Not used (offline evaluation)
        yaml_cfg:    Not used
    """
    y_true = []
    y_pred = []
    per_sample = {}
    unmapped_count = 0

    for p in tqdm(predictions, desc=f"[Aariz CVM] {args.model_name}", dynamic_ncols=True):
        if p.get("failed"):
            continue

        sample_id = p["id"]
        gt_text = p.get("ground_truth", "")
        pred_text = p.get("prediction", "")

        gt_class = gt_text.strip() if isinstance(gt_text, str) else None
        pred_class = extract_predicted_class(pred_text)

        if gt_class not in CVM_CLASSES:
            continue

        if pred_class is None or pred_class == "Not assessable":
            unmapped_count += 1
            pred_class = "UNMAPPED"

        correct = 1 if gt_class == pred_class else 0
        y_true.append(gt_class)
        y_pred.append(pred_class)

        per_sample[sample_id] = {
            "gt": gt_class,
            "pred": pred_class,
            "correct": correct,
            "raw_prediction": pred_text[:200],
        }

    if not y_true:
        print("⚠️  No valid Aariz CVM predictions found.")
        return

    all_labels = CVM_CLASSES + (["UNMAPPED"] if unmapped_count > 0 else [])

    accuracy = accuracy_score(y_true, y_pred)
    w_f1 = f1_score(y_true, y_pred, labels=CVM_CLASSES, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=CVM_CLASSES, average="macro", zero_division=0)
    w_precision = precision_score(y_true, y_pred, labels=CVM_CLASSES, average="weighted", zero_division=0)
    w_recall = recall_score(y_true, y_pred, labels=CVM_CLASSES, average="weighted", zero_division=0)

    report = classification_report(
        y_true, y_pred, labels=CVM_CLASSES,
        output_dict=True, zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)

    classwise = []
    for cls in CVM_CLASSES:
        if cls in report:
            classwise.append({
                "Class": cls,
                "Precision": report[cls]["precision"],
                "Recall": report[cls]["recall"],
                "F1": report[cls]["f1-score"],
                "Support": report[cls]["support"],
            })
    classwise_df = pd.DataFrame(classwise)

    point = "0.000"
    summary = {
        "accuracy": float(Decimal(str(accuracy)).quantize(Decimal(point))),
        "f1_weighted": float(Decimal(str(w_f1)).quantize(Decimal(point))),
        "f1_macro": float(Decimal(str(macro_f1)).quantize(Decimal(point))),
        "precision_weighted": float(Decimal(str(w_precision)).quantize(Decimal(point))),
        "recall_weighted": float(Decimal(str(w_recall)).quantize(Decimal(point))),
        "total_samples": len(y_true),
        "unmapped_predictions": unmapped_count,
    }

    save_json_data(per_sample, output_dir, "per_sample.json",
                   title=f"Aariz CVM Per-Sample - {args.model_name}")
    save_json_data(summary, output_dir, "metrics.json",
                   title=f"Aariz CVM Summary - {args.model_name}")
    save_csv_data(classwise_df, output_dir, "classwise_metrics.csv",
                  title=f"Aariz CVM Classwise - {args.model_name}")
    save_csv_data(cm_df, output_dir, "confusion_matrix.csv",
                  title=f"Aariz CVM Confusion Matrix - {args.model_name}")

    print(f"\n  Aariz CVM Classification Results:")
    print(f"    Accuracy:     {summary['accuracy']:.3f}")
    print(f"    F1 (weighted): {summary['f1_weighted']:.3f}")
    print(f"    F1 (macro):    {summary['f1_macro']:.3f}")
    print(f"    Samples:      {summary['total_samples']}")
    if unmapped_count > 0:
        print(f"    ⚠️  Unmapped predictions: {unmapped_count}")
