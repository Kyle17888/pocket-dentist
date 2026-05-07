"""
COde Classification Evaluation — 6-class single-label metrics.

Reads unified predictions and computes:
  - Per-class Precision, Recall, F1
  - Overall Accuracy, Weighted Precision, Weighted Recall, Weighted F1
  - Confusion matrix (6×6)

Corresponds to COde paper Table 3 metrics.
"""

import json
import os
import re
from collections import defaultdict
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


# The 6 benchmark categories from the COde paper (exact names)
CODE_CLASSES = [
    "Dental Caries",
    "Gingivitis",
    "Class III Malocclusion",
    "Pulpitis",
    "Tooth Loss",
    "Tooth Structure Loss",
]

# Fuzzy matching aliases for each class
_ALIASES = {
    "Dental Caries": ["dental caries", "caries", "tooth decay", "cavity", "cavities"],
    "Gingivitis": ["gingivitis", "gum disease", "gum inflammation", "periodontal", "periodontitis"],
    "Class III Malocclusion": [
        "class iii malocclusion", "malocclusion", "misalignment", "crowding",
        "overbite", "underbite", "crossbite", "open bite",
        "class i malocclusion", "class ii malocclusion",
    ],
    "Pulpitis": ["pulpitis", "pulp infection", "pulp inflammation", "tooth infection", "periapical"],
    "Tooth Loss": ["tooth loss", "missing tooth", "missing teeth", "edentulism", "tooth extraction"],
    "Tooth Structure Loss": [
        "tooth structure loss", "tooth wear", "erosion", "attrition",
        "abrasion", "tooth fracture", "enamel loss", "wedge-shaped",
        "fluorosis", "discoloration", "hypoplasia",
    ],
}

# Build reverse lookup
_ALIAS_MAP = {}
for cls_name, aliases in _ALIASES.items():
    _ALIAS_MAP[cls_name.lower()] = cls_name
    for alias in aliases:
        _ALIAS_MAP[alias.lower()] = cls_name


def extract_predicted_class(raw_prediction: str) -> str | None:
    """Extract the predicted class name from model output using fuzzy matching."""
    if not raw_prediction or not isinstance(raw_prediction, str):
        return None

    text = raw_prediction.strip()

    # Try direct match first (case-insensitive)
    text_lower = text.lower().strip().rstrip(".")
    if text_lower in _ALIAS_MAP:
        return _ALIAS_MAP[text_lower]

    # Try to find a class name anywhere in the text
    for cls_name in CODE_CLASSES:
        if cls_name.lower() in text_lower:
            return cls_name

    # Try aliases in text
    for alias, cls_name in _ALIAS_MAP.items():
        if alias in text_lower:
            return cls_name

    # Try JSON parsing (model might wrap answer in JSON)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for key in ["answer", "class", "label", "category", "disease", "anomaly"]:
                if key in parsed:
                    return extract_predicted_class(str(parsed[key]))
        elif isinstance(parsed, str):
            return extract_predicted_class(parsed)
    except (json.JSONDecodeError, TypeError):
        pass

    return None


def evaluate(predictions: list[dict], output_dir: str, args, model=None, yaml_cfg=None):
    """
    Evaluate COde classification predictions (6-class single-label).

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

    for p in tqdm(predictions, desc=f"[COde Classification] {args.model_name}", dynamic_ncols=True):
        if p.get("failed"):
            continue

        sample_id = p["id"]
        gt_text = p.get("ground_truth", "")
        pred_text = p.get("prediction", "")

        # Ground truth is a plain class name
        gt_class = gt_text.strip() if isinstance(gt_text, str) else None

        # Try to parse prediction
        pred_class = extract_predicted_class(pred_text)

        if gt_class not in CODE_CLASSES:
            continue

        if pred_class is None:
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
        print("⚠️  No valid COde classification predictions found.")
        return

    # Include UNMAPPED as a possible predicted class for metrics
    all_labels = CODE_CLASSES + (["UNMAPPED"] if unmapped_count > 0 else [])

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Weighted metrics (matching COde paper)
    w_precision = precision_score(y_true, y_pred, labels=CODE_CLASSES, average="weighted", zero_division=0)
    w_recall = recall_score(y_true, y_pred, labels=CODE_CLASSES, average="weighted", zero_division=0)
    w_f1 = f1_score(y_true, y_pred, labels=CODE_CLASSES, average="weighted", zero_division=0)

    # Per-class report
    report = classification_report(
        y_true, y_pred, labels=CODE_CLASSES,
        output_dict=True, zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)

    # Per-class metrics table
    classwise = []
    for cls in CODE_CLASSES:
        if cls in report:
            classwise.append({
                "Class": cls,
                "Precision": report[cls]["precision"],
                "Recall": report[cls]["recall"],
                "F1": report[cls]["f1-score"],
                "Support": report[cls]["support"],
            })
    classwise_df = pd.DataFrame(classwise)

    # Summary
    point = "0.000"
    summary = {
        "accuracy": float(Decimal(str(accuracy)).quantize(Decimal(point))),
        "precision_weighted": float(Decimal(str(w_precision)).quantize(Decimal(point))),
        "recall_weighted": float(Decimal(str(w_recall)).quantize(Decimal(point))),
        "f1_weighted": float(Decimal(str(w_f1)).quantize(Decimal(point))),
        "total_samples": len(y_true),
        "unmapped_predictions": unmapped_count,
    }

    # Save outputs
    save_json_data(per_sample, output_dir, "per_sample.json",
                   title=f"COde Classification Per-Sample - {args.model_name}")
    save_json_data(summary, output_dir, "metrics.json",
                   title=f"COde Classification Summary - {args.model_name}")
    save_csv_data(classwise_df, output_dir, "classwise_metrics.csv",
                  title=f"COde Classification Classwise - {args.model_name}")
    save_csv_data(cm_df, output_dir, "confusion_matrix.csv",
                  title=f"COde Classification Confusion Matrix - {args.model_name}")

    # Print summary
    print(f"\n  COde Classification Results:")
    print(f"    Accuracy:  {summary['accuracy']:.3f}")
    print(f"    Precision: {summary['precision_weighted']:.3f} (weighted)")
    print(f"    Recall:    {summary['recall_weighted']:.3f} (weighted)")
    print(f"    F1:        {summary['f1_weighted']:.3f} (weighted)")
    print(f"    Samples:   {summary['total_samples']}")
    if unmapped_count > 0:
        print(f"    ⚠️  Unmapped predictions: {unmapped_count}")
