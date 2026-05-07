"""
DR Multi-label Classification Evaluation — 4-class dental finding detection.

Reads unified predictions and computes:
  - Exact Match accuracy (pred label set == GT label set)
  - Per-class Precision, Recall, F1
  - Macro/Weighted F1
  - Confusion stats

Classes: Cavity, Fillings, Impacted Tooth, Implant
"""

import json
import re
from collections import defaultdict
from decimal import Decimal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from src.utils.file_io import save_csv_data, save_json_data


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

DR_CLASSES = sorted(["Cavity", "Fillings", "Impacted Tooth", "Implant"])

# Fuzzy matching aliases (lowercase)
_ALIASES = {
    "Cavity":         ["cavity", "cavities", "caries", "decay", "decayed"],
    "Fillings":       ["fillings", "filling", "restoration", "restorations", "filled",
                       "filled tooth", "filled teeth"],
    "Impacted Tooth": ["impacted tooth", "impacted", "impaction", "impacted teeth"],
    "Implant":        ["implant", "implants", "dental implant", "dental implants",
                       "tooth implant", "tooth implants"],
}

_ALIAS_MAP = {}
for cls_name, aliases in _ALIASES.items():
    _ALIAS_MAP[cls_name.lower()] = cls_name
    for alias in aliases:
        _ALIAS_MAP[alias.lower()] = cls_name


# ──────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────

def extract_predicted_labels(raw_prediction: str) -> set:
    """
    Extract predicted DR labels from model output.

    Handles:
      - Comma-separated list: "Fillings, Implant"
      - Newline-separated list
      - Bullet-pointed list
      - "None" / empty
      - Free-form text containing label keywords
    """
    if not raw_prediction or not isinstance(raw_prediction, str):
        return set()

    text = raw_prediction.strip()

    # Handle "None" / "No findings" / empty
    if text.lower() in ("none", "no findings", "no findings detected", "n/a", ""):
        return set()

    found = set()

    # Strategy 1: Split by comma, newline, bullet, semicolon and check each chunk
    chunks = re.split(r'[,\n;•\-]+', text)
    for chunk in chunks:
        chunk_clean = chunk.strip().rstrip(".").lower()
        if chunk_clean in _ALIAS_MAP:
            found.add(_ALIAS_MAP[chunk_clean])

    # Strategy 2: Always also search for keywords in the full text
    # (catches partial matches that Strategy 1 misses, e.g. "Filled Tooth")
    text_lower = text.lower()
    for alias, cls_name in _ALIAS_MAP.items():
        if alias in text_lower:
            found.add(cls_name)

    return found


def extract_gt_labels(gt_text: str) -> set:
    """Extract ground truth labels from assistant message text."""
    if not gt_text or not isinstance(gt_text, str):
        return set()

    text = gt_text.strip()
    if text.lower() in ("none", ""):
        return set()

    labels = set()
    for part in text.split(","):
        part = part.strip()
        if part in DR_CLASSES:
            labels.add(part)
        elif part.lower() in _ALIAS_MAP:
            labels.add(_ALIAS_MAP[part.lower()])

    return labels


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate(predictions: list[dict], output_dir: str, args, model=None, yaml_cfg=None):
    """
    Evaluate DR multi-label classification predictions.

    Args:
        predictions: List of {id, task, source, ground_truth, prediction, failed}
        output_dir:  Directory to save evaluation results
        args:        Parsed argparse.Namespace
        model:       Not used (offline evaluation)
        yaml_cfg:    Not used
    """
    per_sample = {}
    gt_rows = []
    pred_rows = []
    unmapped_count = 0

    for p in tqdm(predictions, desc=f"[DR Cls] {args.model_name}", dynamic_ncols=True):
        if p.get("failed"):
            continue

        sample_id = p["id"]
        gt_text = p.get("ground_truth", "")
        pred_text = p.get("prediction", "")

        gt_set = extract_gt_labels(gt_text)
        pred_set = extract_predicted_labels(pred_text)

        # Skip samples with no valid GT
        if not gt_set:
            continue

        if not pred_set:
            unmapped_count += 1

        exact_match = 1 if gt_set == pred_set else 0

        # Per-sample metrics
        tp = len(gt_set & pred_set)
        fn = len(gt_set - pred_set)
        fp = len(pred_set - gt_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_sample[sample_id] = {
            "gt": sorted(gt_set),
            "pred": sorted(pred_set),
            "exact_match": exact_match,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "raw_prediction": pred_text[:300],
        }

        # Binary vectors for sklearn
        gt_row = {"ID": sample_id}
        pred_row = {"ID": sample_id}
        for c in DR_CLASSES:
            gt_row[c] = 1 if c in gt_set else 0
            pred_row[c] = 1 if c in pred_set else 0
        gt_rows.append(gt_row)
        pred_rows.append(pred_row)

    if not per_sample:
        print("⚠️  No valid DR classification predictions found.")
        return

    # Overall metrics
    n = len(per_sample)
    exact_matches = sum(v["exact_match"] for v in per_sample.values())
    accuracy = exact_matches / n
    avg_f1 = sum(v["f1"] for v in per_sample.values()) / n

    point = "0.000"
    summary = {
        "accuracy": float(Decimal(str(accuracy)).quantize(Decimal(point))),
        "f1_weighted": float(Decimal(str(avg_f1)).quantize(Decimal(point))),
        "total_samples": n,
        "unmapped_predictions": unmapped_count,
    }

    # Per-class metrics via sklearn
    columns = ["ID"] + DR_CLASSES
    gt_df = pd.DataFrame(gt_rows, columns=columns).set_index("ID")
    pred_df = pd.DataFrame(pred_rows, columns=columns).set_index("ID")

    classwise = []
    for c in DR_CLASSES:
        y_true = gt_df[c]
        y_pred = pred_df[c]
        if y_true.sum() == 0 and y_pred.sum() == 0:
            classwise.append({"Class": c, "TP": 0, "FP": 0, "FN": 0, "TN": len(y_true),
                              "Precision": 0.0, "Recall": 0.0, "F1": 0.0})
            continue
        tn, fp_c, fn_c, tp_c = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        classwise.append({
            "Class": c,
            "TP": int(tp_c), "FP": int(fp_c), "FN": int(fn_c), "TN": int(tn),
            "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        })

    classwise_df = pd.DataFrame(classwise)
    macro_f1 = classwise_df["F1"].mean()
    summary["f1_macro"] = float(Decimal(str(macro_f1)).quantize(Decimal(point)))

    # Save outputs
    save_json_data(per_sample, output_dir, "per_sample.json",
                   title=f"DR Classification Per-Sample - {args.model_name}")
    save_json_data(summary, output_dir, "metrics.json",
                   title=f"DR Classification Summary - {args.model_name}")
    save_csv_data(classwise_df, output_dir, "classwise_metrics.csv",
                  title=f"DR Classification Classwise - {args.model_name}")

    # Print summary
    print(f"\n  DR Classification Results:")
    print(f"    Accuracy (EM): {summary['accuracy']:.3f}")
    print(f"    F1 (weighted): {summary['f1_weighted']:.3f}")
    print(f"    F1 (macro):    {summary['f1_macro']:.3f}")
    print(f"    Samples:       {summary['total_samples']}")
    if unmapped_count > 0:
        print(f"    ⚠️  Unmapped: {unmapped_count}")
