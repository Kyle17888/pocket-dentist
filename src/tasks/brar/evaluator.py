#!/usr/bin/env python3
from __future__ import annotations
"""
BRAR Evaluation — 3-class single-label classification metrics.

Reads predictions from the unified predictor format and computes:
  - Accuracy
  - F1 Macro / Weighted
  - Precision / Recall (Macro / Weighted)
  - Confusion Matrix (3×3)
  - Per-class metrics (P/R/F1 per grade)
"""

import json
import os
import re

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ──────────────────────────────────────────────────────────────
# Grade parsing (5-level fallback)
# ──────────────────────────────────────────────────────────────

def parse_grade(text: str) -> int | None:
    """
    Extract grade (1, 2, or 3) from model output using 5-level fallback:
      1. Pure digit: "2"
      2. JSON object: {"grade": 2, "reason": "..."}
      3. Markdown code block extraction
      4. Regex: grade = X, grade: X, etc.
      5. First digit 1-3 in text
    Returns None if parsing fails completely.
    """
    if not text:
        return None

    text = text.strip()

    # 1. Pure digit
    if text in ("1", "2", "3"):
        return int(text)

    # 2. JSON parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "grade" in obj:
            g = int(obj["grade"])
            if g in (1, 2, 3):
                return g
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 3. Markdown code block
    md_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if md_match:
        try:
            obj = json.loads(md_match.group(1))
            if isinstance(obj, dict) and "grade" in obj:
                g = int(obj["grade"])
                if g in (1, 2, 3):
                    return g
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # 4. Regex fallback
    regex_match = re.search(r'"?grade"?\s*[:=]\s*(\d)', text, re.IGNORECASE)
    if regex_match:
        g = int(regex_match.group(1))
        if g in (1, 2, 3):
            return g

    # 5. First digit 1-3
    digit_match = re.search(r'[123]', text)
    if digit_match:
        return int(digit_match.group(0))

    return None


# ──────────────────────────────────────────────────────────────
# Metrics computation
# ──────────────────────────────────────────────────────────────

def compute_all_metrics(y_true: list, y_pred: list) -> dict:
    """Compute all classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


# ──────────────────────────────────────────────────────────────
# Unified evaluate interface
# ──────────────────────────────────────────────────────────────

def evaluate(predictions: list[dict], output_dir: str, args, model=None, yaml_cfg=None):
    """
    Evaluate BRAR predictions (3-class classification).

    Args:
        predictions: List of {id, task, source, ground_truth, prediction, failed}
        output_dir:  Directory to save evaluation results
        args:        Parsed argparse.Namespace
        model:       Not used (offline evaluation)
        yaml_cfg:    Not used
    """
    # Parse ground truth and predictions
    valid = []
    failed = []

    for p in predictions:
        gt_grade = parse_grade(p.get("ground_truth", ""))
        pred_grade = parse_grade(p.get("prediction", ""))

        if gt_grade is not None and pred_grade is not None:
            valid.append({"id": p["id"], "ground_truth": gt_grade, "prediction": pred_grade})
        else:
            failed.append(p)

    print(f"  Total:       {len(predictions)}")
    print(f"  Valid:       {len(valid)}")
    print(f"  Failed:      {len(failed)}")

    if not valid:
        print("⚠️  No valid predictions. Cannot compute metrics.")
        empty_metrics = {
            "total_samples": len(predictions),
            "valid_predictions": 0,
            "failed_predictions": len(failed),
            "parse_failure_rate": 1.0,
        }
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(empty_metrics, f, indent=2)
        return

    y_true = [p["ground_truth"] for p in valid]
    y_pred = [p["prediction"] for p in valid]

    # Compute metrics
    metrics = compute_all_metrics(y_true, y_pred)
    metrics.update({
        "total_samples": len(predictions),
        "valid_predictions": len(valid),
        "failed_predictions": len(failed),
        "parse_failure_rate": round(len(failed) / len(predictions), 4) if predictions else 0,
    })

    # Confusion Matrix (3×3)
    labels = [1, 2, 3]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])

    # Per-class metrics
    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=["Grade_1", "Grade_2", "Grade_3"],
                                   output_dict=True, zero_division=0)
    per_class_df = pd.DataFrame(report).transpose()

    # Save outputs
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Saved: {metrics_path}")

    cm_path = os.path.join(output_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_path)
    print(f"  ✅ Saved: {cm_path}")

    pc_path = os.path.join(output_dir, "per_class_metrics.csv")
    per_class_df.to_csv(pc_path)
    print(f"  ✅ Saved: {pc_path}")

    # Print summary
    print(f"\n  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  F1 Macro:          {metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted:       {metrics['f1_weighted']:.4f}")
    print(f"  Precision Macro:   {metrics['precision_macro']:.4f}")
    print(f"  Recall Macro:      {metrics['recall_macro']:.4f}")
    print(f"  Parse Failures:    {metrics['failed_predictions']}/{metrics['total_samples']} ({metrics['parse_failure_rate']:.1%})")
    print(f"\n  Confusion Matrix:")
    print(f"  {cm_df.to_string()}")
