"""
DenPAR Tooth Counting Evaluation.

Evaluates tooth counting predictions using:
  - Exact Match Accuracy (primary metric)
  - MAE (Mean Absolute Error)
  - Distribution of errors (±0, ±1, ±2, ±3+)
"""

import json
import os
import re
from decimal import Decimal

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.file_io import save_csv_data, save_json_data


def extract_count(raw_prediction: str) -> int | None:
    """Extract a number from model output."""
    if not raw_prediction or not isinstance(raw_prediction, str):
        return None

    text = raw_prediction.strip()

    # Direct integer
    try:
        return int(text)
    except ValueError:
        pass

    # Find first number in text
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))

    # Handle "Not assessable"
    if "not assessable" in text.lower():
        return None

    return None


def evaluate(predictions: list[dict], output_dir: str, args, model=None, yaml_cfg=None):
    """
    Evaluate DenPAR tooth counting predictions.

    Metrics:
      - Exact Match Accuracy: predicted count == ground truth count
      - MAE: mean absolute error of count
    """
    gt_counts = []
    pred_counts = []
    per_sample = {}
    unmapped = 0

    for p in tqdm(predictions, desc=f"[DenPAR Count] {args.model_name}", dynamic_ncols=True):
        if p.get("failed"):
            continue

        sample_id = p["id"]
        gt_text = p.get("ground_truth", "")
        pred_text = p.get("prediction", "")

        gt_count = extract_count(gt_text)
        pred_count = extract_count(pred_text)

        if gt_count is None:
            continue

        if pred_count is None:
            unmapped += 1
            pred_count = -1  # Sentinel for unmapped

        error = abs(pred_count - gt_count) if pred_count >= 0 else gt_count
        correct = 1 if pred_count == gt_count else 0

        gt_counts.append(gt_count)
        pred_counts.append(pred_count)

        per_sample[sample_id] = {
            "gt": gt_count,
            "pred": pred_count if pred_count >= 0 else "UNMAPPED",
            "error": error,
            "correct": correct,
            "raw_prediction": pred_text[:200],
        }

    if not gt_counts:
        print("⚠️  No valid DenPAR counting predictions found.")
        return

    gt_arr = np.array(gt_counts)
    pred_arr = np.array(pred_counts)

    # Exact match accuracy
    exact_match = np.mean(gt_arr == pred_arr)

    # MAE (only for mapped predictions)
    valid_mask = pred_arr >= 0
    if valid_mask.any():
        mae = np.mean(np.abs(gt_arr[valid_mask] - pred_arr[valid_mask]))
    else:
        mae = float("inf")

    # Error distribution
    errors = np.abs(gt_arr[valid_mask] - pred_arr[valid_mask]) if valid_mask.any() else np.array([])
    error_dist = {
        "exact (±0)": int(np.sum(errors == 0)),
        "off_by_1 (±1)": int(np.sum(errors == 1)),
        "off_by_2 (±2)": int(np.sum(errors == 2)),
        "off_by_3+ (±3+)": int(np.sum(errors >= 3)),
    }

    point = "0.000"
    summary = {
        "accuracy": float(Decimal(str(exact_match)).quantize(Decimal(point))),
        "mae": float(Decimal(str(mae)).quantize(Decimal(point))),
        "total_samples": len(gt_counts),
        "unmapped_predictions": unmapped,
        "error_distribution": error_dist,
    }

    save_json_data(per_sample, output_dir, "per_sample.json",
                   title=f"DenPAR Counting Per-Sample - {args.model_name}")
    save_json_data(summary, output_dir, "metrics.json",
                   title=f"DenPAR Counting Summary - {args.model_name}")

    print(f"\n  DenPAR Tooth Counting Results:")
    print(f"    Accuracy:    {summary['accuracy']:.3f}")
    print(f"    MAE:         {summary['mae']:.3f}")
    print(f"    Samples:     {summary['total_samples']}")
    print(f"    Error dist:  {error_dist}")
    if unmapped > 0:
        print(f"    ⚠️  Unmapped: {unmapped}")
