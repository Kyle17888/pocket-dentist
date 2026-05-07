"""
DenPAR Site Classification Evaluation — 3-class (Anterior/Left/Right).
"""

import json
from decimal import Decimal

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

from src.utils.file_io import save_csv_data, save_json_data


SITE_CLASSES = ["Anterior", "Left", "Right"]

_ALIASES = {
    "Anterior": ["anterior", "front", "incisors", "canines"],
    "Left": ["left", "left posterior", "left side"],
    "Right": ["right", "right posterior", "right side"],
}

_ALIAS_MAP = {}
for cls_name, aliases in _ALIASES.items():
    _ALIAS_MAP[cls_name.lower()] = cls_name
    for alias in aliases:
        _ALIAS_MAP[alias.lower()] = cls_name


def extract_site(raw_prediction: str) -> str | None:
    if not raw_prediction or not isinstance(raw_prediction, str):
        return None
    text = raw_prediction.strip().rstrip(".").lower()
    if text in _ALIAS_MAP:
        return _ALIAS_MAP[text]
    # Check in order: "anterior" before "left"/"right" to avoid false matches
    for cls in SITE_CLASSES:
        if cls.lower() in text:
            return cls
    for alias, cls in _ALIAS_MAP.items():
        if alias in text:
            return cls
    return None


def evaluate(predictions: list[dict], output_dir: str, args, model=None, yaml_cfg=None):
    y_true, y_pred = [], []
    per_sample = {}
    unmapped = 0

    for p in tqdm(predictions, desc=f"[DenPAR Site] {args.model_name}", dynamic_ncols=True):
        if p.get("failed"):
            continue
        gt = p.get("ground_truth", "").strip()
        pred_text = p.get("prediction", "")
        pred = extract_site(pred_text)

        if gt not in SITE_CLASSES:
            continue
        if pred is None:
            unmapped += 1
            pred = "UNMAPPED"

        y_true.append(gt)
        y_pred.append(pred)
        per_sample[p["id"]] = {"gt": gt, "pred": pred, "correct": int(gt == pred),
                                "raw_prediction": pred_text[:200]}

    if not y_true:
        print("⚠️  No valid DenPAR site predictions found.")
        return

    all_labels = SITE_CLASSES + (["UNMAPPED"] if unmapped > 0 else [])
    acc = accuracy_score(y_true, y_pred)
    w_f1 = f1_score(y_true, y_pred, labels=SITE_CLASSES, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)

    point = "0.000"
    summary = {
        "accuracy": float(Decimal(str(acc)).quantize(Decimal(point))),
        "f1_weighted": float(Decimal(str(w_f1)).quantize(Decimal(point))),
        "total_samples": len(y_true),
        "unmapped_predictions": unmapped,
    }

    save_json_data(per_sample, output_dir, "per_sample.json",
                   title=f"DenPAR Site Per-Sample - {args.model_name}")
    save_json_data(summary, output_dir, "metrics.json",
                   title=f"DenPAR Site Summary - {args.model_name}")
    save_csv_data(cm_df, output_dir, "confusion_matrix.csv",
                  title=f"DenPAR Site Confusion Matrix - {args.model_name}")

    print(f"\n  DenPAR Site Classification Results:")
    print(f"    Accuracy: {summary['accuracy']:.3f}")
    print(f"    F1:       {summary['f1_weighted']:.3f}")
    print(f"    Samples:  {summary['total_samples']}")
