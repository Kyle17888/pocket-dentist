"""
Caries Detection Evaluation — Binary caries presence (Yes / No).

Reads unified predictions and computes:
  - Accuracy
  - Precision, Recall, F1 (for the positive class)
  - Confusion matrix stats
"""

import re
from decimal import Decimal

from tqdm import tqdm

from src.utils.file_io import save_json_data


# ──────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────

_YES_PATTERNS = {"yes", "y", "true", "positive", "caries detected", "caries present"}
_NO_PATTERNS = {"no", "n", "false", "negative", "no caries", "none", "no caries detected"}


def _normalise(text: str) -> str:
    """Strip markdown, punctuation, lowercase."""
    text = text.strip().lower()
    text = re.sub(r"[*_`#\-•]", "", text)
    text = text.strip().rstrip(".")
    return text


def extract_binary_prediction(raw: str) -> str | None:
    """Extract Yes/No from model output. Returns 'Yes', 'No', or None."""
    if not raw or not isinstance(raw, str):
        return None

    norm = _normalise(raw)

    # Exact match first
    if norm in _YES_PATTERNS:
        return "Yes"
    if norm in _NO_PATTERNS:
        return "No"

    # First-word match (handles "Yes, there are caries..." type outputs)
    first_word = norm.split()[0] if norm else ""
    if first_word in ("yes", "yes,", "yes."):
        return "Yes"
    if first_word in ("no", "no,", "no."):
        return "No"

    # Keyword search fallback
    if "yes" in norm:
        return "Yes"
    if "no" in norm:
        return "No"

    return None


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate(predictions: list[dict], output_dir: str, args, model=None, yaml_cfg=None):
    """
    Evaluate binary caries detection predictions.

    Args:
        predictions: List of {id, task, source, ground_truth, prediction, failed}
        output_dir:  Directory to save evaluation results
        args:        Parsed argparse.Namespace
        model:       Not used (offline evaluation)
        yaml_cfg:    Not used
    """
    per_sample = {}
    tp = fp = fn = tn = 0
    unmapped = 0

    for p in tqdm(predictions, desc=f"[Caries Detect] {args.model_name}", dynamic_ncols=True):
        if p.get("failed"):
            continue

        sample_id = p["id"]
        gt_text = _normalise(p.get("ground_truth", ""))
        pred_text = p.get("prediction", "")

        gt_label = "Yes" if gt_text in _YES_PATTERNS else "No"
        pred_label = extract_binary_prediction(pred_text)

        if pred_label is None:
            unmapped += 1
            pred_label = "No"  # default to No for unmapped

        correct = int(gt_label == pred_label)

        # Confusion matrix (positive = Yes = has caries)
        if gt_label == "Yes" and pred_label == "Yes":
            tp += 1
        elif gt_label == "No" and pred_label == "Yes":
            fp += 1
        elif gt_label == "Yes" and pred_label == "No":
            fn += 1
        else:
            tn += 1

        per_sample[sample_id] = {
            "gt": gt_label,
            "pred": pred_label,
            "correct": correct,
            "raw_prediction": pred_text[:300],
        }

    if not per_sample:
        print("⚠️  No valid caries detection predictions found.")
        return

    n = len(per_sample)
    accuracy = (tp + tn) / n if n > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    point = "0.000"
    summary = {
        "accuracy": float(Decimal(str(accuracy)).quantize(Decimal(point))),
        "f1_weighted": float(Decimal(str(f1)).quantize(Decimal(point))),
        "precision": float(Decimal(str(precision)).quantize(Decimal(point))),
        "recall": float(Decimal(str(recall)).quantize(Decimal(point))),
        "total_samples": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "unmapped_predictions": unmapped,
    }

    # Save outputs
    save_json_data(per_sample, output_dir, "per_sample.json",
                   title=f"Caries Detection Per-Sample - {args.model_name}")
    save_json_data(summary, output_dir, "metrics.json",
                   title=f"Caries Detection Summary - {args.model_name}")

    # Print summary
    print(f"\n  Caries Detection Results:")
    print(f"    Accuracy:  {summary['accuracy']:.3f}")
    print(f"    F1:        {summary['f1_weighted']:.3f}")
    print(f"    Precision: {summary['precision']:.3f}")
    print(f"    Recall:    {summary['recall']:.3f}")
    print(f"    TP={tp} FP={fp} FN={fn} TN={tn} (n={n})")
    if unmapped > 0:
        print(f"    ⚠️  Unmapped: {unmapped}")
