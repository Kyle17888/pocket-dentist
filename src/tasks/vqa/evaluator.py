"""
VQA Evaluation — Accuracy metrics for multiple-choice and true/false questions.

Reads unified predictions and computes:
  - Per-source (DS1/DS2/DS3) accuracy breakdown
  - Overall MetaDent accuracy
  - Multiple-choice accuracy and Judge accuracy separately
"""

import json
import os
from collections import defaultdict
from decimal import Decimal

from tqdm import tqdm

from src.utils.file_io import save_json_data


def evaluate(predictions: list[dict], output_dir: str, args, model=None, yaml_cfg=None):
    """
    Evaluate VQA predictions.

    Args:
        predictions: List of {id, task, source, ground_truth, prediction, failed}
        output_dir:  Directory to save evaluation results
        args:        Parsed argparse.Namespace
        model:       Not used (offline evaluation)
        yaml_cfg:    Not used
    """
    # Parse predictions: ground_truth and prediction are JSON strings
    per_sample = defaultdict(dict)

    for p in tqdm(predictions, desc=f"[VQA] {args.model_name}", dynamic_ncols=True):
        sample_id = p["id"]

        # Parse ground truth (the full question list from the JSONL assistant message)
        try:
            gt = json.loads(p["ground_truth"]) if isinstance(p["ground_truth"], str) else p["ground_truth"]
        except (json.JSONDecodeError, TypeError):
            continue

        # Parse model prediction
        try:
            pred = json.loads(p["prediction"]) if isinstance(p["prediction"], str) else p["prediction"]
        except (json.JSONDecodeError, TypeError):
            continue

        if p.get("failed"):
            continue

        # The ground truth for VQA is the answer from the assistant message
        # The prediction is the model's output (should have answer + reason)
        # For VQA, GT is a JSON with {"answer": "B", "reason": "..."}
        # and prediction is also {"answer": "X", "reason": "..."}

        gt_answer = gt.get("answer") if isinstance(gt, dict) else None
        pred_answer = pred.get("answer") if isinstance(pred, dict) else None

        if gt_answer is None or pred_answer is None:
            continue

        # Normalize answers: some SFT models output numeric (1,2,3,4) instead of letter (A,B,C,D)
        _NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        gt_norm = str(gt_answer).strip().upper()
        pred_norm = str(pred_answer).strip().upper()
        gt_norm = _NUM_TO_LETTER.get(gt_norm, gt_norm)
        pred_norm = _NUM_TO_LETTER.get(pred_norm, pred_norm)
        correct = 1 if gt_norm == pred_norm else 0

        per_sample[sample_id] = {
            "source": p.get("source", ""),
            "gt_answer": gt_answer,
            "pred_answer": pred_answer,
            "correct": correct,
        }

    # Aggregate by source
    source_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
    overall = {"correct": 0, "total": 0}

    for sample_id, item in per_sample.items():
        source = item.get("source", "overall")
        source_metrics[source]["correct"] += item["correct"]
        source_metrics[source]["total"] += 1
        overall["correct"] += item["correct"]
        overall["total"] += 1

    # Build results — flat top-level with per_source breakdown
    point = '0.000'
    per_source = {}
    for source, m in sorted(source_metrics.items()):
        acc = m["correct"] / m["total"] if m["total"] > 0 else 0
        per_source[source] = {
            "correct": m["correct"],
            "total": m["total"],
            "accuracy": float(Decimal(str(acc)).quantize(Decimal(point), rounding="ROUND_HALF_UP")),
        }

    overall_acc = overall["correct"] / overall["total"] if overall["total"] > 0 else 0
    results = {
        "accuracy": float(Decimal(str(overall_acc)).quantize(Decimal(point), rounding="ROUND_HALF_UP")),
        "correct": overall["correct"],
        "total_samples": overall["total"],
        "per_source": per_source,
    }

    # Save results
    save_json_data(results, output_dir, "metrics.json",
                   title=f"VQA Evaluation - {args.model_name}")

    # Save per-sample results
    save_json_data(dict(per_sample), output_dir, "per_sample.json",
                   title=f"VQA Per-Sample - {args.model_name}")

    # Print summary
    print(f"\n  VQA Results:")
    print(f"    Overall: {results['accuracy']:.3f} ({results['correct']}/{results['total_samples']})")
    for source, m in per_source.items():
        print(f"    {source}: {m['accuracy']:.3f} ({m['correct']}/{m['total']})")
