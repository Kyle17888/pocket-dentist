"""
Classification Evaluation — 18-class multi-label metrics.

Reads unified predictions and computes:
  - Per-class Precision, Recall, F1
  - Macro/Micro aggregated metrics
  - Exact Match accuracy
  - Per-source breakdown (DS1/DS2/DS3)
"""

import json
import os
import re
from collections import defaultdict
from decimal import Decimal

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm

from src.utils.file_io import save_csv_data, save_json_data


# All 18 classification categories
CLS_LABELS = [f"C{i}" for i in range(1, 19)]
CLS_LABEL_SET = set(CLS_LABELS)


def extract_category_ids(raw_prediction) -> set:
    """
    Extract category IDs (C1-C18) from model prediction output.

    Handles multiple output formats:
      - List of dicts: [{"id": "C1", ...}, ...]
      - String containing category IDs
      - Nested structures
    """
    if isinstance(raw_prediction, str):
        try:
            raw_prediction = json.loads(raw_prediction)
        except (json.JSONDecodeError, TypeError):
            # Fallback: extract C1-C18 from raw string
            return {m for m in CLS_LABEL_SET if m in raw_prediction}

    ids = set()
    if isinstance(raw_prediction, list):
        for item in raw_prediction:
            if not item or isinstance(item, str):
                continue
            if isinstance(item, dict):
                if "id" in item:
                    ids.add(item["id"])
                elif "1" in item:
                    ids.add(item["1"])
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        if "id" in sub:
                            ids.add(sub["id"])
                        elif "1" in sub:
                            ids.add(sub["1"])
    elif isinstance(raw_prediction, dict):
        if "id" in raw_prediction:
            ids.add(raw_prediction["id"])

    return ids & CLS_LABEL_SET


def extract_gt_labels(gt_text: str) -> list:
    """Extract ground truth labels from the assistant message text."""
    try:
        gt = json.loads(gt_text) if isinstance(gt_text, str) else gt_text
    except (json.JSONDecodeError, TypeError):
        return []

    if isinstance(gt, list):
        labels = []
        for item in gt:
            if isinstance(item, dict) and "id" in item:
                labels.append(item["id"])
            elif isinstance(item, str):
                labels.append(item)
        return [l for l in labels if l in CLS_LABEL_SET]

    return []


def compute_sample_metrics(reference: set, prediction: set) -> dict:
    """Compute per-sample confusion matrix and metrics."""
    exact_match = 1 if reference == prediction else 0

    tp = len(reference & prediction)
    fn = len(reference - prediction)
    fp = len(prediction - reference)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Per-category results
    result = {c: None for c in CLS_LABELS}
    result.update({c: 0 for c in reference - prediction})   # FN
    result.update({c: 1 for c in reference & prediction})    # TP

    result.update({
        "TP": tp, "FN": fn, "FP": fp,
        "Exact_Match": exact_match,
        "P": precision, "R": recall, "F1": f1,
    })
    return result


def evaluate(predictions: list[dict], output_dir: str, args, model=None, yaml_cfg=None):
    """
    Evaluate classification predictions (18-class multi-label).

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

    for p in tqdm(predictions, desc=f"[Classification] {args.model_name}", dynamic_ncols=True):
        if p.get("failed"):
            continue

        sample_id = p["id"]

        # Parse ground truth
        gt_labels = extract_gt_labels(p.get("ground_truth", ""))
        gt_set = set(gt_labels)

        # Parse prediction
        pred_ids = extract_category_ids(p.get("prediction", ""))

        # Compute per-sample metrics
        metrics = compute_sample_metrics(gt_set, pred_ids)
        metrics["source"] = p.get("source", "")
        per_sample[sample_id] = metrics

        # Build binary vectors for sklearn
        gt_row = {"ID": sample_id}
        pred_row = {"ID": sample_id}
        for c in CLS_LABELS:
            gt_row[c] = 1 if c in gt_set else 0
            pred_row[c] = 1 if c in pred_ids else 0
        gt_rows.append(gt_row)
        pred_rows.append(pred_row)

    if not per_sample:
        print("⚠️  No valid predictions found.")
        return

    # Aggregate by source
    source_metrics = defaultdict(lambda: {"P": 0, "R": 0, "F1": 0, "EM": 0, "count": 0})
    for sample_id, m in per_sample.items():
        source = m.get("source", "overall")
        source_metrics[source]["P"] += m["P"]
        source_metrics[source]["R"] += m["R"]
        source_metrics[source]["F1"] += m["F1"]
        source_metrics[source]["EM"] += m["Exact_Match"]
        source_metrics[source]["count"] += 1
        # Also aggregate overall
        source_metrics["overall"]["P"] += m["P"]
        source_metrics["overall"]["R"] += m["R"]
        source_metrics["overall"]["F1"] += m["F1"]
        source_metrics["overall"]["EM"] += m["Exact_Match"]
        source_metrics["overall"]["count"] += 1

    point = '0.000'
    per_source = {}
    for source, sm in sorted(source_metrics.items()):
        n = sm["count"]
        if source == "overall":
            continue
        per_source[source] = {
            "count": n,
            "precision_weighted": float(Decimal(str(sm["P"] / n)).quantize(Decimal(point))) if n else 0,
            "recall_weighted": float(Decimal(str(sm["R"] / n)).quantize(Decimal(point))) if n else 0,
            "f1_weighted": float(Decimal(str(sm["F1"] / n)).quantize(Decimal(point))) if n else 0,
            "accuracy": float(Decimal(str(sm["EM"] / n)).quantize(Decimal(point))) if n else 0,
        }

    ov = source_metrics["overall"]
    ov_n = ov["count"]
    summary = {
        "accuracy": float(Decimal(str(ov["EM"] / ov_n)).quantize(Decimal(point))) if ov_n else 0,
        "f1_weighted": float(Decimal(str(ov["F1"] / ov_n)).quantize(Decimal(point))) if ov_n else 0,
        "precision_weighted": float(Decimal(str(ov["P"] / ov_n)).quantize(Decimal(point))) if ov_n else 0,
        "recall_weighted": float(Decimal(str(ov["R"] / ov_n)).quantize(Decimal(point))) if ov_n else 0,
        "total_samples": ov_n,
        "per_source": per_source,
    }

    # Per-class metrics via sklearn
    columns = ["ID"] + CLS_LABELS
    gt_df = pd.DataFrame(gt_rows, columns=columns).set_index("ID")
    pred_df = pd.DataFrame(pred_rows, columns=columns).set_index("ID")

    classwise_results = []
    for c in CLS_LABELS:
        y_true = gt_df[c]
        y_pred = pred_df[c]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        classwise_results.append({
            "Class": c,
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        })

    classwise_df = pd.DataFrame(classwise_results)
    macro_P = classwise_df["Precision"].mean()
    macro_R = classwise_df["Recall"].mean()
    macro_F1 = classwise_df["F1"].mean()
    micro_P = precision_score(gt_df.values.flatten(), pred_df.values.flatten(), zero_division=0)
    micro_R = recall_score(gt_df.values.flatten(), pred_df.values.flatten(), zero_division=0)
    micro_F1 = f1_score(gt_df.values.flatten(), pred_df.values.flatten(), zero_division=0)

    overall_metrics = pd.DataFrame({
        "Metric": ["Macro", "Micro"],
        "Precision": [macro_P, micro_P],
        "Recall": [macro_R, micro_R],
        "F1": [macro_F1, micro_F1],
    })

    # Save outputs
    save_json_data(per_sample, output_dir, "per_sample.json",
                   title=f"Classification Per-Sample - {args.model_name}")
    save_json_data(summary, output_dir, "metrics.json",
                   title=f"Classification Summary - {args.model_name}")
    save_csv_data(classwise_df, output_dir, "classwise_metrics.csv",
                  title=f"Classification Classwise - {args.model_name}")
    save_csv_data(overall_metrics, output_dir, "overall_metrics.csv",
                  title=f"Classification Overall - {args.model_name}")

    # Print summary
    print(f"\n  Classification Results:")
    print(f"    Overall: Acc={summary['accuracy']:.3f} F1={summary['f1_weighted']:.3f} P={summary['precision_weighted']:.3f} R={summary['recall_weighted']:.3f} (n={summary['total_samples']})")
    for source, m in per_source.items():
        print(f"    {source}: Acc={m['accuracy']:.3f} F1={m['f1_weighted']:.3f} (n={m['count']})")
    print(f"    Macro:  P={macro_P:.3f} R={macro_R:.3f} F1={macro_F1:.3f}")
    print(f"    Micro:  P={micro_P:.3f} R={micro_R:.3f} F1={micro_F1:.3f}")
