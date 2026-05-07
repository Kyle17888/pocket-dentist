"""
Captioning Evaluation — BERTScore + LLM-as-judge confusion matrix.

This is a requires_llm_judge evaluator. It has two modes:
  1. BERTScore (automatic) — compares model captions with GT descriptions
  2. LLM-as-judge (requires model) — extracts abnormalities and scores them

Both modes are triggered when --enable_llm_judge is set.
"""

import json
import os
from argparse import Namespace
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal

import numpy as np
from bert_score import score
from tqdm import tqdm
from transformers import logging

from src.models.base_model import BaseModel
from src.utils import prompt
from src.utils.file_io import save_json_data

logging.set_verbosity_error()


# ──────────────────────────────────────────────────────────────
# BERTScore computation (preserved from original)
# ──────────────────────────────────────────────────────────────

def _patch_bert_score_tokenizer():
    """Patch for bert_score + transformers 5.x compatibility."""
    import bert_score.utils as bsu

    if getattr(bsu, "_sd4h_patched", False):
        return

    _orig_sent_encode = bsu.sent_encode

    def _patched_sent_encode(tokenizer, sent):
        sent = sent.strip()
        if sent == "":
            try:
                return tokenizer.build_inputs_with_special_tokens([])
            except AttributeError:
                cls_id = getattr(tokenizer, "cls_token_id",
                                 getattr(tokenizer, "bos_token_id", 0))
                sep_id = getattr(tokenizer, "sep_token_id",
                                 getattr(tokenizer, "eos_token_id", 2))
                return [cls_id, sep_id]
        return tokenizer.encode(
            sent, add_special_tokens=True,
            max_length=tokenizer.model_max_length, truncation=True,
        )

    bsu.sent_encode = _patched_sent_encode
    bsu._sd4h_patched = True


def statistical_BERTScore(cands, refs, chunk=False, chunk_size=512, max_retries=3):
    _patch_bert_score_tokenizer()
    all_P, all_R, all_F1 = [], [], []

    if not chunk:
        device = None
        for attempt in range(1, max_retries + 1):
            try:
                tqdm.write(f"calculating scores... (attempt {attempt}/{max_retries}, device={device or 'auto'})")
                P, R, F1 = score(cands, refs, lang="en", verbose=True,
                                 rescale_with_baseline=True, device=device)
                return P, R, F1
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "out of memory" in str(e):
                    tqdm.write(f"[Attempt {attempt}] CUDA OOM – falling back to CPU")
                    import torch
                    torch.cuda.empty_cache()
                    device = "cpu"
                    continue
                else:
                    raise
            except Exception as e:
                tqdm.write(f"[Attempt {attempt}] Error: {e}")
                if attempt == max_retries:
                    raise
                continue
        tqdm.write("All GPU attempts failed – running BERTScore on CPU")
        P, R, F1 = score(cands, refs, lang="en", verbose=True,
                         rescale_with_baseline=True, device="cpu")
        return P, R, F1

    # Chunked mode
    device = None
    for i in tqdm(range(0, len(cands), chunk_size), desc="Scoring chunks", dynamic_ncols=True):
        batch_cands = cands[i:i+chunk_size]
        batch_refs = refs[i:i+chunk_size]
        for attempt in range(1, max_retries + 1):
            try:
                P, R, F1 = score(batch_cands, batch_refs, lang="en", verbose=False,
                                 rescale_with_baseline=True, device=device)
                all_P.append(P)
                all_R.append(R)
                all_F1.append(F1)
                break
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "out of memory" in str(e):
                    tqdm.write(f"[Chunk {i}, Attempt {attempt}] CUDA OOM – falling back to CPU")
                    import torch
                    torch.cuda.empty_cache()
                    device = "cpu"
                    continue
                else:
                    raise

    from torch import cat
    return cat(all_P), cat(all_R), cat(all_F1)


# ──────────────────────────────────────────────────────────────
# LLM-as-judge: confusion matrix scoring
# ──────────────────────────────────────────────────────────────

def _judge_single(sample_id: str, model: BaseModel, gt_text: str,
                  pred_text: str, lfss_meta_type: str) -> dict:
    """Run LLM-as-judge on a single captioning sample."""
    try:
        # Parse ground truth
        gt = json.loads(gt_text) if isinstance(gt_text, str) else gt_text

        # The GT from JSONL has the captioning description
        # We need to extract abnormalities from the prediction
        refine = model.generate_from_text(
            prompt=prompt.captioning_extraction_intraoral_condition.substitute(
                case=model.j2t(pred_text if isinstance(pred_text, str) else pred_text)
            ),
            output_type=list,
        )

        # Generate confusion matrix
        # For GT, we need the items from the label data
        gt_items = gt.get("items", [gt]) if isinstance(gt, dict) else gt
        res = model.generate_from_text(
            prompt=prompt.captioning_score_intraoral_condition.substitute(
                reference=model.j2t(gt_items),
                prediction=model.j2t(refine),
            )
        )

        return {
            "id": sample_id,
            "refine": refine,
            "confusion": res,
            "failed": False,
        }
    except Exception as e:
        return {
            "id": sample_id,
            "refine": None,
            "confusion": {"failed": str(e)},
            "failed": True,
        }


def compute_confusion_metrics(confusion_dict: dict) -> dict:
    """Compute P/R/F1 from a confusion matrix dict."""
    TP = confusion_dict.get("TP", 0)
    FP = confusion_dict.get("FP", 0)
    FN = confusion_dict.get("FN", 0)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"P": precision, "R": recall, "F1": f1}


# ──────────────────────────────────────────────────────────────
# Unified evaluate interface
# ──────────────────────────────────────────────────────────────

def evaluate(predictions: list[dict], output_dir: str, args, model=None, yaml_cfg=None):
    """
    Evaluate captioning predictions.

    Args:
        predictions: List of {id, task, source, ground_truth, prediction, failed}
        output_dir:  Directory to save evaluation results
        args:        Parsed argparse.Namespace
        model:       LLM model for confusion matrix scoring (required)
        yaml_cfg:    Global config
    """
    bert_dir = os.path.join(output_dir, "BertScore")
    confusion_dir = os.path.join(output_dir, "confusion_matrix")
    os.makedirs(bert_dir, exist_ok=True)
    os.makedirs(confusion_dir, exist_ok=True)

    # Filter valid predictions
    valid = [p for p in predictions if not p.get("failed")]

    if not valid:
        tqdm.write(f"[WARNING] No valid captioning predictions for {args.model_name}. Skipping.")
        save_json_data({}, bert_dir, "per_sample.json", title=f"Captioning BERTScore - {args.model_name}")
        return

    # ── 1. BERTScore ──
    print(f"\n  [1/2] Computing BERTScore for {len(valid)} samples...")

    ids = []
    cands = []
    refs = []
    source_map = {}

    for p in valid:
        sample_id = p["id"]
        try:
            gt = json.loads(p["ground_truth"]) if isinstance(p["ground_truth"], str) else p["ground_truth"]
            pred = json.loads(p["prediction"]) if isinstance(p["prediction"], str) else p["prediction"]
        except (json.JSONDecodeError, TypeError):
            continue

        gt_desc = gt.get("description", str(gt)) if isinstance(gt, dict) else str(gt)
        pred_desc = pred.get("description", str(pred)) if isinstance(pred, dict) else str(pred)

        ids.append(sample_id)
        refs.append(gt_desc)
        cands.append(pred_desc)
        source_map[sample_id] = p.get("source", "")

    if not ids:
        tqdm.write(f"[WARNING] No valid description pairs found. Skipping BERTScore.")
        return

    chunk = getattr(args, "chunk", False)
    chunk_size = getattr(args, "chunk_size", 512)
    P, R, F1 = statistical_BERTScore(cands, refs, chunk=chunk, chunk_size=chunk_size)

    bert_score_json = {}
    for i, key in enumerate(ids):
        bert_score_json[key] = {
            "P": P[i].item(),
            "R": R[i].item(),
            "F1": F1[i].item(),
            "source": source_map.get(key, ""),
        }

    save_json_data(bert_score_json, bert_dir, "per_sample.json",
                   title=f"Captioning BERTScore - {args.model_name}")

    # Aggregate by source
    source_scores = defaultdict(lambda: {"P": [], "R": [], "F1": []})
    for key, v in bert_score_json.items():
        src = v.get("source", "overall")
        source_scores[src]["P"].append(v["P"])
        source_scores[src]["R"].append(v["R"])
        source_scores[src]["F1"].append(v["F1"])
        source_scores["overall"]["P"].append(v["P"])
        source_scores["overall"]["R"].append(v["R"])
        source_scores["overall"]["F1"].append(v["F1"])

    summary = {}
    for src, scores in sorted(source_scores.items()):
        summary[src] = {
            "P": float(np.mean(scores["P"])),
            "R": float(np.mean(scores["R"])),
            "F1": float(np.mean(scores["F1"])),
            "count": len(scores["F1"]),
        }

    save_json_data(summary, bert_dir, "summary.json",
                   title=f"Captioning BERTScore Summary - {args.model_name}")

    # Save unified metrics.json at task root (flat format, consistent with all evaluators)
    overall_bert = summary.get("overall", {})
    unified_metrics = {
        "bertscore_f1": overall_bert.get("F1", 0.0),
        "bertscore_precision": overall_bert.get("P", 0.0),
        "bertscore_recall": overall_bert.get("R", 0.0),
        "total_samples": overall_bert.get("count", 0),
    }
    save_json_data(unified_metrics, output_dir, "metrics.json",
                   title=f"Captioning Metrics - {args.model_name}")

    print(f"\n  BERTScore Results:")
    for src, m in summary.items():
        print(f"    {src}: P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} (n={m['count']})")

    # ── 2. LLM-as-judge confusion matrix (only if model available) ──
    if model is None:
        print(f"\n  [2/2] Skipping confusion matrix (no model provided)")
        return

    print(f"\n  [2/2] Computing confusion matrix for {len(valid)} samples...")

    lfss_meta_type = getattr(args, "lfss_meta_type", "en")
    workers = getattr(args, "workers", 1)

    confusion_results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for p in valid:
            future = executor.submit(
                _judge_single, p["id"], model,
                p["ground_truth"], p["prediction"], lfss_meta_type,
            )
            futures[future] = p["id"]

        for future in tqdm(as_completed(futures), total=len(futures),
                          desc="Confusion Matrix", unit="sample", dynamic_ncols=True):
            result = future.result()
            if not result.get("failed") and isinstance(result.get("confusion"), dict):
                confusion = result["confusion"]
                metrics = compute_confusion_metrics(confusion)
                confusion.update(metrics)
                confusion_results[result["id"]] = confusion

    save_json_data(confusion_results, confusion_dir, "per_sample.json",
                   title=f"Captioning Confusion - {args.model_name}")

    # Aggregate
    if confusion_results:
        source_cm = defaultdict(lambda: {"P": [], "R": [], "F1": []})
        for key, v in confusion_results.items():
            src = source_map.get(key, "overall")
            source_cm[src]["P"].append(v["P"])
            source_cm[src]["R"].append(v["R"])
            source_cm[src]["F1"].append(v["F1"])
            source_cm["overall"]["P"].append(v["P"])
            source_cm["overall"]["R"].append(v["R"])
            source_cm["overall"]["F1"].append(v["F1"])

        cm_summary = {}
        for src, scores in sorted(source_cm.items()):
            cm_summary[src] = {
                "P": float(np.mean(scores["P"])),
                "R": float(np.mean(scores["R"])),
                "F1": float(np.mean(scores["F1"])),
                "count": len(scores["F1"]),
            }

        save_json_data(cm_summary, confusion_dir, "summary.json",
                       title=f"Captioning Confusion Summary - {args.model_name}")

        print(f"\n  Confusion Matrix Results:")
        for src, m in cm_summary.items():
            print(f"    {src}: P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} (n={m['count']})")
