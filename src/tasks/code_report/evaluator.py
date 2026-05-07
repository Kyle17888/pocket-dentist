"""
COde Diagnostic Report Generation Evaluation — BLEU, METEOR, Cosine Similarity.

Reads unified predictions and computes:
  - BLEU score (sacrebleu, sentence-level)
  - METEOR score (nltk)
  - Cosine Similarity (sentence-transformers embeddings)

Corresponds to COde paper Fig. 7 metrics.
"""

import json
import os
from collections import defaultdict
from decimal import Decimal

import numpy as np
from tqdm import tqdm

from src.utils.file_io import save_json_data


def _compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute sentence-level BLEU score."""
    try:
        import sacrebleu
        # sacrebleu expects list of references
        bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
        return bleu.score / 100.0  # Normalize to [0, 1]
    except ImportError:
        # Fallback to nltk BLEU
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        if not hyp_tokens:
            return 0.0
        smoothie = SmoothingFunction().method1
        return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)


def _compute_meteor(reference: str, hypothesis: str) -> float:
    """Compute METEOR score."""
    try:
        from nltk.translate.meteor_score import meteor_score
        # METEOR expects tokenized strings
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        if not hyp_tokens:
            return 0.0
        return meteor_score([ref_tokens], hyp_tokens)
    except ImportError:
        tqdm.write("⚠️  nltk not available, skipping METEOR")
        return 0.0


def _compute_cosine_similarity_batch(references: list[str], hypotheses: list[str]) -> list[float]:
    """Compute cosine similarity using sentence-transformers embeddings.

    Uses a single lightweight model (all-MiniLM-L6-v2) for efficiency.
    The COde paper uses Gemma-2B + Llama-3.2-1B averaged, but we use a
    standard sentence embedding model for reproducibility and speed.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        ref_embeddings = model.encode(references, show_progress_bar=False, convert_to_numpy=True)
        hyp_embeddings = model.encode(hypotheses, show_progress_bar=False, convert_to_numpy=True)

        # Cosine similarity per pair
        similarities = []
        for ref_emb, hyp_emb in zip(ref_embeddings, hyp_embeddings):
            cos_sim = np.dot(ref_emb, hyp_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(hyp_emb) + 1e-8)
            similarities.append(float(cos_sim))
        return similarities
    except ImportError:
        tqdm.write("⚠️  sentence-transformers not available, skipping cosine similarity")
        return [0.0] * len(references)


def evaluate(predictions: list[dict], output_dir: str, args, model=None, yaml_cfg=None):
    """
    Evaluate COde diagnostic report generation predictions.

    Args:
        predictions: List of {id, task, source, ground_truth, prediction, failed}
        output_dir:  Directory to save evaluation results
        args:        Parsed argparse.Namespace
        model:       Not used (offline evaluation)
        yaml_cfg:    Not used
    """
    valid = []
    for p in predictions:
        if p.get("failed"):
            continue
        gt_text = p.get("ground_truth", "")
        pred_text = p.get("prediction", "")
        if not gt_text or not pred_text:
            continue
        valid.append(p)

    if not valid:
        print("⚠️  No valid COde report predictions found.")
        return

    # Compute BLEU and METEOR per sample
    print(f"\n  [1/3] Computing BLEU + METEOR for {len(valid)} samples...")
    per_sample = {}
    refs_for_cosine = []
    hyps_for_cosine = []

    for p in tqdm(valid, desc=f"[COde Report] {args.model_name}", dynamic_ncols=True):
        sample_id = p["id"]
        gt_text = p["ground_truth"].strip()
        pred_text = p["prediction"].strip()

        bleu = _compute_bleu(gt_text, pred_text)
        meteor = _compute_meteor(gt_text, pred_text)

        per_sample[sample_id] = {
            "bleu": bleu,
            "meteor": meteor,
            "cosine_sim": 0.0,  # filled later
        }
        refs_for_cosine.append(gt_text)
        hyps_for_cosine.append(pred_text)

    # Compute cosine similarity in batch
    print(f"\n  [2/3] Computing Cosine Similarity...")
    cosine_scores = _compute_cosine_similarity_batch(refs_for_cosine, hyps_for_cosine)

    sample_ids = list(per_sample.keys())
    for i, sid in enumerate(sample_ids):
        per_sample[sid]["cosine_sim"] = cosine_scores[i]

    # Aggregate summary
    print(f"\n  [3/3] Computing summary statistics...")
    point = "0.000"

    bleu_scores = [v["bleu"] for v in per_sample.values()]
    meteor_scores = [v["meteor"] for v in per_sample.values()]
    cosine_scores_final = [v["cosine_sim"] for v in per_sample.values()]

    summary = {
        "total_samples": len(per_sample),
        "bleu": {
            "mean": float(Decimal(str(np.mean(bleu_scores))).quantize(Decimal(point))),
            "std": float(Decimal(str(np.std(bleu_scores))).quantize(Decimal(point))),
            "median": float(Decimal(str(np.median(bleu_scores))).quantize(Decimal(point))),
        },
        "meteor": {
            "mean": float(Decimal(str(np.mean(meteor_scores))).quantize(Decimal(point))),
            "std": float(Decimal(str(np.std(meteor_scores))).quantize(Decimal(point))),
            "median": float(Decimal(str(np.median(meteor_scores))).quantize(Decimal(point))),
        },
        "cosine_similarity": {
            "mean": float(Decimal(str(np.mean(cosine_scores_final))).quantize(Decimal(point))),
            "std": float(Decimal(str(np.std(cosine_scores_final))).quantize(Decimal(point))),
            "median": float(Decimal(str(np.median(cosine_scores_final))).quantize(Decimal(point))),
        },
    }

    # Save outputs
    save_json_data(per_sample, output_dir, "per_sample.json",
                   title=f"COde Report Per-Sample - {args.model_name}")
    save_json_data(summary, output_dir, "metrics.json",
                   title=f"COde Report Summary - {args.model_name}")

    # Print summary
    print(f"\n  COde Report Generation Results:")
    print(f"    BLEU:      {summary['bleu']['mean']:.3f} ± {summary['bleu']['std']:.3f}")
    print(f"    METEOR:    {summary['meteor']['mean']:.3f} ± {summary['meteor']['std']:.3f}")
    print(f"    CosineSim: {summary['cosine_similarity']['mean']:.3f} ± {summary['cosine_similarity']['std']:.3f}")
    print(f"    Samples:   {summary['total_samples']}")
