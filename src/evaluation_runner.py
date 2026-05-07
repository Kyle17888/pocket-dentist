"""
Unified Evaluation Runner — Loads predictions.jsonl, groups by task, dispatches to evaluators.

Supports two evaluation modes:
  - Offline (default): VQA, Classification, Captioning BERTScore, BRAR — no LLM needed
  - LLM-as-judge (auto or --enable_llm_judge): Captioning confusion matrix — requires model

LLM-as-judge is auto-enabled when LLM_JUDGE_* environment variables are set in .env.

Evaluator registry makes it easy to add new task types.
"""

import json
import os
from collections import defaultdict

from tqdm import tqdm


def _load_dotenv():
    """Load .env file from project root if it exists."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Don't override existing env vars
                if key not in os.environ:
                    os.environ[key] = value


# ──────────────────────────────────────────────────────────────
# Evaluator Registry
# ──────────────────────────────────────────────────────────────

# Each entry: task_name -> (module_path, function_name, requires_llm_judge)
EVALUATOR_REGISTRY = {
    "vqa": ("src.tasks.vqa.evaluator", "evaluate", False),
    "classification": ("src.tasks.classification.evaluator", "evaluate", False),
    "captioning": ("src.tasks.captioning.evaluator", "evaluate", True),
    "brar_classification": ("src.tasks.brar.evaluator", "evaluate", False),
    "code_classification": ("src.tasks.code_classification.evaluator", "evaluate", False),
    "code_report": ("src.tasks.code_report.evaluator", "evaluate", False),
    "aariz_cvm": ("src.tasks.aariz_cvm.evaluator", "evaluate", False),
    "aariz_vqa": ("src.tasks.aariz_vqa.evaluator", "evaluate", False),
    "denpar_count": ("src.tasks.denpar_count.evaluator", "evaluate", False),
    "denpar_arch": ("src.tasks.denpar_arch.evaluator", "evaluate", False),
    "denpar_site": ("src.tasks.denpar_site.evaluator", "evaluate", False),
    "dr_classification": ("src.tasks.dr_classification.evaluator", "evaluate", False),
    "caries_detect": ("src.tasks.caries_detect.evaluator", "evaluate", False),
    "caries_cls": ("src.tasks.caries_cls.evaluator", "evaluate", False),
}



def get_evaluator(task_type: str):
    """Dynamically import and return the evaluate function for a task type."""
    if task_type not in EVALUATOR_REGISTRY:
        tqdm.write(f"⚠️  No evaluator registered for task type: {task_type}")
        return None, False

    module_path, func_name, requires_llm = EVALUATOR_REGISTRY[task_type]
    import importlib
    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    return func, requires_llm


# ──────────────────────────────────────────────────────────────
# Prediction loading & grouping
# ──────────────────────────────────────────────────────────────

def load_predictions(pred_path: str) -> list[dict]:
    """Load predictions.jsonl, return list of dicts."""
    predictions = []
    if not os.path.exists(pred_path):
        return predictions
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                predictions.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return predictions


def group_by_task(predictions: list[dict]) -> dict[str, list[dict]]:
    """Group predictions by their task field."""
    grouped = defaultdict(list)
    for p in predictions:
        grouped[p.get("task", "unknown")].append(p)
    return dict(grouped)


# ──────────────────────────────────────────────────────────────
# LLM Judge auto-detection from .env
# ──────────────────────────────────────────────────────────────

def _resolve_llm_judge(args):
    """
    Auto-detect LLM judge configuration from .env environment variables.

    If LLM_JUDGE_MODEL, LLM_JUDGE_API_KEY, and LLM_JUDGE_API_BASE are set,
    automatically enable LLM-as-judge for captioning evaluation.

    Returns:
        (enable_llm, judge_model, judge_api_base, judge_api_key)
    """
    # Explicit CLI flag takes priority
    if getattr(args, "enable_llm_judge", False):
        return (
            True,
            getattr(args, "evaluator_model_name", None) or args.model_name,
            getattr(args, "api_base_url", None),
            getattr(args, "api_key", None),
        )

    # Auto-detect from environment
    _load_dotenv()
    judge_model = os.environ.get("LLM_JUDGE_MODEL", "").strip()
    judge_key = os.environ.get("LLM_JUDGE_API_KEY", "").strip()
    judge_base = os.environ.get("LLM_JUDGE_API_BASE", "").strip()

    if judge_model and judge_key and judge_base:
        return True, judge_model, judge_base, judge_key

    return False, None, None, None


# ──────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────

def run(args, yaml_cfg, model_cfg):
    # Resolve prediction file path
    output_dir = os.path.join(
        args.save_root_dir,
        args.run_tag,
        args.model_name.split("/")[-1],
    )
    pred_path = os.path.join(output_dir, "predictions.jsonl")

    if not os.path.exists(pred_path):
        print(f"⚠️  Predictions file not found: {pred_path}")
        print(f"   Run prediction first, then evaluate.")
        return None

    print(f"{'=' * 60}")
    print(f"Unified Evaluation")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Predictions:  {pred_path}")
    print(f"  Output:       {output_dir}")
    print(f"{'=' * 60}")

    # 1. Load and group predictions
    predictions = load_predictions(pred_path)
    if not predictions:
        print("⚠️  No predictions found. Exiting.")
        return None

    grouped = group_by_task(predictions)
    print(f"  Total:        {len(predictions)} predictions")
    for task_type, preds in grouped.items():
        print(f"    {task_type}: {len(preds)}")

    # 2. Filter by subtask if specified
    if getattr(args, "subtask", "all") != "all":
        if args.subtask in grouped:
            grouped = {args.subtask: grouped[args.subtask]}
        else:
            print(f"⚠️  No predictions found for subtask: {args.subtask}")
            return None

    # 3. Auto-detect LLM judge configuration
    enable_llm, judge_model, judge_api_base, judge_api_key = _resolve_llm_judge(args)
    if enable_llm:
        print(f"  LLM Judge:    ✅ {judge_model} (via {'CLI' if getattr(args, 'enable_llm_judge', False) else '.env'})")
    else:
        print(f"  LLM Judge:    ❌ disabled (set LLM_JUDGE_* in .env to enable)")

    # 4. Evaluate each task type
    model = None

    for task_type, task_preds in grouped.items():
        evaluate_fn, requires_llm = get_evaluator(task_type)
        if evaluate_fn is None:
            continue

        if requires_llm and not enable_llm:
            print(f"\n⏭  Skipping {task_type} LLM-as-judge evaluation (configure LLM_JUDGE_* in .env to enable)")
            # For captioning: still run BERTScore (offline) even without LLM judge
            if task_type == "captioning":
                print(f"   → Running BERTScore only (offline)")
                eval_output_dir = os.path.join(output_dir, task_type)
                os.makedirs(eval_output_dir, exist_ok=True)
                evaluate_fn(
                    predictions=task_preds,
                    output_dir=eval_output_dir,
                    args=args,
                    model=None,  # No model = BERTScore only
                    yaml_cfg=yaml_cfg,
                )
            continue

        print(f"\n{'─' * 40}")
        print(f"Evaluating: {task_type} ({len(task_preds)} samples)")
        print(f"{'─' * 40}")

        # Load model for LLM-as-judge tasks
        if requires_llm and model is None:
            from src.models.api_model import APIModel
            model = APIModel(
                model_name=judge_model,
                temperature=0.0,
                base_url=judge_api_base,
                api_key=judge_api_key,
                max_tokens=getattr(args, "max_new_tokens", 1024),
            )
            print(f"  Loaded LLM judge: {judge_model}")

        # Build evaluation output directory
        eval_output_dir = os.path.join(output_dir, task_type)
        os.makedirs(eval_output_dir, exist_ok=True)

        evaluate_fn(
            predictions=task_preds,
            output_dir=eval_output_dir,
            args=args,
            model=model if requires_llm else None,
            yaml_cfg=yaml_cfg,
        )

    print(f"\n{'=' * 60}")
    print(f"Evaluation Complete")
    print(f"{'=' * 60}")

    return model
