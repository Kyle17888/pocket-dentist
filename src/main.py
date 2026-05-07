import json
import os

from src import evaluation_runner, prediction_runner
from src.models.api_model import APIModel
from src.utils.config_loader import load_args, load_model_config, load_yaml_config

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if __name__ == '__main__':
    args = load_args()

    # ── API credential checks ──
    # Prediction always needs API creds; evaluation only needs them with --enable_llm_judge
    needs_api = (
        args.task == "prediction"
        or (args.task == "evaluation" and getattr(args, "enable_llm_judge", False))
    )
    if args.client_type == "api" and needs_api:
        if not args.api_base_url:
            args.api_base_url = os.getenv("API_BASE_URL")
            if not args.api_base_url:
                raise ValueError("--api_base_url is required for API mode")
        if not args.api_key:
            args.api_key = os.getenv("API_KEY")
            if not args.api_key:
                raise ValueError("--api_key is required for API mode")
    elif args.client_type != "api":
        args.workers = 1

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # ── Output paths ──
    args.project_root = project_root
    # Unified output: results/<dataset>/
    args.save_root_dir = os.path.join(project_root, "results", "datasets", args.dataset)

    # ── Load config ──
    yaml_cfg = load_yaml_config(args=args)
    model_cfg = load_model_config(args)

    # Resolve served_model_name for API models
    if args.task == "evaluation" and getattr(args, "enable_llm_judge", False) and args.evaluator_model_name:
        fallback_model_name = args.evaluator_model_name
    else:
        fallback_model_name = args.model_name
    args.served_model_name = (
        model_cfg.get("served_model_name")
        if (model_cfg and model_cfg.get("served_model_name"))
        else fallback_model_name
    )

    # ── Dispatch ──
    model = None
    if args.task == "prediction":
        model = prediction_runner.run(args, yaml_cfg, model_cfg)
    elif args.task == "evaluation":
        model = evaluation_runner.run(args, yaml_cfg, model_cfg)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    # ── Token usage logging (API models + LLM judge) ──
    if isinstance(model, APIModel):
        import datetime
        usage = model.get_usage()
        prompt_t = usage["prompt_tokens"]
        completion_t = usage["completion_tokens"]
        total_t = usage["total_tokens"]

        # Resolve pricing: check model_cfg first, then look up judge model
        # in yaml_cfg api_models section (for LLM judge during local model eval)
        pricing = (model_cfg or {}).get("pricing", {})
        if not pricing and yaml_cfg:
            judge_cfg = yaml_cfg.get("api_models", {}).get(model.model_name, {})
            pricing = judge_cfg.get("pricing", {})
        input_rate = pricing.get("input_per_1m", 0)
        output_rate = pricing.get("output_per_1m", 0)
        cost_usd = (prompt_t * input_rate + completion_t * output_rate) / 1_000_000

        # Use the actual model name (could be the judge model, not the VLM)
        actual_model = model.model_name
        subtask_label = getattr(args, "subtask", "all")
        usage_source = "prediction" if args.task == "prediction" else "llm_judge"
        print(f"\n[Token Usage] task={args.task}/{subtask_label}  model={actual_model}  source={usage_source}")
        print(f"  prompt:     {prompt_t:>12,}")
        print(f"  completion: {completion_t:>12,}")
        print(f"  total:      {total_t:>12,}")
        print(f"  cost:       ${cost_usd:>11.4f} USD")

        run_id = os.getenv("PIPELINE_RUN_ID", "manual")
        log_file = os.path.join(project_root, "results", "bills", "token_usage.jsonl")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        record = {
            "run_id": run_id,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "task": args.task,
            "subtask": subtask_label,
            "dataset": args.dataset,
            "model": actual_model,
            "source": usage_source,
            "prompt_tokens": prompt_t,
            "completion_tokens": completion_t,
            "total_tokens": total_t,
            "cost_usd": round(cost_usd, 6),
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Done!")