import argparse
import os

import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))


def load_yaml_config(args: argparse.Namespace):
    """Load dataset-specific config.yaml based on --dataset argument."""
    cfg_dir = os.path.join(project_root, "configs", args.dataset)

    if args.test_mode:
        yaml_path = os.path.join(cfg_dir, "config-dev.yaml")
    else:
        yaml_path = os.path.join(cfg_dir, "config.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_model_config(args: argparse.Namespace):
    """Load model-specific config from the dataset config."""
    cfg = load_yaml_config(args=args)
    # Offline evaluation tasks don't need model config
    if args.task == "evaluation" and not getattr(args, "enable_llm_judge", False):
        return None

    if args.task == "evaluation" and args.enable_llm_judge:
        model_name = args.evaluator_model_name or args.model_name
    else:
        model_name = args.model_name

    if args.client_type == "local":
        model_cfg = cfg.get("local_models", {}).get(model_name)
    elif args.client_type == "api":
        model_cfg = cfg.get("api_models", {}).get(model_name)
    else:
        model_cfg = None

    if args.client_type == "local" and not model_cfg:
        raise ValueError(f"Model '{model_name}' not found in config.yaml")
    return model_cfg

def load_args():
    # Pre-parse just the config argument
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None, help="Path to a YAML configuration file to use instead of passing arguments through the CLI.")
    pre_args, remaining_argv = pre_parser.parse_known_args()
    
    yaml_config = {}
    if pre_args.config:
        if not os.path.exists(pre_args.config):
            raise FileNotFoundError(f"Argument config file not found: {pre_args.config}")
        with open(pre_args.config, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}
        
        # If model_name is not in the task config, try to read it from base.yaml
        # in the same directory. This allows a single base.yaml to control which
        # model identity (base vs SFT) all task configs in the folder use.
        if "model_name" not in yaml_config:
            base_yaml = os.path.join(os.path.dirname(pre_args.config), "base.yaml")
            if os.path.exists(base_yaml):
                with open(base_yaml, "r", encoding="utf-8") as f:
                    base_cfg = yaml.safe_load(f) or {}
                if "model_name" in base_cfg:
                    yaml_config["model_name"] = base_cfg["model_name"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML configuration file to use instead of passing arguments through the CLI.")
    parser.add_argument("--dataset", type=str, help="Dataset name (e.g., metadent, brar). Determines config directory and output paths.")
    parser.add_argument("--model_name", type=str, help="Model name or path to use (for local or API mode).")
    parser.add_argument("--evaluator_model_name", type=str, default=None, help="Name or path of the LLM used for evaluation. Required only for LLM-as-judge tasks (e.g., captioning evaluation).")
    parser.add_argument("--do_sample", action="store_true", default=False, help="Whether to sample from the model.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling. Higher temperature results in more random output.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of tokens to generate.")
    parser.add_argument("--client_type",  type=str, choices=["local", "api"], default="api", help="Choose how to load the model: 'local' for local weights, or 'api' for remote (OpenAI/vLLM-compatible) endpoints.")
    
    parser.add_argument("--gpus", type=str, default=None, help="GPUs to use (comma-separated), e.g. 0,1,2,3. If not specified, all GPUs will be used.")
    parser.add_argument("--api_base_url", type=str, default=None, help="Base URL for the OpenAI-compatible API (e.g. http://localhost:8000/v1)")
    parser.add_argument("--api_key", type=str, default=None, help="API key for the OpenAI-compatible API.")
    
    parser.add_argument("--task", type=str, choices=["prediction", "evaluation"], help="Main task type: 'prediction' to run VLM inference, 'evaluation' to compute metrics")
    parser.add_argument("--subtask", type=str, default="all", help="Sub-task filter (e.g., vqa, classification, captioning). Default 'all' processes all tasks in the JSONL.")
    
    parser.add_argument("--workers", type=int, default=8, help="Number of threads to use")
    
    parser.add_argument("--lfss_meta_type", type=str, choices=["cn", "en"], default="en", help="Language of the meta data, cn or en")
    parser.add_argument("--test_mode", action="store_true", default=False, help="Whether to run in test mode (use private data)")
    parser.add_argument("--few_shot_config", type=str, default=None, help="Path to a few-shot YAML config file.")
    parser.add_argument("--num_shots", type=int, default=None, help="Override the number of shots from the config.")
    parser.add_argument("--run_tag", type=str, default="baseline", help="Tag for this run (e.g., baseline, 1shot, 2shot, sft). Used to isolate output directories.")
    
    parser.add_argument("--enable_llm_judge", action="store_true", default=False, help="Enable LLM-as-judge evaluation (e.g., captioning confusion matrix). Without this flag, only offline metrics are computed.")

    # captioning
    parser.add_argument("--chunk", action="store_true", help="Whether to chunk the data into smaller batches to avoid GPU memory issues")
    parser.add_argument("--chunk_size", type=int, default=512, help="Size of each chunk")
    
    parser.set_defaults(**yaml_config)
    args = parser.parse_args()
    
    if not args.model_name:
        parser.error("--model_name is required either in CLI or config")
    if not args.dataset:
        parser.error("--dataset is required (e.g., metadent, brar)")
    if getattr(args, "task", None) not in ["prediction", "evaluation"]:
        parser.error("--task is required and must be prediction or evaluation")
        
    return args
