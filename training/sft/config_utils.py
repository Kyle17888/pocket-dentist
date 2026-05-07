#!/usr/bin/env python3
"""
config_utils.py — Shared SFT config loading with base_sft.yaml merge chain.

Merge priority (low → high):
    base_config/base_sft.yaml → base_config/models/<model>.yaml → datasets/<dataset>/dataset.yaml → CLI args

Usage in SFT scripts:
    from config_utils import load_merged_config
    config = load_merged_config(args.config)

Standalone verification:
    python config_utils.py --verify training/sft/configs/base_config/models/slms/gemma-4-E2B-it.yaml
    python config_utils.py --verify-all
"""

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------
def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into `base`. Override values win."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ---------------------------------------------------------------------------
# Auto-derive fields
# ---------------------------------------------------------------------------
def _auto_derive(config: Dict[str, Any], model_config_path: str) -> Dict[str, Any]:
    """
    Auto-derive output_dir, wandb.name, wandb.dir from model_name_or_path
    and the config file's location, if not explicitly set in the model config.
    """
    model_path = Path(model_config_path).resolve()

    # tier = parent dir name (slms / llms) — or flat (brar without slms/llms)
    tier = model_path.parent.name
    if tier in ("slms", "llms"):
        dataset_dir = model_path.parent.parent
    else:
        tier = ""
        dataset_dir = model_path.parent

    dataset_name = dataset_dir.name  # e.g. "metadent", "brar", "models"
    # When config is under base_config/models/, derive dataset name from SFT_DATASET env var
    if dataset_name == "models":
        dataset_name = os.environ.get("SFT_DATASET", "unknown")

    # model_short = last segment of HF model path
    model_full = config.get("model_name_or_path", "")
    model_short = model_full.split("/")[-1] if "/" in model_full else model_full

    # Auto-derive output_dir
    if "output_dir" not in config:
        output_base = config.get("output_base", "")
        if output_base:
            parts = [output_base]
            if tier:
                parts.append(tier)
            parts.append(model_short)
            parts.append("source")
            config["output_dir"] = "/".join(parts)
            log.info(f"Auto-derived output_dir: {config['output_dir']}")

    # Auto-derive wandb fields
    wandb = config.setdefault("wandb", {})
    if "name" not in wandb:
        # Use proper casing for dataset labels
        ds_labels = {"metadent": "MetaDent", "brar": "BRAR"}
        ds_label = ds_labels.get(dataset_name, dataset_name.capitalize())
        wandb["name"] = f"{model_short}-{ds_label}"
        log.info(f"Auto-derived wandb.name: {wandb['name']}")
    if "dir" not in wandb:
        parts = ["./logs/wandb", dataset_name]
        if tier:
            parts.append(tier)
        parts.append(model_short)
        wandb["dir"] = "/".join(parts)
        log.info(f"Auto-derived wandb.dir: {wandb['dir']}")

    # Remove intermediate key (not used by training scripts)
    config.pop("output_base", None)

    return config


# ---------------------------------------------------------------------------
# Dynamic training step computation
# ---------------------------------------------------------------------------
def compute_eval_steps(config: Dict[str, Any], num_train_samples: int) -> Dict[str, Any]:
    """
    Auto-compute logging_steps, eval_steps, and save_steps from dataset size.

    Uses config keys:
        - num_logging_points: target number of loss log entries (~50)
        - num_evals_per_epoch: target eval/save events per epoch (~3)

    If config already has explicit eval_steps/save_steps/logging_steps,
    those are used as-is (override wins).

    Args:
        config: The merged config dict.
        num_train_samples: Number of training samples (len(train_dataset)).

    Returns:
        Config with logging_steps, eval_steps, and save_steps populated.
    """
    training = config.get("training", {})

    batch_size = training.get("per_device_train_batch_size", 4)
    grad_accum = training.get("gradient_accumulation_steps", 4)
    num_epochs = training.get("num_train_epochs", 3)

    effective_batch = batch_size * grad_accum
    steps_per_epoch = max(num_train_samples // effective_batch, 1)
    total_steps = steps_per_epoch * num_epochs

    # --- logging_steps ---
    if "logging_steps" not in training:
        num_log_points = training.get("num_logging_points", 100)
        logging_steps = max(total_steps // num_log_points, 1)
        training["logging_steps"] = logging_steps
        log.info(f"Auto-computed logging_steps: {logging_steps} "
                 f"(~{num_log_points} log entries across {total_steps} steps)")

    # --- eval_steps & save_steps (independent) ---
    # With load_best_model_at_end=False, these can be set independently.
    # eval: infrequent (track loss), save: frequent (resume from interruptions).
    if "eval_steps" not in training:
        evals_per_epoch = training.get("num_evals_per_epoch", 4)
        eval_steps = max(steps_per_epoch // evals_per_epoch, 1)
        training["eval_steps"] = eval_steps
        log.info(f"Auto-computed eval_steps={eval_steps} (~{evals_per_epoch}/epoch)")

    if "save_steps" not in training:
        saves_per_epoch = training.get("num_saves_per_epoch", 20)
        save_steps = max(steps_per_epoch // saves_per_epoch, 1)
        training["save_steps"] = save_steps
        log.info(f"Auto-computed save_steps={save_steps} (~{saves_per_epoch}/epoch)")

    log.info(
        f"Training plan: {num_train_samples} samples × {num_epochs} epochs "
        f"÷ batch {effective_batch} = {total_steps} total steps"
    )

    config["training"] = training
    return config


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------
def _find_file_upward(start_dir: Path, filename: str, stop_at: Optional[Path] = None) -> Optional[Path]:
    """Search for `filename` in start_dir and parent directories, stopping at stop_at."""
    current = start_dir.resolve()
    if stop_at:
        stop_at = stop_at.resolve()
    while True:
        candidate = current / filename
        if candidate.exists():
            return candidate
        if stop_at and current == stop_at:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def load_merged_config(model_config_path: str) -> Dict[str, Any]:
    """
    Load config with 4-level merge chain (low → high priority):

        base_config/base_sft.yaml → base_config/models/<model>.yaml → datasets/<dataset>/dataset.yaml → CLI args

    The model_config_path can point to either:
      - A dataset-specific override: configs/datasets/<dataset>/(slms|llms)/<model>.yaml
      - A base_config model default: configs/base_config/models/(slms|llms)/<model>.yaml

    When using run_*_sft.sh, the path typically points to a base_config/models config
    (for models with no dataset-specific overrides) or a dataset override.

    Search strategy:
    1. Load the model config from `model_config_path`
    2. Determine configs_root (directory containing base_config/)
    3. Load `base_config/base_sft.yaml` from configs_root
    4. Load `base_config/models/(slms|llms)/<model>.yaml` if it exists
    5. Load `datasets/<dataset>/dataset.yaml` by searching upward from model config
    6. Load dataset-specific model override (if exists)
    7. Deep merge: base → base_model → dataset → dataset_model_override
    8. Auto-derive output_dir, wandb.name, wandb.dir if not set
    """
    model_path = Path(model_config_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Config file not found: {model_config_path}")

    model_filename = model_path.name  # e.g. "gemma-4-E2B-it.yaml"

    # Determine tier (slms / llms) and parent directory
    tier = model_path.parent.name  # e.g. "slms", "llms", or dataset name for flat layout
    if tier in ("slms", "llms"):
        parent_dir = model_path.parent.parent  # e.g. "models" or dataset name
    else:
        parent_dir = model_path.parent
        tier = ""

    parent_name = parent_dir.name  # e.g. "metadent", "brar", "code", "models"

    # Find configs root (directory containing base_config/)
    # Walk upward from the model config to find configs_root
    configs_root = None
    search_dir = model_path.parent
    for _ in range(6):  # max depth
        if (search_dir / "base_config" / "base_sft.yaml").exists():
            configs_root = search_dir
            break
        search_dir = search_dir.parent
    if not configs_root:
        configs_root = model_path.parent.parent.parent
    log.info(f"Configs root: {configs_root}")

    # ── Layer 1: base_config/base_sft.yaml ──
    base_config = {}
    base_path = configs_root / "base_config" / "base_sft.yaml"
    if base_path.exists():
        with open(base_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f) or {}
        log.info(f"Loaded base config: {base_path}")

    # ── Layer 2: base_config/models/<tier>/<model>.yaml ──
    base_model_config = {}
    if tier:
        base_model_path = configs_root / "base_config" / "models" / tier / model_filename
    else:
        base_model_path = configs_root / "base_config" / "models" / model_filename
    if base_model_path.exists() and parent_name != "models":
        # Only load base_config/models as a separate layer when the model_config_path
        # is NOT already pointing to base_config/models (avoid double-loading)
        with open(base_model_path, "r", encoding="utf-8") as f:
            base_model_config = yaml.safe_load(f) or {}
        log.info(f"Loaded base model config: {base_model_path}")

    # ── Layer 3: datasets/<dataset>/dataset.yaml ──
    dataset_config = {}
    dataset_name = None
    if parent_name == "models":
        # When model_config_path points to base_config/models directly,
        # read dataset name from SFT_DATASET env var (set by run_*_sft.sh)
        dataset_name = os.environ.get("SFT_DATASET", "")
        if dataset_name:
            dataset_path = configs_root / "datasets" / dataset_name / "dataset.yaml"
            if dataset_path.exists():
                with open(dataset_path, "r", encoding="utf-8") as f:
                    dataset_config = yaml.safe_load(f) or {}
                log.info(f"Loaded dataset config (via SFT_DATASET={dataset_name}): {dataset_path}")
            else:
                log.warning(f"SFT_DATASET={dataset_name} but dataset.yaml not found: {dataset_path}")
        else:
            log.warning("model_config_path points to base_config/models/ but SFT_DATASET env var not set — no dataset config loaded")
    else:
        dataset_path = _find_file_upward(
            model_path.parent, "dataset.yaml", stop_at=configs_root
        )
        if dataset_path:
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset_config = yaml.safe_load(f) or {}
            log.info(f"Loaded dataset config: {dataset_path}")

    # ── Layer 4: dataset-specific model override (if model_config_path is under datasets/) ──
    model_override_config = {}
    if parent_name != "models":
        with open(model_path, "r", encoding="utf-8") as f:
            model_override_config = yaml.safe_load(f) or {}
        log.info(f"Loaded model override config: {model_path}")
    else:
        # model_config_path IS the base_config/models config — it becomes the
        # base_model_config layer, and there's no dataset override
        base_model_config = {}
        with open(model_path, "r", encoding="utf-8") as f:
            base_model_config = yaml.safe_load(f) or {}
        log.info(f"Loaded base model config (direct): {model_path}")

    # ── Merge: base → base_model → dataset → model_override ──
    merged = deep_merge(base_config, base_model_config)
    merged = deep_merge(merged, dataset_config)
    merged = deep_merge(merged, model_override_config)

    # Auto-derive fields (uses model_config_path for directory context)
    merged = _auto_derive(merged, model_config_path)

    return merged


# ---------------------------------------------------------------------------
# Verification CLI
# ---------------------------------------------------------------------------
def _verify_single(model_config_path: str, original_configs: Dict) -> bool:
    """Verify that merged config matches original monolithic config."""
    merged = load_merged_config(model_config_path)
    original = original_configs.get(model_config_path)
    if original is None:
        print(f"  ⚠️  No original found for: {model_config_path}")
        return True

    # Compare key by key
    diffs = []
    all_keys = set(list(_flatten(merged).keys()) + list(_flatten(original).keys()))
    merged_flat = _flatten(merged)
    original_flat = _flatten(original)

    # Keys that base.yaml may add but weren't in originals (harmless defaults)
    IGNORABLE_NEW_KEYS = {"vision.fps", "vision.max_pixels", "vision.min_pixels",
                          "wandb.enabled", "output_base", "training.early_stopping_patience"}
    # Auto-derived keys where intentional improvements over originals are OK
    AUTO_DERIVED_KEYS = {"output_dir", "wandb.dir", "wandb.name"}

    for key in sorted(all_keys):
        m_val = merged_flat.get(key, "__MISSING__")
        o_val = original_flat.get(key, "__MISSING__")
        if str(m_val) != str(o_val):
            # Skip keys that are new additions from base.yaml (not in original)
            if o_val == "__MISSING__" and key in IGNORABLE_NEW_KEYS:
                continue
            # Skip auto-derived keys (path normalization improvements are OK)
            if key in AUTO_DERIVED_KEYS:
                continue
            diffs.append((key, o_val, m_val))

    if diffs:
        print(f"  ❌ {model_config_path}")
        for key, o_val, m_val in diffs:
            print(f"     {key}: original={o_val} → merged={m_val}")
        return False
    else:
        print(f"  ✅ {model_config_path}")
        return True


def _flatten(d: Dict, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dict to dot-separated keys."""
    result = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            result.update(_flatten(v, f"{key}."))
        else:
            result[key] = v
    return result


def main():
    parser = argparse.ArgumentParser(description="SFT Config Utilities")
    parser.add_argument("--verify", type=str, help="Verify a single config against original")
    parser.add_argument("--verify-all", action="store_true", help="Verify all configs")
    parser.add_argument("--dump", type=str, help="Dump merged config for a model config path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.dump:
        config = load_merged_config(args.dump)
        print(yaml.dump(config, default_flow_style=False, sort_keys=True))
        return

    if args.verify or args.verify_all:
        # Load original configs
        originals_path = Path(__file__).parent / "configs" / ".original_configs.json"
        if not originals_path.exists():
            print(f"❌ Original configs backup not found: {originals_path}")
            sys.exit(1)
        with open(originals_path) as f:
            original_configs = json.load(f)

        if args.verify:
            ok = _verify_single(args.verify, original_configs)
            sys.exit(0 if ok else 1)
        else:
            # Verify all
            total = 0
            passed = 0
            for path in sorted(original_configs.keys()):
                if Path(path).exists():
                    total += 1
                    if _verify_single(path, original_configs):
                        passed += 1
            print(f"\n{'✅' if passed == total else '❌'} {passed}/{total} configs verified")
            sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
