#!/usr/bin/env python3
"""
LoRA Model Merger — Single & Batch Mode

Supports two modes:
  1. Single mode: merge one specific checkpoint via config file
  2. Batch mode: auto-scan SFT root, find & merge all unmerged models

Usage:
  # ── Batch Mode (recommended) ──────────────────────────────────

  # Auto-scan and merge ALL unmerged models
  python merge_lora.py --batch

  # Dry-run: show what would be merged without executing
  python merge_lora.py --batch --dry-run

  # Only merge specific datasets
  python merge_lora.py --batch --datasets "BRAR,MetaDent"

  # Only merge specific models
  python merge_lora.py --batch --models "Lingshu-32B"

  # Combine filters (dataset + model)
  python merge_lora.py --batch --datasets "COde" --models "Lingshu-32B"

  # ── Single Mode (for specific checkpoint) ─────────────────────

  # Merge one checkpoint via merge_config.yaml
  python merge_lora.py --config merge_config.yaml

NeSI Resource Requirements (CPU only, no GPU needed):
  - 1B-8B models:  --mem=64G  --time=1:00:00
  - 32B models:    --mem=128G --time=2:00:00

  srun --account=uoa04670 --job-name=merge-lora \\
    --partition=milan --cpus-per-task=8 --mem=128G --time=1:00:00 --pty bash
"""

import os
import sys
import glob
import time
import argparse
import yaml
import torch
import gc

SFT_ROOT = "/home/kbia984/00_nesi_projects/uoa04670_nobackup/kbia984/models/Neurlps2026-SFT"


def load_config(yaml_path):
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def load_base_model(base_model_path, dtype, device_map, trust_remote_code):
    """
    Tries to load the base model using the correct AutoClass.
    Different models require different classes (e.g., Llama/Gemma -> AutoModelForCausalLM,
    PaliGemma/QwenVL -> AutoModelForImageTextToText, InternVL -> AutoModel).
    """
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoModel

    print(f"Loading Base Model from: {base_model_path}")

    classes_to_try = [
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoModel
    ]

    model = None
    last_exception = None

    for cls in classes_to_try:
        try:
            print(f"  -> Attempting with {cls.__name__}...")
            model = cls.from_pretrained(
                base_model_path,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code
            )
            print(f"  ✅ Successfully loaded using {cls.__name__}")
            break
        except Exception as e:
            last_exception = e
            print(f"  ❌ Failed with {cls.__name__}: {str(e)[:100]}...")

    if model is None:
        raise RuntimeError(f"Could not load base model. Last error: {last_exception}")

    return model


def load_processor(base_model_path, trust_remote_code):
    from transformers import AutoProcessor, AutoTokenizer
    print("Loading processor/tokenizer...")
    try:
        processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=trust_remote_code)
        return processor
    except Exception as e:
        print(f"  AutoProcessor failed: {str(e)[:100]}. Falling back to AutoTokenizer.")
        return AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=trust_remote_code)


def find_best_checkpoint(model_dir):
    """
    Find the best LoRA checkpoint to merge.

    Directory structure:
      model_dir/
      ├── source/                        ← LoRA training output
      │   ├── adapter_model.safetensors  ← final saved model (best if load_best_model_at_end=true)
      │   ├── checkpoint-100/
      │   └── checkpoint-200/
      └── merged/                        ← merged full model (output)

    Priority:
      1. source/ root (if it has adapter_model — this is the final/best saved model)
      2. Latest checkpoint-* inside source/ (highest step number)
    """
    source_dir = os.path.join(model_dir, "source")

    # If no source/ directory, fall back to model_dir itself
    search_dir = source_dir if os.path.isdir(source_dir) else model_dir

    # Check if search_dir itself has a LoRA adapter (final saved model)
    if os.path.isfile(os.path.join(search_dir, "adapter_model.safetensors")) or \
       os.path.isfile(os.path.join(search_dir, "adapter_model.bin")):
        return search_dir

    # Check checkpoint-* subdirectories (latest first)
    checkpoints = sorted(
        glob.glob(os.path.join(search_dir, "checkpoint-*")),
        key=lambda x: int(os.path.basename(x).split("-")[1]) if os.path.basename(x).split("-")[1].isdigit() else 0
    )

    for ckpt in reversed(checkpoints):  # Start from latest
        if os.path.isfile(os.path.join(ckpt, "adapter_model.safetensors")) or \
           os.path.isfile(os.path.join(ckpt, "adapter_model.bin")):
            return ckpt

    return None


def is_merged(model_dir):
    """Check if merged/ directory exists and contains weight files."""
    merged_dir = os.path.join(model_dir, "merged")
    if not os.path.isdir(merged_dir):
        return False
    weights = glob.glob(os.path.join(merged_dir, "*.safetensors")) + \
              glob.glob(os.path.join(merged_dir, "*.bin"))
    return len(weights) > 0


def merge_single(lora_path, merged_output_dir, base_model_path="",
                 device_map="auto", bf16=True, trust_remote_code=True):
    """Merge a single LoRA adapter into its base model."""
    from peft import PeftModel, PeftConfig

    dtype = torch.bfloat16 if bf16 else torch.float32

    # 1. Resolve base model path
    if not base_model_path:
        print("  Base model path not provided, detecting from adapter_config.json...")
        peft_config = PeftConfig.from_pretrained(lora_path)
        base_model_path = peft_config.base_model_name_or_path
        if not base_model_path:
            raise ValueError("Could not auto-detect base_model_name_or_path from LoRA checkpoint.")

    # 2. Load Base Model and Processor
    base_model = load_base_model(base_model_path, dtype, device_map, trust_remote_code)
    processor = load_processor(base_model_path, trust_remote_code)

    # 3. Load LoRA weights into Base Model
    print(f"\n  Loading LoRA adapter from: {lora_path} ...")
    peft_model = PeftModel.from_pretrained(base_model, lora_path)

    # 4. Merge and Unload
    print("  Merging LoRA weights into base model...")
    merged_model = peft_model.merge_and_unload()

    # 5. Save Model
    print(f"  Saving merged model to: {merged_output_dir}")
    os.makedirs(merged_output_dir, exist_ok=True)
    merged_model.save_pretrained(merged_output_dir)
    processor.save_pretrained(merged_output_dir)

    # 6. Free memory
    del merged_model, peft_model, base_model, processor
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("  🎉 Merge completed!")


def scan_unmerged(sft_root, filter_datasets=None, filter_models=None):
    """Scan SFT root and return list of models that need merging."""
    tasks = []

    for dataset_name in sorted(os.listdir(sft_root)):
        dataset_dir = os.path.join(sft_root, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue

        if filter_datasets and dataset_name not in filter_datasets:
            continue

        for tier in ["llms", "slms"]:
            tier_dir = os.path.join(dataset_dir, tier)
            if not os.path.isdir(tier_dir):
                continue

            for model_name in sorted(os.listdir(tier_dir)):
                model_dir = os.path.join(tier_dir, model_name)
                if not os.path.isdir(model_dir):
                    continue

                if filter_models and model_name not in filter_models:
                    continue

                # Skip if already merged
                if is_merged(model_dir):
                    continue

                # Find best checkpoint
                lora_path = find_best_checkpoint(model_dir)
                if lora_path is None:
                    continue  # No training data found

                merged_dir = os.path.join(model_dir, "merged")
                tasks.append({
                    "dataset": dataset_name,
                    "tier": tier,
                    "model": model_name,
                    "lora_path": lora_path,
                    "merged_dir": merged_dir,
                })

    return tasks


def main():
    parser = argparse.ArgumentParser(description="LoRA Model Merger — Single & Batch")
    parser.add_argument("--config", type=str, default="merge_config.yaml",
                        help="Path to merge config yaml (single mode)")
    parser.add_argument("--batch", action="store_true",
                        help="Batch mode: auto-scan and merge all unmerged models")
    parser.add_argument("--datasets", type=str, default="",
                        help="Comma-separated dataset filter for batch mode (e.g. 'BRAR,MetaDent')")
    parser.add_argument("--models", type=str, default="",
                        help="Comma-separated model filter for batch mode (e.g. 'Lingshu-32B')")
    parser.add_argument("--sft-root", type=str, default=SFT_ROOT,
                        help="SFT output root directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be merged without actually merging")
    args = parser.parse_args()

    if args.batch:
        # ── Batch Mode ──
        filter_datasets = set(args.datasets.split(",")) if args.datasets else None
        filter_models = set(args.models.split(",")) if args.models else None

        print("")
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  🔄 Batch LoRA Merge                                        ║")
        print(f"║  Root: {args.sft_root}")
        if filter_datasets:
            print(f"║  Datasets: {', '.join(filter_datasets)}")
        if filter_models:
            print(f"║  Models: {', '.join(filter_models)}")
        print("╚══════════════════════════════════════════════════════════════╝")

        tasks = scan_unmerged(args.sft_root, filter_datasets, filter_models)

        if not tasks:
            print("\n✅ All models are already merged! Nothing to do.")
            return

        print(f"\n📋 Found {len(tasks)} model(s) to merge:\n")
        for i, t in enumerate(tasks, 1):
            print(f"  [{i}] {t['dataset']}/{t['tier']}/{t['model']}")
            print(f"      LoRA: {t['lora_path']}")
            print(f"      Dest: {t['merged_dir']}")
            print()

        if args.dry_run:
            print("🏁 Dry-run complete. Use without --dry-run to execute.")
            return

        # Execute merges
        succeeded = []
        failed = []
        total = len(tasks)

        for i, t in enumerate(tasks, 1):
            label = f"{t['dataset']}/{t['tier']}/{t['model']}"
            print(f"\n{'━' * 60}")
            print(f"  [{i}/{total}] Merging: {label}")
            print(f"{'━' * 60}")

            start = time.time()
            try:
                merge_single(
                    lora_path=t["lora_path"],
                    merged_output_dir=t["merged_dir"],
                )
                elapsed = time.time() - start
                print(f"  ✅ {label} — {elapsed:.0f}s")
                succeeded.append(label)
            except Exception as e:
                elapsed = time.time() - start
                print(f"  ❌ {label} — FAILED after {elapsed:.0f}s: {e}")
                failed.append(label)

        # Summary
        print(f"\n{'═' * 60}")
        print(f"  📊 Batch Merge Summary")
        print(f"{'═' * 60}")
        print(f"  Total:     {total}")
        print(f"  ✅ Success: {len(succeeded)}")
        print(f"  ❌ Failed:  {len(failed)}")
        if failed:
            print(f"\n  Failed models:")
            for f in failed:
                print(f"    → {f}")
        print()

    else:
        # ── Config Mode (single or multi-job) ──
        cfg = load_config(args.config)

        device_map = cfg.get("device_map", "auto")
        bf16 = cfg.get("bf16", True)
        trust_remote_code = cfg.get("trust_remote_code", True)

        # Check if config has a 'jobs' list (multi-job mode)
        jobs = cfg.get("jobs", None)

        if jobs and isinstance(jobs, list):
            # ── Multi-job mode ──
            total = len(jobs)
            print(f"\n📋 Config contains {total} merge job(s):\n")
            for i, job in enumerate(jobs, 1):
                name = job.get("name", f"Job {i}")
                print(f"  [{i}] {name}")
                print(f"      LoRA: {job['lora_path']}")
                print(f"      Dest: {job['merged_output_dir']}")
                print()

            succeeded = []
            failed = []

            for i, job in enumerate(jobs, 1):
                name = job.get("name", f"Job {i}")
                lora_path = job["lora_path"]
                merged_output_dir = job["merged_output_dir"]
                base_model_path = job.get("base_model_path", "")

                print(f"\n{'━' * 60}")
                print(f"  [{i}/{total}] Merging: {name}")
                print(f"{'━' * 60}")

                if not os.path.isdir(lora_path):
                    print(f"  ❌ LoRA path not found: {lora_path}")
                    failed.append(name)
                    continue

                start = time.time()
                try:
                    merge_single(
                        lora_path=lora_path,
                        merged_output_dir=merged_output_dir,
                        base_model_path=base_model_path,
                        device_map=device_map,
                        bf16=bf16,
                        trust_remote_code=trust_remote_code,
                    )
                    elapsed = time.time() - start
                    print(f"  ✅ {name} — {elapsed:.0f}s")
                    succeeded.append(name)
                except Exception as e:
                    elapsed = time.time() - start
                    print(f"  ❌ {name} — FAILED after {elapsed:.0f}s: {e}")
                    failed.append(name)

            # Summary
            print(f"\n{'═' * 60}")
            print(f"  📊 Config Merge Summary")
            print(f"{'═' * 60}")
            print(f"  Total:     {total}")
            print(f"  ✅ Success: {len(succeeded)}")
            print(f"  ❌ Failed:  {len(failed)}")
            if failed:
                print(f"\n  Failed:")
                for f in failed:
                    print(f"    → {f}")
            print()

        else:
            # ── Single-job mode (legacy) ──
            lora_path = cfg.get("lora_path")
            merged_output_dir = cfg.get("merged_output_dir")
            base_model_path = cfg.get("base_model_path", "")

            if not lora_path or not merged_output_dir:
                raise ValueError("lora_path and merged_output_dir must be specified in the config")

            if not os.path.isdir(lora_path):
                raise FileNotFoundError(f"LoRA checkpoint directory not found: {lora_path}")

            merge_single(
                lora_path=lora_path,
                merged_output_dir=merged_output_dir,
                base_model_path=base_model_path,
                device_map=device_map,
                bf16=bf16,
                trust_remote_code=trust_remote_code,
            )


if __name__ == "__main__":
    main()
