#!/usr/bin/env python3
"""
sft-qwen3_5.py — Qwen3.5-4B LoRA Supervised Fine-Tuning.
Supports training from .jsonl datasets with multi-modal inputs (text + image).
Fully configurable via YAML and CLI args.

Key architecture notes:
  - Qwen3.5 is a unified vision-language model (early fusion)
  - Architecture class: Qwen3_5ForConditionalGeneration
  - Uses Gated DeltaNet (linear attention) + Gated Attention hybrid layers
  - Supports mRoPE (multi-dimensional rotary position embeddings)
  - Vision: patch_size=16, spatial_merge_size=2, temporal_patch_size=2
  - Processor: AutoProcessor with native image/video support
"""

import argparse
import json
import logging
import os
import re
import yaml
import torch
import wandb
from PIL import Image
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

# Suppress harmless but noisy warnings from newer transformers processors
warnings.filterwarnings("ignore", message=".*processor_kwargs.*")
warnings.filterwarnings("ignore", message=".*Kwargs passed to.*")
warnings.filterwarnings("ignore", message=".*You are passing both.*")
logging.getLogger("transformers.processing_utils").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Compatibility patch 1 — torch.load safety check bypass for HPC cluster torch 2.5.1
# ---------------------------------------------------------------------------
def _patch_torch_load_check():
    _noop = lambda: None  # noqa: E731
    try:
        import transformers.utils.import_utils as _iu
        if hasattr(_iu, "check_torch_load_is_safe"):
            _iu.check_torch_load_is_safe = _noop

        import transformers.trainer as _tr
        if hasattr(_tr, "check_torch_load_is_safe"):
            _tr.check_torch_load_is_safe = _noop

        import logging as _l
        _l.getLogger(__name__).info(
            "torch.load safety check bypassed (checkpoint files are self-produced, safe)"
        )
    except Exception as _e:
        import logging as _l
        _l.getLogger(__name__).warning(
            f"Could not patch torch.load safety check: {_e}"
        )

_patch_torch_load_check()

# ---------------------------------------------------------------------------
# Compatibility patch 2 — numpy safe globals for checkpoint RNG state loading
# ---------------------------------------------------------------------------
def _patch_numpy_safe_globals():
    try:
        import torch.serialization as _ts
        import numpy._core.multiarray as _nca
        import numpy as _np

        _safe = []
        for _attr in ("_reconstruct", "scalar"):
            _fn = getattr(_nca, _attr, None)
            if _fn is not None:
                _safe.append(_fn)
        for _attr in ("ndarray", "dtype"):
            _obj = getattr(_np, _attr, None)
            if _obj is not None:
                _safe.append(_obj)

        try:
            import numpy.dtypes as _ndt
            for _name in dir(_ndt):
                if _name.endswith("DType"):
                    _obj = getattr(_ndt, _name, None)
                    if _obj is not None and isinstance(_obj, type):
                        _safe.append(_obj)
        except ImportError:
            pass

        try:
            import numpy.random as _nr
            for _attr in ("RandomState", "Generator", "MT19937", "PCG64", "SeedSequence", "Philox", "SFC64"):
                _obj = getattr(_nr, _attr, None)
                if _obj is not None:
                    _safe.append(_obj)
        except ImportError:
            pass

        _ts.add_safe_globals(_safe)

        import logging as _l
        _l.getLogger(__name__).info(
            f"Registered {len(_safe)} numpy types as torch safe globals for checkpoint loading"
        )
    except Exception as _e:
        import logging as _l
        _l.getLogger(__name__).warning(
            f"Could not register numpy safe globals: {_e}"
        )

_patch_numpy_safe_globals()

# ---------------------------------------------------------------------------
# Imports — use AutoModelForImageTextToText for maximum compatibility
# ---------------------------------------------------------------------------
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from config_utils import load_merged_config, compute_eval_steps


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.input:
        config["input_dir"] = args.input
    if args.output:
        config["output_dir"] = args.output
    if args.model:
        config["model_name_or_path"] = args.model
    return config


def resolve_image_path(base_dir: Path, rel_path: str) -> Path:
    """Find the image file — checks absolute path first, then relative candidates."""
    p = Path(rel_path)
    if p.is_absolute():
        return p
    candidates = [
        base_dir / rel_path,
        base_dir.parent / rel_path,
    ]
    for c in candidates:
        if c.exists():
            return c
    return p  # fallback — will fail gracefully in __getitem__


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DentalDataset(Dataset):
    """Load .jsonl samples; optionally compute per-sample loss weights."""

    def __init__(self, jsonl_path: Path, config: Dict[str, Any], compute_weights: bool = False):
        self.samples: List[Dict] = []
        self.base_dir = jsonl_path.parent
        self.config = config
        self.weights: List[float] = []

        grades_count = {1: 0, 2: 0, 3: 0}

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)

                    if compute_weights:
                        grade = 2  # default
                        for msg in sample.get("messages", []):
                            if msg["role"] == "assistant":
                                text_content = "".join(
                                    item["text"]
                                    for item in msg.get("content", [])
                                    if item["type"] == "text"
                                )
                                m = re.search(r'"grade"\s*:\s*(\d)', text_content)
                                if m:
                                    grade = int(m.group(1))
                        grades_count[grade] = grades_count.get(grade, 0) + 1
                        self.weights.append(grade)

        if compute_weights and self.samples:
            total = sum(grades_count.values())
            num_classes = sum(1 for v in grades_count.values() if v > 0)
            class_weights = {k: total / (num_classes * max(v, 1)) for k, v in grades_count.items()}
            log.info(f"Class counts: {grades_count}")
            log.info(f"Class weights: {class_weights}")
            self.weights = [class_weights.get(w, 1.0) for w in self.weights]

        log.info(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        processed_messages = []
        for msg in sample.get("messages", []):
            role = msg["role"]
            content = []
            for item in msg["content"]:
                if item["type"] == "image":
                    img_path = resolve_image_path(self.base_dir, item["image"])
                    try:
                        img = Image.open(img_path).convert("RGB")
                        content.append({"type": "image", "image": img})
                    except Exception as e:
                        log.warning(f"Failed to load image {img_path}: {e}")
                elif item["type"] == "text":
                    content.append({"type": "text", "text": item["text"]})
            processed_messages.append({"role": role, "content": content})

        res = {"messages": processed_messages}
        if self.weights:
            res["weight"] = self.weights[idx]
        return res


# ---------------------------------------------------------------------------
# Data Collator — Qwen3.5 VLM (batched processor)
# ---------------------------------------------------------------------------
class Qwen35DataCollator:
    """
    Collator for Qwen3.5 unified vision-language model.

    Uses batched processor calls to handle variable-resolution images
    (mRoPE-based dynamic patch sequences). The Qwen3.5 processor natively
    handles padding for pixel_values, image_grid_thw, and attention_mask
    when called in batch mode.
    """

    def __init__(self, processor: AutoProcessor, config: Dict[str, Any]):
        self.processor = processor
        self.config = config

    def __call__(self, samples: List[Any]) -> Dict[str, Any]:
        if isinstance(samples[0], dict) and "messages" in samples[0]:
            messages_list = [s["messages"] for s in samples]
            weights = [s["weight"] for s in samples] if "weight" in samples[0] else None
        else:
            messages_list = samples
            weights = None

        full_texts = []
        prompts = []
        all_images = []

        for messages in messages_list:
            # Full sequence (user + assistant)
            full_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            full_texts.append(full_text)

            # Prompt only (everything before first assistant response)
            prompt_messages = [m for m in messages if m["role"] != "assistant"]
            prompt_text = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt_text)

            # Extract images for this sample
            sample_images = []
            for m in messages:
                for item in m["content"]:
                    if item["type"] == "image":
                        sample_images.append(item["image"])
            all_images.append(sample_images if sample_images else None)

        # Determine if any sample has images
        images_to_pass = all_images if any(x is not None for x in all_images) else None

        vision_cfg = self.config.get("vision", {})

        # Batched encode — the processor natively handles padding for
        # pixel_values, image_grid_thw and attention_mask
        inputs = self.processor(
            text=full_texts,
            images=images_to_pass,
            padding=True,
            return_tensors="pt",
            max_pixels=vision_cfg.get("max_pixels", 1003520),
            min_pixels=vision_cfg.get("min_pixels", 31360),
            fps=vision_cfg.get("fps", 2.0),
        )

        # Create labels — mask prompt tokens and padding tokens
        labels = inputs["input_ids"].clone()

        for i in range(len(full_texts)):
            # Encode prompt with same images to get correct prompt length
            # (image tokens expand prompt length)
            prompt_inputs = self.processor(
                text=[prompts[i]],
                images=[all_images[i]] if all_images[i] else None,
                padding=False,
                return_tensors="pt",
                max_pixels=vision_cfg.get("max_pixels", 1003520),
                min_pixels=vision_cfg.get("min_pixels", 31360),
                fps=vision_cfg.get("fps", 2.0),
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]
            labels[i, :prompt_len] = -100

        # Mask padding tokens
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[inputs["input_ids"] == pad_id] = -100
        inputs["labels"] = labels

        if weights is not None:
            inputs["sample_weights"] = torch.tensor(weights, dtype=torch.float32)

        return inputs


# ---------------------------------------------------------------------------
# Weighted Trainer
# ---------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weights = inputs.pop("sample_weights", None)
        outputs = model(**inputs)

        logits = outputs.logits
        labels = inputs["labels"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size(0), shift_labels.size(1))

        valid_tokens = (shift_labels != -100).float()
        loss_per_sample = (loss * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp(min=1e-5)

        if sample_weights is not None:
            sample_weights = sample_weights.to(loss_per_sample.device)
            loss_per_sample = loss_per_sample * sample_weights

        final_loss = loss_per_sample.mean()
        return (final_loss, outputs) if return_outputs else final_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-4B LoRA SFT")
    parser.add_argument("--config", type=str, default="train_config.yml", help="YAML config path")
    parser.add_argument("--input", type=str, help="Override input_dir")
    parser.add_argument("--output", type=str, help="Override output_dir")
    parser.add_argument("--model", type=str, help="Override model_name_or_path")
    args = parser.parse_args()

    # Load config with base.yaml → dataset.yaml → model.yaml merge chain
    config = load_merged_config(args.config)

    config = merge_config(config, args)
    set_seed(config.get("training", {}).get("seed", 42))

    # WandB — with toggle support
    wb_config = config.get("wandb", {})
    use_wandb = wb_config.get("enabled", True) if wb_config else False
    if use_wandb:
        wb_dir = wb_config.get("dir", "./results/wandb")
        os.makedirs(wb_dir, exist_ok=True)
        # Auto-resume WandB: check for saved run_id in output_dir
        wandb_run_id_file = os.path.join(wb_dir, "wandb_run_id.txt")
        wandb_run_id = None
        if os.path.isfile(wandb_run_id_file):
            wandb_run_id = open(wandb_run_id_file).read().strip()
            log.info(f"Found saved WandB run_id: {wandb_run_id}")

        wb_init_kwargs = dict(
            project=wb_config.get("project", "dental-sft"),
            name=wb_config.get("name", "sft"),
            group=wb_config.get("group"),
            job_type=wb_config.get("job_type", "training"),
            dir=wb_dir,
            config=config,
        )
        # Try to resume existing run; if it was deleted, start fresh
        if wandb_run_id:
            try:
                wandb.init(**wb_init_kwargs, id=wandb_run_id, resume="allow")
            except Exception as e:
                log.warning(f"Cannot resume run {wandb_run_id}: {e}. Starting new run.")
                wandb.init(**wb_init_kwargs)
        else:
            wandb.init(**wb_init_kwargs)

        # Save run_id for future resume
        log.info(f"WandB run_id saved to: {wandb_run_id_file}")
        os.makedirs(os.path.dirname(wandb_run_id_file), exist_ok=True)
        with open(wandb_run_id_file, "w") as f:
            f.write(wandb.run.id)

    # Model — use AutoModelForImageTextToText for architecture auto-detection
    log.info(f"Loading model: {config['model_name_or_path']}")
    dtype = torch.bfloat16 if config.get("use_bf16", True) else torch.float32

    model = AutoModelForImageTextToText.from_pretrained(
        config["model_name_or_path"],
        torch_dtype=dtype,
        device_map=config.get("device", "auto"),
        trust_remote_code=config.get("trust_remote_code", True),
        attn_implementation=config.get("attn_implementation", "eager"),
    )

    processor = AutoProcessor.from_pretrained(
        config["model_name_or_path"],
        trust_remote_code=config.get("trust_remote_code", True),
    )

    # LoRA
    if "lora" in config:
        log.info("Applying LoRA...")
        lora_cfg = config["lora"]
        peft_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Gradient checkpointing — use non-reentrant for hybrid attention models
    if config.get("training", {}).get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Datasets
    input_dir = Path(config["input_dir"])
    train_path = input_dir / config.get("train_file", "train.jsonl")
    val_path = input_dir / config.get("val_file", "val.jsonl")

    train_dataset = DentalDataset(train_path, config, compute_weights=True)
    eval_dataset = DentalDataset(val_path, config, compute_weights=False) if val_path.exists() else None

    # Auto-compute eval/save steps based on dataset size
    config = compute_eval_steps(config, len(train_dataset))

    # Collator
    collator = Qwen35DataCollator(processor, config)

    # Training arguments
    train_args_cfg = config.get("training", {})
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=train_args_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_args_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=train_args_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(train_args_cfg.get("learning_rate", 2e-4)),
        weight_decay=train_args_cfg.get("weight_decay", 0.01),
        num_train_epochs=train_args_cfg.get("num_train_epochs", 3),
        lr_scheduler_type=train_args_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_args_cfg.get("warmup_ratio", 0.1),
        logging_steps=train_args_cfg.get("logging_steps", 10),
        eval_strategy=train_args_cfg.get("evaluation_strategy", "steps"),
        eval_steps=train_args_cfg.get("eval_steps", 200),
        save_strategy=train_args_cfg.get("save_strategy", "steps"),
        save_steps=train_args_cfg.get("save_steps", 200),
        save_total_limit=train_args_cfg.get("save_total_limit", 3),
        bf16=train_args_cfg.get("bf16", True),
        tf32=train_args_cfg.get("tf32", True),
        gradient_checkpointing=train_args_cfg.get("gradient_checkpointing", True),
        logging_dir=f"{config['output_dir']}/logs",
        report_to="wandb" if use_wandb else "none",
        push_to_hub=False,
        remove_unused_columns=False,
        load_best_model_at_end=train_args_cfg.get("load_best_model_at_end", False),
        metric_for_best_model=train_args_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=train_args_cfg.get("greater_is_better", False),
        dataloader_num_workers=train_args_cfg.get("dataloader_num_workers", 4),
        dataloader_pin_memory=train_args_cfg.get("dataloader_pin_memory", True),
    )

    callbacks = []
    if train_args_cfg.get("load_best_model_at_end", False):
        patience = train_args_cfg.get("early_stopping_patience", 5)
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
        log.info(f"Early stopping enabled (patience={patience})")

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks if callbacks else None,
    )

    # Checkpoint resume — auto-detect latest checkpoint in output_dir
    resume_from_checkpoint = None
    if train_args_cfg.get("resume_from_checkpoint", False):
        from transformers.trainer_utils import get_last_checkpoint
        last_ckpt = get_last_checkpoint(config["output_dir"]) if os.path.isdir(config["output_dir"]) else None
        if last_ckpt is not None:
            resume_from_checkpoint = last_ckpt
            log.info(f"Resuming training from checkpoint: {last_ckpt}")
        else:
            log.info("resume_from_checkpoint=true but no checkpoint found in output_dir — training from scratch")

    log.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    log.info(f"Saving model to {config['output_dir']}")
    trainer.save_model(config["output_dir"])
    processor.save_pretrained(config["output_dir"])

    # ── Auto-merge LoRA into base model ──
    # ── Auto-merge LoRA into base model ──
    # For very large models (32B+) this may OOM on CPU RAM.
    # The LoRA adapter is already saved above — use training/model_merge/merge_lora.py
    # on a high-memory CPU node if auto-merge fails.
    try:
        log.info("Auto-merging LoRA weights into base model...")
        merged_output_dir = config.get(
            "merged_output_dir",
            config["output_dir"].replace("/source", "/merged")
        )
        model = model.merge_and_unload()
        model.save_pretrained(merged_output_dir)
        processor.save_pretrained(merged_output_dir)
        log.info(f"Merged model saved to {merged_output_dir}")
    except Exception as e:
        log.warning(f"Auto-merge failed (likely OOM for large model): {e}")
        log.warning("LoRA adapter was saved successfully. Use training/model_merge/merge_lora.py on a high-memory CPU node to merge manually.")

    if use_wandb:
        # Clean up run_id file — training completed successfully, no need to resume
        wandb_run_id_file = os.path.join(wb_dir, "wandb_run_id.txt")
        if os.path.isfile(wandb_run_id_file):
            os.remove(wandb_run_id_file)
            log.info(f"Removed {wandb_run_id_file} (training complete)")
        wandb.finish()


if __name__ == "__main__":
    main()
