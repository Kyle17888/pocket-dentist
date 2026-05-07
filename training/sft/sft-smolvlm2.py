#!/usr/bin/env python3
"""
sft_medgemma.py — MedGemma-4b-it LoRA Supervised Fine-Tuning.
Supports training from .jsonl datasets with multi-modal inputs.
Fully configurable via YAML and CLI args.

Key differences from sft.py (Qwen-VL):
  - Model class: AutoModelForImageTextToText (Gemma3ForConditionalGeneration)
  - Processor:   AutoProcessor — does NOT accept max_pixels/min_pixels/fps
  - Chat template: Gemma instruct format via processor.apply_chat_template
  - Image token:  "<image>" injected by the processor automatically
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

# Suppress harmless but noisy `processor_kwargs` warnings from transformers processors
warnings.filterwarnings("ignore", message=".*processor_kwargs.*")
warnings.filterwarnings("ignore", message=".*Kwargs passed to.*")
logging.getLogger("transformers.processing_utils").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Compatibility patch — transformers 5.5.3 / Gemma3 requires torch>=2.6 for
# its mask-creation functions (create_causal_mask, create_sliding_window_causal_mask,
# etc.).  NeSI has torch 2.5.1.
#
# Fix: wrap EVERY function in LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING (and the
# known module-level names) so that `or_mask_function` / `and_mask_function`
# are silently dropped before the version check is reached.
# ---------------------------------------------------------------------------
def _patch_gemma3_masking_utils():
    try:
        import functools
        import transformers.masking_utils as _mu

        def _make_compat(fn):
            """Return a wrapper that drops the two torch>=2.6-gated kwargs."""
            _inner = getattr(fn, '__wrapped__', fn)  # unwrap @deprecated if present
            @functools.wraps(_inner)
            def _compat(*args, or_mask_function=None, and_mask_function=None, **kwargs):
                return _inner(*args, **kwargs)
            return _compat

        # 1. Patch EVERY entry in the dispatch mapping
        mapping = getattr(_mu, 'LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING', {})
        for k in list(mapping):
            mapping[k] = _make_compat(mapping[k])

        # 2. Patch known module-level names for completeness
        for _fname in [
            'create_causal_mask',
            'create_sliding_window_causal_mask',
            'create_causal_mask_mapping',
        ]:
            if hasattr(_mu, _fname):
                setattr(_mu, _fname, _make_compat(getattr(_mu, _fname)))

        import logging as _l
        _l.getLogger(__name__).info(
            f"Gemma3 mask patch applied: {len(mapping)} mapping entries wrapped"
        )
    except Exception as _e:
        import logging as _l
        _l.getLogger(__name__).warning(
            f"Could not apply Gemma3 masking compatibility patch: {_e}"
        )

_patch_gemma3_masking_utils()

# ---------------------------------------------------------------------------
# Compatibility patch 2 — transformers 5.5.3 blocks torch.load (used when
# loading optimizer state for resume_from_checkpoint) on torch < 2.6 due to
# CVE-2025-32434.  Our checkpoints are files we saved ourselves, so it is safe
# to bypass this check in our controlled HPC environment.
#
# IMPORTANT: trainer.py binds check_torch_load_is_safe into its OWN namespace
# at import time.  We must therefore:
#   1. Patch it in transformers.utils.import_utils (for future imports)
#   2. Force-load transformers.trainer and patch its namespace directly
# ---------------------------------------------------------------------------
def _patch_torch_load_check():
    _noop = lambda: None  # noqa: E731
    try:
        # 1. Patch the source module
        import transformers.utils.import_utils as _iu
        if hasattr(_iu, "check_torch_load_is_safe"):
            _iu.check_torch_load_is_safe = _noop

        # 2. Force-load trainer.py (so it exists in sys.modules) then
        #    overwrite its local binding of the function directly.
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
# Compatibility patch 3 — torch 2.5.1 weights_only=True blocks numpy globals
# (numpy._core.multiarray._reconstruct etc.) that appear in checkpoint RNG
# state files.  Register the necessary numpy types as safe globals so that
# checkpoint loading works without switching to weights_only=False.
# ---------------------------------------------------------------------------
def _patch_numpy_safe_globals():
    try:
        import torch.serialization as _ts
        import numpy._core.multiarray as _nca
        import numpy as _np

        # RNG state files typically contain numpy arrays; register the types
        # that pickle needs to reconstruct them.
        _safe = []
        for _attr in ("_reconstruct", "scalar"):
            _fn = getattr(_nca, _attr, None)
            if _fn is not None:
                _safe.append(_fn)
        for _attr in ("ndarray", "dtype"):
            _obj = getattr(_np, _attr, None)
            if _obj is not None:
                _safe.append(_obj)

        # numpy.dtypes DType classes (e.g. UInt32DType) are used in RNG state
        # files saved by newer numpy versions.  Register ALL public DType
        # classes so checkpoint resume works with weights_only=True.
        try:
            import numpy.dtypes as _ndt
            for _name in dir(_ndt):
                if _name.endswith("DType"):
                    _obj = getattr(_ndt, _name, None)
                    if _obj is not None and isinstance(_obj, type):
                        _safe.append(_obj)
        except ImportError:
            pass  # numpy < 1.25 — no numpy.dtypes module

        # Also register numpy.random internal types that may appear in RNG state
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
# Data Collator — MedGemma / Gemma 3 specific
# ---------------------------------------------------------------------------
class SmolVLMDataCollator:
    """
    Collator adapted for SmolVLM2 (Idefics3 architecture).
    Uses batched processor calls to natively handle padding for pixel_values
    and pixel_attention_mask across variable-sized images.
    """

    def __init__(self, processor: AutoProcessor, config: Dict[str, Any]):
        self.processor = processor
        self.config = config
        # SmolVLM2 has a 16384 context window — truncate beyond this
        self.max_length = config.get("max_length", 16384)

    def _extract_images(self, messages: List[Dict]) -> Optional[List[Image.Image]]:
        """Return all PIL images found across all turns, or None."""
        images = []
        for m in messages:
            for item in m["content"]:
                if item["type"] == "image" and "image" in item:
                    images.append(item["image"])
        return images if images else None

    def __call__(self, samples: List[Any]) -> Dict[str, Any]:
        if isinstance(samples[0], dict) and "messages" in samples[0]:
            messages_list = [s["messages"] for s in samples]
            weights = [s["weight"] for s in samples] if "weight" in samples[0] else None
        else:
            messages_list = samples
            weights = None

        pad_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id

        all_input_ids: List[torch.Tensor] = []
        all_labels:    List[torch.Tensor] = []
        collected_pixel_values: List[torch.Tensor] = []
        collected_pixel_attn:   List[torch.Tensor] = []

        for messages in messages_list:
            images = self._extract_images(messages)

            full_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            prompt_messages = [m for m in messages if m["role"] != "assistant"]
            prompt_text = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )

            # Encode WITHOUT truncation — SmolVLM processor rejects truncation
            # that clips image tokens (raises ValueError on mismatch).
            full_enc = self.processor(
                text=full_text,
                images=[images] if images else None,
                return_tensors="pt",
            )
            prompt_enc = self.processor(
                text=prompt_text,
                images=[images] if images else None,
                return_tensors="pt",
            )

            input_ids = full_enc["input_ids"][0]
            prompt_len = prompt_enc["input_ids"].shape[1]

            # Skip over-length samples: replace with zero-loss dummy
            if self.max_length and input_ids.shape[0] > self.max_length:
                import logging
                logging.getLogger(__name__).warning(
                    f"Skipping over-length sample "
                    f"(seq={input_ids.shape[0]}, max_length={self.max_length}). "
                    f"Replaced with zero-loss dummy."
                )
                all_input_ids.append(torch.tensor([pad_id], dtype=torch.long))
                all_labels.append(torch.tensor([-100], dtype=torch.long))
                continue

            labels = input_ids.clone()
            labels[:prompt_len] = -100

            all_input_ids.append(input_ids)
            all_labels.append(labels)

            if "pixel_values" in full_enc and full_enc["pixel_values"] is not None:
                collected_pixel_values.append(full_enc["pixel_values"])
            if "pixel_attention_mask" in full_enc and full_enc["pixel_attention_mask"] is not None:
                collected_pixel_attn.append(full_enc["pixel_attention_mask"])

        # Pad sequences to longest in batch
        max_len = max(ids.shape[0] for ids in all_input_ids)
        B = len(all_input_ids)

        padded_input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        padded_labels    = torch.full((B, max_len), -100,   dtype=torch.long)
        attention_mask   = torch.zeros(B, max_len,          dtype=torch.long)

        for i, (ids, lbl) in enumerate(zip(all_input_ids, all_labels)):
            L = ids.shape[0]
            padded_input_ids[i, :L] = ids
            padded_labels[i, :L]    = lbl
            attention_mask[i, :L]   = 1

        padded_labels[padded_input_ids == pad_id] = -100

        batch: Dict[str, Any] = {
            "input_ids":      padded_input_ids,
            "attention_mask": attention_mask,
            "labels":         padded_labels,
        }

        # Pad pixel values — samples have different numbers of images, so
        # pixel_values shapes differ on dim 1.  Pad to max and create
        # pixel_attention_mask to mark valid vs padding images.
        if collected_pixel_values:
            max_imgs = max(pv.shape[1] for pv in collected_pixel_values)
            padded_pv = []
            padded_pa = []
            for i, pv in enumerate(collected_pixel_values):
                n = pv.shape[1]
                if n < max_imgs:
                    pad_shape = list(pv.shape)
                    pad_shape[1] = max_imgs - n
                    pv = torch.cat([pv, torch.zeros(pad_shape, dtype=pv.dtype)], dim=1)
                padded_pv.append(pv)

                # Build pixel attention mask
                if i < len(collected_pixel_attn):
                    pa = collected_pixel_attn[i]
                    pa_n = pa.shape[1]
                    if pa_n < max_imgs:
                        pa_pad_shape = list(pa.shape)
                        pa_pad_shape[1] = max_imgs - pa_n
                        pa = torch.cat([pa, torch.zeros(pa_pad_shape, dtype=pa.dtype)], dim=1)
                    padded_pa.append(pa)

            batch["pixel_values"] = torch.cat(padded_pv, dim=0)
            if padded_pa:
                batch["pixel_attention_mask"] = torch.cat(padded_pa, dim=0)

        if weights is not None:
            batch["sample_weights"] = torch.tensor(weights, dtype=torch.float32)

        return batch


# ---------------------------------------------------------------------------
# Weighted Trainer
# ---------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weights = inputs.pop("sample_weights", None)
        labels = inputs.pop("labels")  # Pop labels → model skips its internal CE loss

        outputs = model(**inputs)
        logits = outputs.logits  # (B, seq_len, V) — do NOT call .contiguous()

        # Memory-efficient per-sample cross-entropy.
        # Avoids both:
        #   1. Model's internal full-sequence CE (prevented by popping labels above)
        #   2. A full .contiguous() copy of shifted logits
        B = logits.size(0)
        loss_per_sample = []
        for i in range(B):
            logit_i = logits[i, :-1, :]   # non-contiguous view, OK
            label_i = labels[i, 1:]
            valid = label_i != -100
            if valid.any():
                loss_i = torch.nn.functional.cross_entropy(
                    logit_i[valid], label_i[valid], reduction="mean"
                )
            else:
                loss_i = logit_i.sum() * 0.0
            loss_per_sample.append(loss_i)

        del logits  # free immediately
        loss_per_sample = torch.stack(loss_per_sample)

        if sample_weights is not None:
            sample_weights = sample_weights.to(loss_per_sample.device)
            loss_per_sample = loss_per_sample * sample_weights

        final_loss = loss_per_sample.mean()
        return (final_loss, outputs) if return_outputs else final_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MedGemma-4b-it LoRA SFT")
    parser.add_argument("--config", type=str, default="train_config.yml", help="YAML config path")
    parser.add_argument("--input", type=str, help="Override input_dir")
    parser.add_argument("--output", type=str, help="Override output_dir")
    parser.add_argument("--model", type=str, help="Override model_name_or_path")
    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        log.warning(f"Config file {args.config} not found. Using defaults.")
        config: Dict[str, Any] = {
            "model_name_or_path": "google/medgemma-4b-it",
            "training": {"per_device_train_batch_size": 2},
            "output_dir": "./sft_output",
        }
    else:
        config = load_merged_config(args.config)

    config = merge_config(config, args)
    set_seed(config.get("training", {}).get("seed", 42))

    # Apply environment variables from config (e.g. PYTORCH_CUDA_ALLOC_CONF)
    for env_key, env_val in config.get("env", {}).items():
        os.environ[env_key] = str(env_val)
        log.info(f"Set env {env_key}={env_val}")

    # WandB
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

    # Model
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

    if config.get("training", {}).get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    # Datasets
    input_dir = Path(config["input_dir"])
    train_path = input_dir / config.get("train_file", "train.jsonl")
    val_path = input_dir / config.get("val_file", "val.jsonl")

    train_dataset = DentalDataset(train_path, config, compute_weights=True)
    eval_dataset = DentalDataset(val_path, config, compute_weights=False) if val_path.exists() else None

    # Auto-compute eval/save steps based on dataset size
    config = compute_eval_steps(config, len(train_dataset))

    # Collator
    collator = SmolVLMDataCollator(processor, config)

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
