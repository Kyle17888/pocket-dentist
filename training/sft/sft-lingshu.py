#!/usr/bin/env python3
"""
sft-lingshu.py — Lingshu-32B LoRA Supervised Fine-Tuning.
Supports training from .jsonl datasets with multi-modal inputs.
Fully configurable via YAML and CLI args.

Key architecture notes:
  - Base: Qwen2.5-VL architecture (Qwen2_5_VLForConditionalGeneration)
  - Built on top of Qwen2.5-VL, fine-tuned for medical domain
  - Processor: AutoProcessor with max_pixels/min_pixels/fps support
  - Chat template: Qwen-VL format via processor.apply_chat_template
  - Source: https://huggingface.co/lingshu-medical-mllm/Lingshu-32B
"""

import argparse
import json
import logging
import os
import sys
import yaml
import torch
import wandb
import re
from PIL import Image
from pathlib import Path
from typing import Dict, List, Any

# transformers 5.5.3: top-level __init__.py does not export model classes directly;
# import from submodule path to bypass this limitation.
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration as Qwen2_5VLForConditionalGeneration
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
from torch.utils.data import Dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def set_seed(seed=42):
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
    """Merge CLI arguments into the config dictionary."""
    if args.input:
        config["input_dir"] = args.input
    if args.output:
        config["output_dir"] = args.output
    if args.model:
        config["model_name_or_path"] = args.model
    return config

def resolve_image_path(base_dir: Path, rel_path: str) -> Path:
    """Find the image path relative to the dataset or absolute."""
    p = Path(rel_path)
    if p.is_absolute():
        return p
    
    candidates = [
        base_dir / rel_path,
        base_dir.parent / rel_path,
        # Common project location if relative to data root
        Path("<DATA_ROOT>/BRAR") / rel_path
    ]
    for c in candidates:
        if c.exists():
            return c
    return p

class DentalDataset(Dataset):
    def __init__(self, jsonl_path: Path, config: Dict[str, Any], compute_weights: bool = False):
        self.samples = []
        self.base_dir = jsonl_path.parent
        self.config = config
        self.weights = []
        
        grades_count = {1: 0, 2: 0, 3: 0}
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    self.samples.append(sample)
                    
                    if compute_weights:
                        grade = 2
                        for msg in sample.get("messages", []):
                            if msg["role"] == "assistant":
                                text_content = ""
                                for item in msg.get("content", []):
                                    if item["type"] == "text":
                                        text_content += item["text"]
                                match = re.search(r'"grade"\s*:\s*(\d)', text_content)
                                if match:
                                    grade = int(match.group(1))
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
        messages = sample.get("messages", [])
        
        # Prepare content, loading images
        processed_messages = []
        for msg in messages:
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

class QwenVLDataCollator:
    def __init__(self, processor, config):
        self.processor = processor
        self.config = config

    def __call__(self, samples: List[Any]) -> Dict[str, Any]:
        # samples is a list of processed_messages lists or dicts with weights
        full_texts = []
        prompts = []
        all_images = []
        
        if isinstance(samples[0], dict) and "messages" in samples[0]:
            messages_list = [s["messages"] for s in samples]
            weights = [s["weight"] for s in samples] if "weight" in samples[0] else None
        else:
            messages_list = samples
            weights = None
            
        for messages in messages_list:
            # Full sequence
            full_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            full_texts.append(full_text)
            
            # Prompt sequence (everything up to final assistant response)
            prompt_messages = []
            for m in messages:
                if m["role"] == "assistant":
                    break
                prompt_messages.append(m)
            
            prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt_text)
            
            # Extract images for this sample
            sample_images = []
            for m in messages:
                for item in m["content"]:
                    if item["type"] == "image":
                        sample_images.append(item["image"])
            all_images.append(sample_images if sample_images else None)

        # 2. Process multi-modal inputs
        # If all_images is all None, pass None
        images_to_pass = all_images if any(x is not None for x in all_images) else None
        
        vision_cfg = self.config.get("vision", {})
        inputs = self.processor(
            text=full_texts,
            images=images_to_pass,
            padding=True,
            return_tensors="pt",
            max_pixels=vision_cfg.get("max_pixels", 1003520),
            min_pixels=vision_cfg.get("min_pixels", 31360),
            fps=vision_cfg.get("fps", 2.0)
        )

        # 3. Create labels (masking prompts)
        labels = inputs["input_ids"].clone()
        
        for i in range(len(full_texts)):
            # To find prompt length correctly, we MUST process the prompt with the SAME processor settings
            # and the SAME images (because images expand to multiple tokens)
            prompt_inputs = self.processor(
                text=[prompts[i]],
                images=[all_images[i]] if all_images[i] else None,
                padding=False,
                return_tensors="pt",
                max_pixels=vision_cfg.get("max_pixels", 1003520),
                min_pixels=vision_cfg.get("min_pixels", 31360),
                fps=vision_cfg.get("fps", 2.0)
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]
            # Mask until the end of the prompt
            labels[i, :prompt_len] = -100

        # Mask padding tokens
        labels[inputs["input_ids"] == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        
        if weights is not None:
            inputs["sample_weights"] = torch.tensor(weights, dtype=torch.float32)
            
        return inputs

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

def main():
    parser = argparse.ArgumentParser(description="Lingshu-32B LoRA SFT")
    parser.add_argument("--config", type=str, default="train_config.yml", help="YAML config path")
    parser.add_argument("--input", type=str, help="Override input directory path (folder with train.jsonl/val.jsonl)")
    parser.add_argument("--output", type=str, help="Override output directory")
    parser.add_argument("--model", type=str, help="Override base model path")
    args = parser.parse_args()

    # Load and merge config
    if not os.path.exists(args.config):
        log.warning(f"Config file {args.config} not found. Using defaults.")
        config = {
            "model_name_or_path": "lingshu-medical-mllm/Lingshu-32B",
            "training": {"per_device_train_batch_size": 2},
            "output_dir": "./sft_output"
        }
    else:
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

    # Load model and processor
    log.info(f"Loading model: {config['model_name_or_path']}")
    dtype = torch.bfloat16 if config.get("use_bf16", True) else torch.float32
    
    model = Qwen2_5VLForConditionalGeneration.from_pretrained(
        config["model_name_or_path"],
        torch_dtype=dtype,
        device_map=config.get("device", "auto"),
        trust_remote_code=True,
        attn_implementation=config.get("attn_implementation", "eager"),
    )
    
    processor = AutoProcessor.from_pretrained(
        config["model_name_or_path"],
        trust_remote_code=True
    )

    # Prepare LoRA
    if "lora" in config:
        log.info("Applying LoRA...")
        lora_cfg = config["lora"]
        peft_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if config.get("training", {}).get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    # torch.compile for speedup (~15-25% on H100/A100, no effect on results)
    if config.get("training", {}).get("torch_compile", False):
        log.info("Applying torch.compile (mode=reduce-overhead)...")
        model = torch.compile(model, mode="reduce-overhead")

    # Load datasets
    input_dir = Path(config["input_dir"])
    train_path = input_dir / config.get("train_file", "train.jsonl")
    val_path = input_dir / config.get("val_file", "val.jsonl")
    
    train_dataset = DentalDataset(train_path, config, compute_weights=True)
    eval_dataset = DentalDataset(val_path, config, compute_weights=False) if val_path.exists() else None

    # Data collator
    # Auto-compute eval/save steps based on dataset size
    config = compute_eval_steps(config, len(train_dataset))

    collator = QwenVLDataCollator(processor, config)

    # Training arguments
    train_args_cfg = config.get("training", {})
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=train_args_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_args_cfg.get("per_device_eval_batch_size", 2),
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
        remove_unused_columns=False,  # Important for multi-modal
        load_best_model_at_end=train_args_cfg.get("load_best_model_at_end", False),
        metric_for_best_model=train_args_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=train_args_cfg.get("greater_is_better", False),
        dataloader_num_workers=train_args_cfg.get("dataloader_num_workers", 4),
        dataloader_pin_memory=train_args_cfg.get("dataloader_pin_memory", True),
    )

    # Early stopping callback (only if load_best_model_at_end is enabled)
    callbacks = []
    if train_args_cfg.get("load_best_model_at_end", False):
        patience = train_args_cfg.get("early_stopping_patience", 5)
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
        log.info(f"Early stopping enabled (patience={patience})")

    # Trainer
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

    # Save final model
    log.info(f"Saving model to {config['output_dir']}")
    trainer.save_model(config["output_dir"])
    processor.save_pretrained(config["output_dir"])

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
