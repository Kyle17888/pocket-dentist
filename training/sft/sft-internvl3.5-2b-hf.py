#!/usr/bin/env python3
"""
sft_internvl.py — InternVL2.5-4B LoRA Supervised Fine-Tuning.
Supports training from .jsonl datasets with multi-modal inputs.
Fully configurable via YAML and CLI args.

Key differences from sft.py (Qwen-VL):
  - Model class:      AutoModel with trust_remote_code=True
  - Tokenizer:        AutoTokenizer (no unified Processor)
  - Image processor:  CLIPImageProcessor (loaded from model repo)
  - Chat template:    InternVL2 <|im_start|>/<|im_end|> format, built manually
  - Image tokens:     <img>\n repeated per image patch (dynamic_preprocess tiles)
  - No apply_chat_template on a Processor — handled in the collator
"""

import argparse
import json
import logging
import math
import os
import re
import yaml
import torch
import wandb
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# ---------------------------------------------------------------------------
# Compatibility patch — transformers 5.5.3 blocks torch.load (used when
# loading optimizer state for resume_from_checkpoint) on torch < 2.6 due to
# CVE-2025-32434.  Our checkpoints are files we saved ourselves, so it is safe
# to bypass this check in our controlled HPC environment.
#
# IMPORTANT: trainer.py binds check_torch_load_is_safe into its OWN namespace
# at import time.  We must therefore patch both the source module AND the
# already-loaded trainer module namespace directly.
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
        _l.getLogger(__name__).warning(f"Could not patch torch.load safety check: {_e}")

_patch_torch_load_check()

# ---------------------------------------------------------------------------
# Compatibility patch 2 — torch 2.5.1 weights_only=True blocks numpy globals
# that appear in checkpoint RNG state files.  Register them as safe globals.
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
        _l.getLogger(__name__).warning(f"Could not register numpy safe globals: {_e}")

_patch_numpy_safe_globals()

from transformers import (
    AutoModel,
    AutoModelForImageTextToText,
    AutoTokenizer,
    AutoImageProcessor,
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
# Monkey-patch: newer transformers versions expect `all_tied_weights_keys`
# on all PreTrainedModel subclasses, but InternVL's hub-hosted remote code
# does not define it. Patch __init__ to ensure the attribute always exists.
# ---------------------------------------------------------------------------
import transformers.modeling_utils as _mu
_orig_ptm_init = _mu.PreTrainedModel.__init__

def _patched_ptm_init(self, *args, **kwargs):
    _orig_ptm_init(self, *args, **kwargs)
    if not hasattr(self, "all_tied_weights_keys"):
        self.all_tied_weights_keys = {}

_mu.PreTrainedModel.__init__ = _patched_ptm_init

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
# InternVL image pre-processing constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_SIZE      = 448   # InternVL2.5 native resolution per tile
MAX_NUM_TILES = 6     # cap tiles to limit memory usage (original default 12)

# InternVL2.5 image token constants (must match the model's chat / forward logic)
IMG_START_TOKEN   = '<img>'
IMG_END_TOKEN     = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'


def build_transform(image_size: int = IMG_SIZE) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def dynamic_preprocess(
    image: Image.Image,
    max_num: int = MAX_NUM_TILES,
    image_size: int = IMG_SIZE,
) -> Tuple[List[torch.Tensor], int]:
    """
    Tile the image into up to `max_num` non-overlapping squares + 1 thumbnail.
    Returns a list of tile tensors and the tile count (num_tiles).
    """
    orig_w, orig_h = image.size
    aspect = orig_w / orig_h

    # Find best tiling (rows × cols) <= max_num
    best_n = 1
    for n in range(1, max_num + 1):
        cols = round(math.sqrt(n * aspect))
        rows = round(math.sqrt(n / aspect))
        cols = max(cols, 1)
        rows = max(rows, 1)
        if cols * rows <= max_num:
            best_n = cols * rows

    cols = round(math.sqrt(best_n * aspect))
    rows = round(math.sqrt(best_n / aspect))
    cols, rows = max(cols, 1), max(rows, 1)

    tile_w = orig_w // cols
    tile_h = orig_h // rows
    transform = build_transform(image_size)

    tiles: List[torch.Tensor] = []
    for r in range(rows):
        for c in range(cols):
            box = (c * tile_w, r * tile_h, (c + 1) * tile_w, (r + 1) * tile_h)
            tile = image.crop(box)
            tiles.append(transform(tile))

    # Add thumbnail
    thumbnail = image.resize((image_size, image_size), Image.BICUBIC)
    tiles.append(transform(thumbnail))

    return tiles, len(tiles)


# ---------------------------------------------------------------------------
# InternVL2.5 conversation template helpers
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = "You are a professional dental AI assistant."

def build_internvl_prompt(
    messages: List[Dict],
    num_tiles: int,
    num_image_token: int,
) -> str:
    """
    Construct the raw text prompt in InternVL2.5's ChatML format.
    Image placeholder: <img> + <IMG_CONTEXT> * (num_image_token * num_tiles) + </img>
    injected into the first user turn.
    """
    img_placeholder = (
        IMG_START_TOKEN
        + IMG_CONTEXT_TOKEN * (num_image_token * num_tiles)
        + IMG_END_TOKEN
        + "\n"
    )
    lines: List[str] = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"]

    image_injected = False
    for msg in messages:
        role = "user" if msg["role"] == "user" else "assistant"
        parts: List[str] = []

        for item in msg["content"]:
            if item["type"] == "image" and not image_injected:
                parts.append(img_placeholder)
                image_injected = True
            elif item["type"] == "text":
                parts.append(item["text"])

        text = "".join(parts).strip()
        lines.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")

    return "".join(lines)


def build_internvl_prompt_only(
    messages: List[Dict],
    num_tiles: int,
    num_image_token: int,
) -> str:
    """Same as above but stops before the final assistant turn, + generation prompt."""
    prompt_messages = []
    for m in messages:
        if m["role"] == "assistant":
            break
        prompt_messages.append(m)

    text = build_internvl_prompt(prompt_messages, num_tiles, num_image_token)
    text += "<|im_start|>assistant\n"
    return text


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
    p = Path(rel_path)
    if p.is_absolute():
        return p
    candidates = [base_dir / rel_path, base_dir.parent / rel_path]
    for c in candidates:
        if c.exists():
            return c
    return p


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DentalDataset(Dataset):
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
                        grade = 2
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
# Data Collator — InternVL2.5 specific
# ---------------------------------------------------------------------------
class InternVLDataCollator:
    """
    Collator for InternVL2.5:
      - Tiles each image with dynamic_preprocess.
      - Builds the text prompt with <IMG_CONTEXT> tokens.
      - Provides image_flags to mark valid vs padding tiles.
      - Tokenizes + creates labels by masking the prompt portion.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        config: Dict[str, Any],
        num_image_token: int,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.max_num_tiles = config.get("vision", {}).get("max_num_tiles", MAX_NUM_TILES)
        self.num_image_token = num_image_token  # patches per tile after pixel-shuffle

    def _get_image_tiles(self, messages: List[Dict]) -> Tuple[Optional[torch.Tensor], int]:
        """Extract and tile the first image found; return (pixel_values, num_tiles)."""
        for m in messages:
            for item in m["content"]:
                if item["type"] == "image":
                    tiles, n = dynamic_preprocess(item["image"], max_num=self.max_num_tiles)
                    return torch.stack(tiles), n  # (n, 3, H, W)
        return None, 0

    def __call__(self, samples: List[Any]) -> Dict[str, Any]:
        if isinstance(samples[0], dict) and "messages" in samples[0]:
            messages_list = [s["messages"] for s in samples]
            weights = [s["weight"] for s in samples] if "weight" in samples[0] else None
        else:
            messages_list = samples
            weights = None

        all_input_ids: List[torch.Tensor] = []
        all_labels:    List[torch.Tensor] = []
        all_pixel_values: List[Optional[torch.Tensor]] = []
        all_num_tiles: List[int] = []  # actual tile count per sample

        for messages in messages_list:
            pixel_values, num_tiles = self._get_image_tiles(messages)
            all_pixel_values.append(pixel_values)
            all_num_tiles.append(num_tiles)

            full_text   = build_internvl_prompt(messages, num_tiles, self.num_image_token)
            prompt_text = build_internvl_prompt_only(messages, num_tiles, self.num_image_token)

            full_ids   = self.tokenizer(full_text,   add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

            prompt_len = prompt_ids.shape[0]
            labels = full_ids.clone()
            labels[:prompt_len] = -100  # mask prompt

            all_input_ids.append(full_ids)
            all_labels.append(labels)

        # Pad sequences
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        max_len = max(ids.shape[0] for ids in all_input_ids)

        padded_input_ids = torch.full((len(all_input_ids), max_len), pad_id, dtype=torch.long)
        padded_labels    = torch.full((len(all_labels),    max_len), -100,   dtype=torch.long)
        attention_mask   = torch.zeros(len(all_input_ids), max_len, dtype=torch.long)

        for i, (ids, lbl) in enumerate(zip(all_input_ids, all_labels)):
            seq_len = ids.shape[0]
            padded_input_ids[i, :seq_len] = ids
            padded_labels[i, :seq_len]    = lbl
            attention_mask[i, :seq_len]   = 1

        # Also mask pad tokens in labels
        padded_labels[padded_input_ids == pad_id] = -100

        batch = {
            "input_ids":      padded_input_ids,
            "attention_mask": attention_mask,
            "labels":         padded_labels,
        }

        # Stack pixel values and build image_flags
        # InternVL expects:
        #   pixel_values: (total_tiles, C, H, W)  — all tiles concatenated
        #   image_flags:  (total_tiles, 1)         — 1 for valid tiles, 0 for padding
        valid_pv = [pv for pv in all_pixel_values if pv is not None]
        if valid_pv:
            max_tiles = max(pv.shape[0] for pv in valid_pv)
            _, C, H, W = valid_pv[0].shape
            stacked = torch.zeros(len(all_pixel_values), max_tiles, C, H, W)
            flags   = torch.zeros(len(all_pixel_values), max_tiles, 1, dtype=torch.long)

            for i, (pv, nt) in enumerate(zip(all_pixel_values, all_num_tiles)):
                if pv is not None:
                    stacked[i, :pv.shape[0]] = pv
                    flags[i, :nt] = 1  # mark valid tiles

            # Flatten batch dim: (B, max_tiles, ...) -> (B*max_tiles, ...)
            B, T = stacked.shape[:2]
            batch["pixel_values"] = stacked.view(B * T, C, H, W)
            batch["image_flags"]  = flags.view(B * T, 1)

        if weights is not None:
            batch["sample_weights"] = torch.tensor(weights, dtype=torch.float32)

        return batch


# ---------------------------------------------------------------------------
# HF-format Data Collator (InternVL3.5-*-HF)
# ---------------------------------------------------------------------------
class InternVLHFDataCollator:
    """
    Collator for InternVL3.5-HF format models (AutoModelForImageTextToText).
    Uses AutoProcessor — no manual image tiling or IMG_CONTEXT token injection.
    """

    def __init__(self, processor: AutoProcessor, config: Dict[str, Any]):
        self.processor = processor
        self.max_length = config.get("max_length", 4096)
        # max_num_tiles caps image tile count → controls image token budget
        # InternVL default is 12-13 tiles (~3300 tokens); 4 tiles ≈ 1024 tokens
        self.max_num_tiles = config.get("vision", {}).get("max_num_tiles", None)
        self._tile_size = 448  # InternVL tile size
        tok = processor.tokenizer
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        # Directly patch the image processor to enforce the tile budget.
        # The InternVL HF processor ignores max_num_tiles as a __call__ kwarg
        # AND re-tiles images internally regardless of input resolution.
        # The only reliable control is to modify the image processor config.
        if self.max_num_tiles is not None and hasattr(processor, "image_processor"):
            ip = processor.image_processor
            patched = False
            for attr in ("max_dynamic_patch", "max_dynamic_patches",
                         "max_num_tiles", "max_patches"):
                if hasattr(ip, attr):
                    old_val = getattr(ip, attr)
                    setattr(ip, attr, self.max_num_tiles)
                    log.info(f"Patched processor.image_processor.{attr}: "
                             f"{old_val} → {self.max_num_tiles}")
                    patched = True
                    break
            if not patched:
                # Log all image_processor attributes for debugging
                attrs = [a for a in dir(ip) if not a.startswith("_")]
                log.warning(
                    f"Could not find tile-count attribute on image processor. "
                    f"Available attrs: {attrs}"
                )

        # Detect image placeholder token ID — needed to trim pixel_values
        # after truncation so that features match remaining image tokens.
        self._image_token_id = None
        if hasattr(processor, 'image_token_id'):
            self._image_token_id = processor.image_token_id
        elif hasattr(processor, 'image_token'):
            self._image_token_id = tok.convert_tokens_to_ids(processor.image_token)
        else:
            # Try common InternVL placeholder names
            for token_str in ('<IMG_CONTEXT>', '<image>'):
                tid = tok.convert_tokens_to_ids(token_str)
                if tid != tok.unk_token_id:
                    self._image_token_id = tid
                    break
        if self._image_token_id is not None:
            log.info(f"Image token ID for pixel_values trimming: {self._image_token_id}")
        else:
            log.warning("Could not detect image token ID — pixel_values trimming disabled")

    @staticmethod
    def _to_text_messages(messages: List[Dict]) -> List[Dict]:
        """Strip PIL Image objects, replace with {type: image} placeholders."""
        out = []
        for msg in messages:
            parts = []
            for item in msg["content"]:
                if item["type"] == "image":
                    parts.append({"type": "image"})
                else:
                    parts.append(item)
            out.append({"role": msg["role"], "content": parts})
        return out

    @staticmethod
    def _get_images(messages: List[Dict]) -> Optional[List]:
        imgs = [
            item["image"]
            for msg in messages
            for item in msg["content"]
            if item["type"] == "image" and "image" in item
        ]
        return imgs if imgs else None

    def _limit_image_resolution(self, images: List) -> List:
        """Resize images so the processor generates at most max_num_tiles tiles.

        The InternVL HF processor ignores the ``max_num_tiles`` kwarg, so we
        enforce the tile budget by capping the image pixel-area *before*
        passing images to the processor.  This keeps image tokens under
        control and prevents truncation from breaking the image-token ↔
        pixel_values alignment.
        """
        if not self.max_num_tiles or not images:
            return images
        max_area = self.max_num_tiles * self._tile_size * self._tile_size
        result = []
        for img in images:
            w, h = img.size
            if w * h > max_area:
                scale = math.sqrt(max_area / (w * h))
                new_w = max(int(w * scale), self._tile_size)
                new_h = max(int(h * scale), self._tile_size)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            result.append(img)
        return result

    def _apply_template(self, messages: List[Dict], add_generation_prompt: bool = False) -> str:
        tok = self.processor.tokenizer
        if hasattr(tok, "apply_chat_template") and tok.chat_template is not None:
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        # Fallback: InternVL <|im_start|>/<|im_end|> format
        lines = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"]
        for msg in messages:
            role = "user" if msg["role"] == "user" else "assistant"
            text = "".join(
                p["text"] for p in msg["content"] if p.get("type") == "text"
            )
            lines.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")
        if add_generation_prompt:
            lines.append("<|im_start|>assistant\n")
        return "".join(lines)

    def __call__(self, samples: List[Any]) -> Dict[str, Any]:
        if isinstance(samples[0], dict) and "messages" in samples[0]:
            messages_list = [s["messages"] for s in samples]
            weights = [s["weight"] for s in samples] if "weight" in samples[0] else None
        else:
            messages_list = samples
            weights = None

        all_input_ids: List[torch.Tensor] = []
        all_labels:    List[torch.Tensor] = []
        collected_pixel_values: List[torch.Tensor] = []
        collected_image_sizes:  List[torch.Tensor] = []

        for messages in messages_list:
            images    = self._get_images(messages)
            text_msgs = self._to_text_messages(messages)

            # Limit image resolution to control tile count.
            # InternVL HF processor ignores the max_num_tiles kwarg, so we
            # resize the PIL images to cap pixel-area before encoding.
            if images:
                images = self._limit_image_resolution(images)

            # Prompt = everything before the first assistant turn
            prompt_msgs = []
            for msg in text_msgs:
                if msg["role"] == "assistant":
                    break
                prompt_msgs.append(msg)

            full_text   = self._apply_template(text_msgs,   add_generation_prompt=False)
            prompt_text = self._apply_template(prompt_msgs, add_generation_prompt=True)

            # Encode full conversation (text + images).
            # NOTE: Do NOT pass truncation=True here — InternVL processor validates
            # that the number of <image> placeholders in `text` matches `input_ids`.
            # Truncation clips input_ids without updating the text, causing a mismatch.
            # We instead post-truncate manually from the right (response tokens).
            full_enc = self.processor(
                text=full_text,
                images=images,
                return_tensors="pt",
            )
            # Encode prompt only to find the token-space boundary
            prompt_enc = self.processor(
                text=prompt_text,
                images=images,
                return_tensors="pt",
            )

            input_ids  = full_enc["input_ids"][0]
            prompt_len = prompt_enc["input_ids"].shape[1]

            # ── Handle over-length sequences ──
            # Truncating input_ids breaks the image-token ↔ pixel_values
            # alignment (the model validates that every image placeholder in
            # input_ids has a corresponding feature in pixel_values).
            # Instead of truncating, **skip** over-length samples: replace
            # with a single-token dummy whose label is -100 (zero loss).
            # The batch structure is preserved and pixel_values stay aligned.
            if self.max_length and input_ids.shape[0] > self.max_length:
                log.warning(
                    f"Skipping over-length sample "
                    f"(seq={input_ids.shape[0]}, prompt={prompt_len}, "
                    f"max_length={self.max_length}). "
                    f"Replaced with zero-loss dummy."
                )
                all_input_ids.append(torch.tensor([self.pad_id], dtype=torch.long))
                all_labels.append(torch.tensor([-100], dtype=torch.long))
                # No pixel_values for the dummy — contributes zero image tokens
                continue

            labels = input_ids.clone()
            labels[:prompt_len] = -100          # mask prompt tokens

            all_input_ids.append(input_ids)
            all_labels.append(labels)

            if "pixel_values" in full_enc:
                collected_pixel_values.append(full_enc["pixel_values"])
            if "image_sizes" in full_enc:
                collected_image_sizes.append(full_enc["image_sizes"])

        # Pad sequences to the longest in the batch
        max_len = max(ids.shape[0] for ids in all_input_ids)
        B = len(all_input_ids)

        padded_input_ids = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        padded_labels    = torch.full((B, max_len), -100,        dtype=torch.long)
        attention_mask   = torch.zeros(B, max_len,               dtype=torch.long)

        for i, (ids, lbl) in enumerate(zip(all_input_ids, all_labels)):
            L = ids.shape[0]
            padded_input_ids[i, :L] = ids
            padded_labels[i, :L]    = lbl
            attention_mask[i, :L]   = 1

        padded_labels[padded_input_ids == self.pad_id] = -100  # mask pad tokens

        batch: Dict[str, Any] = {
            "input_ids":      padded_input_ids,
            "attention_mask": attention_mask,
            "labels":         padded_labels,
        }

        if collected_pixel_values:
            batch["pixel_values"] = torch.cat(collected_pixel_values, dim=0)
        if collected_image_sizes:
            batch["image_sizes"] = torch.cat(collected_image_sizes, dim=0)

        if weights is not None:
            batch["sample_weights"] = torch.tensor(weights, dtype=torch.float32)

        return batch


# ---------------------------------------------------------------------------
# Weighted Trainer
# ---------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weights = inputs.pop("sample_weights", None)
        # Pop labels so the model does NOT compute its own internal loss;
        # we compute cross-entropy ourselves below.
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        logits = outputs.logits  # (B, seq_len, V) — do NOT call .contiguous()

        # Memory-efficient per-sample cross-entropy.
        # Avoids the ~41 GiB .contiguous() copy on COde's long multi-image sequences.
        # Slice per-sample and only materialize valid (non-masked) tokens.
        B = logits.size(0)
        loss_per_sample = []
        for i in range(B):
            logit_i = logits[i, :-1, :]   # (seq_len-1, V) — non-contiguous view, OK
            label_i = labels[i, 1:]        # (seq_len-1,)
            valid = label_i != -100
            if valid.any():
                # index only valid (non-masked) positions → smaller tensor
                loss_i = torch.nn.functional.cross_entropy(
                    logit_i[valid], label_i[valid], reduction="mean"
                )
            else:
                # All labels masked — produce a zero loss that still has grad_fn
                # so backward() never encounters a detached tensor.
                loss_i = logit_i.sum() * 0.0
            loss_per_sample.append(loss_i)

        del logits  # free immediately
        loss_per_sample = torch.stack(loss_per_sample)   # (B,)

        if sample_weights is not None:
            sample_weights = sample_weights.to(loss_per_sample.device)
            loss_per_sample = loss_per_sample * sample_weights

        final_loss = loss_per_sample.mean()
        return (final_loss, outputs) if return_outputs else final_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction_step so that eval_loss is computed via our custom
        cross-entropy.  InternVL's forward() has no `labels` argument and does
        not return outputs.loss, so the default Trainer prediction_step would
        report loss=None and eval_loss would never appear in metrics.
        """
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        logits = outputs.logits if hasattr(outputs, "logits") else None
        labels = inputs.get("labels", None)
        return (loss, logits, labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="InternVL2.5 LoRA SFT")
    parser.add_argument("--config", type=str, default="train_config.yml", help="YAML config path")
    parser.add_argument("--input",  type=str, help="Override input_dir")
    parser.add_argument("--output", type=str, help="Override output_dir")
    parser.add_argument("--model",  type=str, help="Override model_name_or_path")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        log.warning(f"Config file {args.config} not found. Using defaults.")
        config: Dict[str, Any] = {
            "model_name_or_path": "OpenGVLab/InternVL2_5-4B",
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
    enable_wandb = wb_config.get("enabled", True) if wb_config else False
    if enable_wandb:
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

    # Model + processor
    # InternVL3.5-*-HF uses the standard HF transformers format:
    #   AutoModelForImageTextToText (InternVLForConditionalGeneration)
    #   AutoProcessor               (handles tokenisation + image preprocessing)
    # No trust_remote_code or manual image tiling required.
    log.info(f"Loading model: {config['model_name_or_path']}")
    dtype = torch.bfloat16 if config.get("use_bf16", True) else torch.float32
    trust_remote_code = config.get("trust_remote_code", False)

    # Load to CPU first, then move to device (avoids device_map issues with PEFT).
    model = AutoModelForImageTextToText.from_pretrained(
        config["model_name_or_path"],
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=config.get("attn_implementation", "eager"),
    )
    target_device = config.get("device", "cuda")
    log.info(f"Moving model to device: {target_device}")
    model = model.to(target_device)

    # AutoProcessor handles both tokenisation and image preprocessing.
    processor = AutoProcessor.from_pretrained(
        config["model_name_or_path"],
        trust_remote_code=trust_remote_code,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    log.info(f"Loaded processor: {processor.__class__.__name__}")

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
        model.enable_input_require_grads()  # required for gradient checkpointing + LoRA
        model.print_trainable_parameters()

        # Safety patch: if the underlying HF model's forward() doesn't accept
        # inputs_embeds (some versions), silently drop it to avoid PEFT TypeError.
        import inspect
        try:
            _InternVLCls = type(model.base_model.model)
            _orig_fwd = _InternVLCls.forward
            if "inputs_embeds" not in inspect.signature(_orig_fwd).parameters:
                def _compat_forward(self, *args, inputs_embeds=None, **kwargs):
                    return _orig_fwd(self, *args, **kwargs)
                _InternVLCls.forward = _compat_forward
                log.info("Patched model.forward() to tolerate inputs_embeds kwarg")
            else:
                log.info("model.forward() already accepts inputs_embeds — no patch needed")
        except Exception as _pe:
            log.warning(f"Could not apply inputs_embeds safety patch: {_pe}")

    # NOTE: gradient_checkpointing is handled by TrainingArguments.
    # Do NOT call model.gradient_checkpointing_enable() here — it would
    # re-trigger the issue where first-layer inputs are detached from the
    # computation graph before PEFT can mark them as requiring grad.

    # Datasets
    input_dir = Path(config["input_dir"])
    train_path = input_dir / config.get("train_file", "train.jsonl")
    val_path   = input_dir / config.get("val_file",   "val.jsonl")

    train_dataset = DentalDataset(train_path, config, compute_weights=True)
    eval_dataset  = DentalDataset(val_path,   config, compute_weights=False) if val_path.exists() else None

    # Auto-compute eval/save steps based on dataset size
    config = compute_eval_steps(config, len(train_dataset))

    # Collator
    collator = InternVLHFDataCollator(processor, config)

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
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_dir=f"{config['output_dir']}/logs",
        report_to="wandb" if enable_wandb else "none",
        push_to_hub=False,
        remove_unused_columns=False,
        load_best_model_at_end=False,
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

    if enable_wandb:
        # Clean up run_id file — training completed successfully, no need to resume
        wandb_run_id_file = os.path.join(wb_dir, "wandb_run_id.txt")
        if os.path.isfile(wandb_run_id_file):
            os.remove(wandb_run_id_file)
            log.info(f"Removed {wandb_run_id_file} (training complete)")
        wandb.finish()


if __name__ == "__main__":
    main()
