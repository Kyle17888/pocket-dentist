"""
Few-shot message builder for MetaDent prediction tasks.

A few-shot YAML config has the structure:

    num_shots: 3          # 1, 3, or 5 — how many examples to inject
    vqa:
      examples:
        - image: dataset/MetaDent/output/images/000000001.png
          answer: '{"answer": "B", "reason": "..."}'
        - image: dataset/MetaDent/output/images/000000002.png
          answer: '{"answer": "A", "reason": "..."}'
        ...
    classification:
      examples:
        - image: dataset/MetaDent/output/images/000000001.png
          answer: '[{"id": "C1", "name": "Dental caries", "evidence": "..."}]'
        ...
    captioning:
      examples:
        - image: dataset/MetaDent/output/images/000000001.png
          answer: '{"description": "..."}'
        ...

Each example becomes two turns in the chat history:
  user   → [prompt_text + image]
  assistant → [answer_json_string]
"""

import os
import yaml

from src.utils.common_utils import encode_image


def load_few_shot_config(few_shot_config_path: str, num_shots_override: int = None) -> dict:
    """Load and return the few-shot YAML config as a dict. Returns {} if path is None."""
    if not few_shot_config_path:
        return {}
    if not os.path.exists(few_shot_config_path):
        raise FileNotFoundError(f"Few-shot config not found: {few_shot_config_path}")
    with open(few_shot_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    
    if num_shots_override is not None:
        cfg["num_shots"] = num_shots_override
    return cfg


def build_few_shot_messages(subtask: str, prompt_text: str, few_shot_cfg: dict, model_name: str = "") -> list:
    """
    Build a list of OpenAI-compatible chat messages representing the few-shot examples.

    Args:
        subtask:       "vqa", "classification", "captioning", "code_classification", etc.
        prompt_text:   The same system/task prompt used for the real query (so the
                       model sees consistent instructions in each example turn).
        few_shot_cfg:  Parsed dict from the few-shot YAML file.
        model_name:    Optionally specify model name to add formatting like <image>.

    Supports two image formats in the YAML:
        - image: /path/to/single.jpg          (single image, backward compatible)
        - images: [/path/to/a.jpg, /path/b.jpg]  (multi-image, for COde etc.)

    Returns:
        A list of dicts in OpenAI messages format, e.g.:
          [ {role: user, content: [...]},
            {role: assistant, content: "..."},
            ... ]
        Returns [] when few-shot is disabled or the subtask has no examples.
    """
    if not few_shot_cfg:
        return []

    num_shots = few_shot_cfg.get("num_shots", 0)
    if num_shots == 0:
        return []

    subtask_cfg = few_shot_cfg.get(subtask, {})
    examples = subtask_cfg.get("examples", [])

    # Limit to the requested number of shots
    examples = examples[:num_shots]

    messages = []
    for ex in examples:
        # Support both single-image and multi-image formats
        image_paths = ex.get("images", [])
        if not image_paths:
            single_image = ex.get("image")
            if single_image:
                image_paths = [single_image]

        answer_str = ex.get("answer", "")

        if not image_paths:
            raise FileNotFoundError(
                f"Few-shot example has no image(s) defined "
                f"(subtask={subtask})"
            )

        # Validate all image paths exist
        for img_path in image_paths:
            if not os.path.exists(img_path):
                raise FileNotFoundError(
                    f"Few-shot example image not found: {img_path!r} "
                    f"(defined in few-shot config for subtask={subtask})"
                )

        # User turn: example image(s) + same prompt template
        user_content = []
        for img_path in image_paths:
            ext = os.path.splitext(img_path)[1].lower()
            mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64,{encode_image(img_path)}"
                },
            })

        user_content.append({
            "type": "text",
            "text": prompt_text if "ovis" not in model_name.lower() else f"<image>\n{prompt_text}"
        })

        messages.append({"role": "user", "content": user_content})

        # Assistant turn: the ground-truth answer
        messages.append({"role": "assistant", "content": answer_str})

    return messages

