#!/usr/bin/env python3
"""
01_build_jsonl.py — Build MetaDent baseline JSONL from raw input data.

Reads:
  - input/labels/bench/vqa.json          (2,588 images × ~7 QA each)
  - input/labels/bench/classification.json (2,588 images × 1 each)
  - input/labels/bench/captioning.json     (2,588 images × 1 each)
  - input/images/data/*.parquet            (for source mapping: DS1/DS2/DS3)

Outputs (baseline — no sampling, all data preserved):
  - <output_dir>/train.jsonl   (2,068 images)
  - <output_dir>/val.jsonl     (259 images)
  - <output_dir>/test.jsonl    (261 images)
  - <output_dir>/images/       (extracted images from parquet, if --extract_images)

This is the canonical baseline JSONL with all VQA questions preserved (1:1 with bench).
Use 02_balance_for_sft.py to create a balanced version for SFT training.

Usage:
  python 01_build_jsonl.py \
      --input_dir  /path/to/MetaDent/input \
      --output_dir /path/to/MetaDent/output \
      [--extract_images]   # also extract images from parquet
      [--seed 42]
"""

import argparse
import glob
import hashlib
import json
import os
import random
import string

import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# Prompt templates (must match the production inference pipeline)
# ─────────────────────────────────────────────────────────────

VQA_PROMPT_TEMPLATE = string.Template("""\
You are a professional dentist. You are now presented with a clinical image of a patient and a multiple-choice question.
Please select only one correct answer based on the visual evidence from the image.

Below is the question:
[Question]:
```json
{
    "question": $question,
    "choice": $choice
}
```

Your output should be a JSON object with the following keys:
- "answer": Your selected option, represented as one of $answer_options.
- "reason": The reasoning and supporting visual evidence for your chosen answer.

Do not include any additional explanations, text, or formatting outside the JSON.

[Output Template]:
```json
{
    "answer": <fill in the selected answer as specified above>,
    "reason": <fill in the reasoning and supporting evidence as specified above>
}
```

 Question: $question_text \
 Choice: $choice_text \
""")

CAPTIONING_PROMPT = """\
You are a professional dentist. You are now given a clinical image of a patient. Please generate a detailed and vivid natural language description based on this image.

Your output should be a JSON object with one key, "description". The description must be written as a coherent paragraph, not as a list or dictionary. It should clearly and naturally describe the imaging direction (the angle or orientation from which the image was captured), the main subject of the image (the primary anatomical focus or structure), and all observed abnormalities (including pathological findings, dental defects, or visible dental instruments related to these abnormalities). You may also describe regions that appear normal if you are confident in their correctness, but your descriptions must remain accurate and factual, without any fabricated or speculative details. Use multiple sentences as needed to make the description fluent, expressive, and clinically meaningful.

[Output Template]:
```json
{
    "description": "<fill in the detailed description including observed content, dental instruments, and abnormal findings>"
}
```
Please output json directly without extra output.
"""

CLASSIFICATION_PROMPT = """\
You are a professional dentist. You are now given a clinical image of a patient.
Please perform multi-class category extraction based on this dental clinical image.

The categories and additional information are as follows:
[Categories]:
{
    "C1": {
        "name": "Dental caries",
        "note": "Clearly visible dental caries; early white-spot lesions are excluded."
    },
    "C2": {
        "name": "Non-carious, unrestored tooth defect",
        "note": "Refers to tooth fractures or cervical defects not caused by caries and not yet restored (e.g. wedge-shaped defects or notching). Excludes physiological or pathological tooth wear."
    },
    "C3": {
        "name": "Tooth wear or erosion",
        "note": "Loss of tooth structure due to physiological or pathological wear, or erosion. Defects caused by caries or minor enamel cracks are excluded."
    },
    "C4": {
        "name": "Gingival inflammation",
        "note": "Gingival redness and swelling, may present with or without bleeding, and may or may not be accompanied by alveolar bone resorption."
    },
    "C5": {
        "name": "Gingival recession",
        "note": "Recession of the gingival margin due to physiological or pathological causes, resulting in root exposure or visible black triangles (interdental gingival recession)."
    },
    "C6": {
        "name": "Dental plaque or calculus",
        "note": "Visible accumulation of plaque or calculus. Excludes occasional food debris."
    },
    "C7": {
        "name": "Tooth discoloration",
        "note": "Abnormal tooth color caused by staining, fluorosis, or pulp necrosis, as well as chalky white spots due to enamel demineralization. Excludes dark discoloration due to caries."
    },
    "C8": {
        "name": "Partial edentulism",
        "note": "One or more missing teeth with no residual roots present and no prosthetic replacement."
    },
    "C9": {
        "name": "Residual root",
        "note": "Complete loss of the clinical crown, with only the root portion remaining."
    },
    "C10": {
        "name": "Dental filling (direct filling)",
        "note": "Includes various direct restorative materials on tooth surfaces, such as composite resin, amalgam, temporary fillings, or gutta-percha."
    },
    "C11": {
        "name": "Fixed prosthesis",
        "note": "Includes crowns, bridges, veneers, inlays, and other fixed dental prostheses."
    },
    "C12": {
        "name": "Removable denture",
        "note": "Includes partial and complete removable dentures."
    },
    "C13": {
        "name": "Interdental spacing",
        "note": "Presence of spaces between teeth without missing teeth, possibly due to diastema or physiological spacing. Excludes black triangles caused by gingival recession when adjacent teeth are in contact."
    },
    "C14": {
        "name": "Malocclusion or dental malalignment",
        "note": "Includes individual or generalized tooth rotation, crowding, or displacement. The presence of orthodontic appliances does not necessarily indicate malalignment."
    },
    "C15": {
        "name": "Conventional orthodontic appliance",
        "note": "Includes brackets, archwires, elastics, and other conventional orthodontic materials."
    },
    "C16": {
        "name": "Clear aligner orthodontic appliance",
        "note": "Includes clear aligners, attachments, retainers, and other components of invisible orthodontic systems."
    },
    "C17": {
        "name": "Oral ulcer",
        "note": "Includes recurrent aphthous ulcers and traumatic ulcers. Excludes gingival redness or swelling caused by periodontal inflammation."
    },
    "C18": {
        "name": "Oral wound",
        "note": "Includes extraction sockets, trauma-related wounds, or surgical wounds of the oral tissues. Excludes gingival redness or bleeding caused by gingivitis."
    }
}

Your output should be a JSON array, where each element is a dictionary containing the following keys:
- "id": The category ID (e.g., "C1", "C2", etc.)
- "name": The category name
- "evidence": The evidence or visual cues observed in the image that support the classification into this category.

Important requirements:
- Only select categories that are visibly present in the image. Do not select or provide explanations for categories that cannot be seen.
- The "id" and "name" must strictly match and correspond to the given categories.
- You must only choose from the listed categories.
- It is acceptable to output an empty array if no categories apply.

[Output Template]:
```json
[
    <fill in the extracted categories as specified above>
]
```
"""

# Category ID → Name mapping (for classification GT)
CATEGORY_NAMES = {
    "C1": "Dental caries", "C2": "Non-carious, unrestored tooth defect",
    "C3": "Tooth wear or erosion", "C4": "Gingival inflammation",
    "C5": "Gingival recession", "C6": "Dental plaque or calculus",
    "C7": "Tooth discoloration", "C8": "Partial edentulism",
    "C9": "Residual root", "C10": "Dental filling (direct filling)",
    "C11": "Fixed prosthesis", "C12": "Removable denture",
    "C13": "Interdental spacing", "C14": "Malocclusion or dental malalignment",
    "C15": "Conventional orthodontic appliance",
    "C16": "Clear aligner orthodontic appliance",
    "C17": "Oral ulcer", "C18": "Oral wound",
}


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_source_mapping(input_dir: str) -> dict[str, str]:
    """Load filename → source (DS1/DS2/DS3) mapping from parquet files."""
    parquet_dir = os.path.join(input_dir, "images", "data")
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not parquet_files:
        print(f"⚠️ No parquet files found in {parquet_dir}. Source field will be empty.")
        return {}

    dfs = [pd.read_parquet(f, columns=["filename", "source"]) for f in parquet_files]
    df = pd.concat(dfs)
    return dict(zip(df["filename"], df["source"]))


def load_bench_data(input_dir: str) -> dict:
    """Load bench JSON files: vqa, classification, captioning."""
    bench_dir = os.path.join(input_dir, "labels", "bench")
    data = {}
    for name in ["vqa", "classification", "captioning"]:
        path = os.path.join(bench_dir, f"{name}.json")
        with open(path, "r", encoding="utf-8") as f:
            data[name] = json.load(f)
    return data


# ─────────────────────────────────────────────────────────────
# Train / Val / Test split
# ─────────────────────────────────────────────────────────────

def split_images(all_images: list[str], seed: int = 42,
                 test_ratio: float = 0.10, val_ratio: float = 0.10) -> dict[str, list[str]]:
    """
    Split image IDs into train/val/test.
    Uses deterministic hashing so the split is reproducible regardless of ordering.
    """
    # Sort + deterministic shuffle for reproducibility
    rng = random.Random(seed)
    images = sorted(all_images)
    rng.shuffle(images)

    n = len(images)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "test": images[:n_test],
        "val": images[n_test:n_test + n_val],
        "train": images[n_test + n_val:],
    }
    return splits


# ─────────────────────────────────────────────────────────────
# JSONL record builders
# ─────────────────────────────────────────────────────────────

_global_id = 0

def _next_id():
    global _global_id
    _global_id += 1
    return _global_id


def build_vqa_records(image_id: str, questions: list[dict], split: str,
                      source: str) -> list[dict]:
    """Build JSONL records for VQA questions of one image."""
    records = []
    for q in questions:
        # Format choice text
        choice = q["choice"]
        if isinstance(choice, dict):
            choice_text = ", ".join(f"{k}: {v}" for k, v in choice.items())
            answer_options = list(choice.keys())
        else:
            choice_text = str(choice)
            answer_options = ["A", "B", "C", "D"]

        prompt = VQA_PROMPT_TEMPLATE.substitute(
            question=json.dumps(q["question"], ensure_ascii=False),
            choice=json.dumps(choice, ensure_ascii=False),
            answer_options=json.dumps(answer_options),
            question_text=q["question"],
            choice_text=choice_text,
        )

        gt = json.dumps({"answer": q["answer"], "reason": q.get("reason", "")},
                        ensure_ascii=False)

        records.append({
            "id": _next_id(),
            "task": "vqa",
            "split": split,
            "source": source,
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": f"images/{image_id}.jpg"},
                    {"type": "text", "text": prompt},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": gt},
                ]},
            ],
        })
    return records


def build_captioning_record(image_id: str, cap_data: dict, split: str,
                            source: str) -> dict:
    """Build a single captioning JSONL record."""
    gt = json.dumps(cap_data, ensure_ascii=False)
    return {
        "id": _next_id(),
        "task": "captioning",
        "split": split,
        "source": source,
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": f"images/{image_id}.jpg"},
                {"type": "text", "text": CAPTIONING_PROMPT},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": gt},
            ]},
        ],
    }


def build_classification_record(image_id: str, cls_labels: list[str], split: str,
                                source: str) -> dict:
    """Build a single classification JSONL record."""
    gt_list = [
        {"id": cid, "name": CATEGORY_NAMES.get(cid, cid), "evidence": ""}
        for cid in cls_labels
    ]
    gt = json.dumps(gt_list, ensure_ascii=False)
    return {
        "id": _next_id(),
        "task": "classification",
        "split": split,
        "source": source,
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": f"images/{image_id}.jpg"},
                {"type": "text", "text": CLASSIFICATION_PROMPT},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": gt},
            ]},
        ],
    }


# ─────────────────────────────────────────────────────────────
# Image extraction
# ─────────────────────────────────────────────────────────────

def extract_images(input_dir: str, output_dir: str):
    """Extract images from parquet files to output_dir/images/."""
    parquet_dir = os.path.join(input_dir, "images", "data")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    print(f"Extracting images from {len(parquet_files)} parquet files...")

    total = 0
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=os.path.basename(pf)):
            out_path = os.path.join(images_dir, row["filename"])
            if os.path.exists(out_path):
                continue
            img = row["image"]
            if img and "bytes" in img:
                with open(out_path, "wb") as f:
                    f.write(img["bytes"])
                total += 1
    print(f"✅ Extracted {total} new images to {images_dir}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build MetaDent baseline JSONL")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to MetaDent/input/")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to MetaDent/output/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extract_images", action="store_true",
                        help="Also extract images from parquet files")
    args = parser.parse_args()

    # 1. Load source mapping
    source_map = load_source_mapping(args.input_dir)
    print(f"Source mapping: {len(source_map)} images")

    # 2. Load bench data
    bench = load_bench_data(args.input_dir)
    all_images = sorted(bench["vqa"].keys())
    print(f"Bench images: {len(all_images)}")

    # 3. Split images
    splits = split_images(all_images, seed=args.seed)
    for name, imgs in splits.items():
        print(f"  {name}: {len(imgs)} images")

    # 4. Extract images (optional)
    if args.extract_images:
        extract_images(args.input_dir, args.output_dir)

    # 5. Build JSONL records
    global _global_id
    _global_id = 0

    split_records = {"train": [], "val": [], "test": []}
    for split_name, image_ids in splits.items():
        for img_id in image_ids:
            source = source_map.get(f"{img_id}.jpg", "")

            # VQA — all questions
            if img_id in bench["vqa"]:
                records = build_vqa_records(img_id, bench["vqa"][img_id],
                                           split_name, source)
                split_records[split_name].extend(records)

            # Captioning
            if img_id in bench["captioning"]:
                rec = build_captioning_record(img_id, bench["captioning"][img_id],
                                             split_name, source)
                split_records[split_name].append(rec)

            # Classification
            if img_id in bench["classification"]:
                rec = build_classification_record(img_id, bench["classification"][img_id],
                                                  split_name, source)
                split_records[split_name].append(rec)

    # 6. Write output
    os.makedirs(args.output_dir, exist_ok=True)
    for split_name, records in split_records.items():
        out_file = os.path.join(args.output_dir, f"{split_name}.jsonl")
        with open(out_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Task breakdown
        tasks = {}
        for r in records:
            tasks[r["task"]] = tasks.get(r["task"], 0) + 1
        print(f"✅ {out_file}: {len(records)} records — {tasks}")

    print("\nDone! Baseline JSONL files created.")


if __name__ == "__main__":
    main()
