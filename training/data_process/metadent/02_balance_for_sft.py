#!/usr/bin/env python3
"""
02_balance_for_sft.py — Balance MetaDent baseline JSONL for SFT LoRA training.

Reads baseline JSONL (from 01_build_jsonl.py, all VQA questions preserved)
and produces a balanced version where VQA is subsampled to ~2 questions/image,
while captioning and classification remain unchanged (already 1/image).

The test set is kept IDENTICAL — only train and val are rebalanced.

Rationale:
  Baseline train has ~7 VQA / 1 cap / 1 cls per image → model biases toward VQA.
  Balanced version reduces to ~2 VQA / 1 cap / 1 cls for more uniform SFT.

Usage:
  python 02_balance_for_sft.py \
      --input_dir  /path/to/MetaDent/output   (contains train.jsonl, val.jsonl, test.jsonl) \
      --output_dir /path/to/MetaDent/output   (writes train_sft.jsonl, val_sft.jsonl, test_sft.jsonl) \
      [--max_vqa_per_image 2]                 (default: 2, cap VQA questions per image) \
      [--seed 42]
"""

import argparse
import json
import os
import random
from collections import defaultdict


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: str):
    """Write a list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def get_image_id(record: dict) -> str:
    """Extract image filename from a record's user message."""
    for part in record["messages"][0]["content"]:
        if part["type"] == "image":
            return part["image"]
    return ""


def balance_split(records: list[dict], max_vqa_per_image: int,
                  rng: random.Random) -> list[dict]:
    """
    Balance a split by subsampling VQA questions.

    Strategy:
      - Keep all captioning and classification records unchanged
      - For VQA: randomly sample up to max_vqa_per_image questions per image
      - Shuffle the final result for training diversity
    """
    non_vqa = [r for r in records if r["task"] != "vqa"]
    vqa = [r for r in records if r["task"] == "vqa"]

    # Group VQA by image
    vqa_by_image = defaultdict(list)
    for r in vqa:
        img = get_image_id(r)
        vqa_by_image[img].append(r)

    # Subsample VQA
    sampled_vqa = []
    for img, img_records in vqa_by_image.items():
        if len(img_records) <= max_vqa_per_image:
            sampled_vqa.extend(img_records)
        else:
            sampled_vqa.extend(rng.sample(img_records, max_vqa_per_image))

    # Combine and shuffle
    balanced = non_vqa + sampled_vqa
    rng.shuffle(balanced)

    return balanced


def main():
    parser = argparse.ArgumentParser(
        description="Balance MetaDent baseline → SFT-balanced")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing baseline train.jsonl/val.jsonl/test.jsonl")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as input_dir)")
    parser.add_argument("--max_vqa_per_image", type=int, default=2,
                        help="Max VQA questions per image (default: 2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir
    os.makedirs(args.output_dir, exist_ok=True)

    rng = random.Random(args.seed)

    for split in ["train", "val", "test"]:
        src_file = os.path.join(args.input_dir, f"{split}.jsonl")
        out_file = os.path.join(args.output_dir, f"{split}_sft.jsonl")

        if not os.path.exists(src_file):
            print(f"⚠️ {src_file} not found, skipping.")
            continue

        records = load_jsonl(src_file)

        # Count original stats
        orig_tasks = {}
        for r in records:
            orig_tasks[r["task"]] = orig_tasks.get(r["task"], 0) + 1

        if split == "test":
            # Test set: keep identical
            balanced = records
            print(f"[{split}] Copied as-is: {len(balanced)} records — {orig_tasks}")
        else:
            # Train / Val: balance VQA
            balanced = balance_split(records, args.max_vqa_per_image, rng)

            # Update split field
            for r in balanced:
                r["split"] = split

            bal_tasks = {}
            for r in balanced:
                bal_tasks[r["task"]] = bal_tasks.get(r["task"], 0) + 1

            print(f"[{split}] {len(records)} → {len(balanced)} records")
            print(f"  Before: {orig_tasks}")
            print(f"  After:  {bal_tasks}")

        write_jsonl(balanced, out_file)
        print(f"  ✅ Written: {out_file}")
        print()

    print("Done! SFT-balanced JSONL files created.")


if __name__ == "__main__":
    main()
