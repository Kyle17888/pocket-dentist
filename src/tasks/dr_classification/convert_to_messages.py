#!/usr/bin/env python3
"""
convert_to_messages.py — Convert DR CSV annotations to unified messages JSONL.

Reads per-split annotation CSVs, aggregates per-image labels (discarding bbox),
and outputs messages-format JSONL for the unified pipeline.

Usage:
  python src/tasks/dr_classification/convert_to_messages.py
  python src/tasks/dr_classification/convert_to_messages.py \
    --input_dir reference/DR/src-DR/dataset/input \
    --output_dir reference/DR/src-DR/dataset/output_cls
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

DR_CLASSES = sorted(["Cavity", "Fillings", "Impacted Tooth", "Implant"])

CLASSIFICATION_PROMPT = (
    "Analyze this dental panoramic radiograph. Identify all pathological or "
    "prosthetic findings present in the image.\n\n"
    "Supported finding categories:\n"
    "- Cavity: dark radiolucent areas within tooth structure indicating decay\n"
    "- Fillings: bright radiopaque material (metal) or tooth-colored restorations within teeth\n"
    "- Impacted Tooth: teeth partially or fully embedded in jawbone, often angled or trapped\n"
    "- Implant: metallic screw-like fixtures anchored in the jawbone replacing tooth roots\n\n"
    "List ONLY the finding categories present, separated by commas.\n"
    'If no findings are detected, respond "None".'
)

SPLIT_CONFIG = {
    "train": {
        "csv_name": "0_train_annotations.csv",
        "output_name": "train.jsonl",
    },
    "valid": {
        "csv_name": "0_valid_annotations.csv",
        "output_name": "val.jsonl",
    },
    "test": {
        "csv_name": "_annotations.csv",
        "output_name": "test.jsonl",
    },
}


# ──────────────────────────────────────────────────────────────
# Core conversion
# ──────────────────────────────────────────────────────────────

def parse_csv_labels(csv_path: str) -> dict[str, set]:
    """
    Parse annotation CSV and group unique labels by filename.

    Returns:
        {filename: set_of_labels}
    """
    grouped = defaultdict(set)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            label = row["class"]
            if label in DR_CLASSES:
                grouped[filename].add(label)

    return dict(grouped)


def convert_split(input_dir: str, split: str, output_dir: str) -> dict:
    """Convert one split's CSV to unified messages JSONL."""
    cfg = SPLIT_CONFIG[split]
    csv_path = os.path.join(input_dir, split, cfg["csv_name"])

    if not os.path.exists(csv_path):
        print(f"  ⚠️  CSV not found: {csv_path}, skipping {split}")
        return {}

    grouped = parse_csv_labels(csv_path)

    output_path = os.path.join(output_dir, cfg["output_name"])
    os.makedirs(output_dir, exist_ok=True)

    total_images = 0
    label_counts = Counter()

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, filename in enumerate(sorted(grouped.keys())):
            labels = sorted(grouped[filename])
            total_images += 1
            for label in labels:
                label_counts[label] += 1

            answer = ", ".join(labels)

            record = {
                "id": f"dr_cls_{split}_{idx:04d}",
                "task": "dr_classification",
                "source": "DR",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"images/{split}/{filename}"},
                            {"type": "text", "text": CLASSIFICATION_PROMPT},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": answer},
                        ],
                    },
                ],
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = {
        "split": split,
        "total_images": total_images,
        "label_distribution": dict(sorted(label_counts.items())),
    }

    print(f"  ✅ {split}: {total_images} images → {output_path}")
    return stats


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Convert DR annotations to unified messages JSONL")
    parser.add_argument("--input_dir", type=str,
                        default="reference/DR/src-DR/dataset/input",
                        help="Root input directory containing train/valid/test subdirs with CSVs")
    parser.add_argument("--output_dir", type=str,
                        default="reference/DR/src-DR/dataset/output_cls",
                        help="Output directory for JSONL files")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"{'=' * 60}")
    print(f"DR Dataset Conversion — CSV → Multi-label Classification JSONL")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Prompt: (classification, no bbox)")
    print(f"{'=' * 60}")

    all_stats = {}
    for split in ["train", "valid", "test"]:
        stats = convert_split(args.input_dir, split, args.output_dir)
        if stats:
            all_stats[split] = stats

    # Save stats
    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Stats saved: {stats_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Summary")
    print(f"{'=' * 60}")
    for split, stats in all_stats.items():
        print(f"  {split}:")
        print(f"    Images:  {stats['total_images']}")
        print(f"    Labels:  {stats['label_distribution']}")

    print(f"\n  Prompt template:")
    print(f"  {CLASSIFICATION_PROMPT}")
    print()


if __name__ == "__main__":
    main()
