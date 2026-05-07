#!/usr/bin/env python3
"""
01_build_jsonl.py — Build COde JSONL from complete_dataset.csv.

Reads:
  - <input_dir>/complete_dataset.csv     (8,775 checkups with bilingual clinical records)
  - <input_dir>/images/                  (Photographs/ and Radiographs/ subdirs)

Outputs:
  - <output_dir>/train.jsonl   (uses CSV split=train, minus 10% val)
  - <output_dir>/val.jsonl     (10% stratified split from train)
  - <output_dir>/test.jsonl    (uses CSV split=test, 600 checkups)

Each checkup produces TWO JSONL records (English only):
  1. task=code_classification  — 6-class single-label anomaly classification
  2. task=code_report          — structured diagnostic report generation

Usage:
  python 01_build_jsonl.py \
      --input_dir  /path/to/COde/input \
      --output_dir /path/to/COde/output \
      [--seed 42]
"""

import argparse
import json
import os
import random

import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# 6-class mapping (COde paper Table 2)
# Maps 117 fine-grained anomalies → 6 benchmark categories
# ─────────────────────────────────────────────────────────────

CLASS_6 = [
    "Dental Caries",
    "Gingivitis",
    "Class III Malocclusion",
    "Pulpitis",
    "Tooth Loss",
    "Tooth Structure Loss",
]

# Mapping rules (from COde paper + data-preprocessing.ipynb diseases.xlsx logic)
_ANOMALY_TO_CLASS = {
    # Dental Caries
    "Dental Caries": "Dental Caries",
    "Demineralization": "Dental Caries",
    "Deep Fissures": "Dental Caries",
    # Gingivitis / Periodontal
    "Gingivitis": "Gingivitis",
    "Periodontitis": "Gingivitis",
    "Gingival Recession": "Gingivitis",
    "Calculus": "Gingivitis",
    "Subgingival Calculus": "Gingivitis",
    "Poor Oral Hygiene": "Gingivitis",
    "Hyperemia": "Gingivitis",
    "Alveolar Bone Resorption": "Gingivitis",
    "Horizontal Alveolar Bone Loss": "Gingivitis",
    # Class III Malocclusion
    "Malocclusion": "Class III Malocclusion",
    "Class I Malocclusion": "Class III Malocclusion",
    "Class II Malocclusion": "Class III Malocclusion",
    "Class III Malocclusion": "Class III Malocclusion",
    "Mixed Dentition Malocclusion": "Class III Malocclusion",
    "Dental Crowding": "Class III Malocclusion",
    "Spacing": "Class III Malocclusion",
    "Crossbite": "Class III Malocclusion",
    "Anterior Crossbite": "Class III Malocclusion",
    "Open Bite": "Class III Malocclusion",
    "Anterior Open Bite": "Class III Malocclusion",
    "Deep Overbite": "Class III Malocclusion",
    "Excessive Overbite": "Class III Malocclusion",
    "Deep Overjet": "Class III Malocclusion",
    "Excessive Overjet": "Class III Malocclusion",
    "Edge-To-Edge Bite": "Class III Malocclusion",
    "Dental Midline Deviation": "Class III Malocclusion",
    "Left Dental Midline Deviation": "Class III Malocclusion",
    "Left Mandibular Midline Deviation": "Class III Malocclusion",
    "Left Maxillary Midline Deviation": "Class III Malocclusion",
    "Right Mandibular Midline Deviation": "Class III Malocclusion",
    "Tooth Rotation": "Class III Malocclusion",
    "Buccal Displacement": "Class III Malocclusion",
    "Buccally Displaced Tooth": "Class III Malocclusion",
    "Ectopic Tooth": "Class III Malocclusion",
    "Impacted Tooth": "Class III Malocclusion",
    "Mesioangular Impaction": "Class III Malocclusion",
    "Mesially Inclined Tooth": "Class III Malocclusion",
    "Labial Inclination": "Class III Malocclusion",
    "Incisor Protrusion": "Class III Malocclusion",
    "Proclined Lower Incisors": "Class III Malocclusion",
    "Proclined Upper Incisors": "Class III Malocclusion",
    "Retroclined Lower Incisors": "Class III Malocclusion",
    "Retroclined Upper Incisors": "Class III Malocclusion",
    "Deep Curve Of Spee": "Class III Malocclusion",
    "Supernumerary Tooth": "Class III Malocclusion",
    "Retained Primary Tooth": "Class III Malocclusion",
    "Retained Tooth": "Class III Malocclusion",
    "Palatal Ectopic Eruption": "Class III Malocclusion",
    "Maxillary Protrusion": "Class III Malocclusion",
    "Mandibular Retrusion": "Class III Malocclusion",
    "Narrow Maxillary Arch": "Class III Malocclusion",
    "Maxillary Expansion": "Class III Malocclusion",
    "Convex Profile": "Class III Malocclusion",
    "Concave Profile": "Class III Malocclusion",
    "High-Angle Mandible": "Class III Malocclusion",
    "Low-Angle Mandible": "Class III Malocclusion",
    "Chin Deviated Left": "Class III Malocclusion",
    "Chin Deviated Right": "Class III Malocclusion",
    "Chin Retrusion": "Class III Malocclusion",
    "Mandibular Skeletal Asymmetry": "Class III Malocclusion",
    "Ramus Height Discrepancy": "Class III Malocclusion",
    "Flat Facial Profile": "Class III Malocclusion",
    "Right Facial Widening": "Class III Malocclusion",
    "Angle Class II Molar Relationship": "Class III Malocclusion",
    "Congenitally Missing Tooth": "Class III Malocclusion",
    "Microdontia": "Class III Malocclusion",
    "Occlusal Abnormality": "Class III Malocclusion",
    "Orthodontically Treated Tooth": "Class III Malocclusion",
    "Post Orthodontic Relapse": "Class III Malocclusion",
    "Mouth Breathing": "Class III Malocclusion",
    "Lip Biting Habit": "Class III Malocclusion",
    "Tongue Thrust": "Class III Malocclusion",
    "Abnormal Tongue Habit": "Class III Malocclusion",
    "Short Labial Frenum": "Class III Malocclusion",
    # Pulpitis
    "Pulpitis": "Pulpitis",
    "Pulp Necrosis": "Pulpitis",
    "Periapical Periodontitis": "Pulpitis",
    "Periapical Radiolucency": "Pulpitis",
    "Distal Periapical Radiolucency": "Pulpitis",
    "Internal Resorption": "Pulpitis",
    "Acute Pericoronitis": "Pulpitis",
    "Cracked Tooth": "Pulpitis",
    # Tooth Loss
    "Tooth Loss": "Tooth Loss",
    "Kennedy Class I Edentulism": "Tooth Loss",
    "Kennedy Class III Edentulism": "Tooth Loss",
    "Post Implant Status": "Tooth Loss",
    "Post And Core Crown": "Tooth Loss",
    "Multiple Restored Teeth": "Tooth Loss",
    # Tooth Structure Loss
    "Tooth Structure Loss": "Tooth Structure Loss",
    "Tooth Wear": "Tooth Structure Loss",
    "Wedge-Shaped Defect": "Tooth Structure Loss",
    "Dental Fluorosis": "Tooth Structure Loss",
    "Enamel Hypoplasia": "Tooth Structure Loss",
    "Dens Evaginatus": "Tooth Structure Loss",
    "Tooth Discoloration": "Tooth Structure Loss",
    "Intrinsic Tooth Discoloration": "Tooth Structure Loss",
    "Extrinsic Tooth Stain": "Tooth Structure Loss",
    "Pigment Deposition": "Tooth Structure Loss",
    "Palatal Pigmentation": "Tooth Structure Loss",
    "Dental Trauma": "Tooth Structure Loss",
    "Tooth Fracture": "Tooth Structure Loss",
    "Jaw Trauma": "Tooth Structure Loss",
    "Tooth Mobility": "Tooth Structure Loss",
    "Abnormal Palatal Groove": "Tooth Structure Loss",
    "Tooth Developmental Anomaly": "Tooth Structure Loss",
    "Post Cleft Repair": "Tooth Structure Loss",
    "Lower Lip Ulcer": "Tooth Structure Loss",
    "Obstructive Sleep Apnea Syndrome": "Tooth Structure Loss",
    # Treatment / restoration related → Tooth Structure Loss (structural compromise)
    "Restoration Defect": "Tooth Structure Loss",
    "Lost Filling": "Tooth Structure Loss",
    "Temporary Restoration": "Tooth Structure Loss",
    "Post Endodontic Treatment": "Tooth Structure Loss",
    "Dry Root Canal": "Tooth Structure Loss",
    "Inadequate Root Canal Filling": "Tooth Structure Loss",
}


def map_to_class6(anomalies_str: str) -> str | None:
    """Map anomalies_en string to the primary 6-class label.

    Returns the most 'severe' class when multiple anomalies map to
    different classes.  Priority order follows clinical severity:
    Pulpitis > Caries > Gingivitis > Tooth Loss > Tooth Structure Loss > Malocclusion
    """
    if not isinstance(anomalies_str, str) or not anomalies_str.strip():
        return None

    priority = {
        "Pulpitis": 0,
        "Dental Caries": 1,
        "Gingivitis": 2,
        "Tooth Loss": 3,
        "Tooth Structure Loss": 4,
        "Class III Malocclusion": 5,
    }

    best_class = None
    best_priority = 999

    for anomaly in anomalies_str.split(","):
        anomaly = anomaly.strip()
        cls = _ANOMALY_TO_CLASS.get(anomaly)
        if cls and priority.get(cls, 999) < best_priority:
            best_class = cls
            best_priority = priority[cls]

    return best_class


# ─────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────

CLS_PROMPT_TEMPLATE = """\
You are a professional dentist. You are now given clinical images of a patient along with their medical information. Please identify the primary oro-dental anomaly from the images.

The categories and their clinical descriptions are as follows:
- Dental Caries: Visible tooth decay including deep caries, secondary caries, and demineralization. Excludes non-carious structural defects.
- Gingivitis: Gingival inflammation with redness, swelling, calculus deposits, or alveolar bone resorption. Includes periodontitis and gingival recession.
- Class III Malocclusion: Angle's Class III malocclusion or related orthodontic anomalies including crowding, spacing, crossbite, open bite, and other dentofacial deformities.
- Pulpitis: Inflammation or infection of the dental pulp, including pulp necrosis, periapical periodontitis, and periapical radiolucency.
- Tooth Loss: One or more missing teeth, including extraction sites, dental arch defects, and edentulism. May include post-implant or post-restoration status.
- Tooth Structure Loss: Loss of tooth structure not caused by caries, including fractures, wedge-shaped defects, fluorosis, enamel hypoplasia, tooth wear, discoloration, and trauma.

Choose exactly one category from above.

Patient Information:
{patient_info}

Return only the disease category name, nothing else."""

REPORT_PROMPT_TEMPLATE = """\
You are a professional dentist. You are now given clinical images of a patient along with their medical information. Please generate a structured diagnostic report based on the images and patient information.

Patient Information:
{patient_info}

Your report should include the following sections (leave blank if not applicable):
- Patient Record: Visit type (initial/follow-up) and preliminary diagnosis with ICD-10 code if applicable.
- Examination: Clinical findings from visual and tactile examination of teeth and soft tissues.
- Radiographic Examination: Findings from X-ray images, if radiographs are provided.
- Diagnosis: Final diagnosis with affected tooth numbers and ICD-10 codes.
- Treatment Plan: Planned procedures for current and future visits.
- Treatment Recommendations: Preventive care or follow-up recommendations for the patient.
- Management: Procedures performed during this visit, including materials and techniques used.
- Medical Instructions: Post-treatment instructions given to the patient.
- Remarks: Additional notes if applicable."""


# ─────────────────────────────────────────────────────────────
# Build patient info string
# ─────────────────────────────────────────────────────────────

def build_patient_info(row: pd.Series, include_history: bool = False) -> str:
    """Build patient info text from CSV row fields (English only)."""
    parts = []
    if pd.notna(row.get("age")) and str(row["age"]).strip():
        parts.append(f"Age: {row['age']}")
    if pd.notna(row.get("gender")) and str(row["gender"]).strip():
        parts.append(f"Gender: {row['gender']}")
    if pd.notna(row.get("chief_complaint")) and str(row["chief_complaint"]).strip():
        parts.append(f"Chief Complaint: {row['chief_complaint']}")
    if include_history:
        if pd.notna(row.get("present_illness")) and str(row["present_illness"]).strip():
            parts.append(f"Present Illness: {row['present_illness']}")
        if pd.notna(row.get("past_medical_record")) and str(row["past_medical_record"]).strip():
            parts.append(f"Past Medical History: {row['past_medical_record']}")
    return "\n".join(parts) if parts else "No additional patient information available."


def build_report_gt(row: pd.Series) -> str:
    """Build ground-truth diagnostic report from CSV fields (English)."""
    sections = [
        ("Patient Record", row.get("patient_record")),
        ("Examination", row.get("examination")),
        ("Radiographic Examination", row.get("radiographs_examination")),
        ("Diagnosis", row.get("diagnosis")),
        ("Treatment Plan", row.get("treatment_plan")),
        ("Treatment Recommendations", row.get("treatment_recommendations")),
        ("Management", row.get("management")),
        ("Medical Instructions", row.get("medical_instructions")),
        ("Remarks", row.get("remarks")),
    ]
    parts = []
    for name, val in sections:
        text = str(val).strip() if pd.notna(val) else ""
        parts.append(f"{name}: {text}")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────
# Build image content list
# ─────────────────────────────────────────────────────────────

def build_image_content(row: pd.Series, image_base_dir: str) -> list[dict]:
    """Build list of {"type": "image", "image": "<relative_path>"} entries."""
    content = []

    # Photographs
    photos = str(row.get("photographs", "")).strip()
    if photos and photos != "nan":
        for fname in photos.split(","):
            fname = fname.strip()
            if fname:
                rel_path = f"Images/Photographs/{fname}"
                content.append({"type": "image", "image": rel_path})

    # Radiographs
    xrays = str(row.get("radiographs", "")).strip()
    if xrays and xrays != "nan":
        for fname in xrays.split(","):
            fname = fname.strip()
            if fname:
                rel_path = f"Images/Radiographs/{fname}"
                content.append({"type": "image", "image": rel_path})

    return content


# ─────────────────────────────────────────────────────────────
# Build JSONL records
# ─────────────────────────────────────────────────────────────

def build_cls_record(row: pd.Series, idx: int, image_base_dir: str) -> dict | None:
    """Build a code_classification JSONL record."""
    label = map_to_class6(row.get("anomalies_en", ""))
    if label is None:
        return None

    patient_info = build_patient_info(row, include_history=False)
    prompt_text = CLS_PROMPT_TEMPLATE.format(patient_info=patient_info)
    image_content = build_image_content(row, image_base_dir)

    if not image_content:
        return None

    user_content = image_content + [{"type": "text", "text": prompt_text}]

    return {
        "id": f"code_cls_{idx:05d}",
        "task": "code_classification",
        "source": "COde",
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [
                {"type": "text", "text": label}
            ]},
        ],
    }


def build_report_record(row: pd.Series, idx: int, image_base_dir: str) -> dict | None:
    """Build a code_report JSONL record."""
    patient_info = build_patient_info(row, include_history=True)
    prompt_text = REPORT_PROMPT_TEMPLATE.format(patient_info=patient_info)
    image_content = build_image_content(row, image_base_dir)

    if not image_content:
        return None

    report_gt = build_report_gt(row)
    if not report_gt.strip():
        return None

    user_content = image_content + [{"type": "text", "text": prompt_text}]

    return {
        "id": f"code_rpt_{idx:05d}",
        "task": "code_report",
        "source": "COde",
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [
                {"type": "text", "text": report_gt}
            ]},
        ],
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build COde JSONL from CSV")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to COde input dir (containing complete_dataset.csv and Images/)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to output directory for JSONL files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of train to use as validation")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load CSV
    csv_path = os.path.join(args.input_dir, "complete_dataset.csv")
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Total rows: {len(df)}")
    print(f"  Split distribution: {df['split'].value_counts().to_dict()}")

    # Separate by CSV split
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # Stratified val split from train (by primary class label)
    train_df["_class6"] = train_df["anomalies_en"].apply(map_to_class6)
    val_indices = []
    train_indices = []

    for cls_label in CLASS_6 + [None]:
        mask = train_df["_class6"] == cls_label if cls_label is not None else train_df["_class6"].isna()
        cls_idx = train_df[mask].index.tolist()
        random.shuffle(cls_idx)
        n_val = max(1, int(len(cls_idx) * args.val_ratio))
        val_indices.extend(cls_idx[:n_val])
        train_indices.extend(cls_idx[n_val:])

    val_df = train_df.loc[val_indices].reset_index(drop=True)
    train_df_final = train_df.loc[train_indices].reset_index(drop=True)

    print(f"\n  Final splits:")
    print(f"    Train: {len(train_df_final)}")
    print(f"    Val:   {len(val_df)}")
    print(f"    Test:  {len(test_df)}")

    # Build JSONL records
    os.makedirs(args.output_dir, exist_ok=True)
    image_base_dir = args.input_dir

    stats = {"train": {}, "val": {}, "test": {}}
    for split_name, split_df in [("train", train_df_final), ("val", val_df), ("test", test_df)]:
        out_path = os.path.join(args.output_dir, f"{split_name}.jsonl")

        cls_count = 0
        rpt_count = 0
        skip_count = 0

        with open(out_path, "w", encoding="utf-8") as f:
            for idx, (_, row) in tqdm(
                enumerate(split_df.iterrows()),
                total=len(split_df),
                desc=f"Building {split_name}",
            ):
                # Classification record
                cls_rec = build_cls_record(row, idx, image_base_dir)
                if cls_rec:
                    f.write(json.dumps(cls_rec, ensure_ascii=False) + "\n")
                    cls_count += 1

                # Report generation record
                rpt_rec = build_report_record(row, idx, image_base_dir)
                if rpt_rec:
                    f.write(json.dumps(rpt_rec, ensure_ascii=False) + "\n")
                    rpt_count += 1

                if not cls_rec and not rpt_rec:
                    skip_count += 1

        stats[split_name] = {
            "total_rows": len(split_df),
            "cls_records": cls_count,
            "rpt_records": rpt_count,
            "skipped": skip_count,
        }
        print(f"  {split_name}: {cls_count} cls + {rpt_count} report = {cls_count + rpt_count} records → {out_path}")

    # Save build stats
    stats_path = os.path.join(args.output_dir, "build_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n  Build stats saved to: {stats_path}")
    print("Done!")


if __name__ == "__main__":
    main()
