#!/bin/bash
# ============================================================================
# rerun_tasks.sh — Remaining tasks to complete the benchmark
#
# Updated: 2026-05-07 (post-audit)
#
# Status:
#   ✅ All baseline (7 datasets × 15 models) — DONE
#   ✅ All SFT (6/7 datasets × 13 models) — DONE (except COde)
#   ⚠️  Few-shot has known gaps (PaliGemma2, some API models)
#   ❌ COde SFT — NOT STARTED
#
# Usage:
#   # This is a reference script — run steps individually, NOT as one script
# ============================================================================

# ╔══════════════════════════════════════════════════════════════╗
# ║  Step 0: Pre-flight Checks                                  ║
# ║  Run these audits first to see current state                 ║
# ╚══════════════════════════════════════════════════════════════╝

# Check SFT training & merge status
bash scripts/audit_sft_status.sh

# Check results completeness
bash scripts/audit_results.sh

# Check merge status specifically
bash scripts/audit_sft_merge.sh


# ╔══════════════════════════════════════════════════════════════╗
# ║  Step 1: COde SFT Merge + Inference (P1 — Critical)         ║
# ║  The only MISSING entire setting                             ║
# ╚══════════════════════════════════════════════════════════════╝

# 1a. Check if COde SFT training is done
# bash scripts/audit_sft_status.sh  # Look for COde section

# 1b. Merge any unmerged COde models
# conda activate NeurlPS2026-train-env
# python training/model_merge/merge_lora.py --batch --datasets "COde" --dry-run
# python training/model_merge/merge_lora.py --batch --datasets "COde"

# 1c. Run COde SFT inference
# conda activate NeurlPS2026-benchmark-env
# bash run_code_sft.sh --tiers all --tasks sft


# ╔══════════════════════════════════════════════════════════════╗
# ║  Step 2: Re-run Evaluators for Missing Metrics (P2)          ║
# ║  7 model/setting combos have predictions but no metrics      ║
# ╚══════════════════════════════════════════════════════════════╝

# These need their evaluation scripts re-run.
# The run_*_api.sh scripts with the same model+setting should
# detect existing predictions and just run evaluation.

# --- Aariz ---
# bash run_aariz_api.sh --model gpt-4o-mini --tasks 2shot --evaluate-only

# --- COde ---
# bash run_code_api.sh --model gpt-4o-mini --tasks "1shot,2shot" --evaluate-only

# --- DentalCaries ---
# bash run_dentalcaries_api.sh --model gemini-2.5-flash --tasks 2shot --evaluate-only

# --- MetaDent ---
# bash run_metadent_api.sh --model gemini-2.5-flash --tasks "1shot,2shot" --evaluate-only
# bash run_metadent_api.sh --model gpt-4o-mini --tasks "1shot,2shot" --evaluate-only


# ╔══════════════════════════════════════════════════════════════╗
# ║  Step 3: Verify gemma-4-E2B-it DentalCaries SFT             ║
# ║  Was just fixed (k_norm weight patch)                        ║
# ╚══════════════════════════════════════════════════════════════╝

# Should already be running. Verify results exist after completion:
# ls results/datasets/dentalcaries/sft/gemma-4-E2B-it-SFT/


# ╔══════════════════════════════════════════════════════════════╗
# ║  Step 4: Final Analysis                                      ║
# ║  [WAIT] Run after ALL above steps are complete               ║
# ╚══════════════════════════════════════════════════════════════╝

# python results/analysis/analysis.py


# ============================================================================
# Known Acceptable Gaps (NOT bugs — document in paper)
# ============================================================================
#
# PaliGemma2 few-shot (12 gaps):
#   Architecture only supports single image input.
#   All 7 datasets × {1shot, 2shot} = 14 slots, 12 have 0 predictions.
#   → Report as "—" in tables, document in paper.
#
# SmolVLM2 COde 2shot (1 gap):
#   Context length exceeded with 2 exemplar images.
#   → Report as "—" in table.
#
# Aariz SFT mode collapse:
#   All models predict majority class due to extreme label imbalance.
#   → Document in Threats to Validity.
#
# Gemini-2.5-flash partial completions:
#   Some safety-filtered responses reduce prediction count.
#   → Metrics still computed on available predictions.
# ============================================================================

# ============================================================================
# Completion Checklist
# ============================================================================
# [x] All baseline complete (7 datasets)
# [x] All 1shot/2shot complete (except PaliGemma2 + documented gaps)
# [x] BRAR SFT complete (13 models)
# [x] DR SFT complete (13 models)
# [x] MetaDent SFT complete (13 models)
# [x] Aariz SFT complete (13 models) — mode collapse documented
# [x] DenPAR SFT complete (13 models)
# [x] DentalCaries SFT complete (13 models) — gemma-4-E2B-it fixed
# [ ] COde SFT (13 models) — PENDING
# [ ] Missing metrics re-evaluation (7 combos) — PENDING
# [ ] Final analysis.py run — PENDING
# ============================================================================
