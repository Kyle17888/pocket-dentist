#!/bin/bash
# ============================================================================
# audit_sft_status.sh — Check SFT training status for all datasets & models
#
# Shows which dataset × model combinations have:
#   ✅ Completed (adapter + merged)
#   ⚠️  Trained but not merged (adapter only)
#   ❌ Not trained (missing)
#
# Usage:
#   bash scripts/audit_sft_status.sh
# ============================================================================

SFT_ROOT="/home/kbia984/00_nesi_projects/uoa04670_nobackup/kbia984/models/Neurlps2026-SFT"

# All expected datasets
DATASETS=(BRAR DR MetaDent Aariz COde DenPAR DentalCaries)

# All expected models (slms + llms)
SLMS=(InternVL3_5-1B-HF InternVL3_5-2B-HF SmolVLM2-2.2B-Instruct gemma-4-E2B-it paligemma2-3b-mix-448 gemma-4-E4B-it Qwen3-VL-4B-Instruct Qwen3.5-4B medgemma-4b-it)
LLMS=(Qwen2.5-VL-7B-Instruct MedMO-8B-Next Lingshu-32B)

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🔍 SFT Training Status Audit                               ║"
echo "║  Root: $SFT_ROOT"
echo "╚══════════════════════════════════════════════════════════════╝"

if [ ! -d "$SFT_ROOT" ]; then
    echo "❌ SFT root not found: $SFT_ROOT"
    exit 1
fi

TOTAL=0
COMPLETE=0
TRAINED=0
MISSING=0

MISSING_LIST=()
NEED_MERGE_LIST=()

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "━━━ $ds ━━━"

    for tier_name in slms llms; do
        if [ "$tier_name" == "slms" ]; then
            MODELS=("${SLMS[@]}")
        else
            MODELS=("${LLMS[@]}")
        fi

        for model in "${MODELS[@]}"; do
            TOTAL=$((TOTAL + 1))
            model_dir="$SFT_ROOT/$ds/$tier_name/$model"
            source_dir="$model_dir/source"
            merged_dir="$model_dir/merged"

            has_adapter=false
            has_merged=false
            has_checkpoints=false
            training_progress=""

            # Check adapter
            if [ -f "$source_dir/adapter_model.safetensors" ] || [ -f "$source_dir/adapter_model.bin" ]; then
                has_adapter=true
            fi

            # Check merged
            if [ -d "$merged_dir" ]; then
                merged_count=$(find "$merged_dir" -maxdepth 1 \( -name "*.safetensors" -o -name "*.bin" \) 2>/dev/null | wc -l | tr -d ' ')
                [ "$merged_count" -gt 0 ] && has_merged=true
            fi

            # Check checkpoints (in-progress training)
            if [ -d "$source_dir" ]; then
                ckpt_count=$(ls -d "$source_dir"/checkpoint-* 2>/dev/null | wc -l | tr -d ' ')
                [ "$ckpt_count" -gt 0 ] && has_checkpoints=true && training_progress="ckpts=$ckpt_count"
            fi

            # Status
            if $has_merged; then
                status="✅ COMPLETE"
                COMPLETE=$((COMPLETE + 1))
            elif $has_adapter; then
                status="⚠️  NEED MERGE"
                TRAINED=$((TRAINED + 1))
                NEED_MERGE_LIST+=("$ds/$tier_name/$model")
            elif $has_checkpoints; then
                status="🔄 IN PROGRESS"
                TRAINED=$((TRAINED + 1))
                NEED_MERGE_LIST+=("$ds/$tier_name/$model (in-progress)")
            else
                status="❌ MISSING"
                MISSING=$((MISSING + 1))
                MISSING_LIST+=("$ds/$tier_name/$model")
            fi

            printf "  %-5s %-30s %s  %s\n" "$tier_name" "$model" "$status" "$training_progress"
        done
    done
done

# ===== Summary =====
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  📊 Summary                                                  ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Total expected:  %d\n" "$TOTAL"
printf "║  ✅ Complete:      %d\n" "$COMPLETE"
printf "║  ⚠️  Need merge:   %d\n" "$TRAINED"
printf "║  ❌ Missing:       %d\n" "$MISSING"
echo "╚══════════════════════════════════════════════════════════════╝"

if [ ${#MISSING_LIST[@]} -gt 0 ]; then
    echo ""
    echo "❌ Models that need SFT training:"
    for item in "${MISSING_LIST[@]}"; do
        echo "  → $item"
    done
fi

if [ ${#NEED_MERGE_LIST[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  Models that need merge:"
    for item in "${NEED_MERGE_LIST[@]}"; do
        echo "  → $item"
    done
fi
