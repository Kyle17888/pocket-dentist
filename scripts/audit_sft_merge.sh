#!/bin/bash
# ============================================================================
# audit_sft_merge.sh — Check SFT merge status for all datasets & models
#
# Usage:
#   bash scripts/audit_sft_merge.sh
#
# Run from HPC cluster project root or anywhere — paths are absolute.
# ============================================================================

SFT_ROOT="<MODEL_ROOT>"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🔍 SFT Merge Status Audit                                  ║"
echo "║  Root: $SFT_ROOT"
echo "╚══════════════════════════════════════════════════════════════╝"

if [ ! -d "$SFT_ROOT" ]; then
    echo "❌ SFT root not found: $SFT_ROOT"
    exit 1
fi

TOTAL=0
MERGED=0
NEED_MERGE=0
NO_TRAINING=0

# Track models that need merge for summary
NEED_MERGE_LIST=()

for dataset_dir in "$SFT_ROOT"/*/; do
    [ ! -d "$dataset_dir" ] && continue
    dataset=$(basename "$dataset_dir")

    echo ""
    echo "━━━ $dataset ━━━"

    found_any=false
    for tier_dir in "$dataset_dir"llms "$dataset_dir"slms; do
        [ ! -d "$tier_dir" ] && continue
        for model_dir in "$tier_dir"/*/; do
            [ ! -d "$model_dir" ] && continue
            found_any=true
            model=$(basename "$model_dir")
            tier=$(basename "$tier_dir")
            TOTAL=$((TOTAL + 1))

            has_adapter=false
            has_merged=false
            has_checkpoints=false

            # Check for LoRA adapter inside source/ (training completed)
            source_dir="$model_dir/source"
            if [ -f "$source_dir/adapter_model.safetensors" ] || [ -f "$source_dir/adapter_model.bin" ]; then
                has_adapter=true
            fi

            # Check for merged directory with actual model weight files
            merged_dir="$model_dir/merged"
            merged_weights=0
            merged_size=""
            if [ -d "$merged_dir" ]; then
                merged_weights=$(find "$merged_dir" -maxdepth 1 \( -name "*.safetensors" -o -name "*.bin" \) | wc -l | tr -d ' ')
                if [ "$merged_weights" -gt 0 ]; then
                    has_merged=true
                    # Get total size of weight files in human-readable format
                    merged_size=$(du -sh "$merged_dir" 2>/dev/null | cut -f1)
                fi
            fi

            # Check for checkpoint directories inside source/
            checkpoint_count=$(ls -d "$source_dir"/checkpoint-* 2>/dev/null | wc -l | tr -d ' ')
            [ "$checkpoint_count" -gt 0 ] && has_checkpoints=true

            # Determine status
            if $has_merged; then
                status="✅ MERGED"
                MERGED=$((MERGED + 1))
            elif $has_adapter; then
                status="⚠️  NEED MERGE (adapter saved, no merged/)"
                NEED_MERGE=$((NEED_MERGE + 1))
                NEED_MERGE_LIST+=("$dataset/$tier/$model")
            elif $has_checkpoints; then
                status="⚠️  NEED MERGE (checkpoints exist, no merged/)"
                NEED_MERGE=$((NEED_MERGE + 1))
                NEED_MERGE_LIST+=("$dataset/$tier/$model")
            else
                status="❌ NO TRAINING (empty directory)"
                NO_TRAINING=$((NO_TRAINING + 1))
            fi

            # Extra info
            extra=""
            [ "$checkpoint_count" -gt 0 ] && extra="ckpts=$checkpoint_count"
            $has_adapter && extra="$extra adapter=✅"
            $has_merged && extra="$extra weights=${merged_weights}files size=${merged_size}"

            printf "  %-5s %-30s %s  %s\n" "$tier" "$model" "$status" "$extra"
        done
    done

    if ! $found_any; then
        echo "  (empty)"
    fi
done

# ===== Summary =====
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  📊 Summary                                                  ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Total models:    $TOTAL"
echo "║  ✅ Merged:        $MERGED"
echo "║  ⚠️  Need merge:   $NEED_MERGE"
echo "║  ❌ No training:   $NO_TRAINING"
echo "╚══════════════════════════════════════════════════════════════╝"

if [ ${#NEED_MERGE_LIST[@]} -gt 0 ]; then
    echo ""
    echo "🔧 Models that need manual merge:"
    for item in "${NEED_MERGE_LIST[@]}"; do
        echo "  → $item"
    done
    echo ""
    echo "💡 To merge, edit training/model_merge/merge_config.yaml and run:"
    echo "   python training/model_merge/merge_lora.py --config training/model_merge/merge_config.yaml"
fi
