#!/bin/bash
# ============================================================================
# run_dentalcaries_sft.sh — DentalCaries SFT Training Pipeline
#
# Batch LoRA SFT training for one or more models on the DentalCaries dataset.
# Supports the same --models / --tiers interface as scripts/run_dentalcaries.sh.
#
# Usage:
#   # Single model:
#   bash scripts/run_dentalcaries_sft.sh --models InternVL3_5-1B-HF
#
#   # Multiple models:
#   bash scripts/run_dentalcaries_sft.sh --models "InternVL3_5-1B-HF,gemma-4-E2B-it"
#
#   # By GPU tier:
#   bash scripts/run_dentalcaries_sft.sh --tiers t1              # 1-2B  (L4 24GB / A100 21GB)
#   bash scripts/run_dentalcaries_sft.sh --tiers t2              # 3-4B  (A100 40GB)
#   bash scripts/run_dentalcaries_sft.sh --tiers t3              # 7-8B  (A100 80GB)
#   bash scripts/run_dentalcaries_sft.sh --tiers t4              # 32B   (H100 96GB)
#   bash scripts/run_dentalcaries_sft.sh --tiers t1,t2           # combine tiers
#   bash scripts/run_dentalcaries_sft.sh --tiers all             # every model
#
#   # Resume from a specific model:
#   bash scripts/run_dentalcaries_sft.sh --tiers t2 --resume-from gemma-4-E4B-it
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET="dentalcaries"

# ===== GPU-tier model groups (same as scripts/run_dr.sh) =====
TIER1_MODELS=("InternVL3_5-1B-HF" "InternVL3_5-2B-HF" "SmolVLM2-2.2B-Instruct" "gemma-4-E2B-it")
TIER2_MODELS=("paligemma2-3b-mix-448" "gemma-4-E4B-it" "Qwen3-VL-4B-Instruct" "Qwen3.5-4B" "medgemma-4b-it")
TIER3_MODELS=("Qwen2.5-VL-7B-Instruct" "MedMO-8B-Next")
TIER4_MODELS=("Lingshu-32B")

# ===== Model → SFT script mapping =====
# Each model maps to a specific training script in training/sft/
declare -A SFT_SCRIPTS=(
    ["InternVL3_5-1B-HF"]="sft-internvl3.5-2b-hf.py"
    ["InternVL3_5-2B-HF"]="sft-internvl3.5-2b-hf.py"
    ["SmolVLM2-2.2B-Instruct"]="sft-smolvlm2.py"
    ["gemma-4-E2B-it"]="sft-gemma4.py"
    ["paligemma2-3b-mix-448"]="sft-paligemma2.py"

    ["gemma-4-E4B-it"]="sft-gemma4.py"
    ["Qwen3-VL-4B-Instruct"]="sft-qwen3.py"
    ["Qwen3.5-4B"]="sft-qwen3.5.py"
    # REMOVED: ["dentalgemma-1.5-4b-it"]="sft-dentalgemma.py"
    ["medgemma-4b-it"]="sft-medgemma.py"
    ["Qwen2.5-VL-7B-Instruct"]="sft-qwen2.5.py"
    ["MedMO-8B-Next"]="sft-medmo.py"
    ["Lingshu-32B"]="sft-lingshu.py"
)

# ===== Model → config file mapping =====
# Resolves to training/sft/configs/datasets/dentalcaries/{slms,llms}/<config>.yaml
declare -A SFT_CONFIGS=(
    ["InternVL3_5-1B-HF"]="slms/InternVL3_5-1B-HF.yaml"
    ["InternVL3_5-2B-HF"]="slms/InternVL3_5-2B-HF.yaml"
    ["SmolVLM2-2.2B-Instruct"]="slms/SmolVLM2-2.2B-Instruct.yaml"
    ["gemma-4-E2B-it"]="slms/gemma-4-E2B-it.yaml"
    ["paligemma2-3b-mix-448"]="slms/paligemma2-3b-mix-448.yaml"

    ["gemma-4-E4B-it"]="slms/gemma-4-E4B-it.yaml"
    ["Qwen3-VL-4B-Instruct"]="slms/Qwen3-VL-4B-Instruct.yaml"
    ["Qwen3.5-4B"]="slms/Qwen3.5-4B.yaml"
    # REMOVED: ["dentalgemma-1.5-4b-it"]="slms/dentalgemma-1.5-4b-it.yaml"
    ["medgemma-4b-it"]="slms/medgemma-4b-it.yaml"
    ["Qwen2.5-VL-7B-Instruct"]="llms/qwen2.5-vl-7b-instruct.yaml"
    ["MedMO-8B-Next"]="llms/medmo-8b-next.yaml"
    ["Lingshu-32B"]="llms/lingshu-32b.yaml"
)

# ===== Defaults =====
MODEL_LIST=""
TIERS=""
RESUME_FROM=""

# ===== Parse arguments =====
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --models=*)      MODEL_LIST="${1#*=}"; shift ;;
        --models)        MODEL_LIST="$2"; shift 2 ;;
        --tiers=*)       TIERS="${1#*=}"; shift ;;
        --tiers)         TIERS="$2"; shift 2 ;;
        --resume-from=*) RESUME_FROM="${1#*=}"; shift ;;
        --resume-from)   RESUME_FROM="$2"; shift 2 ;;
        *)               echo "❌ Unknown argument: $1"; exit 1 ;;
    esac
done

# ===== Resolve model list =====
if [ -n "$TIERS" ]; then
    MODELS=()
    IFS=',' read -ra TIERS_ARRAY <<< "$TIERS"
    for T in "${TIERS_ARRAY[@]}"; do
        T=$(echo "$T" | xargs)
        case "$T" in
            t1)  MODELS+=("${TIER1_MODELS[@]}") ;;
            t2)  MODELS+=("${TIER2_MODELS[@]}") ;;
            t3)  MODELS+=("${TIER3_MODELS[@]}") ;;
            t4)  MODELS+=("${TIER4_MODELS[@]}") ;;
            all) MODELS+=("${TIER1_MODELS[@]}" "${TIER2_MODELS[@]}" "${TIER3_MODELS[@]}" "${TIER4_MODELS[@]}") ;;
            *)   echo "❌ Unknown tier: $T (expected: t1, t2, t3, t4, all)"; exit 1 ;;
        esac
    done
elif [ -n "$MODEL_LIST" ]; then
    IFS=',' read -ra MODELS <<< "$MODEL_LIST"
else
    echo "❌ --models or --tiers is required"
    echo "Usage: bash scripts/run_dentalcaries_sft.sh --models <name>                # single model"
    echo "       bash scripts/run_dentalcaries_sft.sh --models <name1,name2>          # multiple models"
    echo "       bash scripts/run_dentalcaries_sft.sh --tiers <t1|t2|t3|t4|t1,t2|all> # GPU tier(s)"
    exit 1
fi

# ===== Handle --resume-from =====
if [ -n "$RESUME_FROM" ]; then
    SKIP=true
    FILTERED=()
    for m in "${MODELS[@]}"; do
        [ "$m" = "$RESUME_FROM" ] && SKIP=false
        [ "$SKIP" = false ] && FILTERED+=("$m")
    done
    if [ ${#FILTERED[@]} -eq 0 ]; then
        echo "❌ Model '$RESUME_FROM' not found in selected model list"
        exit 1
    fi
    SKIPPED=$(( ${#MODELS[@]} - ${#FILTERED[@]} ))
    echo "⏭️  Resuming from: $RESUME_FROM (skipping $SKIPPED models)"
    MODELS=("${FILTERED[@]}")
fi

# ============================================================================
# Main: iterate over all models
# ============================================================================
MODEL_TOTAL=${#MODELS[@]}
MODEL_IDX=0
FAILED=()
SUCCEEDED=()

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  🎓 DentalCaries SFT Training Pipeline                      ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Models: $MODEL_TOTAL"
if [ -n "$TIERS" ]; then
echo "║  Tiers:  $TIERS"
fi
echo "║  Start:  $(date)"
echo "╚══════════════════════════════════════════════════════════╝"

for M in "${MODELS[@]}"; do
    M=$(echo "$M" | xargs)  # trim whitespace
    MODEL_IDX=$((MODEL_IDX + 1))

    # Resolve script and config
    SFT_SCRIPT="${SFT_SCRIPTS[$M]}"
    SFT_CONFIG="${SFT_CONFIGS[$M]}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$MODEL_IDX/$MODEL_TOTAL] $M"
    echo "  Script: training/sft/$SFT_SCRIPT"
    echo "  Config: training/sft/configs/datasets/${DATASET}/$SFT_CONFIG"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Validate
    if [ -z "$SFT_SCRIPT" ]; then
        echo "  ❌ No SFT script mapping for model: $M"
        FAILED+=("$M")
        continue
    fi
    if [ ! -f "${SCRIPT_DIR}/training/sft/$SFT_SCRIPT" ]; then
        echo "  ❌ Script not found: training/sft/$SFT_SCRIPT"
        FAILED+=("$M")
        continue
    fi
    FULL_CONFIG="${SCRIPT_DIR}/training/sft/configs/datasets/${DATASET}/$SFT_CONFIG"
    FALLBACK_CONFIG="${SCRIPT_DIR}/training/sft/configs/base_config/models/$SFT_CONFIG"
    if [ ! -f "$FULL_CONFIG" ]; then
        if [ -f "$FALLBACK_CONFIG" ]; then
            FULL_CONFIG="$FALLBACK_CONFIG"
            echo "  ℹ️  Using base_config/models config (no dataset override)"
        else
            echo "  ❌ Config not found: $FULL_CONFIG (nor base_config/models fallback)"
            FAILED+=("$M")
            continue
        fi
    fi

    # Run SFT training
    START_TIME=$(date +%s)
    export SFT_DATASET="${DATASET}"
    if python "${SCRIPT_DIR}/training/sft/$SFT_SCRIPT" --config "$FULL_CONFIG"; then
        ELAPSED=$(( $(date +%s) - START_TIME ))
        echo "  ✅ $M completed in ${ELAPSED}s"
        SUCCEEDED+=("$M")
    else
        ELAPSED=$(( $(date +%s) - START_TIME ))
        echo "  ❌ $M FAILED after ${ELAPSED}s (continuing...)"
        FAILED+=("$M")
    fi
done

# ===== Summary =====
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  📊 SFT Training Summary                                ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Total:     $MODEL_TOTAL models"
echo "║  Succeeded: ${#SUCCEEDED[@]}"
echo "║  Failed:    ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
echo "║  ❌ Failed: ${FAILED[*]}"
fi
echo "║  End: $(date)"
echo "╚══════════════════════════════════════════════════════════╝"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "💡 To re-run failed: bash scripts/run_dentalcaries_sft.sh --models $(IFS=,; echo "${FAILED[*]}")"
    exit 1
fi
