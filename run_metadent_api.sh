#!/bin/bash
# ============================================================================
# run_metadent_api.sh — MetaDent Pipeline (API Models: GPT, Gemini)
#
# Runs prediction + evaluation for closed-source API models
# using the unified src/ framework.
#
# Usage:
#   # GPT-4o-mini
#   bash run_metadent_api.sh --model gpt-4o-mini --tasks baseline \
#     --api_base https://api.openai.com/v1 --api_key $OPENAI_API_KEY
#
#   # Gemini 2.5 Flash (via OpenAI-compatible proxy)
#   bash run_metadent_api.sh --model gemini-2.5-flash --tasks "baseline,1shot,2shot" \
#     --api_base https://generativelanguage.googleapis.com/v1beta/openai/ \
#     --api_key $GEMINI_API_KEY
# ============================================================================

set -e

# ===== Default configuration =====
MODEL_NAME=""
TASKS="baseline"
API_BASE=""
API_KEY=""
WORKERS=1           # API models: use 1 worker to respect rate limits
DATASET="metadent"
FEW_SHOT_CONFIG="configs/metadent/few-shots.yaml"

# ===== Parse arguments =====
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model=*)    MODEL_NAME="${1#*=}"; shift ;;
        --model)      MODEL_NAME="$2"; shift 2 ;;
        --tasks=*)    TASKS="${1#*=}"; shift ;;
        --tasks|-t)   TASKS="$2"; shift 2 ;;
        --api_base=*) API_BASE="${1#*=}"; shift ;;
        --api_base)   API_BASE="$2"; shift 2 ;;
        --api_key=*)  API_KEY="${1#*=}"; shift ;;
        --api_key)    API_KEY="$2"; shift 2 ;;
        --workers=*)  WORKERS="${1#*=}"; shift ;;
        --workers)    WORKERS="$2"; shift 2 ;;
        *)            echo "❌ Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL_NAME" ]; then
    echo "❌ --model is required"
    echo "Usage: bash run_metadent_api.sh --model <model_name> --tasks <baseline,1shot,2shot> --api_base <url> --api_key <key>"
    exit 1
fi

if [ -z "$API_BASE" ]; then
    echo "❌ --api_base is required for API models"
    exit 1
fi

if [ -z "$API_KEY" ]; then
    # Try environment variables
    if [ -n "$OPENAI_API_KEY" ] && [[ "$MODEL_NAME" == *"gpt"* ]]; then
        API_KEY="$OPENAI_API_KEY"
        echo "ℹ️  Using OPENAI_API_KEY from environment"
    elif [ -n "$GEMINI_API_KEY" ] && [[ "$MODEL_NAME" == *"gemini"* ]]; then
        API_KEY="$GEMINI_API_KEY"
        echo "ℹ️  Using GEMINI_API_KEY from environment"
    else
        echo "❌ --api_key is required (or set OPENAI_API_KEY / GEMINI_API_KEY)"
        exit 1
    fi
fi

# ===== Execution =====
export PIPELINE_RUN_ID="$(date +%Y%m%d_%H%M%S)_${MODEL_NAME}_metadent"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  MetaDent API Pipeline                 ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Model:     $MODEL_NAME"
echo "║  Tasks:     $TASKS"
echo "║  API Base:  $API_BASE"
echo "║  Workers:   $WORKERS"
echo "║  Dataset:   $DATASET"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

IFS=',' read -ra TASK_ARRAY <<< "$TASKS"
TOTAL=${#TASK_ARRAY[@]}
CURRENT=0

for TASK in "${TASK_ARRAY[@]}"; do
    CURRENT=$((CURRENT + 1))
    TASK=$(echo "$TASK" | xargs)

    echo ""
    echo "━━━━━ [$CURRENT/$TOTAL] $MODEL_NAME / $TASK ━━━━━"

    # ========== Parse num_shots ==========
    NUM_SHOTS_ARG=""
    FEW_SHOT_ARG=""
    if [[ "$TASK" =~ ^([0-9]+)shot$ ]]; then
        NUM_SHOTS_ARG="--num_shots ${BASH_REMATCH[1]}"
        FEW_SHOT_ARG="--few_shot_config ${FEW_SHOT_CONFIG}"
    elif [[ "$TASK" == "baseline" || "$TASK" == "sft" ]]; then
        NUM_SHOTS_ARG="--num_shots 0"
    fi

    # Step 1: Prediction
    echo "🚀 [Step 1/2] Prediction..."
    python -m src.main \
        --dataset "${DATASET}" \
        --task prediction \
        --model_name "${MODEL_NAME}" \
        --client_type api \
        --api_base_url "${API_BASE}" \
        --api_key "${API_KEY}" \
        --workers "${WORKERS}" \
        --run_tag "${TASK}" \
        ${NUM_SHOTS_ARG} ${FEW_SHOT_ARG}

    # Step 2: Evaluation
    echo "📊 [Step 2/2] Evaluation..."
    python -m src.main \
        --dataset "${DATASET}" \
        --task evaluation \
        --model_name "${MODEL_NAME}" \
        --run_tag "${TASK}"

    echo "✅ [$TASK] Complete"
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  🎉 All tasks complete: $TASKS"
echo "║  Model: $MODEL_NAME"
echo "║  Dataset: $DATASET"
echo "╚══════════════════════════════════════════════════════════╝"
