#!/bin/bash
# ============================================================================
# run_dr.sh — DR Unified Pipeline (Local vLLM Models)
#
# Runs prediction + evaluation for one or more models across specified tasks.
#
# Usage:
#   # Single model:
#   bash run_dr.sh --models Qwen3-VL-4B-Instruct --tasks baseline
#
#   # Multiple models (serial execution):
#   bash run_dr.sh --models "Qwen3-VL-4B-Instruct,InternVL3_5-2B-HF" --tasks "baseline,1shot,sft"
#
#   # Run all models in a GPU tier (parallel-friendly for SLURM):
#   bash run_dr.sh --tiers t1 --tasks "baseline,1shot,2shot,sft"   # 1-2B  (A100 21GB)
#   bash run_dr.sh --tiers t2 --tasks "baseline,1shot,2shot,sft"   # 3-4B  (A100 40GB)
#   bash run_dr.sh --tiers t3 --tasks "baseline,1shot,2shot,sft"   # 7-8B  (A100 80GB)
#   bash run_dr.sh --tiers t4 --tasks "baseline,1shot,2shot,sft"   # 32B   (H100 80GB)
#   bash run_dr.sh --tiers t1,t2 --tasks "baseline,sft"            # combine tiers
#   bash run_dr.sh --tiers all                                     # every model
#
#   # Resume from a specific model (skip already-completed ones):
#   bash run_dr.sh --tiers t2 --resume-from gemma-4-E4B-it
#
#   # Manual mode: point to a running vLLM server:
#   bash run_dr.sh --models Qwen3-VL-4B-Instruct --tasks baseline --vllm_server http://localhost:9015/v1
# ============================================================================

set -e

# ===== GPU-tier model groups =====
# t1: 1-2B  → Inference: A100 21GB  | SFT: A100 21GB
# t2: 3-4B  → Inference: A100 40GB  | SFT: A100 40GB
# t3: 7-8B  → Inference: A100 40GB  | SFT: A100 80GB
# t4: 32B   → Inference: H100 80GB  | SFT: H100 80GB
TIER1_MODELS=("InternVL3_5-1B-HF" "InternVL3_5-2B-HF" "SmolVLM2-2.2B-Instruct" "gemma-4-E2B-it")
TIER2_MODELS=("paligemma2-3b-mix-448" "gemma-4-E4B-it" "Qwen3-VL-4B-Instruct" "Qwen3.5-4B" "medgemma-4b-it")
TIER3_MODELS=("Qwen2.5-VL-7B-Instruct" "MedMO-8B-Next")
TIER4_MODELS=("Lingshu-32B")

# ===== Default configuration =====
MODEL_LIST=""
TIERS=""
TASKS="baseline"
API_BASE=""
API_BASE_OVR=""
API_KEY="EMPTY"
WORKERS=16
DATASET="dr"
FEW_SHOT_CONFIG="configs/dr/few-shots.yaml"
AUTO_VLLM=true
NO_KILL_VLLM=false
RESUME_FROM=""

# ===== Parse arguments =====
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --models=*)      MODEL_LIST="${1#*=}"; shift ;;
        --models)        MODEL_LIST="$2"; shift 2 ;;
        --tiers=*)        TIERS="${1#*=}"; shift ;;
        --tiers)          TIERS="$2"; shift 2 ;;
        --tasks=*)       TASKS="${1#*=}"; shift ;;
        --tasks|-t)      TASKS="$2"; shift 2 ;;
        --api_base=*)    API_BASE_OVR="${1#*=}"; shift ;;
        --api_base)      API_BASE_OVR="$2"; shift 2 ;;
        --api_key=*)     API_KEY="${1#*=}"; shift ;;
        --api_key)       API_KEY="$2"; shift 2 ;;
        --workers=*)     WORKERS="${1#*=}"; shift ;;
        --workers)       WORKERS="$2"; shift 2 ;;
        --vllm_server=*) AUTO_VLLM=false; API_BASE_OVR="${1#*=}"; shift ;;
        --vllm_server)   AUTO_VLLM=false; API_BASE_OVR="$2"; shift 2 ;;
        --no_kill_vllm)  NO_KILL_VLLM=true; shift ;;
        --resume-from=*) RESUME_FROM="${1#*=}"; shift ;;
        --resume-from)   RESUME_FROM="$2"; shift 2 ;;
        *)               echo "❌ Unknown argument: $1"; exit 1 ;;
    esac
done

# ===== Resolve model list =====
if [ -n "$TIERS" ]; then
    # Tier-based model selection (supports comma-separated: t1,t2)
    MODELS=()
    IFS=',' read -ra TIERS_ARRAY <<< "$TIERS"
    for T in "${TIERS_ARRAY[@]}"; do
        T=$(echo "$T" | xargs)  # trim whitespace
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
    echo "Usage: bash run_dr.sh --models <name>                    # single model"
    echo "       bash run_dr.sh --models <name1,name2>              # multiple models"
    echo "       bash run_dr.sh --tiers <t1|t2|t3|t4|t1,t2|all>      # GPU tier(s)"
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

# ===== Source vLLM utilities if auto mode =====
if [ "$AUTO_VLLM" = true ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    source "${SCRIPT_DIR}/scripts/vllm_utils.sh"
    trap 'stop_vllm' EXIT INT TERM
fi

# ============================================================================
# run_single_model — Run all tasks for one model
# ============================================================================
run_single_model() {
    local CUR_MODEL="$1"

    # ── Auto-find config directory ──
    local CONFIG_DIR=""
    for search_path in configs/dr/llms-api/${CUR_MODEL} configs/dr/llms/${CUR_MODEL} configs/dr/slms/${CUR_MODEL}; do
        if [ -d "$search_path" ]; then
            CONFIG_DIR="$search_path"
            break
        fi
    done

    if [ -z "$CONFIG_DIR" ]; then
        echo "❌ Config directory not found for model: ${CUR_MODEL}"
        echo "Searched: configs/dr/{llms-api,llms,slms}/${CUR_MODEL}"
        return 1
    fi

    # ── Resolve vLLM config paths ──
    local VLLM_YAML="${CONFIG_DIR}/vllm.yaml"
    local SFT_VLLM_YAML="${CONFIG_DIR}/vllm-sft.yaml"

    # ── Pipeline run ID for cost tracking ──
    export PIPELINE_RUN_ID="$(date +%Y%m%d_%H%M%S)_${CUR_MODEL}"

    # ── Split tasks into non-sft and sft groups ──
    IFS=',' read -ra TASK_ARRAY <<< "$TASKS"
    local NON_SFT_TASKS=()
    local SFT_TASKS=()
    for TASK in "${TASK_ARRAY[@]}"; do
        TASK=$(echo "$TASK" | xargs)
        if [ "$TASK" = "sft" ]; then
            SFT_TASKS+=("$TASK")
        else
            NON_SFT_TASKS+=("$TASK")
        fi
    done

    local ALL_TASKS=("${TASK_ARRAY[@]}")
    local TOTAL=${#ALL_TASKS[@]}
    local CURRENT=0

    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  DR Unified Pipeline                              ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║  Model:     $CUR_MODEL"
    echo "║  Tasks:     $TASKS"
    echo "║  Auto vLLM: $AUTO_VLLM"
    if [ "$AUTO_VLLM" = false ]; then
    echo "║  vLLM URL:  ${API_BASE_OVR:-auto-detect}"
    fi
    echo "║  Workers:   $WORKERS"
    echo "║  Config:    $CONFIG_DIR"
    echo "╚══════════════════════════════════════════════════════════╝"

    # ================================================================
    # AUTO_VLLM mode: manage vLLM lifecycle
    # ================================================================
    if [ "$AUTO_VLLM" = true ]; then

        # ── Phase 1: Non-SFT tasks (baseline, 1shot, 2shot, ...) ──
        if [ ${#NON_SFT_TASKS[@]} -gt 0 ]; then
            local VLLM_PORT
            VLLM_PORT=$(get_vllm_port "$VLLM_YAML")
            local CUR_API_BASE="http://localhost:${VLLM_PORT:-9000}/v1"

            stop_vllm
            start_vllm "$VLLM_YAML" "${CUR_MODEL}_zs-fs"
            wait_for_vllm "$CUR_API_BASE"

            for TASK in "${NON_SFT_TASKS[@]}"; do
                CURRENT=$((CURRENT + 1))
                _run_task "$CUR_MODEL" "$TASK" "$CUR_API_BASE" "$CURRENT" "$TOTAL"
            done

            stop_vllm
        fi

        # ── Phase 2: SFT tasks ──
        if [ ${#SFT_TASKS[@]} -gt 0 ]; then
            if [ ! -f "$SFT_VLLM_YAML" ]; then
                echo "⚠️  SFT vLLM config not found: $SFT_VLLM_YAML, skipping SFT tasks"
            else
                local SFT_PORT
                SFT_PORT=$(get_vllm_port "$SFT_VLLM_YAML")
                local SFT_API_BASE="http://localhost:${SFT_PORT:-9000}/v1"

                stop_vllm
                start_vllm "$SFT_VLLM_YAML" "${CUR_MODEL}_sft"
                wait_for_vllm "$SFT_API_BASE"

                for TASK in "${SFT_TASKS[@]}"; do
                    CURRENT=$((CURRENT + 1))
                    _run_task "$CUR_MODEL" "$TASK" "$SFT_API_BASE" "$CURRENT" "$TOTAL"
                done

                stop_vllm
            fi
        fi

    # ================================================================
    # Manual mode: user manages vLLM themselves (original behavior)
    # ================================================================
    else
        # Auto-detect port from vllm.yaml
        local CUR_API_BASE="http://localhost:9000/v1"
        if [ -n "$API_BASE_OVR" ]; then
            CUR_API_BASE="$API_BASE_OVR"
        elif [ -f "$VLLM_YAML" ]; then
            local DETECTED_PORT
            DETECTED_PORT=$(get_vllm_port "$VLLM_YAML" 2>/dev/null || grep '^port:' "$VLLM_YAML" | head -1 | awk '{print $2}')
            if [ -n "$DETECTED_PORT" ]; then
                CUR_API_BASE="http://localhost:${DETECTED_PORT}/v1"
                echo "🔍 [Auto-detect] Found port $DETECTED_PORT in $VLLM_YAML"
            fi
        fi

        for TASK in "${ALL_TASKS[@]}"; do
            CURRENT=$((CURRENT + 1))
            TASK=$(echo "$TASK" | xargs)

            # SFT task: try to use sft port
            local TASK_API_BASE="$CUR_API_BASE"
            if [ "$TASK" = "sft" ]; then
                if [ -f "$SFT_VLLM_YAML" ] && [ -z "$API_BASE_OVR" ]; then
                    local SFT_PORT
                    SFT_PORT=$(grep '^port:' "$SFT_VLLM_YAML" | head -1 | awk '{print $2}')
                    if [ -n "$SFT_PORT" ]; then
                        TASK_API_BASE="http://localhost:${SFT_PORT}/v1"
                        echo "🔍 [Auto-detect] SFT port $SFT_PORT from $SFT_VLLM_YAML"
                    fi
                fi
            fi

            _run_task "$CUR_MODEL" "$TASK" "$TASK_API_BASE" "$CURRENT" "$TOTAL"
        done
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  🎉 All tasks complete: $TASKS"
    echo "║  Model: $CUR_MODEL"
    echo "║  Dataset: $DATASET"
    echo "╚══════════════════════════════════════════════════════════╝"
}

# ============================================================================
# _run_task — Execute prediction + evaluation for a single task
# ============================================================================
_run_task() {
    local CUR_MODEL="$1"
    local TASK="$2"
    local TASK_API_BASE="$3"
    local CURRENT="$4"
    local TOTAL="$5"
    TASK=$(echo "$TASK" | xargs)

    echo ""
    echo "━━━━━ [$CURRENT/$TOTAL] $CUR_MODEL / $TASK ━━━━━"

    # Parse num_shots
    local NUM_SHOTS_ARG=""
    local FEW_SHOT_ARG=""
    if [[ "$TASK" =~ ^([0-9]+)shot$ ]]; then
        NUM_SHOTS_ARG="--num_shots ${BASH_REMATCH[1]}"
        FEW_SHOT_ARG="--few_shot_config ${FEW_SHOT_CONFIG}"
    elif [[ "$TASK" == "baseline" || "$TASK" == "sft" ]]; then
        NUM_SHOTS_ARG="--num_shots 0"
    fi

    # Resolve the model name that vLLM is serving
    local SERVED_MODEL="${CUR_MODEL}"
    if [ "$TASK" = "sft" ]; then
        # SFT vLLM uses a different served_model_name
        local SFT_YAML_PATH=""
        for sp in configs/${DATASET}/llms-api/${CUR_MODEL} configs/${DATASET}/llms/${CUR_MODEL} configs/${DATASET}/slms/${CUR_MODEL}; do
            if [ -f "$sp/vllm-sft.yaml" ]; then
                SFT_YAML_PATH="$sp/vllm-sft.yaml"
                break
            fi
        done
        if [ -n "$SFT_YAML_PATH" ]; then
            local sft_name
            sft_name=$(grep '^served_model_name:' "$SFT_YAML_PATH" | head -1 | awk '{print $2}' | tr -d '"')
            [ -n "$sft_name" ] && SERVED_MODEL="$sft_name"
        fi
    fi

    # Step 1: Prediction
    echo "🚀 [Step 1/2] Prediction..."
    python -m src.main \
        --dataset "${DATASET}" \
        --task prediction \
        --model_name "${SERVED_MODEL}" \
        --run_tag "${TASK}" \
        --api_base_url "${TASK_API_BASE}" \
        --api_key "${API_KEY}" \
        --workers "${WORKERS}" \
        --client_type api \
        ${NUM_SHOTS_ARG} ${FEW_SHOT_ARG}

    # Step 2: Evaluation
    echo "📊 [Step 2/2] Evaluation..."
    python -m src.main \
        --dataset "${DATASET}" \
        --task evaluation \
        --model_name "${SERVED_MODEL}" \
        --run_tag "${TASK}"

    echo "✅ [$TASK] Complete"
}

# ============================================================================
# Main: iterate over all models
# ============================================================================
MODEL_TOTAL=${#MODELS[@]}
MODEL_IDX=0
FAILED=()
SUCCEEDED=()

if [ "$MODEL_TOTAL" -gt 1 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  🚀 Batch Run: $MODEL_TOTAL models | Tasks: $TASKS"
    if [ -n "$TIERS" ]; then
    echo "║  Tier: $TIERS"
    fi
    echo "╚══════════════════════════════════════════════════════════╝"
fi

for M in "${MODELS[@]}"; do
    M=$(echo "$M" | xargs)  # trim whitespace
    MODEL_IDX=$((MODEL_IDX + 1))

    if [ "$MODEL_TOTAL" -gt 1 ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  [$MODEL_IDX/$MODEL_TOTAL] $M"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    fi


    START_TIME=$(date +%s)
    if run_single_model "$M"; then
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
if [ "$MODEL_TOTAL" -gt 1 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  📊 Batch Summary: ${#SUCCEEDED[@]} succeeded, ${#FAILED[@]} failed / $MODEL_TOTAL total"
    if [ ${#FAILED[@]} -gt 0 ]; then
    echo "║  ❌ Failed: ${FAILED[*]}"
    fi
    echo "╚══════════════════════════════════════════════════════════╝"
fi

echo ""
echo "💡 Captioning LLM-as-judge evaluation is auto-enabled when LLM_JUDGE_* vars are set in .env"
echo "   See .env.example for configuration details."

# Exit with error if any model failed
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "💡 To re-run failed models: bash run_dr.sh --models $(IFS=,; echo "${FAILED[*]}") --tasks $TASKS"
    exit 1
fi
