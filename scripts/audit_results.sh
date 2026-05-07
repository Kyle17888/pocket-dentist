#!/bin/bash
# ============================================================================
# audit_results.sh — Audit benchmark results completeness on NeSI
#
# Scans results/datasets/ to check predictions and metrics for all
# 7 datasets × 4 settings × 15 models.
#
# Usage:
#   bash scripts/audit_results.sh
#   bash scripts/audit_results.sh --dataset aariz     # single dataset
#   bash scripts/audit_results.sh --setting sft       # single setting
#   bash scripts/audit_results.sh --issues-only       # only show problems
# ============================================================================

RESULTS_ROOT="results/datasets"

# Parse args
FILTER_DS=""
FILTER_SETTING=""
ISSUES_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) FILTER_DS="$2"; shift 2 ;;
        --setting) FILTER_SETTING="$2"; shift 2 ;;
        --issues-only) ISSUES_ONLY=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🔍 Benchmark Results Audit                                 ║"
echo "║  Root: $RESULTS_ROOT"
[ -n "$FILTER_DS" ] && echo "║  Filter dataset: $FILTER_DS"
[ -n "$FILTER_SETTING" ] && echo "║  Filter setting: $FILTER_SETTING"
echo "╚══════════════════════════════════════════════════════════════╝"

if [ ! -d "$RESULTS_ROOT" ]; then
    echo "❌ Results root not found: $RESULTS_ROOT"
    exit 1
fi

TOTAL=0
OK=0
NO_PREDS=0
NO_METRICS=0
MISSING_SETTINGS=0
ISSUES=()

SETTINGS=("baseline" "1shot" "2shot" "sft")

for ds_dir in "$RESULTS_ROOT"/*/; do
    [ ! -d "$ds_dir" ] && continue
    ds=$(basename "$ds_dir")

    [ -n "$FILTER_DS" ] && [ "$ds" != "$FILTER_DS" ] && continue

    echo ""
    echo "━━━ ${ds^^} ━━━"

    for setting in "${SETTINGS[@]}"; do
        [ -n "$FILTER_SETTING" ] && [ "$setting" != "$FILTER_SETTING" ] && continue

        setting_dir="$ds_dir/$setting"
        if [ ! -d "$setting_dir" ]; then
            echo "  ❌ [$setting] MISSING"
            MISSING_SETTINGS=$((MISSING_SETTINGS + 1))
            ISSUES+=("MISSING_SETTING|$ds|$setting")
            continue
        fi

        model_count=0
        ok_count=0
        issue_models=()

        for model_dir in "$setting_dir"/*/; do
            [ ! -d "$model_dir" ] && continue
            model=$(basename "$model_dir")
            TOTAL=$((TOTAL + 1))
            model_count=$((model_count + 1))

            # Count predictions
            pred_file="$model_dir/predictions.jsonl"
            pred_count=0
            if [ -f "$pred_file" ]; then
                pred_count=$(wc -l < "$pred_file" | tr -d ' ')
            fi

            # Count metrics files
            metrics_count=$(find "$model_dir" -name "metrics.json" | wc -l | tr -d ' ')

            if [ "$pred_count" -gt 0 ] && [ "$metrics_count" -gt 0 ]; then
                OK=$((OK + 1))
                ok_count=$((ok_count + 1))
                if ! $ISSUES_ONLY; then
                    printf "    ✅ %-40s preds=%-5s metrics=%s\n" "$model" "$pred_count" "$metrics_count"
                fi
            elif [ "$pred_count" -eq 0 ]; then
                NO_PREDS=$((NO_PREDS + 1))
                printf "    ❌ %-40s preds=0     NO PREDICTIONS\n" "$model"
                ISSUES+=("NO_PREDS|$ds|$setting|$model")
            else
                NO_METRICS=$((NO_METRICS + 1))
                printf "    ⚠️  %-40s preds=%-5s NO METRICS\n" "$model" "$pred_count"
                ISSUES+=("NO_METRICS|$ds|$setting|$model|$pred_count")
            fi
        done

        if ! $ISSUES_ONLY || [ "$ok_count" -ne "$model_count" ]; then
            echo "  [$setting] $ok_count/$model_count OK"
        fi
    done
done

# ===== Summary =====
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  📊 Summary                                                  ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Total model-setting pairs: %d\n" "$TOTAL"
printf "║  ✅ Complete:               %d\n" "$OK"
printf "║  ❌ No predictions:         %d\n" "$NO_PREDS"
printf "║  ⚠️  No metrics:             %d\n" "$NO_METRICS"
printf "║  ❌ Missing settings:       %d\n" "$MISSING_SETTINGS"
echo "╚══════════════════════════════════════════════════════════════╝"

# ===== Actionable issues =====
if [ ${#ISSUES[@]} -gt 0 ]; then
    echo ""
    echo "🔧 Issues to fix:"
    for issue in "${ISSUES[@]}"; do
        IFS='|' read -ra parts <<< "$issue"
        case "${parts[0]}" in
            MISSING_SETTING)
                echo "  ❌ ${parts[1]}/${parts[2]} — entire setting missing"
                ;;
            NO_PREDS)
                echo "  ❌ ${parts[1]}/${parts[2]}/${parts[3]} — no predictions"
                ;;
            NO_METRICS)
                echo "  ⚠️  ${parts[1]}/${parts[2]}/${parts[3]} — has ${parts[4]} preds but no metrics"
                ;;
        esac
    done
fi
