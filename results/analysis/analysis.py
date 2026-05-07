#!/usr/bin/env python3
"""
Unified Benchmark — Results Analysis & Visualization

Scans results/datasets/<dataset>/<setting>/<model>/<task>/metrics.json
to extract all metrics, then generates:
  1. CSV summary tables (one per setting + one combined)
  2. LaTeX table code for baseline, 1shot, 2shot, sft (written to .md)
  3. Per-dataset visualization charts (models × settings × tasks)
  4. Combined summary chart

Usage (from the project root):
    python results/analysis/analysis.py
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
SETTING_ORDER = ["baseline", "1shot", "2shot", "sft"]
SKIP_DIRS = {"figures", "__pycache__", "analysis", "bills"}

# Model display order: Large VLMs first, then Compact VLMs
API_VLMS = ["gemini-2.5-flash", "gpt-4o-mini"]
LARGE_VLMS = ["Lingshu-32B", "MedMO-8B-Next", "Qwen2.5-VL-7B-Instruct"]
COMPACT_VLMS = [
    "Qwen3.5-4B", "Qwen3-VL-4B-Instruct", "gemma-4-E4B-it",
    "InternVL2_5-4B",
    "medgemma-4b-it", "paligemma2-3b-mix-448",
    "SmolVLM2-2.2B-Instruct", "InternVL3_5-2B-HF",
    "gemma-4-E2B-it", "InternVL3_5-1B-HF",
]
ALL_MODELS_ORDER = API_VLMS + LARGE_VLMS + COMPACT_VLMS

# Primary metric to extract from each task's metrics.json
# Key = task subfolder name, Value = (metric_key, display_name)
TASK_METRIC_MAP = {
    # Aariz
    "aariz_cvm":    ("accuracy", "CVM (Acc)"),
    "aariz_vqa":    ("accuracy", "VQA (Acc)"),
    # BRAR
    "brar_classification": ("accuracy", "BRAR (Acc)"),
    # COde
    "code_classification": ("accuracy", "Cls (Acc)"),
    "code_report":  ("meteor.mean", "Report (METEOR)"),
    # MetaDent
    "vqa":          ("accuracy", "VQA (Acc)"),
    "classification": ("f1_weighted", "Cls (F1)"),
    "captioning":   ("bertscore_f1", "Cap (BERT-F1)"),
    # DenPAR
    "denpar_count": ("accuracy", "Count (Acc)"),
    "denpar_arch":  ("accuracy", "Arch (Acc)"),
    "denpar_site":  ("accuracy", "Site (Acc)"),
    # DR
    "dr_classification": ("f1_weighted", "DR Cls (F1w)"),
    # DentalCaries
    "caries_detect": ("accuracy", "Detect (Acc)"),
    "caries_cls":    ("accuracy", "Cls (Acc)"),
}

# Fallback: if the task name is not in TASK_METRIC_MAP,
# try these keys in order from metrics.json
FALLBACK_METRIC_KEYS = ["accuracy", "f1_weighted", "f1_macro"]

# Color palette for settings (used in LaTeX / summary)
SETTING_COLORS = {
    "baseline": "#4C72B0",
    "1shot":    "#55A868",
    "2shot":    "#C44E52",
    "sft":      "#8172B3",
}

# Color palette for models (used in bar charts)
MODEL_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E58606", "#5D69B1", "#52BCA3", "#99C945", "#CC61B0",
]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATASETS_DIR = PROJECT_ROOT / "results" / "datasets"
OUTPUT_DIR = SCRIPT_DIR  # results/analysis/
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"


# ──────────────────────────────────────────────────────────────
# 1. Extract metrics from all datasets
# ──────────────────────────────────────────────────────────────
def extract_all_metrics() -> pd.DataFrame:
    """
    Walk results/datasets/<dataset>/<setting>/<model>/<task>/metrics.json
    and return a flat DataFrame with columns:
    [dataset, setting, model, task, metric_name, value]
    """
    rows = []

    if not DATASETS_DIR.is_dir():
        print(f"⚠ Datasets directory not found: {DATASETS_DIR}")
        return pd.DataFrame()

    for dataset_dir in sorted(DATASETS_DIR.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name in SKIP_DIRS:
            continue
        dataset_name = dataset_dir.name

        for setting_dir in sorted(dataset_dir.iterdir()):
            if not setting_dir.is_dir() or setting_dir.name in SKIP_DIRS:
                continue
            setting = setting_dir.name

            for model_dir in sorted(setting_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                # Normalize SFT model names: strip '-SFT' suffix so SFT models
                # share the same name as their base counterparts (distinguished by setting column)
                model = model_dir.name.removesuffix("-SFT")

                # Look for task subdirectories containing metrics.json
                for task_dir in sorted(model_dir.iterdir()):
                    if not task_dir.is_dir():
                        continue
                    metrics_file = task_dir / "metrics.json"
                    if not metrics_file.is_file():
                        continue

                    task_name = task_dir.name
                    try:
                        with open(metrics_file) as f:
                            data = json.load(f)

                        # Determine which metric to extract
                        if task_name in TASK_METRIC_MAP:
                            metric_key, display_name = TASK_METRIC_MAP[task_name]
                            value = _extract_value(data, metric_key)
                        else:
                            # Fallback: try common keys
                            display_name = task_name
                            value = None
                            for key in FALLBACK_METRIC_KEYS:
                                value = _extract_value(data, key)
                                if value is not None:
                                    break

                        if value is not None:
                            # Clamp negative values (e.g. negative BERTScore from
                            # models outputting garbage/repetitive text)
                            value = max(0.0, value)
                            rows.append({
                                "dataset": dataset_name,
                                "setting": setting,
                                "model": model,
                                "task": task_name,
                                "display_name": display_name,
                                "value": round(value, 3),
                            })
                    except Exception as e:
                        print(f"  ⚠ Error reading {metrics_file}: {e}")

    return pd.DataFrame(rows)


def _extract_value(data: dict, key: str):
    """Extract a numeric value from metrics dict, handling nested keys like 'meteor.mean'."""
    # Handle dotted keys (e.g. "meteor.mean")
    if "." in key:
        parts = key.split(".", 1)
        if parts[0] in data and isinstance(data[parts[0]], dict):
            return _extract_value(data[parts[0]], parts[1])
        return None
    # Direct key
    if key in data and isinstance(data[key], (int, float)):
        return float(data[key])
    # Nested under 'overall'
    if "overall" in data and isinstance(data["overall"], dict):
        if key in data["overall"]:
            return float(data["overall"][key])
    return None


# ──────────────────────────────────────────────────────────────
# 2. Build pivot tables (one per setting)
# ──────────────────────────────────────────────────────────────
def build_setting_table(df: pd.DataFrame, setting: str) -> pd.DataFrame:
    """
    Build a table: rows = models, columns = dataset/task display_names.
    """
    sub = df[df["setting"] == setting].copy()
    if sub.empty:
        return pd.DataFrame()

    # Create a unique column label: "Dataset / Task"
    sub["col_label"] = sub["dataset"].str.upper() + " / " + sub["display_name"]

    pivot = sub.pivot_table(
        index="model", columns="col_label", values="value", aggfunc="first"
    )

    # Sort models by our predefined order
    model_order = [m for m in ALL_MODELS_ORDER if m in pivot.index]
    remaining = [m for m in pivot.index if m not in model_order]
    pivot = pivot.reindex(model_order + remaining)

    return pivot


# ──────────────────────────────────────────────────────────────
# 3. Generate LaTeX tables
# ──────────────────────────────────────────────────────────────
def generate_latex_tables(df: pd.DataFrame) -> str:
    """Generate LaTeX table code for all 4 settings."""
    latex_parts = []

    for setting in SETTING_ORDER:
        pivot = build_setting_table(df, setting)
        if pivot.empty:
            latex_parts.append(f"% No data for setting: {setting}\n")
            continue

        latex_parts.append(generate_single_latex_table(pivot, setting))

    return "\n\n".join(latex_parts)


def generate_single_latex_table(pivot: pd.DataFrame, setting: str) -> str:
    """Generate a single LaTeX table for one setting."""
    setting_labels = {
        "baseline": "Zero-Shot (Baseline)",
        "1shot": "One-Shot",
        "2shot": "Two-Shot",
        "sft": "Instruction Tuning (LoRA)",
    }

    cols = list(pivot.columns)
    n_cols = len(cols)
    models = list(pivot.index)

    # Find best and second-best per column
    best_vals = {}
    second_vals = {}
    for col in cols:
        valid = pivot[col].dropna()
        if len(valid) >= 1:
            sorted_vals = valid.sort_values(ascending=False)
            best_vals[col] = sorted_vals.iloc[0]
            if len(sorted_vals) >= 2:
                second_vals[col] = sorted_vals.iloc[1]

    # Build LaTeX
    lines = []
    lines.append(f"% ~~~~~~~~~~~~~~~~~~~~~~~~ TABLE: {setting.upper()} ~~~~~~~~~~~~~~~~~~~~~~~~")
    lines.append("\\begin{table*}[!t]")
    lines.append(f"    \\caption{{Performance under \\textbf{{{setting_labels.get(setting, setting)}}}.")
    lines.append(f"    Best in \\textbf{{bold}}, second-best \\underline{{underlined}}.}}")
    lines.append(f"    \\label{{tab:{setting}}}")
    lines.append("    \\centering")
    lines.append("    \\small")
    lines.append("    \\setlength{\\tabcolsep}{3pt}")

    # Column spec
    col_spec = "ll" + "c" * n_cols
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append("        \\toprule")

    # Group columns by dataset
    dataset_groups = defaultdict(list)
    for i, col in enumerate(cols):
        parts = col.split(" / ", 1)
        ds = parts[0] if len(parts) > 1 else ""
        dataset_groups[ds].append((i, col))

    # Header row 1: dataset names with cmidrule
    header1_parts = ["", "Model"]
    cmidrules = []
    col_idx = 3  # starts at column 3 (after row label + Model)
    for ds, group_cols in dataset_groups.items():
        n = len(group_cols)
        header1_parts.append(f"\\multicolumn{{{n}}}{{c}}{{{ds}}}")
        cmidrules.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + n - 1}}}")
        col_idx += n

    lines.append("        " + " & ".join(header1_parts) + " \\\\")
    lines.append("        " + " ".join(cmidrules))

    # Header row 2: task metric names
    header2_parts = ["", ""]
    for col in cols:
        parts = col.split(" / ", 1)
        task_label = parts[1] if len(parts) > 1 else parts[0]
        header2_parts.append(f"{task_label} ($\\uparrow$)")
    lines.append("        " + " & ".join(header2_parts) + " \\\\")

    # Separate Large VLMs and Compact VLMs
    large_models = [m for m in models if m in LARGE_VLMS]
    compact_models = [m for m in models if m not in LARGE_VLMS]

    def format_model_name(name):
        """Escape underscores for LaTeX."""
        return name.replace("_", "\\_")

    def format_value(val, col):
        """Format a value with bold/underline if best/second."""
        if pd.isna(val):
            return "—"
        s = f"{val:.3f}"
        if col in best_vals and val == best_vals[col]:
            return f"\\textbf{{{s}}}"
        if col in second_vals and val == second_vals[col]:
            return f"\\underline{{{s}}}"
        return s

    def add_model_rows(model_list, group_label, lines_out):
        if not model_list:
            return
        n_models = len(model_list)
        lines_out.append(f"        \\midrule \\multirow{{{n_models}}}{{*}}{{\\shortstack{{{group_label}}}}}")
        for model in model_list:
            parts = [f"& {format_model_name(model)}"]
            for col in cols:
                val = pivot.loc[model, col] if model in pivot.index else None
                parts.append(format_value(val, col))
            lines_out.append("            " + " & ".join(parts) + " \\\\")

        # Add Mean and Best rows
        mean_parts = ["", "\\textit{Mean}"]
        best_parts = ["", "\\textit{Best}"]
        for col in cols:
            vals = [pivot.loc[m, col] for m in model_list if m in pivot.index and pd.notna(pivot.loc[m, col])]
            if vals:
                mean_parts.append(f"\\textit{{{np.mean(vals):.3f}}}")
                best_parts.append(f"\\textit{{{np.max(vals):.3f}}}")
            else:
                mean_parts.append("—")
                best_parts.append("—")
        lines_out.append("        \\cmidrule{2-" + str(n_cols + 2) + "}")
        lines_out.append("            " + " & ".join(mean_parts) + " \\\\")
        lines_out.append("        \\cmidrule{2-" + str(n_cols + 2) + "}")
        lines_out.append("            " + " & ".join(best_parts) + " \\\\")

    add_model_rows(large_models, "Large\\\\VLMs", lines)
    add_model_rows(compact_models, "Compact\\\\VLMs", lines)

    lines.append("        \\bottomrule")
    lines.append(f"    \\end{{tabular}}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 4. Per-dataset visualization
#    X-axis = settings (baseline, 1shot, 2shot, sft)
#    Each bar within a group = one model
# ──────────────────────────────────────────────────────────────
def plot_per_dataset(df: pd.DataFrame):
    """
    For each dataset, create a chart with one subplot per task.
    X-axis groups are settings; each bar is a model.
    """
    datasets = sorted(df["dataset"].unique())
    chart_paths = []

    for dataset in datasets:
        ds_df = df[df["dataset"] == dataset]
        tasks = sorted(ds_df["display_name"].unique())
        settings = [s for s in SETTING_ORDER if s in ds_df["setting"].unique()]
        models = _sort_models(ds_df["model"].unique())
        model_colors = {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(models)}

        n_tasks = len(tasks)
        fig, axes = plt.subplots(1, n_tasks, figsize=(max(7 * n_tasks, 9), 6.5))
        if n_tasks == 1:
            axes = [axes]

        for ax, task in zip(axes, tasks):
            task_df = ds_df[ds_df["display_name"] == task]
            n_models = len(models)
            x = np.arange(len(settings))
            width = 0.8 / max(n_models, 1)

            for i, model in enumerate(models):
                vals = []
                for s in settings:
                    match = task_df[(task_df["model"] == model) & (task_df["setting"] == s)]
                    vals.append(match["value"].values[0] if len(match) else 0)

                offset = (i - n_models / 2 + 0.5) * width
                bars = ax.bar(
                    x + offset, vals, width,
                    label=model, color=model_colors[model],
                    edgecolor="white", linewidth=0.5,
                )
                # Value labels
                for bar, v in zip(bars, vals):
                    if v > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{v:.3f}", ha="center", va="bottom",
                            fontsize=5, rotation=60,
                        )

            ax.set_xticks(x)
            ax.set_xticklabels(settings, fontsize=10)
            ax.set_title(task, fontsize=12, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.grid(axis="y", alpha=0.3)

        # Shared legend (model names)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center",
            ncol=min(len(models), 5),
            fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.02),
        )
        fig.suptitle(
            f"{dataset.upper()} — Model Performance Across Settings",
            fontsize=14, fontweight="bold", y=1.08,
        )
        fig.tight_layout()

        chart_path = FIGURES_DIR / f"chart_{dataset}.png"
        fig.savefig(chart_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ Chart saved → {chart_path.relative_to(PROJECT_ROOT)}")
        chart_paths.append(chart_path)

    return chart_paths


# ──────────────────────────────────────────────────────────────
# 5. Combined summary chart
#    X-axis = settings, each bar = model (avg across all tasks)
#    One subplot per dataset
# ──────────────────────────────────────────────────────────────
def plot_combined_summary(df: pd.DataFrame):
    """
    Create a single combined figure with one subplot per dataset.
    X-axis groups are settings; each bar is a model (averaged across tasks).
    """
    datasets = sorted(df["dataset"].unique())
    n_datasets = len(datasets)
    if n_datasets == 0:
        return

    # Collect all models across datasets for consistent coloring
    all_models = _sort_models(df["model"].unique())
    model_colors = {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(all_models)}

    fig, axes = plt.subplots(1, n_datasets, figsize=(max(8 * n_datasets, 10), 7))
    if n_datasets == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        ds_df = df[df["dataset"] == dataset]
        settings = [s for s in SETTING_ORDER if s in ds_df["setting"].unique()]
        models = _sort_models(ds_df["model"].unique())

        # Compute mean across all tasks for each model/setting
        avg_df = ds_df.groupby(["model", "setting"])["value"].mean().reset_index()

        n_models = len(models)
        x = np.arange(len(settings))
        width = 0.8 / max(n_models, 1)

        for i, model in enumerate(models):
            vals = []
            for s in settings:
                match = avg_df[(avg_df["model"] == model) & (avg_df["setting"] == s)]
                vals.append(match["value"].values[0] if len(match) else 0)

            offset = (i - n_models / 2 + 0.5) * width
            ax.bar(
                x + offset, vals, width,
                label=model, color=model_colors[model],
                edgecolor="white", linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(settings, fontsize=10)
        ax.set_title(f"{dataset.upper()}\n(avg across tasks)", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", alpha=0.3)

    # Shared legend (model names)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        ncol=min(len(all_models), 5),
        fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Combined Benchmark Summary — All Datasets",
        fontsize=14, fontweight="bold", y=1.10,
    )
    fig.tight_layout()

    chart_path = FIGURES_DIR / "chart_combined_summary.png"
    fig.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Combined chart saved → {chart_path.relative_to(PROJECT_ROOT)}")


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _sort_models(model_list):
    """Sort models according to predefined order."""
    ordered = [m for m in ALL_MODELS_ORDER if m in model_list]
    remaining = sorted(set(model_list) - set(ALL_MODELS_ORDER))
    return ordered + remaining


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    # Create output subdirectories
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("━" * 70)
    print("  Unified Benchmark — Results Analysis")
    print("━" * 70)
    print(f"  Data source: {DATASETS_DIR}")
    print(f"  Output dir:  {OUTPUT_DIR}")
    print(f"    ├── tables/  → CSV + LaTeX (.md)")
    print(f"    └── figures/ → charts (.png)")
    print()

    # 1. Extract all metrics
    df = extract_all_metrics()
    if df.empty:
        print("⚠ No results found!")
        sys.exit(1)

    print(f"  📊 Found {len(df)} metric entries across "
          f"{df['dataset'].nunique()} datasets, "
          f"{df['model'].nunique()} models, "
          f"{df['setting'].nunique()} settings")
    print()

    # 2. Save combined CSV
    csv_path = TABLES_DIR / "all_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ Combined CSV saved → {csv_path.relative_to(PROJECT_ROOT)}")

    # 3. Save per-setting CSVs
    for setting in SETTING_ORDER:
        pivot = build_setting_table(df, setting)
        if not pivot.empty:
            setting_csv = TABLES_DIR / f"table_{setting}.csv"
            pivot.to_csv(setting_csv)
            print(f"  ✅ {setting} CSV saved → {setting_csv.relative_to(PROJECT_ROOT)}")

    # 4. Generate LaTeX tables and save as .md
    latex_code = generate_latex_tables(df)
    md_path = TABLES_DIR / "latex_tables.md"
    with open(md_path, "w") as f:
        f.write("# Benchmark LaTeX Tables\n\n")
        f.write("Copy the table code below into your Overleaf document.\n\n")
        for setting in SETTING_ORDER:
            f.write(f"## {setting.capitalize()} Table\n\n")
            f.write("```latex\n")
            # Extract the relevant section
            marker = f"% ~~~~~~~~~~~~~~~~~~~~~~~~ TABLE: {setting.upper()}"
            if marker in latex_code:
                start = latex_code.index(marker)
                end_marker = "\\end{table*}"
                end = latex_code.index(end_marker, start) + len(end_marker)
                f.write(latex_code[start:end])
            f.write("\n```\n\n")

    print(f"  ✅ LaTeX tables saved → {md_path.relative_to(PROJECT_ROOT)}")

    # 5. Generate per-dataset charts
    print()
    print("  📈 Generating per-dataset charts...")
    plot_per_dataset(df)

    # 6. Generate combined summary chart
    print()
    print("  📈 Generating combined summary chart...")
    plot_combined_summary(df)

    print()
    print("━" * 70)
    print("  ✅ Analysis complete!")
    print("━" * 70)


if __name__ == "__main__":
    main()
