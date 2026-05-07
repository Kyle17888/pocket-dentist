#!/usr/bin/env python3
"""
summary_overleaf_table.py — Extract metrics & generate Overleaf tables

Reads every metrics.json under results/datasets/<dataset>/<setting>/<model>/<task>/
and outputs:
  1. overleaf_table_metrics.json — raw 3-decimal precision values (ground truth)
  2. tables_all.tex — LaTeX tables with 2-decimal display, bold/underline based on
     3-decimal ranking (ties broken by first-in-table-order)

Usage (from project root):
    python results/datasets/summary_overleaf_table.py                # extract JSON only
    python results/datasets/summary_overleaf_table.py --verify       # extract + verify vs tex
    python results/datasets/summary_overleaf_table.py --generate     # extract + generate tex
    python results/datasets/summary_overleaf_table.py --generate --verify  # full pipeline
"""

import json
import re
import sys
from pathlib import Path

# ─── Paths ───
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATASETS_DIR = SCRIPT_DIR  # results/datasets/
OUTPUT_JSON = SCRIPT_DIR / "overleaf_table_metrics.json"
TEX_PATH = PROJECT_ROOT / "neurips-paper" / "overleaf" / "v0.1.0" / "tables_all.tex"

# ─── Model display names: internal folder name → paper name ───
MODEL_DISPLAY = {
    "Lingshu-32B": "Lingshu-32B",
    "MedMO-8B-Next": "MedMO-8B-Next",
    "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "Qwen3.5-4B": "Qwen3.5-4B",
    "Qwen3-VL-4B-Instruct": "Qwen3-VL-4B",
    "gemma-4-E4B-it": "gemma-4-E4B-it",
    "medgemma-4b-it": "medgemma-4b-it",
    "paligemma2-3b-mix-448": "paligemma2-3b",
    "SmolVLM2-2.2B-Instruct": "SmolVLM2-2.2B",
    "InternVL3_5-2B-HF": "InternVL3.5-2B",
    "gemma-4-E2B-it": "gemma-4-E2B-it",
    "InternVL3_5-1B-HF": "InternVL3.5-1B",
}

# ─── Column definitions ───
# (col_key, dataset_folder, task_folder, metric_key, lower_is_better)
COLUMN_DEFS = [
    ("BRAR_Acc",         "brar",         "brar_classification",  "accuracy",      False),
    ("BRAR_F1",          "brar",         "brar_classification",  "f1_macro",      False),
    ("DR_F1w",           "dr",           "dr_classification",    "f1_weighted",   False),
    ("MetaDent_VQA",     "metadent",     "vqa",                  "accuracy",      False),
    ("MetaDent_Cap",     "metadent",     "captioning",           "bertscore_f1",  False),
    ("MetaDent_Cls",     "metadent",     "classification",       "f1_weighted",   False),
    ("Aariz_VQA",        "aariz",        "aariz_vqa",            "accuracy",      False),
    ("Aariz_CVM",        "aariz",        "aariz_cvm",            "accuracy",      False),
    ("COde_Cls",         "code",         "code_classification",  "f1_weighted",   False),
    ("DenPAR_Arch",      "denpar",       "denpar_arch",          "accuracy",      False),
    ("DenPAR_Site",      "denpar",       "denpar_site",          "f1_weighted",   False),
    ("DenPAR_MAE",       "denpar",       "denpar_count",         "mae",           True),
    ("DentalCaries_Det", "dentalcaries", "caries_detect",        "accuracy",      False),
    ("DentalCaries_Cls", "dentalcaries", "caries_cls",           "f1_weighted",   False),
]

COL_KEYS = [c[0] for c in COLUMN_DEFS]
LOWER_IS_BETTER = {c[0]: c[4] for c in COLUMN_DEFS}

# Settings
SETTINGS = {"baseline": "", "1shot": "", "2shot": "", "sft": "-SFT"}
SETTING_ORDER = ["baseline", "1shot", "2shot", "sft"]

# Model order
LARGE_VLMS_ZS = ["Lingshu-32B", "MedMO-8B-Next", "Qwen2.5-VL-7B-Instruct",
                 "gemini-2.0-flash", "gemini-2.5-flash"]
LARGE_VLMS_SFT = ["Lingshu-32B", "MedMO-8B-Next", "Qwen2.5-VL-7B-Instruct"]
COMPACT_VLMS = [
    "Qwen3.5-4B", "Qwen3-VL-4B-Instruct", "gemma-4-E4B-it",
    "medgemma-4b-it", "paligemma2-3b-mix-448",
    "SmolVLM2-2.2B-Instruct", "InternVL3_5-2B-HF",
    "gemma-4-E2B-it", "InternVL3_5-1B-HF",
]

# LaTeX table metadata
DATASET_GROUPS = [("BRAR", 2), ("DR", 1), ("MetaDent", 3), ("Aariz", 2),
                  ("COde", 1), ("DenPAR", 3), ("DentalCaries", 2)]
COL_HEADERS = ["Acc", "F1", "F1w", "VQA", "Cap", "Cls", "VQA", "CVM",
               "Cls", "Arch", "Site", r"MAE$\downarrow$", "Det", "Cls"]

SETTING_LABELS = {
    "baseline": ("ZS", "tab:zs",
        r"Performance of large-scale and compact VLMs under zero-shot (ZS) setting across all seven benchmarks. Best result is \textbf{bold}, second-best is \underline{underlined}. ``--'' indicates architectural incompatibility or complete parse failure."),
    "1shot": ("FS-1", "tab:fs1",
        r"Performance under 1-shot (FS-1) setting across all seven benchmarks. Best result is \textbf{bold}, second-best is \underline{underlined}."),
    "2shot": ("FS-2", "tab:fs2",
        r"Performance under 2-shot (FS-2) setting across all seven benchmarks. Best result is \textbf{bold}, second-best is \underline{underlined}."),
    "sft": ("SFT", "tab:sft",
        r"Performance under LoRA instruction tuning (SFT) across all seven benchmarks. Best result is \textbf{bold}, second-best is \underline{underlined}. Aariz SFT results show mode collapse due to extreme class imbalance (see Section~\ref{sec:threats})."),
}


# ═══════════════════════════════════════════════════
#  1. EXTRACTION
# ═══════════════════════════════════════════════════

def extract_value(data: dict, key: str):
    """Extract a numeric value, clamp negatives to 0, keep 3-decimal precision."""
    if key in data and isinstance(data[key], (int, float)):
        val = max(0.0, float(data[key]))
        return round(val, 3)
    return None


def extract_all_metrics() -> dict:
    """Scan results/datasets/ and extract all table-relevant metrics (3-decimal)."""
    result = {}
    for setting, suffix in SETTINGS.items():
        models = (LARGE_VLMS_SFT if setting == "sft" else LARGE_VLMS_ZS) + COMPACT_VLMS
        setting_data = {}
        for model_id in models:
            display = MODEL_DISPLAY.get(model_id, model_id)
            model_folder = model_id + suffix
            row = {}
            for col_key, ds, task, metric_key, _ in COLUMN_DEFS:
                metrics_file = DATASETS_DIR / ds / setting / model_folder / task / "metrics.json"
                if metrics_file.is_file():
                    try:
                        with open(metrics_file) as f:
                            data = json.load(f)
                        row[col_key] = extract_value(data, metric_key)
                    except Exception as e:
                        print(f"  ⚠ Error reading {metrics_file}: {e}")
                        row[col_key] = None
                else:
                    row[col_key] = None
            setting_data[display] = row
        result[setting] = setting_data
    return result


# ═══════════════════════════════════════════════════
#  2. TEX GENERATION
# ═══════════════════════════════════════════════════

def fmt2(val):
    """Format value to 2 decimal places for display."""
    if val is None:
        return "--"
    if val >= 10:
        return f"{val:.1f}"
    return f"{val:.2f}"


def generate_tex(metrics: dict) -> str:
    """Generate tables_all.tex from metrics JSON.

    Bold/underline logic:
    - Ranking uses 3-decimal precision (from JSON)
    - Display uses 2-decimal precision
    - Bold = best unique 3-decimal value (all ties get bold)
    - Underline = second-best unique 3-decimal value (only FIRST occurrence)
    """
    lines = []
    lines.append(r"% ═══════════════════════════════════════════════════")
    lines.append(r"% AUTO-GENERATED BENCHMARK TABLES")
    lines.append(r"% Generated by: results/datasets/summary_overleaf_table.py --generate")
    lines.append(r"% Ranking uses 3-decimal precision; display uses 2-decimal")
    lines.append(r"% Bold = best, Underline = second-best (first occurrence only)")
    lines.append(r"% ═══════════════════════════════════════════════════")
    lines.append("")

    for setting in SETTING_ORDER:
        short, tab_label, caption = SETTING_LABELS[setting]
        data = metrics[setting]

        if setting == "sft":
            large_ids = [MODEL_DISPLAY[m] for m in LARGE_VLMS_SFT]
            large_tag = r"Large\\VLMs\\(LoRA)"
            compact_tag = r"Compact\\VLMs\\(LoRA)"
        else:
            large_ids = [MODEL_DISPLAY[m] for m in LARGE_VLMS_ZS]
            large_tag = rf"Large\\VLMs\\({short})"
            compact_tag = rf"Compact\\VLMs\\({short})"

        compact_ids = [MODEL_DISPLAY[m] for m in COMPACT_VLMS]
        all_models = [m for m in (large_ids + compact_ids) if m in data]

        # ── Compute best / second-best per column (3-decimal) ──
        col_best = {}      # col_key → best 3-decimal value
        col_second = {}    # col_key → second-best 3-decimal value
        for ck in COL_KEYS:
            lower = LOWER_IS_BETTER[ck]
            vals = [data[m][ck] for m in all_models if data[m].get(ck) is not None]
            if not vals:
                continue
            unique = sorted(set(vals), reverse=not lower)
            col_best[ck] = unique[0]
            if len(unique) >= 2:
                col_second[ck] = unique[1]

        # Track which columns already used their one bold / underline
        bold_used = set()
        underline_used = set()

        def fmt_cell(model: str, ck: str) -> str:
            val = data[model].get(ck)
            display = fmt2(val)
            if val is None:
                return "--"
            if ck in col_best and val == col_best[ck] and ck not in bold_used:
                bold_used.add(ck)
                return rf"\textbf{{{display}}}"
            if ck in col_second and val == col_second[ck] and ck not in underline_used:
                underline_used.add(ck)
                return rf"\underline{{{display}}}"
            return display

        # ── Build table ──
        lines.append(f"% ~~~~~~~~~~~~~~~~~~~~~~~~ TABLE: {short} ~~~~~~~~~~~~~~~~~~~~~~~~")
        lines.append(r"\begin{table*}[!t]")
        lines.append(f"    \\caption{{{caption}}}")
        lines.append(f"    \\label{{{tab_label}}}")
        lines.append(r"    \centering\scriptsize\setlength{\tabcolsep}{2pt}")
        lines.append(r"    \resizebox{\textwidth}{!}{%")
        lines.append(r"    \begin{tabular}{llcccccccccccccc}")
        lines.append(r"        \toprule")

        # Multi-column header
        hdr_parts = [r"\multirow{2}{*}{}", r"\multirow{2}{*}{Model}"]
        cmr_parts = []
        col_start = 3
        for ds_name, ds_cnt in DATASET_GROUPS:
            hdr_parts.append(rf"\multicolumn{{{ds_cnt}}}{{c}}{{{ds_name}}}")
            cmr_parts.append(rf"\cmidrule(lr){{{col_start}-{col_start + ds_cnt - 1}}}")
            col_start += ds_cnt
        lines.append("        " + " & ".join(hdr_parts) + r" \\")
        lines.append("        " + " ".join(cmr_parts))

        # Sub-header
        lines.append("        & & " + " & ".join(COL_HEADERS) + r" \\")
        lines.append(r"        \midrule")

        # Large VLMs block
        large_in = [m for m in large_ids if m in data]
        first = large_in[0]
        cells = " & ".join(fmt_cell(first, ck) for ck in COL_KEYS)
        lines.append(
            rf"        \multirow{{{len(large_in)}}}*{{\shortstack{{{large_tag}}}}} "
            rf" & {first}  & {cells} \\")
        for m in large_in[1:]:
            cells = " & ".join(fmt_cell(m, ck) for ck in COL_KEYS)
            lines.append(rf"           & {m}   & {cells} \\")

        lines.append(r"        \midrule")

        # Compact VLMs block
        compact_in = [m for m in compact_ids if m in data]
        first = compact_in[0]
        cells = " & ".join(fmt_cell(first, ck) for ck in COL_KEYS)
        lines.append(
            rf"        \multirow{{{len(compact_in)}}}*{{\shortstack{{{compact_tag}}}}} "
            rf" & {first}  & {cells} \\")
        for m in compact_in[1:]:
            cells = " & ".join(fmt_cell(m, ck) for ck in COL_KEYS)
            lines.append(rf"           & {m}   & {cells} \\")

        lines.append(r"        \bottomrule")
        lines.append(r"    \end{tabular}%")
        lines.append(r"    }")
        lines.append(r"\end{table*}")
        lines.append("")

    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════
#  3. VERIFICATION
# ═══════════════════════════════════════════════════

def strip_tex_fmt(s: str) -> str:
    s = s.strip()
    m = re.match(r'\\textbf\{([^}]+)\}', s) or re.match(r'\\underline\{([^}]+)\}', s)
    return m.group(1) if m else s


def parse_tex_value(s: str):
    raw = strip_tex_fmt(s).strip()
    if raw in ('--', '—', '', '-'):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def verify_against_tex(metrics: dict):
    """Compare 2-decimal rounded JSON values against tables_all.tex cell values."""
    if not TEX_PATH.is_file():
        print(f"⚠ LaTeX file not found: {TEX_PATH}")
        return False

    with open(TEX_PATH) as f:
        tex_lines = f.readlines()

    tables = []
    i = 0
    while i < len(tex_lines):
        if '\\begin{table*}' in tex_lines[i]:
            start = i
            while i < len(tex_lines) and '\\end{table*}' not in tex_lines[i]:
                i += 1
            tables.append((start, i))
        i += 1

    if len(tables) != 4:
        print(f"⚠ Expected 4 tables, found {len(tables)}")
        return False

    all_ok = True
    total_cells = 0
    mismatches = []

    for ti, (ts, te) in enumerate(tables):
        setting = SETTING_ORDER[ti]
        setting_metrics = metrics.get(setting, {})

        for li in range(ts, te):
            line = tex_lines[li]
            if '&' not in line or '\\\\' not in line:
                continue
            if any(kw in line for kw in ['Model', 'multicolumn', 'cmidrule',
                                          'toprule', 'bottomrule', 'midrule']):
                continue

            parts = line.split('&')
            if len(parts) < 10:
                continue

            model_raw = parts[1].strip()
            model_match = None
            for display in setting_metrics:
                if display in model_raw:
                    model_match = display
                    break
            if model_match is None:
                continue

            tex_vals = []
            for p in parts[2:]:
                clean = p.replace('\\\\', '').strip()
                tex_vals.append(parse_tex_value(clean))

            json_row = setting_metrics.get(model_match, {})
            for ci, col_key in enumerate(COL_KEYS):
                if ci >= len(tex_vals):
                    break
                tex_v = tex_vals[ci]
                json_v = json_row.get(col_key)
                total_cells += 1

                # Round JSON to 2 decimals for comparison with tex display
                json_v_display = round(json_v, 2) if json_v is not None else None

                if tex_v is None and json_v_display is None:
                    continue
                if tex_v is None or json_v_display is None:
                    mismatches.append({
                        "setting": setting, "model": model_match, "column": col_key,
                        "tex": tex_v, "json": json_v, "json_2d": json_v_display,
                        "line": li + 1
                    })
                    all_ok = False
                    continue
                if round(tex_v, 2) != json_v_display:
                    mismatches.append({
                        "setting": setting, "model": model_match, "column": col_key,
                        "tex": tex_v, "json": json_v, "json_2d": json_v_display,
                        "line": li + 1
                    })
                    all_ok = False

    print(f"\n{'━' * 70}")
    print(f"  Verification: {total_cells} cells checked")
    print(f"{'━' * 70}")
    if all_ok:
        print(f"  ✅ ALL CELLS MATCH — tables_all.tex is consistent with source data")
    else:
        print(f"  ❌ {len(mismatches)} MISMATCH(ES) FOUND:\n")
        for m in mismatches:
            print(f"    [{m['setting']}] {m['model']} / {m['column']}: "
                  f"tex={m['tex']}  vs  json_3d={m['json']}  json_2d={m['json_2d']}  "
                  f"(line {m['line']})")
    print(f"{'━' * 70}")
    return all_ok


# ═══════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════

def main():
    do_verify = "--verify" in sys.argv
    do_generate = "--generate" in sys.argv

    print("━" * 70)
    print("  Overleaf Table Metrics Pipeline")
    print("━" * 70)
    print(f"  Source:  {DATASETS_DIR}")
    print(f"  Output:  {OUTPUT_JSON}")
    if do_generate:
        print(f"  TeX:     {TEX_PATH}")
    print()

    # 1. Extract (3-decimal precision)
    metrics = extract_all_metrics()
    n_filled = sum(1 for s in metrics.values() for m in s.values()
                   for v in m.values() if v is not None)
    n_total = sum(len(m) for s in metrics.values() for m in s.values())
    print(f"  📊 Extracted: {len(metrics)} settings, "
          f"{sum(len(v) for v in metrics.values())} model-rows, "
          f"{n_filled}/{n_total} cells filled (3-decimal precision)")

    # 2. Save JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✅ JSON saved → {OUTPUT_JSON.relative_to(PROJECT_ROOT)}")

    # 3. Generate LaTeX
    if do_generate:
        tex = generate_tex(metrics)
        with open(TEX_PATH, 'w') as f:
            f.write(tex)
        print(f"  ✅ LaTeX generated → {TEX_PATH.relative_to(PROJECT_ROOT)}")

    # 4. Verify
    if do_verify:
        verify_against_tex(metrics)
    elif not do_generate:
        print(f"\n  💡 --generate  to regenerate tables_all.tex")
        print(f"  💡 --verify    to check tables_all.tex vs source data")

    return 0


if __name__ == "__main__":
    sys.exit(main())
