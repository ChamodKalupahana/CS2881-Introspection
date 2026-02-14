"""Plot patching results (strength-based): attn vs mlp across layers and concepts.

Reads all CSVs from a results directory (auto-detects ablate or patch mode).
Generates:
  1. Multi-concept heatmap (layers × sublayer → reported strength)
  2. Strength score line plot (score across layers)
  3. Per-concept stacked bars

Usage:
    python plots/plot_ablation_attn_mlp_strength.py
    python plots/plot_ablation_attn_mlp_strength.py --input_dir acitvation_patching/results/strength
"""

import argparse
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CATEGORIES = [
    "very_strong",
    "strong",
    "moderate",
    "weak",
    "unknown",
]

# Color scheme for strength categories
CAT_COLORS = {
    "very_strong": "#e74c3c",  # red
    "strong":      "#e67e22",  # orange
    "moderate":    "#f1c40f",  # yellow
    "weak":        "#2ecc71",  # green
    "unknown":     "#95a5a6",  # grey
}

# Numeric score: higher = stronger reported strength
CAT_SCORE = {
    "very_strong": 1.0,
    "strong":      0.75,
    "moderate":    0.5,
    "weak":        0.25,
    "unknown":     0.0,
}

# Numeric encoding for heatmap
CAT_HEATMAP_VAL = {
    "very_strong": 4,
    "strong":      3,
    "moderate":    2,
    "weak":        1,
    "unknown":     0,
}

# Map expected_strength label (title-case from CSV) to CAT_COLORS key
_EXPECTED_TO_KEY = {
    "weak": "weak",
    "moderate": "moderate",
    "strong": "strong",
    "very strong": "very_strong",
}

def parse_csv(csv_path):
    """Parse a strength patching/ablation CSV.
    Columns: patch_target, reported_strength, expected_strength, strength_judge, raw_response_prefix
    """
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = row["patch_target"].strip()
            m = re.match(r"L(\d+)_(attn|mlp)", target)
            if not m:
                continue
            layer = int(m.group(1))
            sublayer = m.group(2)

            cat = row.get("reported_strength", "unknown").strip()
            if cat not in CATEGORIES:
                cat = "unknown"

            rows.append({
                "layer": layer,
                "sublayer": sublayer,
                "category": cat,
                "expected": row.get("expected_strength", ""),
                "judge": row.get("strength_judge", ""),
                "raw": row.get("raw_response_prefix", ""),
            })
    return rows


def extract_concept_from_filename(filename):
    """Extract injected concept name from various filename patterns."""
    # Pattern: patch_attn_mlp_inject_Dust_clean_Thunder_coeff8.0_...
    m = re.search(r"inject_(.+?)_clean_", filename)
    if m:
        return m.group(1)
    # Pattern: ablate_attn_mlp_Magnetism_coeff8.0_...
    # Pattern: patch_attn_mlp_Phosphorus_coeff8.0_...
    m = re.search(r"(?:ablate|patch)_attn_mlp_(.+?)_coeff", filename)
    if m:
        return m.group(1)
    return filename


def load_all_csvs(input_dir, mode=None):
    """Load ablation/patch CSVs from directory, filtered by mode. Returns dict: concept -> rows."""
    data = {}
    for csv_path in sorted(input_dir.glob("*.csv")):
        # Skip files that don't look like attn_mlp results
        if "attn_mlp" not in csv_path.name:
            continue
        # Filter by mode prefix (patch_ or ablate_)
        if mode and not csv_path.name.startswith(mode):
            continue
        concept = extract_concept_from_filename(csv_path.name)
        data[concept] = parse_csv(csv_path)
    return data


def detect_mode(input_dir):
    """Auto-detect whether directory contains ablation or patch results."""
    for f in input_dir.glob("*.csv"):
        if f.name.startswith("ablate"):
            return "ablate"
        if f.name.startswith("patch"):
            return "patch"
    return "unknown"


# ── Plot 1: Multi-concept heatmap ────────────────────────────────────────────

def plot_multi_concept_heatmap(data, save_dir, mode="ablate", expected_strength=None):
    """
    Heatmap grid: rows = layers (16-31), columns = concept × sublayer.
    Cell color = detection category.
    """
    mode_title = "Ablation" if mode == "ablate" else "Patch"
    concepts = sorted(data.keys())
    layers = list(range(16, 32))
    n_layers = len(layers)
    n_cols = len(concepts) * 2  # attn + mlp per concept

    # Build matrix
    matrix = np.full((n_layers, n_cols), -1, dtype=int)
    col_labels = []

    for ci, concept in enumerate(concepts):
        rows = data[concept]
        for sublayer_idx, sublayer in enumerate(["attn", "mlp"]):
            col = ci * 2 + sublayer_idx
            col_labels.append(f"{concept}\n{sublayer}")
            for row in rows:
                if row["sublayer"] == sublayer and row["layer"] in layers:
                    li = row["layer"] - 16
                    matrix[li, col] = CAT_HEATMAP_VAL[row["category"]]

    # Custom colormap: grey(unknown) → green(weak) → yellow(moderate) → orange(strong) → red(very_strong)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap_colors = [
        CAT_COLORS["unknown"],
        CAT_COLORS["weak"],
        CAT_COLORS["moderate"],
        CAT_COLORS["strong"],
        CAT_COLORS["very_strong"],
    ]
    cmap = ListedColormap(cmap_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.2), n_layers * 0.55 + 2))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=9, ha="center")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{l}" for l in layers], fontsize=10)

    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(f"{mode_title} Results: Reported Strength per Layer × Sublayer",
                 fontsize=14, fontweight="bold", pad=12)

    # Add vertical separators between concepts
    for ci in range(1, len(concepts)):
        ax.axvline(x=ci * 2 - 0.5, color="white", linewidth=2.5)

    # Add subtle separator between attn/mlp within each concept
    for ci in range(len(concepts)):
        ax.axvline(x=ci * 2 + 0.5, color="white", linewidth=0.8, alpha=0.5)

    # Cell text
    cat_short = {0: "?", 1: "W", 2: "M", 3: "S", 4: "VS"}
    for i in range(n_layers):
        for j in range(n_cols):
            val = matrix[i, j]
            if val >= 0:
                txt = cat_short[val]
                text_color = "white" if val in [0, 4] else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, fontweight="bold", color=text_color)

    # Legend
    legend_elements = [
        Patch(facecolor=CAT_COLORS["very_strong"], label="Very Strong (VS)"),
        Patch(facecolor=CAT_COLORS["strong"], label="Strong (S)"),
        Patch(facecolor=CAT_COLORS["moderate"], label="Moderate (M)"),
        Patch(facecolor=CAT_COLORS["weak"], label="Weak (W)"),
        Patch(facecolor=CAT_COLORS["unknown"], label="Unknown (?)"),
    ]
    if expected_strength:
        key = _EXPECTED_TO_KEY.get(expected_strength.lower())
        color = CAT_COLORS.get(key, "#95a5a6") if key else "#95a5a6"
        legend_elements.append(Patch(facecolor=color, edgecolor="black", linewidth=2,
                                     label=f"Ground Truth: {expected_strength}"))
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=9, frameon=True, title="Strength", title_fontsize=10)

    fig.tight_layout()
    path = save_dir / f"{mode}_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {path}")


# ── Plot 2: Detection survival rate ──────────────────────────────────────────

def plot_strength_score(data, save_dir, mode="patch", expected_strength=None):
    """
    Line plot: strength score across layers, separate lines for attn/mlp.
    One subplot per concept.
    """
    mode_label = "Ablation" if mode == "ablate" else "Patch"
    concepts = sorted(data.keys())
    n_concepts = len(concepts)
    layers = list(range(16, 32))

    cols = min(3, n_concepts)
    rows_grid = (n_concepts + cols - 1) // cols
    fig, axes = plt.subplots(rows_grid, cols, figsize=(6 * cols, 4 * rows_grid),
                              squeeze=False, sharey=True)

    for ci, concept in enumerate(concepts):
        r, c = divmod(ci, cols)
        ax = axes[r][c]

        attn_scores = []
        mlp_scores = []

        for layer in layers:
            for row in data[concept]:
                if row["layer"] == layer:
                    score = CAT_SCORE[row["category"]]
                    if row["sublayer"] == "attn":
                        attn_scores.append(score)
                    else:
                        mlp_scores.append(score)

        ax.plot(layers, attn_scores, 'o-', color="#3498db", linewidth=2,
                markersize=6, label="attn patched", alpha=0.9)
        ax.plot(layers, mlp_scores, 's-', color="#e67e22", linewidth=2,
                markersize=6, label="mlp patched", alpha=0.9)

        ax.set_title(concept, fontsize=13, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_xticks(layers)
        ax.set_xticklabels([str(l) for l in layers], fontsize=8, rotation=45)
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(y=0.5, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)

        if c == 0:
            ax.set_ylabel("Strength Score", fontsize=10)
        if ci == 0:
            ax.legend(fontsize=9, loc="lower right")

    # Hide unused
    for ci in range(n_concepts, rows_grid * cols):
        r, c = divmod(ci, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"Reported Strength After {mode_label} (attn vs mlp)",
                 fontsize=15, fontweight="bold", y=1.02)
    if expected_strength:
        key = _EXPECTED_TO_KEY.get(expected_strength.lower())
        color = CAT_COLORS.get(key, "#95a5a6") if key else "#95a5a6"
        gt_patch = Patch(facecolor=color, edgecolor="black", linewidth=2,
                         label=f"Ground Truth: {expected_strength}")
        fig.legend(handles=[gt_patch], loc="upper right", fontsize=9, frameon=True,
                   bbox_to_anchor=(0.99, 0.99))
    fig.tight_layout()
    path = save_dir / f"{mode}_survival.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {path}")


# ── Plot 3: Stacked bars per concept ─────────────────────────────────────────

def plot_stacked_bars(data, save_dir, mode="ablate", expected_strength=None):
    """
    For each concept: paired stacked bars (attn|mlp) across layers.
    Segments colored by detection category.
    """
    concepts = sorted(data.keys())
    layers = list(range(16, 32))
    n_layers = len(layers)

    n_concepts = len(concepts)
    cols = min(2, n_concepts)
    rows_grid = (n_concepts + cols - 1) // cols

    fig, axes = plt.subplots(rows_grid, cols,
                              figsize=(max(12, n_layers * 0.9), 5 * rows_grid),
                              squeeze=False)

    for ci, concept in enumerate(concepts):
        r, c = divmod(ci, cols)
        ax = axes[r][c]

        x = np.arange(n_layers)
        width = 0.35

        # Build category arrays for attn and mlp
        for si, (sublayer, offset) in enumerate([("attn", -width/2), ("mlp", width/2)]):
            bottoms = np.zeros(n_layers)
            for cat in CATEGORIES:
                heights = np.zeros(n_layers)
                for row in data[concept]:
                    if row["sublayer"] == sublayer and row["layer"] in layers:
                        li = row["layer"] - 16
                        if row["category"] == cat:
                            heights[li] = 1.0

                ax.bar(x + offset, heights, width, bottom=bottoms,
                       color=CAT_COLORS[cat], edgecolor="white", linewidth=0.3,
                       label=cat if (ci == 0 and si == 0) else "_nolegend_")
                bottoms += heights

        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8, rotation=45)
        ax.set_title(concept, fontsize=13, fontweight="bold")
        ax.set_ylabel("Strength", fontsize=10)
        ax.set_ylim(0, 1.15)

        # Label attn/mlp
        ax.text(0 - width/2, 1.05, "A", ha="center", fontsize=7, color="#3498db", fontweight="bold")
        ax.text(0 + width/2, 1.05, "M", ha="center", fontsize=7, color="#e67e22", fontweight="bold")

    # Hide unused
    for ci in range(n_concepts, rows_grid * cols):
        r, c = divmod(ci, cols)
        axes[r][c].set_visible(False)

    # Single legend
    legend_elements = [Patch(facecolor=CAT_COLORS[cat], label=cat.replace("_", " ").title())
                       for cat in CATEGORIES]
    if expected_strength:
        key = _EXPECTED_TO_KEY.get(expected_strength.lower())
        color = CAT_COLORS.get(key, "#95a5a6") if key else "#95a5a6"
        legend_elements.append(Patch(facecolor=color, edgecolor="black", linewidth=2,
                                     label=f"Ground Truth: {expected_strength}"))
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=len(legend_elements), fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.02))

    mode_label = "Ablation" if mode == "ablate" else "Patch"
    fig.suptitle(f"{mode_label}: Strength Breakdown by Layer (A=attn, M=mlp)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = save_dir / f"{mode}_stacked_bars.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot ablation/patch results (attn vs mlp)")
    parser.add_argument("--input_dir", type=str,
                        default=str(PROJECT_ROOT / "acitvation_patching" / "results" / "strength"))
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: plots/ablation or plots/patch)")
    parser.add_argument("--mode", type=str, choices=["ablate", "patch"], default=None,
                        help="Force mode (default: auto-detect from filenames)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    mode = args.mode or detect_mode(input_dir)
    print(f"📂 Mode: {mode}")

    if args.output_dir:
        save_dir = Path(args.output_dir)
    else:
        save_dir = PROJECT_ROOT / "plots" / ("ablation" if mode == "ablate" else "patch")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading CSVs from: {input_dir}")
    data = load_all_csvs(input_dir, mode=mode)

    if not data:
        print("❌ No ablation CSVs found!")
        return

    concepts = sorted(data.keys())
    print(f"   Found {len(concepts)} concepts: {concepts}")
    for c in concepts:
        print(f"   {c}: {len(data[c])} rows")

    # Extract expected strength from CSV data (all rows share the same value)
    expected_strength = None
    for concept_rows in data.values():
        for row in concept_rows:
            if row.get("expected"):
                expected_strength = row["expected"]
                break
        if expected_strength:
            break
    if expected_strength:
        print(f"   Ground Truth: {expected_strength}")

    print(f"\n📊 Generating plots → {save_dir}/")
    plot_multi_concept_heatmap(data, save_dir, mode=mode, expected_strength=expected_strength)
    plot_strength_score(data, save_dir, mode=mode, expected_strength=expected_strength)
    plot_stacked_bars(data, save_dir, mode=mode, expected_strength=expected_strength)

    print(f"\n🎉 All plots saved to {save_dir}/")


if __name__ == "__main__":
    main()
