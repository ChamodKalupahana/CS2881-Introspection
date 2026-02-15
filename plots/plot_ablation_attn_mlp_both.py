"""Plot patch/ablation results for BOTH attn+mlp patched simultaneously.

Reads CSVs from the "both" results directory where each row is
L{layer}_attn+mlp (one result per layer, not separate attn/mlp).

Generates:
  1. Multi-concept heatmap (layers × concepts → category)
  2. Detection survival rate (line plot, one line per concept)
  3. Per-concept stacked bars (one bar per layer)

Usage:
    python plots/plot_ablation_attn_mlp_both.py
    python plots/plot_ablation_attn_mlp_both.py --input_dir acitvation_patching/results/both/patch
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
    "detected_correct",
    "detected_parallel",
    "detected_orthogonal",
    "detected_opposite",
    "not_detected",
]

# Color scheme for categories
CAT_COLORS = {
    "detected_correct":    "#2ecc71",  # green
    "detected_parallel":   "#f1c40f",  # yellow
    "detected_orthogonal": "#e67e22",  # orange
    "detected_opposite":   "#e74c3c",  # red
    "not_detected":        "#95a5a6",  # grey
}

# Numeric score: how much "detection" survived the patching
CAT_SCORE = {
    "detected_correct":    1.0,
    "detected_parallel":   0.75,
    "detected_orthogonal": 0.25,
    "detected_opposite":   0.5,
    "not_detected":        0.0,
}

# Numeric encoding for heatmap
CAT_HEATMAP_VAL = {
    "detected_correct":    4,
    "detected_parallel":   3,
    "detected_opposite":   2,
    "detected_orthogonal": 1,
    "not_detected":        0,
}


def parse_csv(csv_path):
    """Parse a both-patching CSV.
    Format: patch_target, detection_category, raw_response_prefix
    where patch_target is L{layer}_attn+mlp
    """
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = row["patch_target"].strip()
            m = re.match(r"L(\d+)_attn\+mlp", target)
            if not m:
                continue
            layer = int(m.group(1))

            if "detection_category" in row:
                cat = row["detection_category"].strip()
                if cat not in CATEGORIES:
                    cat = "not_detected"
            else:
                cat = "not_detected"
                for c in CATEGORIES:
                    if row.get(c, "0").strip() == "1":
                        cat = c
                        break

            rows.append({
                "layer": layer,
                "category": cat,
                "raw": row.get("raw_response_prefix", ""),
            })
    return rows


def extract_concept_from_filename(filename):
    """Extract concept name from filename patterns like:
    patch_attn_mlp_both_Illusions_coeff8.0_...
    ablate_attn_mlp_both_Phosphorus_coeff8.0_...
    """
    m = re.search(r"(?:ablate|patch)_attn_mlp_both_(.+?)_coeff", filename)
    if m:
        return m.group(1)
    return filename


def load_all_csvs(input_dir):
    """Load all both-patching CSVs from directory. Returns dict: concept -> rows."""
    data = {}
    for csv_path in sorted(input_dir.glob("*.csv")):
        if "attn_mlp" not in csv_path.name:
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

def plot_multi_concept_heatmap(data, save_dir, mode="patch"):
    """
    Heatmap grid: rows = layers (16-31), columns = concepts.
    Cell color = detection category.
    """
    mode_title = "Ablation" if mode == "ablate" else "Patch"
    concepts = sorted(data.keys())
    layers = list(range(16, 32))
    n_layers = len(layers)
    n_cols = len(concepts)  # one column per concept (both sublayers patched)

    # Build matrix
    matrix = np.full((n_layers, n_cols), -1, dtype=int)
    col_labels = concepts

    for ci, concept in enumerate(concepts):
        rows = data[concept]
        for row in rows:
            if row["layer"] in layers:
                li = row["layer"] - 16
                matrix[li, ci] = CAT_HEATMAP_VAL[row["category"]]

    # Custom colormap: grey → orange → red → yellow → green
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap_colors = [
        CAT_COLORS["not_detected"],
        CAT_COLORS["detected_orthogonal"],
        CAT_COLORS["detected_opposite"],
        CAT_COLORS["detected_parallel"],
        CAT_COLORS["detected_correct"],
    ]
    cmap = ListedColormap(cmap_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.5), n_layers * 0.55 + 2))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=10, ha="center")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{l}" for l in layers], fontsize=10)

    ax.set_ylabel("Layer", fontsize=12)
    ax.set_xlabel("Concept", fontsize=12)
    ax.set_title(f"{mode_title} (attn+mlp both): Detection Category per Layer × Concept",
                 fontsize=14, fontweight="bold", pad=12)

    # Add vertical separators between concepts
    for ci in range(1, len(concepts)):
        ax.axvline(x=ci - 0.5, color="white", linewidth=2)

    # Cell text
    cat_short = {0: "✗", 1: "orth", 2: "opp", 3: "par", 4: "✓"}
    for i in range(n_layers):
        for j in range(n_cols):
            val = matrix[i, j]
            if val >= 0:
                txt = cat_short[val]
                text_color = "white" if val in [0, 2] else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=9, fontweight="bold", color=text_color)

    # Legend
    legend_elements = [
        Patch(facecolor=CAT_COLORS["detected_correct"], label="Correct (✓)"),
        Patch(facecolor=CAT_COLORS["detected_parallel"], label="Parallel (par)"),
        Patch(facecolor=CAT_COLORS["detected_opposite"], label="Opposite (opp)"),
        Patch(facecolor=CAT_COLORS["detected_orthogonal"], label="Orthogonal (orth)"),
        Patch(facecolor=CAT_COLORS["not_detected"], label="Not Detected (✗)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=9, frameon=True, title="Category", title_fontsize=10)

    fig.tight_layout()
    path = save_dir / f"{mode}_both_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {path}")


# ── Plot 2: Detection survival rate ──────────────────────────────────────────

def plot_detection_survival(data, save_dir, mode="patch"):
    """
    Line plot: detection score across layers, one line per concept.
    All concepts on a single plot.
    """
    mode_label = "Ablation" if mode == "ablate" else "Patch"
    concepts = sorted(data.keys())
    layers = list(range(16, 32))

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(concepts)))

    for ci, concept in enumerate(concepts):
        scores = []
        for layer in layers:
            score = 0.0
            for row in data[concept]:
                if row["layer"] == layer:
                    score = CAT_SCORE[row["category"]]
                    break
            scores.append(score)

        ax.plot(layers, scores, 'o-', color=colors[ci], linewidth=2,
                markersize=6, label=concept, alpha=0.9)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Detection Score", fontsize=12)
    ax.set_title(f"Detection Survival After {mode_label} (attn+mlp both patched)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers], fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(y=0.5, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="best", ncol=2)

    fig.tight_layout()
    path = save_dir / f"{mode}_both_survival.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {path}")


# ── Plot 3: Stacked bars per concept ─────────────────────────────────────────

def plot_stacked_bars(data, save_dir, mode="patch"):
    """
    For each concept: one stacked bar per layer (attn+mlp patched together).
    Segments colored by detection category.
    """
    concepts = sorted(data.keys())
    layers = list(range(16, 32))
    n_layers = len(layers)

    n_concepts = len(concepts)
    cols = min(3, n_concepts)
    rows_grid = (n_concepts + cols - 1) // cols

    fig, axes = plt.subplots(rows_grid, cols,
                              figsize=(max(12, n_layers * 0.9), 5 * rows_grid),
                              squeeze=False)

    for ci, concept in enumerate(concepts):
        r, c = divmod(ci, cols)
        ax = axes[r][c]

        x = np.arange(n_layers)
        width = 0.6

        bottoms = np.zeros(n_layers)
        for cat in CATEGORIES:
            heights = np.zeros(n_layers)
            for row in data[concept]:
                if row["layer"] in layers:
                    li = row["layer"] - 16
                    if row["category"] == cat:
                        heights[li] = 1.0

            ax.bar(x, heights, width, bottom=bottoms,
                   color=CAT_COLORS[cat], edgecolor="white", linewidth=0.3,
                   label=cat if ci == 0 else "_nolegend_")
            bottoms += heights

        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8, rotation=45)
        ax.set_title(concept, fontsize=13, fontweight="bold")
        ax.set_ylabel("Detection", fontsize=10)
        ax.set_ylim(0, 1.15)

    # Hide unused
    for ci in range(n_concepts, rows_grid * cols):
        r, c = divmod(ci, cols)
        axes[r][c].set_visible(False)

    # Single legend
    legend_elements = [Patch(facecolor=CAT_COLORS[cat], label=cat.replace("_", " ").title())
                       for cat in CATEGORIES]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=len(CATEGORIES), fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.02))

    mode_label = "Ablation" if mode == "ablate" else "Patch"
    fig.suptitle(f"{mode_label} (attn+mlp both): Category Breakdown by Layer",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = save_dir / f"{mode}_both_stacked_bars.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot patch/ablation results (attn+mlp both)")
    parser.add_argument("--input_dir", type=str,
                        default=str(PROJECT_ROOT / "acitvation_patching" / "results" / "both" / "patch"))
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: plots/both_patch or plots/both_ablate)")
    parser.add_argument("--mode", type=str, choices=["ablate", "patch"], default=None,
                        help="Force mode (default: auto-detect from filenames)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    mode = args.mode or detect_mode(input_dir)
    print(f"📂 Mode: {mode}")

    if args.output_dir:
        save_dir = Path(args.output_dir)
    else:
        save_dir = PROJECT_ROOT / "plots" / (f"both_{'ablation' if mode == 'ablate' else 'patch'}")
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading CSVs from: {input_dir}")
    data = load_all_csvs(input_dir)

    if not data:
        print("❌ No CSVs found!")
        return

    concepts = sorted(data.keys())
    print(f"   Found {len(concepts)} concepts: {concepts}")
    for c in concepts:
        print(f"   {c}: {len(data[c])} rows")

    print(f"\n📊 Generating plots → {save_dir}/")
    plot_multi_concept_heatmap(data, save_dir, mode=mode)
    plot_detection_survival(data, save_dir, mode=mode)
    plot_stacked_bars(data, save_dir, mode=mode)

    print(f"\n🎉 All plots saved to {save_dir}/")


if __name__ == "__main__":
    main()
