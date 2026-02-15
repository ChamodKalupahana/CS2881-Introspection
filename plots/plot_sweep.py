"""Plot sweep results from a CSV file.

Generates:
1. Stacked bar chart (all categories)
2. Heatmap (color intensity = count)
3. Success Rate bar chart (only detected_correct)

Usage:
    python plots/plot_sweep.py --csv success_programs/sweep_results/sweep_L17H3_L31H3_MLP17-21-29_02_14_26_19_08.csv
"""

import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

# Color scheme (consistent with other plots)
CAT_COLORS = {
    "detected_correct":    "#2ecc71",  # green
    "detected_parallel":   "#f1c40f",  # yellow
    "detected_orthogonal": "#e67e22",  # orange
    "detected_opposite":   "#e74c3c",  # red
    "not_detected":        "#95a5a6",  # grey
}

CATEGORIES_ORDER = [
    "detected_correct",
    "detected_parallel",
    "detected_orthogonal",
    "detected_opposite",
    "not_detected",
]


def load_data(csv_path):
    experiments = []
    data = {cat: [] for cat in CATEGORIES_ORDER}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_name = row['experiment']
            # Clean up experiment names for better labels
            if exp_name.startswith("L") or exp_name.startswith("mlp"):
                parts = exp_name.split('_', 1)
                if len(parts) > 1:
                    exp_name = f"{parts[0]}\n{parts[1]}"
            
            experiments.append(exp_name)
            
            total = int(row.get('total', 0))
            if total == 0:
                print(f"Warning: Total is 0 for {exp_name}")
                continue

            for cat in CATEGORIES_ORDER:
                count = int(row.get(cat, 0))
                data[cat].append(count)
                
    return experiments, data


def plot_stacked_bars(experiments, data, title_suffix, output_path):
    n_exp = len(experiments)
    fig, ax = plt.subplots(figsize=(10, max(6, n_exp * 0.5)))
    
    y_pos = np.arange(n_exp)
    left = np.zeros(n_exp)
    
    for cat in CATEGORIES_ORDER:
        counts = np.array(data[cat])
        ax.barh(y_pos, counts, left=left, height=0.6, 
                label=cat.replace('_', ' ').title(), 
                color=CAT_COLORS[cat], edgecolor='white', linewidth=0.5)
        
        for i, count in enumerate(counts):
            if count > 0:
                x = left[i] + count / 2
                ax.text(x, y_pos[i], str(count), ha='center', va='center', 
                        color='white' if cat != "detected_parallel" else "black", 
                        fontsize=8, fontweight='bold')
        left += counts

    ax.set_yticks(y_pos)
    ax.set_yticklabels(experiments)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(f"Sweep Results: Stacked Bars\n{title_suffix}", fontsize=12, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved stacked plot to {output_path}")
    plt.close()


def plot_heatmap(experiments, data, title_suffix, output_path):
    # Convert data to matrix: concepts (rows) x categories (cols)
    # Categories order for heatmap: Correct -> Parallel -> etc.
    cols = CATEGORIES_ORDER
    matrix = []
    
    for i in range(len(experiments)):
        row_vals = []
        for cat in cols:
            row_vals.append(data[cat][i])
        matrix.append(row_vals)
    
    matrix = np.array(matrix)
    
    n_exp = len(experiments)
    fig, ax = plt.subplots(figsize=(8, max(6, n_exp * 0.5)))
    
    # Use seaborn heatmap if available, else simple matshow
    try:
        sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu", 
                    xticklabels=[c.replace("detected_", "") for c in cols],
                    yticklabels=experiments, ax=ax, cbar_kws={'label': 'Count'})
    except NameError:
        # Fallback if seaborn not requested/installed (though I imported it, so let's handle if import fails)
        im = ax.imshow(matrix, aspect='auto', cmap='YlGnBu')
        # ... verbose manual annotation code skipped for brevity, user likely has seaborn or matplotlib
        # If seaborn fails, this block won't run correctly without more code. 
        # But let's assume matplotlib is enough for basics if I used matshow properly.
        # Actually, let's just stick to matplotlib for robustness as I didn't install seaborn.
        pass

    ax.set_title(f"Sweep Results: Heatmap\n{title_suffix}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    plt.close()

def plot_heatmap_matplotlib(experiments, data, title_suffix, output_path):
    """Robust heatmap using only matplotlib"""
    cols = ["detected_correct", "detected_parallel", "detected_orthogonal", "detected_opposite", "not_detected"]
    # Shorten labels
    col_labels = ["Correct", "Parallel", "Orthogonal", "Opposite", "Not Detected"]
    
    matrix = []
    for i in range(len(experiments)):
        row = [data[cat][i] for cat in cols]
        matrix.append(row)
    matrix = np.array(matrix)
    
    n_rows = len(experiments)
    n_cols = len(cols)
    
    fig, ax = plt.subplots(figsize=(8, max(5, n_rows * 0.5)))
    im = ax.imshow(matrix, cmap="Blues", aspect='auto')
    
    # Axes
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(experiments)
    
    # Loop over data dimensions and create text annotations.
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            text_color = "white" if val > matrix.max()/2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=text_color, fontweight='bold')

    ax.set_title(f"Sweep Results: Heatmap\n{title_suffix}", fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    plt.close()


def plot_success_rate(experiments, data, title_suffix, output_path):
    """Bar chart of just 'detected_correct' counts"""
    counts = np.array(data["detected_correct"])
    
    # Sort by count for better readability? Optional. Let's keep input order for now to match other plots.
    # Actually, sorting makes it easier to pick winners.
    
    # Create sorted indices
    indices = np.argsort(counts)
    sorted_exp = [experiments[i] for i in indices]
    sorted_counts = counts[indices]
    
    n_exp = len(experiments)
    fig, ax = plt.subplots(figsize=(10, max(6, n_exp * 0.4)))
    
    y_pos = np.arange(n_exp)
    ax.barh(y_pos, sorted_counts, color=CAT_COLORS["detected_correct"], height=0.7)
    
    # Add labels
    for i, v in enumerate(sorted_counts):
        ax.text(v + 0.1, i, str(v), va='center', fontweight='bold')
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_exp)
    ax.set_xlabel("Count (Correct Detections)")
    ax.set_title(f"Sweep Results: Success Rate (Sorted)\n{title_suffix}", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved success rate plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot sweep results from CSV")
    parser.add_argument("--csv", type=str, required=True, help="Path to the sweep CSV file")
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        # Try finding it relative to project root
        project_root = Path(__file__).resolve().parent.parent
        csv_path = project_root / args.csv
        
    if not csv_path.exists():
        print(f"Error: File not found: {args.csv}")
        return

    print(f"Reading {csv_path}...")
    experiments, data = load_data(csv_path)
    
    # Title suffix from filename
    title = csv_path.stem
    if title.startswith("sweep_"):
        title = title[6:]
        
    # Generate plots
    base_name = csv_path.parent / csv_path.stem
    
    plot_stacked_bars(experiments, data, title, Path(f"{base_name}_stacked.png"))
    plot_heatmap_matplotlib(experiments, data, title, Path(f"{base_name}_heatmap.png"))
    plot_success_rate(experiments, data, title, Path(f"{base_name}_success.png"))


if __name__ == "__main__":
    main()
