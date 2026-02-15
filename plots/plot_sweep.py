"""Plot sweep results from a CSV file.

New features:
- Group experiments by Head/MLP (e.g., L17H3, mlp17)
- Cleaner chart titles (remove timestamps, better formatting)
- Visual separators between groups
- Bold component names in labels (e.g., **L17H3** input amplify)

Usage:
    python plots/plot_sweep.py --csv success_programs/sweep_results/sweep_L17H3_L31H3_MLP17-21-29_02_14_26_19_08.csv
"""

import argparse
import csv
import re
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


def clean_title(filename_stem):
    """
    Example: sweep_L17H3_L31H3_MLP17-21-29_02_14_26_19_08
    Returns: Activation Engineering Sweep: L17H3, L31H3, MLP17-21-29
    """
    title = filename_stem
    if title.startswith("sweep_"):
        title = title[6:]
    
    # Remove timestamp suffix (e.g. _02_14_26...)
    title = re.sub(r'_\d{2}_\d{2}_\d{2}.*$', '', title)
    
    # Replace remaining underscores with spaces or commas
    title = title.replace('_', ', ')
    return f"Activation Engineering Sweep: {title}"


def get_group(exp_name):
    """
    Extract grouping key:
    L17H3_input_amplify -> L17H3
    mlp17_ablate -> mlp17
    control -> Control
    """
    if exp_name.lower() == "control":
        return "Control"
    
    m = re.match(r"(L\d+H\d+)", exp_name)
    if m:
        return m.group(1)
        
    m = re.match(r"(mlp\d+)", exp_name)
    if m:
        return m.group(1)
        
    return "Other"


def format_exp_label(exp_name):
    """
    Format label with bold component name.
    L17H3_input_amplify -> $\bf{L17H3}$ input amplify
    control -> $\bf{Control}$
    """
    # Check for Control
    if exp_name.lower() == "control":
        return r"$\bf{Control}$"
    
    # Split on first underscore
    parts = exp_name.split('_', 1)
    if len(parts) == 2:
        head, suffix = parts
        suffix = suffix.replace('_', ' ')
        
        # Check if head looks like a component we want to bold
        if re.match(r"^(L\d+H\d+|mlp\d+)$", head, re.IGNORECASE):
             # Use mathtext for bolding. Note: spaces must be outside match or escaped.
             # We put the component name inside $\bf{...}$
             return f"$\\bf{{{head}}}$ {suffix}"
             
    # Fallback: just replace underscores
    return exp_name.replace('_', ' ')


def load_data(csv_path):
    raw_experiments = []
    raw_data = {cat: {} for cat in CATEGORIES_ORDER} # store by exp name
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_name = row['experiment']
            raw_experiments.append(exp_name)
            
            for cat in CATEGORIES_ORDER:
                count = int(row.get(cat, 0))
                raw_data[cat][exp_name] = count

    # Sort logic: Control first, then Heads, then MLPs
    def sort_key(name):
        group = get_group(name)
        if group == "Control":
            return (0, name)
        elif group.startswith("L"):
            # Sort by layer number (L17 < L31)
            try:
                layer = int(re.search(r"L(\d+)", group).group(1))
                return (1, layer, group, name)
            except:
                return (1, 999, group, name)
        elif group.startswith("mlp"):
             # Sort by layer number (mlp17 < mlp21)
            try:
                layer = int(re.search(r"mlp(\d+)", group).group(1))
                return (2, layer, group, name)
            except:
                return (2, 999, group, name)
        return (3, name)

    experiments = sorted(raw_experiments, key=sort_key)
    
    # Rebuild data lists in sorted order
    data = {cat: [] for cat in CATEGORIES_ORDER}
    for exp in experiments:
        for cat in CATEGORIES_ORDER:
            data[cat].append(raw_data[cat][exp])
            
    return experiments, data


def add_group_separators(ax, experiments):
    # Detect group boundaries
    current_group = get_group(experiments[0])
    for i, exp in enumerate(experiments):
        grp = get_group(exp)
        if grp != current_group:
            # Draw line between i-1 and i
            ax.axhline(i - 0.5, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
            current_group = grp


def plot_stacked_bars(experiments, data, title_base, output_path):
    n_exp = len(experiments)
    fig, ax = plt.subplots(figsize=(12, max(6, n_exp * 0.5)))
    
    y_pos = np.arange(n_exp)
    left = np.zeros(n_exp)
    
    for cat in CATEGORIES_ORDER:
        counts = np.array(data[cat])
        ax.barh(y_pos, counts, left=left, height=0.7, 
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
    
    # Format labels with bolding
    labels = [format_exp_label(e) for e in experiments]
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    
    add_group_separators(ax, experiments)

    ax.set_xlabel("Count")
    ax.set_title(f"{title_base} - Stacked Bars", fontsize=13, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved stacked plot to {output_path}")
    plt.close()


def plot_heatmap_matplotlib(experiments, data, title_base, output_path):
    cols = ["detected_correct", "detected_parallel", "detected_orthogonal", "detected_opposite", "not_detected"]
    col_labels = ["Correct", "Parallel", "Orthogonal", "Opposite", "Not Detected"]
    
    matrix = []
    for i in range(len(experiments)):
        row = [data[cat][i] for cat in cols]
        matrix.append(row)
    matrix = np.array(matrix)
    
    n_rows = len(experiments)
    n_cols = len(cols)
    
    fig, ax = plt.subplots(figsize=(10, max(5, n_rows * 0.5)))
    im = ax.imshow(matrix, cmap="Blues", aspect='auto')
    
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels)
    
    labels = [format_exp_label(e) for e in experiments]
    ax.set_yticklabels(labels)
    
    # Add separators
    add_group_separators(ax, experiments)
    
    # Annotate
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            text_color = "white" if val > matrix.max()/2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=text_color, fontweight='bold')

    ax.set_title(f"{title_base} - Heatmap", fontsize=13, fontweight='bold')
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    plt.close()


def plot_success_rate(experiments, data, title_base, output_path):
    """Bar chart of 'detected_correct', but keeping the Grouped order (don't sort by count)"""
    counts = np.array(data["detected_correct"])
    
    n_exp = len(experiments)
    fig, ax = plt.subplots(figsize=(11, max(6, n_exp * 0.4)))
    
    y_pos = np.arange(n_exp)
    ax.barh(y_pos, counts, color=CAT_COLORS["detected_correct"], height=0.7)
    
    for i, v in enumerate(counts):
        ax.text(v + 0.1, i, str(v), va='center', fontweight='bold')
    
    add_group_separators(ax, experiments)
    
    ax.set_yticks(y_pos)
    labels = [format_exp_label(e) for e in experiments]
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    
    ax.set_xlabel("Count (Correct Detections)")
    ax.set_title(f"{title_base} - Success Rate", fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved success rate plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot sweep results from CSV (Grouped)")
    parser.add_argument("--csv", type=str, required=True, help="Path to the sweep CSV file")
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        project_root = Path(__file__).resolve().parent.parent
        csv_path = project_root / args.csv
        
    if not csv_path.exists():
        print(f"Error: File not found: {args.csv}")
        return

    print(f"Reading {csv_path}...")
    experiments, data = load_data(csv_path)
    
    # Generate clean title
    title = clean_title(csv_path.stem)
    print(f"Title: {title}")
        
    base_name = csv_path.parent / csv_path.stem
    
    plot_stacked_bars(experiments, data, title, Path(f"{base_name}_stacked.png"))
    plot_heatmap_matplotlib(experiments, data, title, Path(f"{base_name}_heatmap.png"))
    plot_success_rate(experiments, data, title, Path(f"{base_name}_success.png"))


if __name__ == "__main__":
    main()
