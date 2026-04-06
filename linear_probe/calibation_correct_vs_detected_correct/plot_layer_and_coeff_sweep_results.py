import os
import json
import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    import sys
    sys.path.append(str(PROJECT_ROOT))

def load_sweep_results(base_dir):
    """
    Scans the base directory for summary.json files in run folders.
    """
    data = []
    base_path = Path(base_dir)
    
    # Look for any subdirectories in the base_path
    run_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    for run_dir in run_dirs:
        summary_file = run_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file, "r") as f:
                run_data = json.load(f)
                
                # Extract relevant metrics
                metadata = run_data.get("metadata", {})
                stats = run_data.get("stats", {})
                
                total = sum(stats.values())
                if total == 0: continue
                
                entry = {
                    "layer": metadata.get("layer"),
                    "coeff": metadata.get("coeff"),
                    "not_detected": (stats.get("not_detected", 0) / total) * 100,
                    "detected_correct": ((stats.get("detected_correct", 0) + stats.get("detected_parallel", 0)) / total) * 100,
                    "incoherent": (stats.get("incoherent", 0) / total) * 100
                }
                data.append(entry)
                
    return data

def plot_results(data, output_dir=None, xlim=None):
    if not data:
        print("No data found to plot!")
        return

    if output_dir is None:
        output_dir = PROJECT_ROOT / "plots" / "linear_probe" / "calibation_correct_vs_detected_correct"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Metrics to plot
    metrics = ["not_detected", "detected_correct", "incoherent"]
    titles = ["Not Detected (%)", "Correct/Parallel Detection (%)", "Incoherent (%)"]
    
    # Group data by coeff
    coeffs = sorted(list(set(d["coeff"] for d in data)))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for coeff in coeffs:
            # Filter data for this coeff and sort by layer
            coeff_data = sorted([d for d in data if d["coeff"] == coeff], key=lambda x: x["layer"])
            
            layers = [d["layer"] for d in coeff_data]
            values = [d[metric] for d in coeff_data]
            
            ax.plot(layers, values, marker='o', label=f"Coeff {coeff}")
            
        ax.set_title(titles[i], fontweight='bold')
        ax.set_xlabel("Layer Index")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if i == 0:
            ax.set_ylabel("Percentage (%)")
        
        # Consistent Y-axis for percentages
        ax.set_ylim(-5, 105)
        
        if xlim:
            ax.set_xlim(xlim)

    # Shared Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(coeffs), frameon=False)

    plt.tight_layout()
    save_file = output_path / "sweep_results.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Plot saved to: {save_file}")

if __name__ == "__main__":
    default_out = PROJECT_ROOT / "plots" / "linear_probe" / "calibation_correct_vs_detected_correct"
    parser = argparse.ArgumentParser(description="Plot results from the layer/coeff sweep.")
    parser.add_argument("--base_dir", type=str, default="saved_activations", help="Directory containing sweep runs.")
    parser.add_argument("--output_dir", type=str, default=str(default_out), help="Where to save the plot.")
    parser.add_argument("--xlim", type=int, nargs=2, help="X-axis limits (min_layer max_layer).")
    args = parser.parse_args()

    results = load_sweep_results(args.base_dir)
    plot_results(results, args.output_dir, args.xlim)