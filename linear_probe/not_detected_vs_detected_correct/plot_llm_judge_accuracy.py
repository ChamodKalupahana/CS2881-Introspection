"""
# load in results

# take in args of models

# plot column chart
# y axis: final category accuracy, not_detected accuracy, detected_correct accuracy
# x axis: models
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Resolve project root and plot directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PLOT_DIR = PROJECT_ROOT / "plots" / "linear_probe" / "not_detected_vs_detected_correct"

def main():
    parser = argparse.ArgumentParser(description="Plot LLM judge accuracy comparison.")
    parser.add_argument("results_paths", nargs="+", help="Paths to the results.json files.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_PLOT_DIR / "judge_accuracy_comparison.png"), 
                        help="Output plot filename.")
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    models = []
    final_accs = []
    not_detected_accs = []
    detected_correct_accs = []
    avg_times = []

    for path_str in args.results_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: File not found: {path}")
            continue

        with open(path, "r") as f:
            data = json.load(f)

        model_name = data.get("model", "Unknown")
        models.append(model_name)

        # Timing
        total_time = data.get("total_time", 0)
        total_cases = data.get("total_cases", 1)
        avg_times.append(total_time / total_cases)

        # Overall Final Category Accuracy
        final_acc = data["overall_accuracies"]["final_category"]["percentage"] * 100
        final_accs.append(final_acc)

        # Per-category accuracies
        breakdown = data.get("per_category_breakdown", {})
        
        not_detected_acc = breakdown.get("not_detected", {}).get("accuracy", 0) * 100
        not_detected_accs.append(not_detected_acc)
        
        detected_correct_acc = breakdown.get("detected_correct", {}).get("accuracy", 0) * 100
        detected_correct_accs.append(detected_correct_acc)

    if not models:
        print("No valid results files provided.")
        return

    # Plotting
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))

    rects1 = ax.bar(x - width, final_accs, width, label='Final Category (Overall)', color='#4285F4')
    rects2 = ax.bar(x, not_detected_accs, width, label='Not Detected Accuracy', color='#EA4335')
    rects3 = ax.bar(x + width, detected_correct_accs, width, label='Detected Correct Accuracy', color='#FBBC05')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('LLM Judge Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower left')

    # Add labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # Add Avg Time annotations
    for i, avg_time in enumerate(avg_times):
        ax.text(i, 102, f'Avg: {avg_time:.2f}s/prompt', 
                ha='center', va='bottom', fontsize=10, 
                fontweight='bold', color='#444444',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"📈 Plot saved to: {args.output}")

if __name__ == "__main__":
    main()
