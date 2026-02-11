import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def main():
    data_path = "success_results/head_contributions_layers16-31.pt"
    if not Path(data_path).exists():
        print(f"File not found: {data_path}")
        print("Please run compute_head_contributions.py first to generate data.")
        return

    print(f"Loading data from {data_path}...")
    results = torch.load(data_path, map_location="cpu") # {case: {layer: tensor(num_heads)}}

    cases = list(results.keys())
    num_cases = len(cases)
    
    # Extract data layout
    # Assuming all cases have same layers
    first_case = results[cases[0]]
    layers = sorted(list(first_case.keys()))
    num_layers = len(layers)
    num_heads = len(first_case[layers[0]])
    
    # Prepare data matrices
    # Shape: (num_cases, num_layers, num_heads)
    matrices = {}
    
    # Store min/max for global scale
    all_values = []
    
    for case in cases:
        mat = np.zeros((num_layers, num_heads))
        for i, layer in enumerate(layers):
            vec = results[case][layer]
            if isinstance(vec, torch.Tensor):
                vec = vec.numpy()
            mat[i, :] = vec
        matrices[case] = mat
        all_values.extend(mat.flatten())
        
    
    print(f"Plotting {num_cases} cases with independent scales.")

    # Create figure
    fig, axes = plt.subplots(1, num_cases, figsize=(6 * num_cases, 8), constrained_layout=True)
    if num_cases == 1:
        axes = [axes]
        
    for ax, case in zip(axes, cases):
        mat = matrices[case]
        
        # Per-subplot symmetric scale
        limit = np.nanmax(np.abs(mat))
        if limit == 0: limit = 1.0
        vmin, vmax = -limit, limit
        
        # Plot heatmap
        sns.heatmap(mat, ax=ax, cmap="RdBu_r", center=0, vmin=vmin, vmax=vmax,
                    cbar=True, cbar_kws={"shrink": 0.6}, square=True)
        
        ax.set_title(case)
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Layer Index")
        
        # Set y-ticks to actual layer numbers
        ax.set_yticks(np.arange(num_layers) + 0.5)
        ax.set_yticklabels(layers, rotation=0)

    plt.suptitle("Head Contributions to Success Vector (Layers 16-31)", fontsize=16)
    
    out_path = "success_results/head_contributions_plot.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")
    
    # Also save as PDF
    pdf_path = out_path.replace(".png", ".pdf")
    plt.savefig(pdf_path)
    print(f"Saved PDF to {pdf_path}")

if __name__ == "__main__":
    main()
