import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import argparse

def main():
    parser = argparse.ArgumentParser(description="Analyze PCA Components")
    parser.add_argument("--plot_pc1", action="store_true", help="Plot Explained Variance of PC1 instead of num components for 80% variance")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    input_dir = script_dir / "PCA_components"
    
    layers = []
    metric_values = [] # Will hold either num_components or pc1_variance
    
    print(f"Reading from: {input_dir}")
    if args.plot_pc1:
        print("Layer\t| PC1 Explained Variance")
    else:
        print("Layer\t| Components for 80% Variance")
    print("-" * 40)
    
    for layer in range(16, 32):
        file_path = input_dir / f"all_deltas_layer{layer}.pt"
        
        if not file_path.exists():
            print(f"{layer}\t| File not found")
            continue
            
        try:
            data = torch.load(file_path, map_location="cpu")
            # data structure from compute_delta_per_layer.py: 
            # {'deltas': tensor, 'concepts': list, 'layer': int, 'coeff': float}
            deltas = data['deltas'] # [num_concepts, hidden_size]
            
            # Center the data
            mean = deltas.mean(dim=0)
            centered = deltas - mean
            
            # Perform PCA (SVD)
            # Ensure float32 for CPU SVD
            centered_float = centered.to(dtype=torch.float32)
            U, S, Vh = torch.linalg.svd(centered_float, full_matrices=False)
            
            # Compute explained variance ratio
            eigenvalues = S.pow(2)
            total_variance = eigenvalues.sum()
            explained_variance_ratio = eigenvalues / total_variance
            
            layers.append(layer)

            if args.plot_pc1:
                pc1_var = explained_variance_ratio[0].item()
                metric_values.append(pc1_var)
                print(f"{layer}\t| {pc1_var:.4f}")
            else:
                cumulative_variance = torch.cumsum(explained_variance_ratio, dim=0)
                # Find number of components for 80% variance
                target_variance = 0.80
                
                if cumulative_variance[-1] < target_variance:
                    print(f"{layer}\t| Max variance {cumulative_variance[-1]:.4f} < {target_variance}")
                    n_components = len(cumulative_variance)
                else:
                    n_components = (cumulative_variance >= target_variance).nonzero()[0].item() + 1
                
                metric_values.append(n_components)
                print(f"{layer}\t| {n_components}")
            
        except Exception as e:
            print(f"{layer}\t| Error: {e}")
            continue

    if not layers:
        print("No data processed.")
        return

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(layers, metric_values, marker='o', linestyle='-', color='b')
    
    if args.plot_pc1:
        plt.title("Explained Variance of PC1 by Layer")
        plt.ylabel("Explained Variance Ratio (PC1)")
        output_filename = "pc1_explained_variance.png"
    else:
        plt.title("Number of PCA Components Needed to Explain 80% Variance by Layer")
        plt.ylabel("Number of Components")
        output_filename = "components_needed_80_variance.png"

    plt.xlabel("Layer")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(layers) # Ensure all layer numbers are shown
    
    output_dir = PROJECT_ROOT / "plots" / "PCA"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_plot_path = output_dir / output_filename
    plt.savefig(output_plot_path)
    print(f"\nPlot saved to {output_plot_path}")

if __name__ == "__main__":
    main()
