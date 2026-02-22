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
    parser.add_argument("--plot_pc1", action="store_true", help="Plot Explained Variance of PC1 instead of num components for 80%% variance")
    parser.add_argument("--input_dir", type=str, default="PCA_components",
                        help="Input directory with delta .pt files (default: PCA_components)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = script_dir / args.input_dir
    
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
            file_path = input_dir / f"deltas_layer{layer}.pt"
        
        if not file_path.exists():
            print(f"{layer}\t| File not found")
            continue
            
        try:
            data = torch.load(file_path, map_location="cpu")
            
            # Handle both data formats:
            #   compute_delta_per_layer.py:              {'deltas': tensor}
            #   compute_delta_per_layer_not_detected.py: {'detected_vectors': tensor, 'not_detected_vectors': tensor}
            if 'deltas' in data:
                deltas = data['deltas']  # [num_concepts, hidden_size]
            elif 'detected_vectors' in data and 'not_detected_vectors' in data:
                # Combine both classes for PCA analysis
                deltas = torch.cat([data['detected_vectors'], data['not_detected_vectors']], dim=0)
            else:
                print(f"{layer}\t| Unknown data format, keys: {list(data.keys())}")
                continue
            
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
    
    # Use input dir name as suffix for plot filenames to differentiate
    dir_suffix = Path(args.input_dir).name.replace("PCA_components", "").strip("_")
    suffix = f"_{dir_suffix}" if dir_suffix else ""

    if args.plot_pc1:
        plt.title("Explained Variance of PC1 by Layer")
        plt.ylabel("Explained Variance Ratio (PC1)")
        output_filename = f"pc1_explained_variance{suffix}.png"
    else:
        plt.title("Number of PCA Components Needed to Explain 80% Variance by Layer")
        plt.ylabel("Number of Components")
        output_filename = f"components_needed_80_variance{suffix}.png"

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
