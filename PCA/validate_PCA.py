import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description="Validate PCA by projecting validation vectors")
    parser.add_argument("--pca_dir", type=str, default="PCA_components", 
                        help="Directory containing reference PCA deltas (e.g. PCA/PCA_components)")
    parser.add_argument("--val_dir", type=str, default="validation/components", 
                        help="Directory containing validation deltas (e.g. PCA/validation/components)")
    parser.add_argument("--output_plot", type=str, default="validation_projection.png",
                        help="Filename for the output plot")
    
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    pca_root = Path(args.pca_dir)
    if not pca_root.is_absolute():
        pca_root = script_dir / args.pca_dir
        
    val_root = Path(args.val_dir)
    if not val_root.is_absolute():
        val_root = script_dir / args.val_dir
        
    print(f"PCA Reference Dir: {pca_root}")
    print(f"Validation Dir:    {val_root}")

    plot_layers = []
    avg_projections = []
    std_projections = []

    print("\nLayer\t| Mean Projection (Cosine Sim)")
    print("-" * 40)

    for layer in range(16, 32):
        # Load reference PCA deltas
        pca_file = pca_root / f"all_deltas_layer{layer}.pt"
        # Load validation deltas
        val_file = val_root / f"all_deltas_layer{layer}.pt"
        
        if not pca_file.exists():
            print(f"{layer}\t| PCA file missing: {pca_file.name}")
            continue
        if not val_file.exists():
            print(f"{layer}\t| Val file missing: {val_file.name}")
            continue
            
        try:
            # Load data
            pca_data = torch.load(pca_file, map_location="cpu")
            val_data = torch.load(val_file, map_location="cpu")
            
            # Extract deltas: [num_concepts, hidden_size]
            ref_deltas = pca_data['deltas'].to(torch.float32)
            val_deltas = val_data['deltas'].to(torch.float32)
            
            # --- Analysis Strategy ---
            # Idea: Check if validation deltas align with the "primary direction" of reference deltas.
            # 1. Compute mean direction of reference deltas (or PC1)
            # 2. Compute cosine similarity of validation deltas to this mean direction.
            
            # 1. Reference Mean Direction
            ref_mean = ref_deltas.mean(dim=0)
            ref_mean_norm = ref_mean / torch.norm(ref_mean)
            
            # 2. Project validation deltas onto this direction (Cosine Similarity)
            # Normalize validation deltas first for cosine similarity
            val_norms = torch.norm(val_deltas, dim=1, keepdim=True)
            # Avoid division by zero
            val_norms[val_norms == 0] = 1.0
            val_normalized = val_deltas / val_norms
            
            # Cosine similarity = dot product of normalized vectors
            cosine_sims = torch.matmul(val_normalized, ref_mean_norm)
            
            # Statistics
            mean_sim = cosine_sims.mean().item()
            std_sim = cosine_sims.std().item()
            
            plot_layers.append(layer)
            avg_projections.append(mean_sim)
            std_projections.append(std_sim)
            
            print(f"{layer}\t| {mean_sim:.4f} ± {std_sim:.4f}")
            
        except Exception as e:
            print(f"{layer}\t| Error: {e}")
            continue

    if not plot_layers:
        print("No valid layers processed.")
        return

    # Plotting
    output_dir = PROJECT_ROOT / "plots" / "PCA"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(plot_layers, avg_projections, yerr=std_projections, fmt='o-', capsize=5, ecolor='gray', color='b')
    plt.title("Alignment of Validation Vectors with Train Mean Direction")
    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity (Mean ± Std)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(plot_layers)
    plt.ylim(-1.1, 1.1) # Cosine similarity range
    
    save_path = output_dir / args.output_plot
    plt.savefig(save_path)
    print(f"\nPlot saved to {save_path}")

if __name__ == "__main__":
    main()
