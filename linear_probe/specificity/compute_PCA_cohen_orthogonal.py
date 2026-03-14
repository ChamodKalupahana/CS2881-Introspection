import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


def load_activations(directory: Path):
    """
    Loads all .pt files in directory and returns a dict mapping
    (concept, prompt_id) -> data dict
    """
    if not directory.exists():
        return {}

    data_map = {}
    files = sorted(directory.glob("*.pt"))
    for f in files:
        try:
            data = torch.load(f, map_location="cpu", weights_only=False)
            concept = data.get("concept", f.name.split("_")[0])
            pid = data.get("prompt_id", -1)
            # Use concept alone if pid is -1 for better matching on older runs
            key = (concept, pid) if pid != -1 else concept
            data_map[key] = data
        except Exception as e:
            print(f"  ⚠ Failed to load {f.name}: {e}")
            
    return data_map


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Compute PCA on positive-negative deltas")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Path to the prompt_activations run directory (default: latest in saved_activations/)")
    parser.add_argument("--token_type", type=str, default="last_token",
                        choices=["last_token", "prompt_last_token"],
                        help="Which token position to read (default: last_token)")
    parser.add_argument("--max_components", type=int, default=20, 
                        help="Maximum number of PCA components to plot in the heatmap (default: 20)")
    parser.add_argument("--max_layer", type=int, default=None, 
                        help="Maximum layer to plot (inclusive)")
    parser.add_argument("--plot_pc1", action="store_true", 
                        help="Plot Explained Variance of PC1 as a line graph instead of the 2D heatmap")
    parser.add_argument("--plot_cohens_d", action="store_true", 
                        help="Plot Cohen's d between orthogonal and correct projected onto PCA")
    parser.add_argument("--plot_mean_diff_dot_product", action="store_true", 
                        help="Plot scalar multiplication of projections: Cohen's d * (parallel - correct)")
    parser.add_argument("--input_vector", type=str, default=None,
                        help="Path to an activation .pt file to compute dot product against PCA components")
    args = parser.parse_args()

    token_type = args.token_type

    # ── Resolve run directory ────────────────────────────────────────────
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        sa_dir = script_dir / "saved_activations"
        candidates = sorted(sa_dir.glob("run_*"))
        if not candidates:
            print("ERROR: No run_* directories found in saved_activations/")
            return
        run_dir = candidates[-1]

    correct_dir = run_dir / "detected_correct"
    orthogonal_dir = run_dir / "detected_orthogonal"
    parallel_dir = run_dir / "detected_parallel"

    print(f"Run directory: {run_dir}")
    print(f"Token type: {token_type}")
    print(f"Loading Correct from: {correct_dir}")
    print(f"Loading Orthogonal from: {orthogonal_dir}")
    print(f"Loading Parallel from: {parallel_dir}")

    correct_map = load_activations(correct_dir)
    orthogonal_map = load_activations(orthogonal_dir)
    parallel_map = load_activations(parallel_dir)

    print(f"Loaded {len(correct_map)} correct examples, {len(orthogonal_map)} orthogonal examples, and {len(parallel_map)} parallel examples.")

    input_data = None
    if args.input_vector:
        input_path = Path(args.input_vector)
        if input_path.exists():
            input_data = torch.load(input_path, map_location="cpu", weights_only=False)
            print(f"Loaded input vector from: {input_path}")
        else:
            print(f"ERROR: Input vector file not found at {input_path}")
            return

    # Collect all available keys
    all_correct_keys = sorted(correct_map.keys())
    all_orthogonal_keys = sorted(orthogonal_map.keys())
    all_parallel_keys = sorted(parallel_map.keys())

    if not all_correct_keys or not all_orthogonal_keys:
        print("ERROR: Missing correct or orthogonal data.")
        return

    # Check which layers are available in the first example
    sample_data = correct_map[all_correct_keys[0]]
    available_layers = [k for k in sample_data["activations"].keys() if isinstance(k, int)]
    
    if args.max_layer is not None:
        available_layers = [k for k in available_layers if k <= args.max_layer]
        
    available_layers = sorted(available_layers)
    
    if not available_layers:
        print("ERROR: No layer data found in activations dict.")
        return

    layers_plotted = []
    variance_matrix = []
    pc1_values = []
    input_dots_matrix = []
    parallel_cohens_d_matrix = []
    cohens_d_matrix = []
    mean_diff_dot_product_matrix = []

    print(f"\nProcessing layers...")
    if args.plot_pc1:
        print("\nLayer\t| PC1 Explained Variance")
    print("-" * 40)

    for layer in available_layers:
        correct_vecs = []
        for key in all_correct_keys:
            data = correct_map[key]
            if layer in data["activations"]:
                vec = data["activations"][layer].get(token_type)
                if vec is not None:
                    correct_vecs.append(vec.float())
        
        orthogonal_vecs = []
        for key in all_orthogonal_keys:
            data = orthogonal_map[key]
            if layer in data["activations"]:
                vec = data["activations"][layer].get(token_type)
                if vec is not None:
                    orthogonal_vecs.append(vec.float())

        if not correct_vecs or not orthogonal_vecs:
            print(f"{layer}\t| Missing correct/orthogonal data, skipping")
            continue

        correct_tensor = torch.stack(correct_vecs)
        orthogonal_tensor = torch.stack(orthogonal_vecs)
        mean_orthogonal = orthogonal_tensor.mean(dim=0)

        # Delta population: each success relative to the average "wrong" baseline
        deltas_tensor = correct_tensor - mean_orthogonal
        
        parallel_vecs = []
        for key in all_parallel_keys:
            data = parallel_map[key]
            if layer in data["activations"]:
                vec = data["activations"][layer].get(token_type)
                if vec is not None:
                    parallel_vecs.append(vec.float())

        correct_all_vecs = correct_vecs # Already collected above
        orthogonal_all_vecs = orthogonal_vecs # Already collected above

        try:
            # PCA matching previous script
            mean = deltas_tensor.mean(dim=0)
            centered = deltas_tensor - mean
            centered_float = centered.to(dtype=torch.float32)
            
            U, S, Vh = torch.linalg.svd(centered_float, full_matrices=False)
            
            eigenvalues = S.pow(2)
            total_variance = eigenvalues.sum()
            explained_variance_ratio = eigenvalues / total_variance

            layers_plotted.append(layer)
            
            # Save PCA components
            pca_save_dir = run_dir / "pca_components"
            pca_save_dir.mkdir(parents=True, exist_ok=True)
            
            save_dict = {
                "components": Vh,
                "mean": mean,
                "eigenvalues": eigenvalues,
                "explained_variance_ratio": explained_variance_ratio,
            }
            
            input_dots = ""
            if input_data and "activations" in input_data and layer in input_data["activations"]:
                in_vec = input_data["activations"][layer].get(token_type)
                if in_vec is not None:
                    # Dot product over all components
                    dot_products_raw = torch.matmul(Vh, in_vec.float())
                    dot_products_centered = torch.matmul(Vh, in_vec.float() - mean)
                    save_dict["input_vector_dot_products_raw"] = dot_products_raw
                    save_dict["input_vector_dot_products_centered"] = dot_products_centered
                    input_dots = f" | Input dots (pc0-2 raw): {dot_products_raw[:3].numpy().round(2)}"

            torch.save(save_dict, pca_save_dir / f"{token_type}_layer_{layer}_pca.pt")

            # Store dot products for plotting up to max_components
            if input_data and "input_vector_dot_products_raw" in save_dict:
                dots_array = save_dict["input_vector_dot_products_raw"].numpy()
                if len(dots_array) > args.max_components:
                    dots_array = dots_array[:args.max_components]
                elif len(dots_array) < args.max_components:
                    dots_array = np.pad(dots_array, (0, args.max_components - len(dots_array)), 'constant')
                input_dots_matrix.append(dots_array)

            # Store difference in means for parallel vs correct
            parallel_cohens_d_signed = None
            if parallel_vecs and correct_all_vecs:
                parallel_tensor = torch.stack(parallel_vecs)
                correct_tensor = torch.stack(correct_all_vecs)
                
                Vh_f = Vh.to(parallel_tensor.dtype)
                
                parallel_proj = torch.matmul(parallel_tensor, Vh_f.T)
                correct_proj = torch.matmul(correct_tensor, Vh_f.T)
                
                n1 = parallel_proj.shape[0]
                n2 = correct_proj.shape[0]
                
                mu1 = parallel_proj.mean(dim=0).numpy()
                mu2 = correct_proj.mean(dim=0).numpy()
                
                s1_sq = parallel_proj.var(dim=0, unbiased=True).numpy()
                s2_sq = correct_proj.var(dim=0, unbiased=True).numpy()
                
                s_pooled = np.sqrt(((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2))
                s_pooled[s_pooled == 0] = 1e-8
                
                parallel_cohens_d = np.abs(mu1 - mu2) / s_pooled
                parallel_cohens_d_signed = (mu1 - mu2) / s_pooled
                
                if len(parallel_cohens_d) > args.max_components:
                    parallel_cohens_d = parallel_cohens_d[:args.max_components]
                    parallel_cohens_d_signed = parallel_cohens_d_signed[:args.max_components]
                elif len(parallel_cohens_d) < args.max_components:
                    parallel_cohens_d = np.pad(parallel_cohens_d, (0, args.max_components - len(parallel_cohens_d)), 'constant')
                    parallel_cohens_d_signed = np.pad(parallel_cohens_d_signed, (0, args.max_components - len(parallel_cohens_d_signed)), 'constant')
                    
                parallel_cohens_d_matrix.append(parallel_cohens_d)
            else:
                parallel_cohens_d_matrix.append(np.zeros(args.max_components))

            # Compute Cohen's d for orthogonal vs correct
            if args.plot_cohens_d or args.plot_mean_diff_dot_product:
                if orthogonal_all_vecs and correct_all_vecs:
                    orthogonal_tensor = torch.stack(orthogonal_all_vecs)
                    correct_tensor = torch.stack(correct_all_vecs)
                    
                    # Project both populations onto the PCA components (Vh is [n_components, hidden_size])
                    # Ensure Vh has same dtype
                    Vh_f = Vh.to(orthogonal_tensor.dtype)
                    
                    orthogonal_proj = torch.matmul(orthogonal_tensor, Vh_f.T) # [n1, n_components]
                    correct_proj = torch.matmul(correct_tensor, Vh_f.T) # [n2, n_components]
                    
                    n1 = orthogonal_proj.shape[0]
                    n2 = correct_proj.shape[0]
                    
                    mu1 = orthogonal_proj.mean(dim=0).numpy()
                    mu2 = correct_proj.mean(dim=0).numpy()
                    
                    # Unbiased sample variances (ddof=1)
                    s1_sq = orthogonal_proj.var(dim=0, unbiased=True).numpy()
                    s2_sq = correct_proj.var(dim=0, unbiased=True).numpy()
                    
                    # Pooled standard deviation
                    s_pooled = np.sqrt(((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2))
                    
                    # Avoid division by zero
                    s_pooled[s_pooled == 0] = 1e-8
                    
                    # Cohen's d: |mu1 - mu2| / s_pooled
                    # But we also need the raw non-absolute effect size for dot products if needed
                    cohens_d = np.abs(mu1 - mu2) / s_pooled
                    cohens_d_signed = (mu1 - mu2) / s_pooled
                    
                    if len(cohens_d) > args.max_components:
                        cohens_d = cohens_d[:args.max_components]
                        cohens_d_signed = cohens_d_signed[:args.max_components]
                    elif len(cohens_d) < args.max_components:
                        cohens_d = np.pad(cohens_d, (0, args.max_components - len(cohens_d)), 'constant')
                        cohens_d_signed = np.pad(cohens_d_signed, (0, args.max_components - len(cohens_d_signed)), 'constant')
                    
                    if args.plot_cohens_d:
                        cohens_d_matrix.append(cohens_d)
                    
                    if args.plot_mean_diff_dot_product and parallel_cohens_d_signed is not None:
                        # normalize before multiplication
                        norm_calib = np.linalg.norm(cohens_d_signed)
                        norm_parallel = np.linalg.norm(parallel_cohens_d_signed)
                        normed_calib = cohens_d_signed / norm_calib if norm_calib > 0 else cohens_d_signed
                        normed_parallel = parallel_cohens_d_signed / norm_parallel if norm_parallel > 0 else parallel_cohens_d_signed
                        
                        # element-wise multiplication
                        mean_diff_dot_product_matrix.append(normed_calib * normed_parallel)
                    elif args.plot_mean_diff_dot_product:
                        mean_diff_dot_product_matrix.append(np.zeros(args.max_components))
                        
                else:
                    if args.plot_cohens_d:
                        cohens_d_matrix.append(np.zeros(args.max_components))
                    if args.plot_mean_diff_dot_product:
                        mean_diff_dot_product_matrix.append(np.zeros(args.max_components))

            if args.plot_pc1:
                pc1_var = explained_variance_ratio[0].item()
                pc1_values.append(pc1_var)
                print(f"{layer}\t| PC1 var: {pc1_var:.4f}{input_dots}")
            else:
                var_array = explained_variance_ratio.numpy()
                if len(var_array) > args.max_components:
                    var_array = var_array[:args.max_components]
                elif len(var_array) < args.max_components:
                    var_array = np.pad(var_array, (0, args.max_components - len(var_array)), 'constant')
                    
                variance_matrix.append(var_array)
                print(f"{layer}\t| Extracted {len(explained_variance_ratio)} components{input_dots}")

        except Exception as e:
            print(f"{layer}\t| Error: {e}")
            continue

    if not layers_plotted:
        print("No layers processed successfully.")
        return

    # ── Plotting ─────────────────────────────────────────────────────────
    dir_suffix = run_dir.name.replace("prompt_activations_", "")

    if args.plot_pc1:
        plt.figure(figsize=(10, 6))
        plt.plot(layers_plotted, pc1_values, marker='o', linestyle='-', color='b')
        plt.title(f"Explained Variance of PC1 by Layer ({token_type})")
        plt.ylabel("Explained Variance Ratio (PC1)")
        plt.xlabel("Layer")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(layers_plotted)
        output_filename = f"pc1_explained_variance_deltas_{dir_suffix}.png"
    else:
        variance_matrix = np.array(variance_matrix)
        
        plt.figure(figsize=(10, 8))
        
        # Plot as a heatmap: Y-axis is Layers, X-axis is Components
        plt.imshow(variance_matrix, aspect='auto', cmap='viridis', origin='lower')
        
        # Configure axes
        plt.yticks(ticks=np.arange(len(layers_plotted)), labels=layers_plotted)
        
        # For X-axis, don't label every single tick if there are too many components
        step = max(1, args.max_components // 10)
        xticks = np.arange(0, args.max_components, step)
        xlabels = [str(i + 1) for i in xticks]
        plt.xticks(ticks=xticks, labels=xlabels)
        
        plt.colorbar(label='Explained Variance Ratio')
        
        plt.title(f"Explained Variance by PCA Component per Layer ({token_type})")
        plt.ylabel("Layer")
        plt.xlabel("PCA Component")
        output_filename = f"pca_variance_heatmap_deltas_{dir_suffix}.png"
    
        output_dir = PROJECT_ROOT / "plots" / "linear_probe" / "specificity" / "PCA"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_plot_path = output_dir / output_filename
        plt.savefig(output_plot_path)
        print(f"\nPlot saved to {output_plot_path}")

        # If input vector was provided, plot the dot products heatmap
        if args.input_vector and input_dots_matrix:
            # Map to absolute values
            dots_matrix_np = np.abs(np.array(input_dots_matrix))
            plt.figure(figsize=(10, 8))
            
            # Now using a sequential colormap since values are strictly non-negative
            vmax = np.max(dots_matrix_np)
            plt.imshow(dots_matrix_np, aspect='auto', cmap='Reds', origin='lower', 
                       vmin=0, vmax=vmax)
            
            plt.yticks(ticks=np.arange(len(layers_plotted)), labels=layers_plotted)
            plt.xticks(ticks=xticks, labels=xlabels)
            
            plt.colorbar(label='Dot Product Value')
            
            plt.title(f"Dot Products with PCA Components per Layer ({token_type})")
            plt.ylabel("Layer")
            plt.xlabel("PCA Component")
            
            dots_filename = f"pca_input_dot_products_heatmap_{dir_suffix}.png"
            dots_plot_path = output_dir / dots_filename
            plt.savefig(dots_plot_path)
            print(f"Dot products heatmap saved to {dots_plot_path}")

        # Plot parallel vs correct diff heatmap
        if parallel_cohens_d_matrix:
            diff_matrix_np = np.abs(np.array(parallel_cohens_d_matrix))
            plt.figure(figsize=(10, 8))
            
            vmax = np.max(diff_matrix_np)
            if vmax == 0:
                vmax = 1.0  # avoid division by zero in cmap if empty
                
            plt.imshow(diff_matrix_np, aspect='auto', cmap='Reds', origin='lower', 
                       vmin=0, vmax=vmax)
            
            plt.yticks(ticks=np.arange(len(layers_plotted)), labels=layers_plotted)
            plt.xticks(ticks=xticks, labels=xlabels)
            
            plt.colorbar(label="Cohen's d (Parallel vs Correct)")
            
            plt.title(f"Cohen's d of (Parallel vs Correct) onto PCA ({token_type})")
            plt.ylabel("Layer")
            plt.xlabel("PCA Component")
            
            diff_filename = f"pca_parallel_cohens_d_heatmap_{dir_suffix}.png"
            diff_plot_path = output_dir / diff_filename
            plt.savefig(diff_plot_path)
            print(f"Parallel Cohen's d heatmap saved to {diff_plot_path}")

        # Plot Cohen's d heatmap
        if args.plot_cohens_d and cohens_d_matrix:
            cohens_d_np = np.array(cohens_d_matrix)
            plt.figure(figsize=(10, 8))
            
            # Use 'Reds' because Cohen's d is plotted as absolute value
            vmax = np.max(cohens_d_np)
            if vmax == 0:
                vmax = 1.0  # avoid division by zero in cmap if empty
                
            plt.imshow(cohens_d_np, aspect='auto', cmap='Reds', origin='lower', 
                       vmin=0, vmax=vmax)
            
            plt.yticks(ticks=np.arange(len(layers_plotted)), labels=layers_plotted)
            plt.xticks(ticks=xticks, labels=xlabels)
            
            plt.colorbar(label="Cohen's d (Orthogonal vs Correct)")
            
            plt.title(f"Cohen's d of (Orthogonal vs Correct) onto PCA ({token_type})")
            plt.ylabel("Layer")
            plt.xlabel("PCA Component")
            
            diff_filename2 = f"pca_cohens_d_heatmap_{dir_suffix}.png"
            diff_plot_path2 = output_dir / diff_filename2
            plt.savefig(diff_plot_path2)
            print(f"Cohen's d heatmap saved to {diff_plot_path2}")

        # Plot dot product of the two difference vectors projected onto PCA
        if args.plot_mean_diff_dot_product and mean_diff_dot_product_matrix:
            dot_prod_np = np.abs(np.array(mean_diff_dot_product_matrix))
            plt.figure(figsize=(10, 8))
            
            vmax = np.max(dot_prod_np)
            if vmax == 0:
                vmax = 1.0  # avoid division by zero in cmap if empty
                
            plt.imshow(dot_prod_np, aspect='auto', cmap='Reds', origin='lower', 
                       vmin=0, vmax=vmax)
            
            plt.yticks(ticks=np.arange(len(layers_plotted)), labels=layers_plotted)
            plt.xticks(ticks=xticks, labels=xlabels)
            
            plt.colorbar(label='Absolute Scalar Multiplication Value')
            
            plt.title(f"Abs(Cohen's d (Parallel - Correct) * Cohen's d (Orthogonal - Correct)) ({token_type})")
            plt.ylabel("Layer")
            plt.xlabel("PCA Component")
            
            diff_filename3 = f"pca_mean_diff_scalar_mult_heatmap_{dir_suffix}.png"
            diff_plot_path3 = output_dir / diff_filename3
            plt.savefig(diff_plot_path3)
            print(f"Scalar multiplication heatmap saved to {diff_plot_path3}")

if __name__ == "__main__":
    main()
