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
    (concept, prompt_id) -> activations dict
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
            data_map[(concept, pid)] = data["activations"]
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
    parser.add_argument("--plot_pc1", action="store_true", 
                        help="Plot Explained Variance of PC1 as a line graph instead of the 2D heatmap")
    parser.add_argument("--input_vector", type=str, default=None,
                        help="Path to an activation .pt file to compute dot product against PCA components")
    args = parser.parse_args()

    token_type = args.token_type

    # ── Resolve run directory ────────────────────────────────────────────
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        sa_dir = script_dir / "saved_activations"
        candidates = sorted(sa_dir.glob("prompt_activations_*"))
        if not candidates:
            print("ERROR: No prompt_activations_* directories found in saved_activations/")
            return
        run_dir = candidates[-1]

    pos_dir = run_dir / "positive" / "detected_correct"
    neg_dir = run_dir / "negative" / "detected_correct"
    notdet_dir = run_dir / "positive" / "not_detected"

    print(f"Run directory: {run_dir}")
    print(f"Token type: {token_type}")
    print(f"Loading Positive (injected + correct) from: {pos_dir}")
    print(f"Loading Negative (calibration + correct) from: {neg_dir}")
    print(f"Loading Positive (not detected) from: {notdet_dir}")

    pos_map = load_activations(pos_dir)
    neg_map = load_activations(neg_dir)
    notdet_map = load_activations(notdet_dir)

    print(f"Loaded {len(pos_map)} positive examples, {len(neg_map)} negative examples, and {len(notdet_map)} not_detected examples.")

    input_data = None
    if args.input_vector:
        input_path = Path(args.input_vector)
        if input_path.exists():
            input_data = torch.load(input_path, map_location="cpu", weights_only=False)
            print(f"Loaded input vector from: {input_path}")
        else:
            print(f"ERROR: Input vector file not found at {input_path}")
            return

    # Intersect keys
    common_keys = set(pos_map.keys()).intersection(set(neg_map.keys()))
    print(f"Found {len(common_keys)} matched (concept, prompt_id) pairs.")

    if not common_keys:
        print("ERROR: No matching positive/negative pairs found.")
        return

    # Check which layers are available in the first example
    sample_acts = pos_map[list(common_keys)[0]]
    available_layers = [k for k in sample_acts.keys() if isinstance(k, int)]
    available_layers = sorted(available_layers)
    
    if not available_layers:
        print("ERROR: No layer data found in activations dict.")
        return

    layers_plotted = []
    variance_matrix = []
    pc1_values = []
    input_dots_matrix = []
    notdet_diff_matrix = []

    print(f"\nProcessing layers...")
    if args.plot_pc1:
        print("\nLayer\t| PC1 Explained Variance")
    print("-" * 40)

    for layer in available_layers:
        deltas = []
        skip_layer = False

        for key in common_keys:
            pos_layer_data = pos_map[key].get(layer)
            neg_layer_data = neg_map[key].get(layer)

            if pos_layer_data is None or neg_layer_data is None:
                skip_layer = True
                break

            pos_vec = pos_layer_data.get(token_type)
            neg_vec = neg_layer_data.get(token_type)

            if pos_vec is None or neg_vec is None:
                skip_layer = True
                break

            delta = pos_vec.float() - neg_vec.float()
            deltas.append(delta)

        if skip_layer:
            print(f"{layer}\t| Missing data, skipping")
            continue

        deltas_tensor = torch.stack(deltas)  # [num_pairs, hidden_size]
        
        # Collect not_detected and pos_all for this layer
        notdet_vecs = []
        for key, acts in notdet_map.items():
            if layer in acts:
                vec = acts[layer].get(token_type)
                if vec is not None:
                    notdet_vecs.append(vec.float())
        
        pos_all_vecs = []
        for key, acts in pos_map.items():
            if layer in acts:
                vec = acts[layer].get(token_type)
                if vec is not None:
                    pos_all_vecs.append(vec.float())

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

            # Store difference in means for not_detected vs pos_correct
            if notdet_vecs and pos_all_vecs:
                mean_notdet = torch.stack(notdet_vecs).mean(dim=0)
                mean_pos_all = torch.stack(pos_all_vecs).mean(dim=0)
                mean_diff = mean_notdet - mean_pos_all
                diff_proj = torch.matmul(Vh, mean_diff).numpy()
                
                if len(diff_proj) > args.max_components:
                    diff_proj = diff_proj[:args.max_components]
                elif len(diff_proj) < args.max_components:
                    diff_proj = np.pad(diff_proj, (0, args.max_components - len(diff_proj)), 'constant')
                notdet_diff_matrix.append(diff_proj)
            else:
                notdet_diff_matrix.append(np.zeros(args.max_components))

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
    
        output_dir = PROJECT_ROOT / "plots" / "linear_probe" / "prompt_senstivity" / "PCA"
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

        # Plot not_detected vs pos_correct diff heatmap
        if notdet_diff_matrix:
            diff_matrix_np = np.array(notdet_diff_matrix)
            plt.figure(figsize=(10, 8))
            
            vmax = np.max(np.abs(diff_matrix_np))
            if vmax == 0:
                vmax = 1.0  # avoid division by zero in cmap if empty
                
            plt.imshow(diff_matrix_np, aspect='auto', cmap='RdBu_r', origin='lower', 
                       vmin=-vmax, vmax=vmax)
            
            plt.yticks(ticks=np.arange(len(layers_plotted)), labels=layers_plotted)
            plt.xticks(ticks=xticks, labels=xlabels)
            
            plt.colorbar(label='Dot Product Value (Mean Not Detected - Mean Pos Correct)')
            
            plt.title(f"Projection of (Not Detected - Pos Correct) onto PCA ({token_type})")
            plt.ylabel("Layer")
            plt.xlabel("PCA Component")
            
            diff_filename = f"pca_notdet_diff_heatmap_{dir_suffix}.png"
            diff_plot_path = output_dir / diff_filename
            plt.savefig(diff_plot_path)
            print(f"Not Detected diff heatmap saved to {diff_plot_path}")

if __name__ == "__main__":
    main()
