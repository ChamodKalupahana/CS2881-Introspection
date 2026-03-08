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

    print(f"Run directory: {run_dir}")
    print(f"Token type: {token_type}")
    print(f"Loading Positive (injected + correct) from: {pos_dir}")
    print(f"Loading Negative (calibration + correct) from: {neg_dir}")

    pos_map = load_activations(pos_dir)
    neg_map = load_activations(neg_dir)

    print(f"Loaded {len(pos_map)} positive examples and {len(neg_map)} negative examples.")

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

    print(f"\nProcessing layers...")
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

            var_array = explained_variance_ratio.numpy()
            if len(var_array) > args.max_components:
                var_array = var_array[:args.max_components]
            elif len(var_array) < args.max_components:
                var_array = np.pad(var_array, (0, args.max_components - len(var_array)), 'constant')
                
            variance_matrix.append(var_array)
            print(f"{layer}\t| Extracted {len(explained_variance_ratio)} components")

        except Exception as e:
            print(f"{layer}\t| Error: {e}")
            continue

    if not layers_plotted:
        print("No layers processed successfully.")
        return

    # ── Plotting ─────────────────────────────────────────────────────────
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
    
    dir_suffix = run_dir.name.replace("prompt_activations_", "")
    
    plt.title(f"Explained Variance by PCA Component per Layer ({token_type})")
    plt.ylabel("Layer")
    plt.xlabel("PCA Component")
    output_filename = f"pca_variance_heatmap_deltas_{dir_suffix}.png"
    
    output_dir = PROJECT_ROOT / "plots" / "linear_probe" / "prompt_senstivity" / "PCA"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_plot_path = output_dir / output_filename
    plt.savefig(output_plot_path)
    print(f"\nPlot saved to {output_plot_path}")

if __name__ == "__main__":
    main()
