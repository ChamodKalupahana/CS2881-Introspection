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

    parser = argparse.ArgumentParser(description="Plot Cohen's d scatter graph for PCA components")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Path to the prompt_activations run directory (default: latest in saved_activations/)")
    parser.add_argument("--token_type", type=str, default="last_token",
                        choices=["last_token", "prompt_last_token"],
                        help="Which token position to read (default: last_token)")
    parser.add_argument("--max_components", type=int, default=20, 
                        help="Maximum number of PCA components to process (default: 20)")
    parser.add_argument("--max_layer", type=int, default=None, 
                        help="Maximum layer to process (inclusive)")
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

    # Intersect keys and sort them so order is deterministic
    common_keys = sorted(list(set(pos_map.keys()).intersection(set(neg_map.keys()))))
    print(f"Found {len(common_keys)} matched (concept, prompt_id) pairs.")

    if not common_keys:
        print("ERROR: No matching positive/negative pairs found.")
        return

    # Check which layers are available in the first example
    sample_acts = pos_map[list(common_keys)[0]]
    available_layers = [k for k in sample_acts.keys() if isinstance(k, int)]
    
    if args.max_layer is not None:
        available_layers = [k for k in available_layers if k <= args.max_layer]
        
    available_layers = sorted(available_layers)
    
    if not available_layers:
        print("ERROR: No layer data found in activations dict.")
        return

    calib_cohens_d_all = []
    notdet_cohens_d_all = []
    labels = []  # format: 'L16-PC0'

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

        deltas_tensor = torch.stack(deltas)
        
        # Collect not_detected and pos_all for this layer
        notdet_vecs = []
        for key in sorted(notdet_map.keys()):
            acts = notdet_map[key]
            if layer in acts:
                vec = acts[layer].get(token_type)
                if vec is not None:
                    notdet_vecs.append(vec.float())
        
        pos_all_vecs = []
        for key in sorted(pos_map.keys()):
            acts = pos_map[key]
            if layer in acts:
                vec = acts[layer].get(token_type)
                if vec is not None:
                    pos_all_vecs.append(vec.float())

        neg_all_vecs = []
        for key in sorted(neg_map.keys()):
            acts = neg_map[key]
            if layer in acts:
                vec = acts[layer].get(token_type)
                if vec is not None:
                    neg_all_vecs.append(vec.float())

        try:
            # PCA matching previous script
            mean = deltas_tensor.mean(dim=0)
            centered = deltas_tensor - mean
            centered_float = centered.to(dtype=torch.float32)
            
            U, S, Vh = torch.linalg.svd(centered_float, full_matrices=False)
            
            if notdet_vecs and pos_all_vecs and neg_all_vecs:
                notdet_tensor = torch.stack(notdet_vecs)
                pos_tensor = torch.stack(pos_all_vecs)
                neg_tensor = torch.stack(neg_all_vecs)
                
                Vh_f = Vh.to(notdet_tensor.dtype)
                
                notdet_proj = torch.matmul(notdet_tensor, Vh_f.T)
                pos_proj = torch.matmul(pos_tensor, Vh_f.T)
                neg_proj = torch.matmul(neg_tensor, Vh_f.T)
                
                # N
                n_notdet = notdet_proj.shape[0]
                n_pos = pos_proj.shape[0]
                n_neg = neg_proj.shape[0]
                
                # Means
                mu_notdet = notdet_proj.mean(dim=0).numpy()
                mu_pos = pos_proj.mean(dim=0).numpy()
                mu_neg = neg_proj.mean(dim=0).numpy()
                
                # Variances
                s2_notdet = notdet_proj.var(dim=0, unbiased=True).numpy()
                s2_pos = pos_proj.var(dim=0, unbiased=True).numpy()
                s2_neg = neg_proj.var(dim=0, unbiased=True).numpy()
                
                # Pooled SDs
                s_pooled_notdet = np.sqrt(((n_notdet - 1) * s2_notdet + (n_pos - 1) * s2_pos) / (n_notdet + n_pos - 2))
                s_pooled_neg = np.sqrt(((n_neg - 1) * s2_neg + (n_pos - 1) * s2_pos) / (n_neg + n_pos - 2))
                
                s_pooled_notdet[s_pooled_notdet == 0] = 1e-8
                s_pooled_neg[s_pooled_neg == 0] = 1e-8
                
                # Cohen's d (Absolute values for the dot)
                notdet_cohens_d = np.abs(mu_notdet - mu_pos) / s_pooled_notdet
                neg_cohens_d = np.abs(mu_neg - mu_pos) / s_pooled_neg
                
                if len(notdet_cohens_d) > args.max_components:
                    notdet_cohens_d = notdet_cohens_d[:args.max_components]
                    neg_cohens_d = neg_cohens_d[:args.max_components]
                    
                calib_cohens_d_all.extend(neg_cohens_d)
                notdet_cohens_d_all.extend(notdet_cohens_d)
                
                for pc in range(len(notdet_cohens_d)):
                    labels.append(f"L{layer}-PC{pc+1}")

                print(f"{layer}\t| Processed {len(notdet_cohens_d)} components")

        except Exception as e:
            print(f"{layer}\t| Error: {e}")
            continue

    if not calib_cohens_d_all:
        print("No valid data processed successfully.")
        return

    # ── Plotting ─────────────────────────────────────────────────────────
    dir_suffix = run_dir.name.replace("prompt_activations_", "")

    plt.figure(figsize=(12, 10))
    
    # Create the scatter plot
    plt.scatter(notdet_cohens_d_all, calib_cohens_d_all, alpha=0.6, edgecolors='w', s=50)
    
    # Annotate points
    for i, label in enumerate(labels):
        plt.annotate(label, (notdet_cohens_d_all[i], calib_cohens_d_all[i]), 
                     fontsize=7, alpha=0.7, xytext=(3, 3), textcoords='offset points')
                     
    # Add a diagonal line for reference (y = x)
    max_val = max(max(notdet_cohens_d_all), max(calib_cohens_d_all))
    plt.plot([0, max_val * 1.05], [0, max_val * 1.05], 'k--', alpha=0.3, label='y = x')
    
    plt.title(f"Cohen's d: Calibration vs Not Detected ({token_type})")
    plt.ylabel("Cohen's d (Calibration Correct vs Detected Correct)")
    plt.xlabel("Cohen's d (Not Detected vs Detected Correct)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    output_dir = PROJECT_ROOT / "plots" / "linear_probe" / "prompt_senstivity" / "PCA"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"pca_cohens_d_scatter_{dir_suffix}.png"
    output_plot_path = output_dir / output_filename
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_plot_path}")

if __name__ == "__main__":
    main()
