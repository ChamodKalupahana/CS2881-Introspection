import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Compute DLA of activations across layers towards a probe direction.")
    parser.add_argument("--activation_dir", type=str, required=True, help="Directory containing .pt files with intermediate activations")
    parser.add_argument("--probe_file", type=str, required=True, help="Path to the probe vector .pt file")
    parser.add_argument("--independent_layers", action="store_true", help="Calculate exact layer contribution sequentially instead of cumulative projection.")
    args = parser.parse_args()

    act_dir = Path(args.activation_dir)
    probe_path = Path(args.probe_file)

    if not act_dir.is_dir():
        print(f"Error: Activation directory {act_dir} not found or is not a directory.")
        return
    if not probe_path.exists():
        print(f"Error: Probe file {probe_path} not found.")
        return

    print(f"Loading probe vector from: {probe_path}")
    probe = torch.load(probe_path, map_location="cpu", weights_only=False).float()
    if probe.dim() > 1:
        probe = probe.squeeze()
    
    # Ensure probe direction is normalized
    probe_dir = probe / torch.norm(probe)

    # Iterate through all .pt files in the directory recursively (since structure is often nested)
    pt_files = list(act_dir.rglob("*.pt"))
    if not pt_files:
        print(f"Error: No .pt files found in {act_dir}")
        return

    print(f"Found {len(pt_files)} activation files. Processing...")
    
    all_dlas = []
    layer_ticks = None

    for act_path in tqdm(pt_files, desc="Processing files"):
        data = torch.load(act_path, map_location="cpu", weights_only=False)
        if 'activations' not in data:
            continue
            
        acts = data['activations']
        
        layers = []
        dlas = []
        prev_proj = 0.0  # Keep track of the previous layer's score

        # Sort the layers
        layer_keys = sorted([k for k in acts.keys() if isinstance(k, int)])

        for layer in layer_keys:
            act_dict = acts[layer]
            if 'last_token' in act_dict:
                vec = act_dict['last_token'].float()
                if vec.dim() > 1:
                    vec = vec.squeeze()
                
                # 1. Get the total presence of the feature at this layer
                current_proj = torch.dot(vec, probe_dir).item()
                
                if args.independent_layers:
                    # 2. Subtract the previous layer to find THIS layer's exact contribution
                    layer_contribution = current_proj - prev_proj
                    dlas.append(layer_contribution)
                    # 3. Update previous projection for the next loop iteration
                    prev_proj = current_proj
                else:
                    dlas.append(current_proj)
                
                layers.append(layer)
        
        if layers:
            all_dlas.append(dlas)
            if layer_ticks is None:
                layer_ticks = layers
                
    if not all_dlas:
        print("Error: Could not extract valid DLA trajectories from the files.")
        return
        
    all_dlas_np = np.array(all_dlas)  # Shape: (num_files, num_layers)
    mean_dlas = np.mean(all_dlas_np, axis=0)
    std_dlas = np.std(all_dlas_np, axis=0)
    
    print(f"Extracted trajectories across {len(layer_ticks)} layers: {layer_ticks[0]} to {layer_ticks[-1]}")

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Plot the mean trajectory
    plt.plot(layer_ticks, mean_dlas, marker='o', linestyle='-', color='b', linewidth=2, markersize=6, label=f"Mean (n={len(pt_files)})")
    
    # Shade the standard deviation bounds
    plt.fill_between(layer_ticks, mean_dlas - std_dlas, mean_dlas + std_dlas, color='b', alpha=0.2, label="±1 Std Dev")
    
    # Highlight the zero line if it crosses
    if np.min(mean_dlas - std_dlas) < 0 < np.max(mean_dlas + std_dlas):
        plt.axhline(0, color='black', linestyle='--', linewidth=1)

    plt.xlabel("Layer")
    if args.independent_layers:
        plt.ylabel("Independent DLA Contribution")
        plt.title(f"Independent Layer Contribution towards Probe Direction\nDir: {act_dir.name}\nProbe: {probe_path.name}")
    else:
        plt.ylabel("Cumulative DLA Projection onto Probe")
        plt.title(f"Cumulative DLA towards Probe Direction across Layers\nDir: {act_dir.name}\nProbe: {probe_path.name}")
        
    plt.grid(True, alpha=0.3)
    plt.xticks(layer_ticks)
    plt.legend()
    
    # Find script directory and map plot output
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir.parent / "plots" / "linear_probe" / "DLA_towards_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{act_dir.name}_vs_{probe_path.stem}.png"
    
    plt.savefig(out_path, bbox_inches='tight')
    print(f"\nSUCCESS: Saved DLA layer progression plot to {out_path}")

if __name__ == "__main__":
    main()
