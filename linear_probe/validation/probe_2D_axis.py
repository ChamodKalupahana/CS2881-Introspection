import torch
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import einops
import numpy as np
import matplotlib.pyplot as plt
import re

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def extract_layer_from_filename(path):
    """
    Extracts the layer number (e.g., L14) from a filename using regex.
    """
    name = Path(path).name
    match = re.search(r"L(\d+)", name)
    if match:
        return int(match.group(1))
    return None

def load_activations(run_dir):
    """
    Loads saved activations from the given run directory.
    Reorganizes them into a dictionary: data[category][(concept, layer, position)] = tensor
    """
    run_path = Path(run_dir).resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    activations_by_category = defaultdict(dict)
    
    # 1. Iterate through categories (subdirectories)
    categories = [d.name for d in run_path.iterdir() if d.is_dir()]
    
    print(f"📂 Loading activations from: {run_path}")
    for category in categories:
        category_path = run_path / category
        pt_files = list(category_path.glob("*.pt"))
        
        if not pt_files:
            continue
            
        print(f"  📁 Category: {category} ({len(pt_files)} files)")
        for pf in tqdm(pt_files, desc=f"Loading {category}", leave=False):
            # Filename format: {concept}_c{coeff}_l{layer}_v{vector_type}.pt
            concept = pf.stem.split("_c")[0]
            
            try:
                # Load activations_dict: (layer, position) -> tensor
                activations_dict = torch.load(pf, map_location=torch.device('cpu'), weights_only=True)
                
                for (layer, position), tensor in activations_dict.items():
                    activations_by_category[category][(concept, layer, position)] = tensor
                    
            except Exception as e:
                print(f"    ❌ Error loading {pf.name}: {e}")
                
    return activations_by_category

def get_category_tensors(activations):
    """
    Converts raw activation dictionaries into structured 4D tensors per category.
    Returns: category_tensors[category] = {tensor, concepts, layers, positions}
    """
    category_tensors = {}
    print("\n📦 CONVERTING TO TENSORS")
    print(f"{'='*30}")
    
    for category, samples in activations.items():
        if not samples:
            continue
            
        # Identify unique dimensions
        concepts = sorted(list(set(k[0] for k in samples.keys())))
        layers = sorted(list(set(k[1] for k in samples.keys())))
        positions = sorted(list(set(k[2] for k in samples.keys())))
        
        # Determine d_model (assumes uniform vector size)
        first_vec = next(iter(samples.values()))
        d_model = first_vec.shape[0]
        
        # Initialize 4D tensor: [num_concepts, num_layers, num_positions, d_model]
        cat_tensor = torch.zeros((len(concepts), len(layers), len(positions), d_model), 
                                 dtype=first_vec.dtype, device='cpu')
        
        # Index mappings
        concept_to_idx = {c: i for i, c in enumerate(concepts)}
        layer_to_idx = {l: i for i, l in enumerate(layers)}
        pos_to_idx = {p: i for i, p in enumerate(positions)}
        
        # Fill tensor
        for (c, l, p), vec in samples.items():
            cat_tensor[concept_to_idx[c], layer_to_idx[l], pos_to_idx[p]] = vec
            
        category_tensors[category] = {
            "tensor": cat_tensor,
            "concepts": concepts,
            "layers": layers,
            "positions": positions
        }
        print(f"  ✅ {category:<20}: {cat_tensor.shape}")

    print(f"{'='*30}\n")
    return category_tensors

def main():
    parser = argparse.ArgumentParser(description="Visualize activations on a 2D axis defined by two probes.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the saved_activations run directory.")
    parser.add_argument("--probe1", type=str, required=True, help="Path to the first .pt probe vector.")
    parser.add_argument("--probe2", type=str, required=True, help="Path to the second .pt probe vector.")
    parser.add_argument("--layer", type=int, default=None, help="Optional layer override/fallback.")
    parser.add_argument("--position", type=int, default=0, help="Token position to project (default: 0).")
    parser.add_argument("--output", type=str, default=None, help="Save path for the plot.")
    args = parser.parse_args()

    # 1. Resolve layers for each probe
    layer1 = extract_layer_from_filename(args.probe1)
    layer2 = extract_layer_from_filename(args.probe2)
    
    # Fallback/inheritance logic
    if layer1 is None:
        layer1 = layer2 or args.layer
    if layer2 is None:
        layer2 = layer1 or args.layer
        
    if layer1 is None or layer2 is None:
        print("❌ Error: No layer number found in probe filenames or --layer argument.")
        sys.exit(1)
        
    # 2. Load Probes
    print(f"📡 Loading probes...")
    p1 = torch.load(args.probe1, map_location='cpu', weights_only=True).float()
    p2 = torch.load(args.probe2, map_location='cpu', weights_only=True).float()
    
    # Ensure they are 1D vectors
    if p1.dim() > 1: p1 = p1.flatten()
    if p2.dim() > 1: p2 = p2.flatten()
    
    # Normalize
    p1_unit = p1 / (torch.norm(p1) + 1e-9)
    p2_unit = p2 / (torch.norm(p2) + 1e-9)

    # 3. Load activations
    activations = load_activations(args.run_dir)
    
    # 4. Summary of loaded data
    print("\n📊 LOAD SUMMARY")
    print(f"{'='*30}")
    for category, dict_data in activations.items():
        print(f"  {category:<20}: {len(dict_data)} vectors")
    print(f"{'='*30}\n")
    
    # 5. Convert into tensor for each category
    category_info = get_category_tensors(activations)

    # 6. Compute Projections
    print(f"📐 Projecting activations onto 2D axes (X-Layer: {layer1}, Y-Layer: {layer2}, Pos: {args.position})...")
    
    plt.figure(figsize=(12, 8))
    
    # Define colors for common categories
    category_colors = {
        "not_detected": "red",
        "detected_correct": "blue",
        "calibration_correct": "green",
        "detected_parallel": "cyan",
        "detected_orthogonal": "orange"
    }

    for category, info in category_info.items():
        layers = info["layers"]
        positions = info["positions"]
        
        if layer1 not in layers or layer2 not in layers or args.position not in positions:
            print(f"  ⚠️ Skipping {category}: Required layers ({layer1}, {layer2}) or Pos {args.position} not found.")
            continue
            
        l1_idx = layers.index(layer1)
        l2_idx = layers.index(layer2)
        pos_idx = positions.index(args.position)
        
        # [num_concepts, d_model]
        # Project activations from their respective layers
        acts1 = info["tensor"][:, l1_idx, pos_idx, :]
        acts2 = info["tensor"][:, l2_idx, pos_idx, :]
        
        # [num_concepts]
        x_projs = (acts1 @ p1_unit).numpy()
        y_projs = (acts2 @ p2_unit).numpy()
        
        color = category_colors.get(category, None)
        plt.scatter(x_projs, y_projs, label=category, alpha=0.6, edgecolors='none', c=color)

    plt.xlabel(f"Projection onto {Path(args.probe1).name} (L{layer1})")
    plt.ylabel(f"Projection onto {Path(args.probe2).name} (L{layer2})")
    plt.title(f"2D Activation Subspace (Pos {args.position})\nX: L{layer1}, Y: L{layer2}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if args.output:
        save_path = Path(args.output)
    else:
        save_dir = PROJECT_ROOT / "plots" / "linear_probe" / "validation"
        os.makedirs(save_dir, exist_ok=True)
        run_name = Path(args.run_dir).name
        # Filename reflects the layers used
        l_suffix = f"L{layer1}" if layer1 == layer2 else f"L{layer1}_L{layer2}"
        save_path = save_dir / f"{run_name}_{l_suffix}_P{args.position}.png"
        
    plt.savefig(save_path, bbox_inches='tight')
    print(f"🎨 Plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()