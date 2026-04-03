import argparse
import sys
import re
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(
        description="Compare refusal probes (MMV/PCA) against a ground truth refusal vector."
    )
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--probes_folder_path", type=str, required=True,
                        help="Path to the folder containing mass_mass_vector and PCA files.")
    parser.add_argument("--ground_truth_path", type=str, 
                        default="ground_truth_probe/direction.pt",
                        help="Path to the ground truth direction.pt file.")
    parser.add_argument("--save_dir", type=str, default="../../../plots/control/refusal_control",
                        help="Directory to save the resulting scatter plot.")
    args = parser.parse_args()

    # 1. Load Model (optional reference)
    print(f"\n⏳ Loading tokenizer/config for model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Note: We don't necessarily need the full weights just to compare vectors, if we trust the pt files.
    # but the user's snippet included model loading.

    # ── Discover and Load Probes ──────────────────────────────────────────
    folder = Path(args.probes_folder_path)
    print(f"📐 Scanning probes in: {folder}")
    
    mmv_probes = {}  # layer -> vector
    pca_probes = []  # list of (layer, comp_idx, vector)
    
    # Load Mass-Mass Vectors (dict of layers)
    mmv_files = list(folder.glob("mass_mass_vector*.pt"))
    if mmv_files:
        print(f"   Found Mass-Mass Vector file: {mmv_files[0].name}")
        loaded_mmv = torch.load(mmv_files[0], map_location="cpu")
        if isinstance(loaded_mmv, dict):
            mmv_probes = loaded_mmv
        else:
            print("   ⚠ Warning: mass_mass_vector file is not a dictionary. Skipping.")
    
    # Load PCA files
    # Format: PCA_num_{comp_idx}_layer_{layer}_{concept_name}.pt
    pca_pattern = re.compile(r"PCA_num_(\d+)_layer_(\d+)_")
    for f in folder.glob("PCA_num_*.pt"):
        match = pca_pattern.search(f.name)
        if match:
            comp_idx = int(match.group(1))
            layer = int(match.group(2))
            vector = torch.load(f, map_location="cpu")
            pca_probes.append((layer, comp_idx, vector))
            
    print(f"   Loaded {len(mmv_probes)} layers from MMV and {len(pca_probes)} individual PCA probes.")

    # ── Load Reference Data ──────────────────────────────────────────────
    print(f"🎯 Loading Ground Truth: {args.ground_truth_path}")
    ground_truth = torch.load(args.ground_truth_path, map_location="cpu").detach().float()
    if ground_truth.dim() > 1:
        ground_truth = ground_truth.squeeze()
    ground_truth_np = ground_truth.numpy().reshape(1, -1)

    # ── Comparison & Results Analysis ─────────────────────────────────────
    all_layers = sorted(set(mmv_probes.keys()) | set(l for l, ci, v in pca_probes))
    print(f"📊 Comparing across {len(all_layers)} layers detected in probes...")

    results = []  # List of {'type': ..., 'layer': ..., 'cos_sim': ..., 'dist': ..., 'label': ...}

    for layer in all_layers:
        # A. Mass-Mass Vector comparison
        if layer in mmv_probes:
            probe_v = mmv_probes[layer].detach().cpu().float().numpy().reshape(1, -1)
            cos_sim = float(cosine_similarity(probe_v, ground_truth_np)[0, 0])
            dist = float(np.linalg.norm(probe_v - ground_truth_np))
            results.append({
                'type': 'MMV',
                'layer': layer,
                'cos_sim': cos_sim,
                'dist': dist,
                'label': f"L{layer}"
            })
            
        # B. PCA components comparison
        for l, comp_idx, v in pca_probes:
            if l == layer:
                probe_v = v.detach().cpu().float().numpy().reshape(1, -1)
                cos_sim = float(cosine_similarity(probe_v, ground_truth_np)[0, 0])
                dist = float(np.linalg.norm(probe_v - ground_truth_np))
                results.append({
                    'type': f'PCA {comp_idx}',
                    'layer': layer,
                    'cos_sim': cos_sim,
                    'dist': dist,
                    'label': f"PCA{comp_idx} L{layer}"
                })

    # ── Plotting ──────────────────────────────────────────────────────────
    if not results:
        print("❌ No probes found for comparison. Exiting.")
        return

    plt.figure(figsize=(12, 8))
    
    # Colors for different layers to see evolution
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=min(all_layers), vmax=max(all_layers))

    mmv_results = [r for r in results if r['type'] == 'MMV']
    pca_results = [r for r in results if 'PCA' in r['type']]
    
    # Plot Mass-Mass Vectors (Circles)
    if mmv_results:
        scatter_mmv = plt.scatter(
            [r['dist'] for r in mmv_results], 
            [r['cos_sim'] for r in mmv_results],
            c=[r['layer'] for r in mmv_results], cmap=cmap, norm=norm,
            alpha=0.9, s=200, label='Mass-Mean Vector', marker='o', edgecolors='black'
        )
    
    # Plot PCA Components (Diamonds)
    if pca_results:
        plt.scatter(
            [r['dist'] for r in pca_results], 
            [r['cos_sim'] for r in pca_results],
            c=[r['layer'] for r in pca_results], cmap=cmap, norm=norm,
            alpha=0.6, s=120, marker='D', edgecolors='grey'
        )

    # Label points (only those above threshold of interest)
    for r in results:
        # Highlight interesting probes (Top Cosine Sim or low Distance)
        if r['cos_sim'] > 0.3 or r['dist'] < 10:
            plt.annotate(r['label'], (r['dist'], r['cos_sim']), 
                         textcoords="offset points", xytext=(0,12), ha='center', fontsize=9)

    plt.title(f"Probe Alignment with Refusal Ground Truth")
    plt.xlabel("L2 Distance to Ground Truth")
    plt.ylabel("Cosine Similarity to Ground Truth")
    plt.gca().invert_xaxis()  # Low distance on the right (higher quality)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Colorbar for layers
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=plt.gca(), label='Layer Index')
    
    plt.legend()
    
    # Save results
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / "refusal_ground_truth_comparison.png")
    print(f"\n🎨 Comparison plot saved to: {save_path / 'refusal_ground_truth_comparison.png'}")
    
    plt.show()

if __name__ == "__main__":
    main()