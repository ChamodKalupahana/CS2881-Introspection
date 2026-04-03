import argparse
import sys
import re
import os
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from linear_probe.control.cohens_d import compute_cohens_d

def main():
    parser = argparse.ArgumentParser(
        description="Compare refusal probes (MMV/PCA) against a ground truth refusal vector and compute discriminability (Cohen's d)."
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

    # 1. Discover and Load Probes
    folder = Path(args.probes_folder_path)
    print(f"\n📐 Scanning probes in: {folder}")
    
    mmv_probes = {}  # layer -> vector
    pca_probes = []  # list of (layer, comp_idx, vector)
    
    # Load Mass-Mean Vectors
    mmv_files = list(folder.glob("mass_mean_vector*.pt"))
    if mmv_files:
        print(f"   Found Mass-Mass Vector file: {mmv_files[0].name}")
        loaded_mmv = torch.load(mmv_files[0], map_location="cpu")
        if isinstance(loaded_mmv, dict):
            mmv_probes = loaded_mmv
        else:
            print("   ⚠ Warning: mass_mass_vector file is not a dictionary. Skipping.")
    
    # Load PCA files
    pca_pattern = re.compile(r"PCA_num_(\d+)_layer_(\d+)_")
    for f in folder.glob("PCA_num_*.pt"):
        match = pca_pattern.search(f.name)
        if match:
            comp_idx = int(match.group(1))
            layer = int(match.group(2))
            vector = torch.load(f, map_location="cpu")
            pca_probes.append((layer, comp_idx, vector))
            
    print(f"   Loaded {len(mmv_probes)} layers from MMV and {len(pca_probes)} individual PCA probes.")

    # 2. Load Ground Truth Reference Data
    print(f"🎯 Loading Ground Truth: {args.ground_truth_path}")
    ground_truth = torch.load(args.ground_truth_path, map_location="cpu").detach().float()
    if ground_truth.dim() > 1:
        ground_truth = ground_truth.squeeze()
    ground_truth_np = ground_truth.numpy().reshape(1, -1)

    # 3. Smart Discovery and Loading of Activations
    # Expected structure: linear_probe/control/refusal_control/saved_activations/run_XX_YY_HH_MM
    run_folder_name = folder.name # e.g. run_04_03_19_46
    
    # Try local search first (relative to this script)
    current_script_dir = Path(__file__).resolve().parent
    activations_run_dir = current_script_dir / "saved_activations" / run_folder_name
    
    if not activations_run_dir.exists():
        # Fallback to absolute project-root-relative path
        activations_run_dir = PROJECT_ROOT / "linear_probe" / "control" / "refusal_control" / "saved_activations" / run_folder_name
        
    print(f"📦 Loading activations from: {activations_run_dir}")
    if not activations_run_dir.exists():
        print(f"❌ Error: Could not find matching activation folder at {activations_run_dir}")
        return

    pos_act_path = activations_run_dir / "positive_activations_refusal.pt"
    neg_act_path = activations_run_dir / "negative_activations_refusal.pt"
    
    positive_full = torch.load(pos_act_path, map_location="cpu")
    negative_full = torch.load(neg_act_path, map_location="cpu")

    # 4. Metric Computation (Cosine Sim to GT and Cohen's d across distributions)
    all_layers = sorted(set(mmv_probes.keys()) | set(l for l, ci, v in pca_probes))
    print(f"📊 Computing discriminability scores across {len(all_layers)} layers...")

    results = []  # List of {'type': ..., 'layer': ..., 'cos_sim': ..., 'd_score': ..., 'label': ...}

    for layer in all_layers:
        if layer not in positive_full or layer not in negative_full:
            continue
            
        # Get raw distributions for this layer
        pos_dist = positive_full[layer].detach().float().numpy()
        neg_dist = negative_full[layer].detach().float().numpy()

        # A. Mass-Mass Vector comparison
        if layer in mmv_probes:
            probe_v = mmv_probes[layer].detach().cpu().float().numpy()
            
            # 1. Cosine similarity to the single ground truth direction
            cos_sim_gt = float(cosine_similarity(probe_v.reshape(1, -1), ground_truth_np)[0, 0])
            
            # 2. Cohen's d (separation score on current data)
            pos_projs = pos_dist @ probe_v
            neg_projs = neg_dist @ probe_v
            d_score = compute_cohens_d(pos_projs, neg_projs)
            
            results.append({
                'type': 'MMV',
                'layer': layer,
                'cos_sim': cos_sim_gt,
                'd_score': d_score,
                'label': f"L{layer} "
            })
            
        # B. PCA components comparison
        for l, comp_idx, v in pca_probes:
            if l == layer:
                probe_v = v.detach().cpu().float().numpy()
                
                # 1. Cosine similarity to the ground truth
                cos_sim_gt = float(cosine_similarity(probe_v.reshape(1, -1), ground_truth_np)[0, 0])

                # for PCA, direction is arbitrary ⇒ always take magnitude
                cos_sim_gt = abs(cos_sim_gt)
                
                # 2. Cohen's d
                pos_projs = pos_dist @ probe_v
                neg_projs = neg_dist @ probe_v
                d_score = compute_cohens_d(pos_projs, neg_projs)
                
                results.append({
                    'type': f'PCA {comp_idx}',
                    'layer': layer,
                    'cos_sim': cos_sim_gt,
                    'd_score': d_score,
                    'label': f"PCA{comp_idx} L{layer}"
                })

    # ── Plotting ──────────────────────────────────────────────────────────
    if not results:
        print("❌ No valid probe/activation combinations found. Exiting.")
        return

    plt.figure(figsize=(14, 10))
    
    # Split results into groups for markers/colors
    mmv_results = [r for r in results if r['type'] == 'MMV']
    pca_results = [r for r in results if 'PCA' in r['type']]
    
    # Plot Mass-Mean Vectors (Blue circles)
    if mmv_results:
        plt.scatter(
            [r['d_score'] for r in mmv_results], 
            [r['cos_sim'] for r in mmv_results],
            c='blue', alpha=0.8, s=150, label='Mass-Mean Vector', marker='o', edgecolors='black'
        )
    
    # Plot PCA Components (Red diamonds)
    if pca_results:
        plt.scatter(
            [r['d_score'] for r in pca_results], 
            [r['cos_sim'] for r in pca_results],
            c='red', alpha=0.6, s=100, label='PCA Components', marker='D', edgecolors='grey'
        )

    # Label points (only those above threshold of interest)
    for r in results:
        is_mmv = r['type'] == 'MMV'
        # Labels: Always plot MMV, otherwise use threshold for PCA
        if is_mmv or r['d_score'] > 6 or abs(r['cos_sim']) > 0.15:
            # Shorten label for PCA: "PCA2 L20" -> "P2" or just keep it simple
            # As requested: "plot layer by colour and PCA component by num. inside box"
            # We'll use the layer for the label color or just use a standard label for now.
            label = r['label']
            if 'PCA' in r['type']:
                # Extract the number from "PCA 2"
                match = re.search(r'PCA (\d+)', r['type'])
                pca_num = match.group(1) if match else "?"
                label = f"P{pca_num} L{r['layer']}"
            
            plt.annotate(
                label, 
                (r['d_score'], r['cos_sim']), 
                textcoords="offset points", 
                xytext=(0, 10), 
                ha='center', 
                fontsize=8
            )

    plt.title(f"Probe Discriminability (Cohen's d) vs Alignment (Refusal)")
    plt.xlabel("Separation Quality (Cohen's d)")
    plt.ylabel("Cosine Similarity (Absolute for PCA)")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.legend()
    
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    file_name = "refusal_discriminability_vs_alignment.png"
    plt.savefig(save_path / file_name)
    print(f"\n🎨 Analysis plot saved to: {save_path / file_name}")
    
    plt.show()

if __name__ == "__main__":
    main()