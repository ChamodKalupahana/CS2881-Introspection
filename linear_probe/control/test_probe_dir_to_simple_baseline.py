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
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from linear_probe.control.concept_vector_functions import extract_control_from_baseline


def main():
    parser = argparse.ArgumentParser(
        description="Compare trained probe vectors with simple baseline vectors (word - mean_baseline)."
    )
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--probes_folder_path", type=str, required=True,
                        help="Path to the folder containing mass_mass_vector and PCA files.")
    parser.add_argument("--dataset_name", type=str, default="simple_data",
                        help="Dataset for baseline words (default: simple_data).")
    parser.add_argument("--min_layer_to_save", type=int, default=16,
                        help="Minimum layer for baseline extraction (default: 16).")
    parser.add_argument("--concept_word", type=str, required=True,
                        help="The word to compute the baseline steering vector for (e.g. Magnetism).")
    parser.add_argument("--vec_type", type=str, default="avg", choices=["avg", "last"],
                        help="Choose 'avg' (average activation) or 'last' (last token activation).")
    parser.add_argument("--save_dir", type=str, default="../../plots/control",
                        help="Directory to save the resulting scatter plot.")
    args = parser.parse_args()

    # load model
    print(f"\n⏳ Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model loaded on {device}\n")

    # ── Discover and Load Probes ──────────────────────────────────────────
    folder = Path(args.probes_folder_path)
    print(f"📐 Scanning probes in: {folder}")
    
    mmv_probes = {}  # layer -> vector
    pca_probes = []  # list of (layer, comp_idx, vector)
    
    # 1. Load Mass-Mass Vectors (dict of layers)
    mmv_files = list(folder.glob("mass_mass_vector*.pt"))
    if mmv_files:
        print(f"   Found Mass-Mass Vector file: {mmv_files[0].name}")
        loaded_mmv = torch.load(mmv_files[0], map_location="cpu")
        if isinstance(loaded_mmv, dict):
            mmv_probes = loaded_mmv
        else:
            print("   ⚠ Warning: mass_mass_vector file is not a dictionary. Skipping.")
    
    # 2. Load PCA files
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

    # ── Extract Baselines ────────────────────────────────────────────────
    print(f"🛠  Extracting baseline-corrected vectors for '{args.concept_word}'...")
    baseline_vectors_all_layers = extract_control_from_baseline(
        model,
        tokenizer,
        args.dataset_name,
        args.min_layer_to_save,
        args.concept_word
    )
    
    # Unique layers to iterate over
    all_layers = sorted(set(mmv_probes.keys()) | set(l for l, ci, v in pca_probes))
    all_layers = [l for l in all_layers if l in baseline_vectors_all_layers]
    
    print(f"📊 Comparing across {len(all_layers)} layers: {all_layers}")

    results = []  # List of {'type': ..., 'layer': ..., 'cos_sim': ..., 'dist': ..., 'label': ...}

    for layer in all_layers:
        baseline_v = baseline_vectors_all_layers[layer][args.vec_type].detach().cpu().float().numpy().reshape(1, -1)
        
        # A. Mass-Mass Vector comparison
        if layer in mmv_probes:
            probe_v = mmv_probes[layer].detach().cpu().float().numpy().reshape(1, -1)
            cos_sim = float(cosine_similarity(probe_v, baseline_v)[0, 0])
            dist = float(np.linalg.norm(probe_v - baseline_v))
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
                cos_sim = float(cosine_similarity(probe_v, baseline_v)[0, 0])
                dist = float(np.linalg.norm(probe_v - baseline_v))
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
    
    # Split results into groups for markers/colors
    mmv_results = [r for r in results if r['type'] == 'MMV']
    pca_results = [r for r in results if 'PCA' in r['type']]
    
    # Plot Mass-Mass Vectors (Blue circles)
    if mmv_results:
        plt.scatter(
            [r['dist'] for r in mmv_results], 
            [r['cos_sim'] for r in mmv_results],
            c='blue', alpha=0.8, s=150, label='Mass-Mean Vector', marker='o'
        )
    
    # Plot PCA Components (Red diamonds)
    if pca_results:
        plt.scatter(
            [r['dist'] for r in pca_results], 
            [r['cos_sim'] for r in pca_results],
            c='red', alpha=0.6, s=100, label='PCA Components', marker='D'
        )

    # Label points (only those above threshold of interest)
    for r in results:
        if r['dist'] > 6 or abs(r['cos_sim']) > 0.15:
            # TODO: always plot MMV label
            # TODO: plot layer by colour and PCA componenet by num. inside box
            plt.annotate(r['label'], (r['dist'], r['cos_sim']), 
                         textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    plt.title(f"Probe vs Baseline Comparison ({args.concept_word})")
    plt.xlabel("Euclidean Distance (L2 norm)")
    plt.ylabel("Cosine Similarity")
    plt.gca().invert_xaxis()  # Low distance on the right (ideal)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Save plot
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    file_name = f"comparison_{args.concept_word}_{args.vec_type}_multi.png"
    plt.savefig(save_path / file_name)
    print(f"\n🎨 Results summary for '{args.concept_word}':")
    print(f"   Plotted {len(mmv_results)} MMV points and {len(pca_results)} PCA points.")
    print(f"   Plot saved to: {save_path / file_name}")
    
    plt.show()

if __name__ == "__main__":
    main()