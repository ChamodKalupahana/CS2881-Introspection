import argparse
import math
import sys
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
    parser.add_argument("--probe_path", type=str, required=True,
                        help="Path to the trained probe dictionary .pt file.")
    parser.add_argument("--dataset_name", type=str, default="simple_data",
                        help="Dataset for baseline words (default: simple_data).")
    parser.add_argument("--min_layer_to_save", type=int, default=16,
                        help="Minimum layer for baseline extraction (default: 16).")
    parser.add_argument("--concept_word", type=str, required=True,
                        help="The word to compute the baseline steering vector for (e.g. Magnetism).")
    parser.add_argument("--vec_type", type=str, default="avg", choices=["avg", "last"],
                        help="Choose 'avg' (average activation) or 'last' (last token activation).")
    parser.add_argument("--save_dir", type=str, default="plots",
                        help="Directory to save the resulting scatter plot.")
    args = parser.parse_args()

    # load model
    print(f"\n⏳ Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model loaded on {device}\n")

    # ── Load probe vector dictionary ─────────────────────────────────────
    print(f"📐 Loading probe vectors from: {args.probe_path}")
    probe_dict = torch.load(args.probe_path, map_location="cpu")
    print(f"   Loaded probe dictionary keys: {list(probe_dict.keys())}")

    # extract concept vector from baseline by layer
    print(f"🛠  Extracting baseline-corrected vectors for '{args.concept_word}'...")
    baseline_vectors_all_layers = extract_control_from_baseline(
        model,
        tokenizer,
        args.dataset_name,
        args.min_layer_to_save,
        args.concept_word
    )
    
    layers_to_compare = sorted(set(probe_dict.keys()) & set(baseline_vectors_all_layers.keys()))
    print(f"📊 Comparing across {len(layers_to_compare)} layers: {layers_to_compare}")

    cos_sims = []
    euclidean_dists = []
    labels = []

    for layer in layers_to_compare:
        probe_v = probe_dict[layer].detach().cpu().float().numpy().reshape(1, -1)
        baseline_v = baseline_vectors_all_layers[layer][args.vec_type].detach().cpu().float().numpy().reshape(1, -1)

        # compute cosine sim
        cos_sim = float(cosine_similarity(probe_v, baseline_v)[0, 0])
        
        # euclidean Distance
        dist = float(np.linalg.norm(probe_v - baseline_v))
        
        cos_sims.append(cos_sim)
        euclidean_dists.append(dist)
        labels.append(f"L{layer}")
        
        print(f"  Layer {layer:>2}: Cosine Sim = {cos_sim:.4f} | Euclidean Dist = {dist:.4f}")

    # ── Plotting ──────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    plt.scatter(euclidean_dists, cos_sims, color='blue', alpha=0.6, s=100)

    # Label points
    for i, label in enumerate(labels):
        plt.annotate(label, (euclidean_dists[i], cos_sims[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    plt.title(f"Comparison: Trained Probe vs Baseline ({args.concept_word})")
    plt.xlabel("Euclidean Distance (L2 norm)")
    plt.ylabel("Cosine Similarity")
    plt.gca().invert_xaxis()  # Low distance on the right
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    file_name = f"comparison_{args.concept_word}_{args.vec_type}.png"
    plt.savefig(save_path / file_name)
    print(f"\n🎨 Plot saved to: {save_path / file_name}")
    
    plt.show()

if __name__ == "__main__":
    main()