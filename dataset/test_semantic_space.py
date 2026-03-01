import sys
import argparse
from pathlib import Path
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

# Need to import from original_paper
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from original_paper.compute_concept_vector_utils import compute_vector_single_prompt, get_data


def extract_concept_vectors(model, tokenizer, dataset_name, layer_idx):
    """
    Extract baseline vectors and concept vectors, and return the raw difference vectors (word - baseline_mean).
    Returns lists of numpy arrays.
    """
    data = get_data(dataset_name)
    concept_words = data["concept_vector_words"]
    
    # We use all words for baseline if expanded dataset, else top 50
    if "expanded" not in dataset_name:
        baseline_words = data["baseline_words"][:50]
    else:
        baseline_words = data["baseline_words"]

    # 1. Compute all baseline vectors
    print(f"\nComputing vectors for {len(baseline_words)} baseline words...")
    baseline_vecs_last = []
    baseline_vecs_avg = []
    
    for word in tqdm(baseline_words, desc="Baseline"):
        vec_last, vec_avg = compute_vector_single_prompt(model, tokenizer, dataset_name, word, layer_idx)
        baseline_vecs_last.append(vec_last.squeeze())
        baseline_vecs_avg.append(vec_avg.squeeze())
        
    baseline_tensor_last = torch.stack(baseline_vecs_last, dim=0) # [num_baselines, hidden_dim]
    baseline_tensor_avg = torch.stack(baseline_vecs_avg, dim=0)
    
    # The true "concept vector" representing the baseline is just the baseline itself minus the mean of baselines
    baseline_mean_last = baseline_tensor_last.mean(dim=0)
    baseline_mean_avg = baseline_tensor_avg.mean(dim=0)
    
    # Centered baselines - convert to float before numpy to avoid BFloat16 errors
    centered_baseline_last = (baseline_tensor_last - baseline_mean_last).float().numpy()
    centered_baseline_avg = (baseline_tensor_avg - baseline_mean_avg).float().numpy()
    
    # 2. Compute all concept vectors
    print(f"\nComputing vectors for {len(concept_words)} concept words...")
    concept_vecs_last = []
    concept_vecs_avg = []
    
    for word in tqdm(concept_words, desc="Concepts"):
        vec_last, vec_avg = compute_vector_single_prompt(model, tokenizer, dataset_name, word, layer_idx)
        
        # A "concept vector" is (word_act - baseline_mean)
        cv_last = vec_last.squeeze() - baseline_mean_last
        cv_avg = vec_avg.squeeze() - baseline_mean_avg
        
        concept_vecs_last.append(cv_last)
        concept_vecs_avg.append(cv_avg)
        
    concept_tensor_last = torch.stack(concept_vecs_last, dim=0).float().numpy()
    concept_tensor_avg = torch.stack(concept_vecs_avg, dim=0).float().numpy()
    
    return {
        "baseline_last": centered_baseline_last,
        "baseline_avg": centered_baseline_avg,
        "concept_last": concept_tensor_last,
        "concept_avg": concept_tensor_avg
    }


def analyze_pca_variance(label, data_matrix):
    """
    Fit PCA on the data matrix and strictly print the % of variance explained by PC1.
    """
    pca = PCA()
    pca.fit(data_matrix)
    
    explained_variance_ratios = pca.explained_variance_ratio_
    pc1_var = explained_variance_ratios[0] * 100
    
    print(f"  {label:25s} | Extracted PCA shape: {pca.components_.shape} | PC1 Explained Variance: {pc1_var:0.2f}%")
    return pc1_var


def main():
    parser = argparse.ArgumentParser(description="Test Semantic Space PCA Variance")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--datasets", type=str, nargs="+", default=["simple_data_expanded"])
    parser.add_argument("--layer", type=int, default=19, help="Layer index to compute representations at")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")    
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    for dataset in args.datasets:
        print(f"\nExtracting representations at Layer {args.layer} for dataset: {dataset}")
        vectors = extract_concept_vectors(model, tokenizer, dataset, args.layer)
        
        print(f"\n{'='*60}")
        print(f"PCA Variance Analysis (Layer {args.layer} | Dataset: {dataset})")
        print(f"{'='*60}")
        
        print("\n--- Based on Prompt Average pooling ---")
        analyze_pca_variance("Baseline Words (Centered)", vectors["baseline_avg"])
        analyze_pca_variance("Concept Words", vectors["concept_avg"])
        
        print("\n--- Based on Prompt Last Token pooling ---")
        analyze_pca_variance("Baseline Words (Centered)", vectors["baseline_last"])
        analyze_pca_variance("Concept Words", vectors["concept_last"])
        
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
