import torch
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import einops
import numpy as np
from sklearn.decomposition import PCA

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Internal project imports
from model_utils.cohens_d import compute_cohens_d
import matplotlib.pyplot as plt

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
            # But the activations_dict inside already has (layer, position)
            # Extract concept name from filename
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

def plot_discriminability_scatter(results, save_path):
    """
    Creates a scatter plot comparing primary vs validation discriminability scores.
    """
    mm_scores = results["mass_mean_d_scores"]
    mm_val_scores = results["mass_mean_val_d_scores"]
    pca_scores = results["pca_d_scores"]
    pca_val_scores = results["pca_val_d_scores"]
    
    plt.figure(figsize=(14, 10))
    
    # 1. Collect and Plot Mass-Mean points
    mm_x = []
    mm_y = []
    for (layer, pos), score in mm_scores.items():
        mm_x.append(score)
        mm_y.append(mm_val_scores[(layer, pos)])
        # Optional: Annotate only if one of the scores is significant
        if score > 1.5 or mm_val_scores[(layer, pos)] > 1.5:
             plt.annotate(f"MM L{layer} P{pos}", (score, mm_val_scores[(layer, pos)]), 
                          fontsize=8, alpha=0.7)
    
    plt.scatter(mm_x, mm_y, color='teal', label='Mass-Mean', alpha=0.6, s=50, marker='o')

    # 2. Collect and Plot PCA points
    pca_x = []
    pca_y = []
    for (layer, pos), scores in pca_scores.items():
        v_scores = pca_val_scores[(layer, pos)]
        for i, (s, vs) in enumerate(zip(scores, v_scores)):
            pca_x.append(s)
            pca_y.append(vs)
            # Annotate only top PCA components or significant ones
            if i == 0 and (s > 2.0 or vs > 2.0):
                plt.annotate(f"PCA{i} L{layer} P{pos}", (s, vs), 
                             fontsize=7, alpha=0.5)

    plt.scatter(pca_x, pca_y, color='orange', label='PCA Components', alpha=0.3, s=20, marker='x')

    # Reference Line (y=x)
    max_val = max(max(mm_x + mm_y + [0.5]), max(pca_x + pca_y + [0.5]))
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='y=x (Equal Discrim.)')

    plt.xlabel("Primary Discriminability (Cohen's d: Parallel vs Not-Detected)")
    plt.ylabel("Validation Discriminability (Cohen's d: Orthogonal vs Not-Detected)")
    plt.title("Discriminability Comparison: Target vs Orthogonal Concepts")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📊 Discriminability scatter plot saved to: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compute Mass-Mean vectors and PCA on saved activations.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the saved_activations run directory.")
    parser.add_argument("--n_components", type=int, default=10, help="Number of PCA components to compute per pair.")
    args = parser.parse_args()

    # 1. Load activations
    activations = load_activations(args.run_dir)
    
    # 2. Summary of loaded data
    print("\n📊 LOAD SUMMARY")
    print(f"{'='*30}")
    for category, dict_data in activations.items():
        print(f"  {category:<20}: {len(dict_data)} vectors keys {list(dict_data.keys())[0:5]}")
    print(f"{'='*30}\n")
    
    # User can continue from here
    print("🚀 Ready for further computation (Mass-Mean, PCA, etc.)")

    # 3. Convert into tensor for each category [concept, layer, position, d_model]
    category_tensors = get_category_tensors(activations)

    # Check for required categories
    required = ["detected_correct", "detected_parallel", "not_detected", "detected_orthogonal"]
    missing = [r for r in required if r not in category_tensors]
    if missing:
        print(f"⚠️  Missing required categories for analysis: {missing}. Found: {list(category_tensors.keys())}")
        return

    pos_correct = category_tensors["detected_correct"]["tensor"]
    pos_parallel = category_tensors["detected_parallel"]["tensor"]
    negative = category_tensors["not_detected"]["tensor"]
    validation = category_tensors["detected_orthogonal"]["tensor"]

    # Merge positive categories
    positive = torch.cat([pos_correct, pos_parallel], dim=0)

    # 4. Compute concept-wise mean
    print(f"📊 Computing means...")
    print(f"  Positive shape: {positive.shape}")
    print(f"  Negative shape: {negative.shape}")
    print(f"  Validation shape: {validation.shape}")

    positive_concept_mean = einops.reduce(positive, "concept layer position d_model -> layer position d_model", "mean")
    negative_concept_mean = einops.reduce(negative, "concept layer position d_model -> layer position d_model", "mean")
    validation_concept_mean = einops.reduce(validation, "concept layer position d_model -> layer position d_model", "mean")
    
    print(f"  Result mean shape: {negative_concept_mean.shape}")
    
    # 5. Mass-mean and PCA for each layer and position
    mean_mean_vectors = {}
    pca_results = {}
    mass_mean_d_scores = {}
    pca_d_scores = {}
    mass_mean_val_d_scores = {}
    pca_val_d_scores = {}
    
    layers = category_tensors["not_detected"]["layers"]
    positions = category_tensors["not_detected"]["positions"]
    
    diff = positive_concept_mean - negative_concept_mean
    
    print(f"🔬 Running PCA, Mass-Mean, and Cohen's d Analysis across {len(layers)} layers and {len(positions)} positions...")
    for l_idx, layer in enumerate(tqdm(layers, desc="Layers")):
        for p_idx, position in enumerate(positions):
            # 1. Distributions
            pos_dist = positive[:, l_idx, p_idx, :].detach().cpu().numpy()
            print(f"shape: {pos_dist.shape}")
            neg_dist = negative[:, l_idx, p_idx, :].detach().cpu().numpy()
            val_dist = validation[:, l_idx, p_idx, :].detach().cpu().numpy()
            X = np.concatenate([pos_dist, neg_dist], axis=0)
            
            # 2. Mass-mean vector (for each layer and position)
            mm_vec = diff[l_idx, p_idx].detach().cpu().numpy()
            mean_mean_vectors[(layer, position)] = mm_vec
            
            # Mass-mean Cohen's d (Primary)
            unit_mm = mm_vec / (np.linalg.norm(mm_vec) + 1e-9)
            pos_mm_projs = pos_dist @ unit_mm
            neg_mm_projs = neg_dist @ unit_mm
            mass_mean_d_scores[(layer, position)] = compute_cohens_d(pos_mm_projs, neg_mm_projs)
            
            # Mass-mean Cohen's d (Validation)
            val_mm_projs = val_dist @ unit_mm
            mass_mean_val_d_scores[(layer, position)] = compute_cohens_d(val_mm_projs, neg_mm_projs)
            
            # 3. PCA
            pca = PCA(n_components=args.n_components)
            pca.fit(X)
            pca_results[(layer, position)] = pca
            
            # PCA components Cohen's d
            p_d_scores = []
            p_val_d_scores = []
            for comp_idx in range(args.n_components):
                component = pca.components_[comp_idx]
                unit_comp = component / (np.linalg.norm(component) + 1e-9)
                
                # Primary
                p_projs = pos_dist @ unit_comp
                n_projs = neg_dist @ unit_comp
                p_d_scores.append(compute_cohens_d(p_projs, n_projs))
                
                # Validation
                v_projs = val_dist @ unit_comp
                p_val_d_scores.append(compute_cohens_d(v_projs, n_projs))
                    
            pca_d_scores[(layer, position)] = p_d_scores
            pca_val_d_scores[(layer, position)] = p_val_d_scores
    
    # 6. Consolidate Results
    results = {
        "mean_mean_vectors": mean_mean_vectors,
        "pca_results": pca_results,
        "mass_mean_d_scores": mass_mean_d_scores,
        "pca_d_scores": pca_d_scores,
        "mass_mean_val_d_scores": mass_mean_val_d_scores,
        "pca_val_d_scores": pca_val_d_scores
    }

    print(f"✅ Analysis complete for {len(mean_mean_vectors)} (layer, position) pairs.")

    # 7. Create the 2D plot
    run_id = Path(args.run_dir).name
    save_dir = PROJECT_ROOT / "plots" / "linear_probe" / "not_detected_vs_detected_correct" / "layer_and_coeff_sweep"
    os.makedirs(save_dir, exist_ok=True)
    
    plot_name = save_dir / f"{run_id}_discriminability_comparison_PCA.png"
    plot_discriminability_scatter(results, plot_name)
    
    return results
    

if __name__ == "__main__":
    main()
