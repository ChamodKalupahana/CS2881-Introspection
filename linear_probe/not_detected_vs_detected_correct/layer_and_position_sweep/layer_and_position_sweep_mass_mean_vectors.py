import torch
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import einops
import numpy as np
import numpy as np

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

def plot_mass_mean_discriminability(results, save_path):
    """
    Creates a scatter plot comparing primary vs validation discriminability scores for mass-mean vectors.
    """
    mm_scores = results["mass_mean_d_scores"]
    mm_val_scores = results["mass_mean_val_d_scores"]
    
    plt.figure(figsize=(10, 8))
    
    # 1. Collect and Plot Mass-Mean points
    mm_x = []
    mm_y = []
    for (layer, pos), score in mm_scores.items():
        mm_x.append(score)
        mm_y.append(mm_val_scores[(layer, pos)])
        # Annotate significant points
        if score > 1.2 or mm_val_scores[(layer, pos)] > 1.2:
             plt.annotate(f"L{layer}P{pos}", (score, mm_val_scores[(layer, pos)]), 
                          fontsize=8, alpha=0.7)
    
    plt.scatter(mm_x, mm_y, color='teal', label='Mass-Mean Directions', alpha=0.7, s=60, marker='o', edgecolors='k')

    # Reference Line (y=x)
    # max_val = max(max(mm_x + mm_y + [0.5]), 1.0)
    # plt.plot([1.3, max_val], [1.3, max_val], 'k--', alpha=0.3, label='y=x (Equal Discrim.)')

    plt.xlabel("Primary Discriminability (Cohen's d: Parallel vs Not-Detected)")
    plt.ylabel("Validation Discriminability (Cohen's d: Orthogonal vs Not-Detected)")
    plt.title("Mass-Mean Vector Discriminability: Target vs Orthogonal Concepts")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📊 Mass-mean discriminability plot saved to: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compute Mass-Mean vectors on saved activations.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the saved_activations run directory.")
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
    print("🚀 Ready for further computation (Mass-Mean, etc.)")

    # 3. Convert into tensor for each category [concept, layer, position, d_model]
    category_tensors = get_category_tensors(activations)

    # Check for required categories
    required = ["detected_correct", "not_detected", "detected_orthogonal"]
    missing = [r for r in required if r not in category_tensors]
    if missing:
        print(f"⚠️  Missing required categories for analysis: {missing}. Found: {list(category_tensors.keys())}")
        return

    positive = category_tensors["detected_correct"]["tensor"]
    negative = category_tensors["not_detected"]["tensor"]
    validation = category_tensors["detected_orthogonal"]["tensor"]

    # 4. Compute concept-wise mean
    print(f"📊 Computing means...")
    print(f"  Positive shape: {positive.shape}")
    print(f"  Negative shape: {negative.shape}")
    print(f"  Validation shape: {validation.shape}")

    positive_concept_mean = einops.reduce(positive, "concept layer position d_model -> layer position d_model", "mean")
    negative_concept_mean = einops.reduce(negative, "concept layer position d_model -> layer position d_model", "mean")
    validation_concept_mean = einops.reduce(validation, "concept layer position d_model -> layer position d_model", "mean")
    
    print(f"  Result mean shape: {negative_concept_mean.shape}")
    
    # 5. Mass-mean for each layer and position
    mean_mean_vectors = {}
    mass_mean_d_scores = {}
    mass_mean_val_d_scores = {}
    
    layers = category_tensors["not_detected"]["layers"]
    positions = category_tensors["not_detected"]["positions"]
    
    diff = positive_concept_mean - negative_concept_mean
    
    print(f"🔬 Running Mass-Mean, and Cohen's d Analysis across {len(layers)} layers and {len(positions)} positions...")
    for l_idx, layer in enumerate(tqdm(layers, desc="Layers")):
        for p_idx, position in enumerate(positions):
            # 1. Distributions
            pos_dist = positive[:, l_idx, p_idx, :].detach().cpu().numpy()
            neg_dist = negative[:, l_idx, p_idx, :].detach().cpu().numpy()
            val_dist = validation[:, l_idx, p_idx, :].detach().cpu().numpy()
            
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
    
    # 6. Consolidate Results
    results = {
        "mean_mean_vectors": mean_mean_vectors,
        "mass_mean_d_scores": mass_mean_d_scores,
        "mass_mean_val_d_scores": mass_mean_val_d_scores
    }

    print(f"✅ Analysis complete for {len(mean_mean_vectors)} (layer, position) pairs.")

    # 7. Create the 2D plot
    run_id = Path(args.run_dir).name
    save_dir = PROJECT_ROOT / "plots" / "linear_probe" / "not_detected_vs_detected_correct" / "layer_and_coeff_sweep"
    os.makedirs(save_dir, exist_ok=True)
    
    plot_name = save_dir / f"{run_id}_mass_mean_discriminability.png"
    plot_mass_mean_discriminability(results, plot_name)
    
    return results
    

if __name__ == "__main__":
    main()
