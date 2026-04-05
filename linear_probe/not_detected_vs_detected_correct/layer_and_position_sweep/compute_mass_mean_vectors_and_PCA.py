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

def plot_discriminability_scatter(results, save_path, top_right_only=False):
    """
    Creates a scatter plot comparing primary vs validation discriminability scores.
    Styled after test_probe_dir_to_ground_truth.py.
    """
    mm_scores = results["mass_mean_d_scores"]
    mm_val_scores = results["mass_mean_val_d_scores"]
    pca_scores = results["pca_d_scores"]
    pca_val_scores = results["pca_val_d_scores"]
    
    plt.figure(figsize=(14, 10))
    
    # 1. Plot Mass-Mean Vectors (Blue circles)
    mm_x = []
    mm_y = []
    mm_labels = []
    for (layer, pos), score in mm_scores.items():
        mm_x.append(score)
        val_score = mm_val_scores[(layer, pos)]
        mm_y.append(val_score)
        mm_labels.append(f"L{layer}")
        
    plt.scatter(mm_x, mm_y, c='blue', alpha=0.8, s=150, label='Mass-Mean Vector', marker='o', edgecolors='black')

    # 2. Plot PCA Components (Red diamonds)
    pca_x = []
    pca_y = []
    pca_labels = []
    for (layer, pos), scores in pca_scores.items():
        v_scores = pca_val_scores[(layer, pos)]
        for i, (s, vs) in enumerate(zip(scores, v_scores)):
            pca_x.append(s)
            pca_y.append(vs)
            if i == 0: # Only annotate top component or high scores
                pca_labels.append((s, vs, f"P{i}L{layer}"))

    plt.scatter(pca_x, pca_y, c='red', alpha=0.6, s=100, label='PCA Components', marker='D', edgecolors='grey')

    # 3. Annotations (Compact styling)
    # Mass-Mean Labels (Always plot MMV or significant ones)
    for x, y, label in zip(mm_x, mm_y, mm_labels):
        if x > 1.2 or y > 1.2:
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), 
                         ha='center', fontsize=8)
            
    # PCA Labels (Only significant ones)
    for x, y, label in pca_labels:
        if x > 1.0 or y > 1.0:
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), 
                         ha='center', fontsize=8)

    if top_right_only:
        # Calculate max values for axis scaling
        all_x = mm_x + pca_x
        all_y = mm_y + pca_y
        max_val = max(all_x + all_y)
        
        # Focus on the highly discriminative cluster
        plt.xlim(1.3, 1.71)
        plt.ylim(1.0, 1.4)
        plt.title("Discriminability Comparison: Top-Right Cluster (High Separation)")
    else:
        plt.title("Discriminability Comparison: Target vs Orthogonal Concepts")

    plt.xlabel("Primary Discriminability (Cohen's d: Correct vs Not-Detected)")
    plt.ylabel("Validation Discriminability (Cohen's d: Orthogonal vs Not-Detected)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📊 Styled discriminability scatter plot saved to: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compute Mass-Mean vectors and PCA on saved activations.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the saved_activations run directory.")
    parser.add_argument("--n_components", type=int, default=10, help="Number of PCA components to compute per pair.")
    parser.add_argument("--position", type=int, default=None, help="Optional: specific position to analyze across all layers.")
    parser.add_argument("--top_right_only", action="store_true", help="Plot only the top-right cluster (highly discriminative directions).")
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
    # positive = torch.cat([pos_correct, pos_parallel], dim=0)
    positive = pos_correct

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
    candidates = [] # Top vector candidates
    
    layers = category_tensors["not_detected"]["layers"]
    positions = category_tensors["not_detected"]["positions"]
    
    # Optional: Filter by position
    if args.position is not None:
        if args.position not in positions:
            print(f"⚠️  Position {args.position} not found in available positions: {positions}")
            return
        positions = [args.position]
        print(f"📍 Filtering analysis to position: {args.position}")
    
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
            mm_val_d = compute_cohens_d(val_mm_projs, neg_mm_projs)
            mass_mean_val_d_scores[(layer, position)] = mm_val_d
            
            # Record Candidate
            mm_score = np.sqrt(mass_mean_d_scores[(layer, position)]**2 + mm_val_d**2)
            candidates.append({
                'type': 'MM',
                'layer': layer,
                'pos': position,
                'vector': mm_vec,
                'score': mm_score,
                'd_prim': mass_mean_d_scores[(layer, position)],
                'd_val': mm_val_d
            })
            
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
                val_d = compute_cohens_d(v_projs, n_projs)
                p_val_d_scores.append(val_d)
                
                # Record Candidate
                p_score = np.sqrt(p_d_scores[-1]**2 + val_d**2)
                candidates.append({
                    'type': f'PCA{comp_idx}',
                    'layer': layer,
                    'pos': position,
                    'vector': component,
                    'score': p_score,
                    'd_prim': p_d_scores[-1],
                    'd_val': val_d
                })
                    
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
    
    pos_suffix = f"_pos{args.position}" if args.position is not None else ""
    top_right_suffix = "_top_right" if args.top_right_only else ""
    plot_name = save_dir / f"{run_id}_discriminability_comparison_PCA{pos_suffix}{top_right_suffix}.png"
    plot_discriminability_scatter(results, plot_name, top_right_only=args.top_right_only)
    
    # 8. Save Top 5 Vectors
    print(f"\n🏅 Saving Top 5 most discriminative vectors...")
    candidates.sort(key=lambda x: x['score'], reverse=True)
    top_5 = candidates[:5]
    
    probe_vectors_dir = Path(__file__).parent / "probe_vectors"
    os.makedirs(probe_vectors_dir, exist_ok=True)
    
    for i, item in enumerate(top_5):
        # Precise naming with scores
        filename = f"{item['type']}_L{item['layer']}_P{item['pos']}_dPrim{item['d_prim']:.2f}_dVal{item['d_val']:.2f}.pt"
        vector_tensor = torch.tensor(item['vector'], dtype=torch.float32)
        torch.save(vector_tensor, probe_vectors_dir / filename)
        print(f"  ⭐ Saved {item['type']} (Rank {i+1}, Score {item['score']:.2f}) -> {filename}")
        
    return results
    

if __name__ == "__main__":
    main()
