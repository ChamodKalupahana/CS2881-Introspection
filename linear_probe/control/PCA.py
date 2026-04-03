import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

def extract_PCA_from_activations(positive_avg: dict, negative_avg: dict, num_components: int = 20, concept_name: str = "concept"):
    """
    Performs per-layer PCA on the concatenated positive and negative distributions 
    and finds directions that maximize the difference in projected means.
    
    Args:
        positive_avg: dict mapping layer_idx -> [num_instances, hidden_dim]
        negative_avg: dict mapping layer_idx -> [num_instances, hidden_dim]
        num_components: Number of PCA components to extract per layer.
        concept_name: Name of the concept for labeling saving plots.
        
    Returns:
        all_results: dict mapping layer_idx -> {
            'pca': PCA object,
            'diffs': list of differences per component,
            'top_probes': list of (component_idx, diff_score, vector)
        }
    """
    layers = sorted(positive_avg.keys())
    all_results = {}
    
    # Grid for heatmap: Layers x Components
    heatmap_data = np.zeros((len(layers), num_components))

    print(f"🔬 Running Discriminative PCA Analysis across {len(layers)} layers...")

    for i, l in enumerate(layers):
        pos_data = positive_avg[l].detach().cpu().float().numpy()
        neg_data = negative_avg[l].detach().cpu().float().numpy()
        
        # Combine distributions for PCA
        # PCA finds directions of maximum variance in the data
        X = np.concatenate([pos_data, neg_data], axis=0)
        
        pca = PCA(n_components=num_components)
        pca.fit(X)
        
        # Means for each distribution
        p_mean = pos_data.mean(axis=0)
        n_mean = neg_data.mean(axis=0)
        
        layer_diffs = []
        top_probes_for_layer = []
        
        for comp_idx in range(num_components):
            component = pca.components_[comp_idx]
            
            # Project means onto the component (dot product)
            p_proj = np.dot(p_mean, component)
            n_proj = np.dot(n_mean, component)
            
            # Discriminability score: absolute difference in projected means
            diff = abs(p_proj - n_proj)
            layer_diffs.append(diff)
            heatmap_data[i, comp_idx] = diff
            
            # Store full vector for top probes
            top_probes_for_layer.append((comp_idx, diff, component))
            
        # Sort probes by discriminability diff
        top_probes_for_layer.sort(key=lambda x: x[1], reverse=True)
        
        all_results[l] = {
            'pca': pca,
            'diffs': layer_diffs,
            'top_probes': top_probes_for_layer[:5] # Keep top 5
        }

    # ── Heatmap Visualization ──────────────────────────────────────────────
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap_data, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Discriminability (Mean Diff)')
    plt.title(f"PCA Discriminability Heatmap - {concept_name}")
    plt.xlabel("PCA Component Index")
    plt.ylabel("Layer Index (Offset)")
    
    # Set y-ticks to actual layer labels
    plt.yticks(range(len(layers)), layers)
    
    save_dir = Path("../../plots/control")
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"pca_discriminability_{concept_name}.png")
    print(f"🎨 Heatmap saved to: {save_dir / f'pca_discriminability_{concept_name}.png'}")
    plt.close()

    return all_results
