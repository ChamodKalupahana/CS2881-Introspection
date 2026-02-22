import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Train a linear probe for introspection detection.")
    parser.add_argument("--layer", type=int, default=31, help="Readout layer (default: 31)")
    parser.add_argument("--coeff", type=float, default=8.0, help="Injection coefficient (default: 8.0)")
    parser.add_argument("--injected-dir", type=str, default=None, help="Injected vectors dir (default: <script_dir>/injected_correct)")
    parser.add_argument("--clean-dir", type=str, default=None, help="Clean vectors dir (default: <script_dir>/no_inject)")
    args = parser.parse_args()

    layer = args.layer
    coeff = args.coeff
    injected_dir = Path(args.injected_dir) if args.injected_dir else script_dir / "injected_correct"
    clean_dir = Path(args.clean_dir) if args.clean_dir else script_dir / "no_inject"

    X_list, y_list = [], []

    print(f"Loading data for Layer {layer}...")

    # Load vectors
    for concept_dir in [d for d in injected_dir.iterdir() if d.is_dir()]:
        concept = concept_dir.name
        
        # Glob all runs for this concept
        inj_files = list(concept_dir.glob(f"*_layer{layer}_coeff{coeff}_*.pt"))
        cln_files = list(clean_dir.joinpath(concept).glob(f"*_layer{layer}_noinject_*.pt"))
        
        # Pair them up
        for inj_f, cln_f in zip(inj_files, cln_files):
            try:
                vec_inj = torch.load(inj_f, map_location="cpu")['activations']['last_token']
                vec_cln = torch.load(cln_f, map_location="cpu")['activations']['last_token']
                
                X_list.append(vec_inj.float().numpy())
                y_list.append(1) # INJECTED
                
                X_list.append(vec_cln.float().numpy())
                y_list.append(0) # CLEAN
            except Exception:
                continue

    X = np.stack(X_list)
    y = np.array(y_list)
    print(f"Loaded {len(X)} examples. (Positive: {sum(y)}, Negative: {len(y)-sum(y)})")

    # ---------------------------------------------------------
    # Train the Probe (Linear Support Vector Classification)
    # C=0.1 provides strong regularization to ignore the "parallel concept" noise
    # ---------------------------------------------------------
    probe = LinearSVC(C=0.1, max_iter=10000, dual=False)
    
    # Check robustness with Cross-Validation
    cv_scores = cross_val_score(probe, X, y, cv=5)
    print(f"\nProbe Cross-Validation Accuracy: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

    # Fit on all data to extract the final vector
    probe.fit(X, y)
    train_acc = probe.score(X, y)
    print(f"Final Train Accuracy: {train_acc:.1%}")

    # Extract the true "Introspection Direction"
    # This vector ignores the semantics and only looks for the "Detection Flag"
    introspection_vector = torch.tensor(probe.coef_[0], dtype=torch.float32)
    
    # Print vector statistics
    norm = torch.norm(introspection_vector, p=2).item()
    nonzero = torch.count_nonzero(introspection_vector).item()
    total_dims = introspection_vector.shape[0]
    rank = int(torch.linalg.matrix_rank(introspection_vector.unsqueeze(0)).item())
    print(f"\n--- Probe Direction Stats ---")
    print(f"  Norm (L2):       {norm:.6f}")
    print(f"  Dimensions:      {total_dims}")
    print(f"  Non-zero dims:   {nonzero} / {total_dims}")
    print(f"  Matrix rank:     {rank}")
    
    # Save it so we can use it to score attention heads next!
    probe_dir = script_dir / "probe_vectors"
    probe_dir.mkdir(parents=True, exist_ok=True)
    out_path = probe_dir / f"introspection_probe_vector_layer{layer}.pt"
    torch.save(introspection_vector, out_path)
    print(f"\nSUCCESS: Saved Introspection Vector to {out_path}")

    # ---------------------------------------------------------
    # The Math: Project the data onto the Probe's vector
    # ---------------------------------------------------------
    # probe.decision_function calculates exactly: (X dot w) + b
    # A score > 0 means the probe predicts "Injected"
    # A score < 0 means the probe predicts "Clean"
    projection_scores = probe.decision_function(X)

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 4))
    
    # Add random Y jitter to separate dots visually
    jitter = np.random.uniform(-0.1, 0.1, size=len(y))

    # Plot Clean (Class 0)
    clean_scores = projection_scores[y == 0]
    plt.scatter(clean_scores, jitter[y == 0], 
                color='blue', label='Clean (Normal Stream)', alpha=0.7, edgecolors='w', s=80)
    
    # Plot Injected (Class 1)
    injected_scores = projection_scores[y == 1]
    plt.scatter(injected_scores, jitter[y == 1], 
                color='red', label='Injected (Tampered Stream)', alpha=0.7, edgecolors='w', s=80, marker='^')

    # Draw the Decision Boundary (where Score = 0)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
    
    # Formatting
    plt.title(f"1D Projection onto the Learned Introspection Vector (Layer {layer})")
    plt.xlabel("Introspection Score \n(Projection onto Probe Direction)")
    plt.yticks([]) # Hide Y axis because it's just random jitter
    plt.legend(loc='upper right')
    plt.grid(True, axis='x', alpha=0.3)
    
    # Automatically scale the x-axis to fit the data comfortably
    max_abs_score = max(abs(projection_scores.min()), abs(projection_scores.max())) * 1.2
    plt.xlim(-max_abs_score, max_abs_score)
    
    plot_dir = script_dir.parent / "plots" / "linear_probe"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"probe_separation_layer{layer}.png"
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    main()