"""
Validate an input projection direction (e.g., PCA component or mass-mean vector)
against validation classes (pos_correct, neg_correct, not_detected, etc.).

Usage:
    python validate_vector.py --input_vector path/to/vector.pt --run-dir path/to/activations
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


def load_class_vectors(root_dir: Path, layer: int, token_type: str = "last_token", allowed_prompts: list[int] = None):
    """Load activation vectors for a given layer from all .pt files in root_dir."""
    vectors = []
    concepts = []
    prompt_ids = []

    if not root_dir.exists():
        return vectors, concepts, prompt_ids

    files = sorted(root_dir.glob("*.pt"))

    for f in files:
        try:
            data = torch.load(f, map_location="cpu", weights_only=False)
            concept = data.get("concept", f.name.split("_")[0])
            pid = data.get("prompt_id", -1)

            if allowed_prompts is not None and pid not in allowed_prompts:
                continue

            acts = data["activations"]

            if layer in acts:
                vec = acts[layer].get(token_type)
                if vec is None:
                    continue
            elif token_type in acts:
                vec = acts[token_type]
            else:
                continue

            vectors.append(vec.float())
            concepts.append(concept)
            prompt_ids.append(pid)
        except Exception as e:
            continue

    return vectors, concepts, prompt_ids


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Validate an input projection vector")
    parser.add_argument("--input_vector", type=str, required=True,
                        help="Path to the input vector .pt file. If PCA, expects a tensor or dict with 'components'. "
                             "We will extract the required vector.")
    parser.add_argument("--pca_index", type=int, default=0,
                        help="If input vector is a PCA components matrix, which index to use (default: 0).")
    parser.add_argument("--layer", type=int, default=24,
                        help="Readout layer (default: 24)")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Path to the prompt_activations run directory (default: latest in saved_activations/)")
    parser.add_argument("--token_type", type=str, default="last_token",
                        choices=["last_token", "prompt_last_token"],
                        help="Which token position to read (default: last_token)")
    parser.add_argument("--hide_intermediate", action="store_true",
                        help="Only plot the two main classes")
    parser.add_argument("--prompts", type=int, nargs="+", default=None,
                        help="List of prompt IDs to include. If not set, all are included.")
    args = parser.parse_args()

    layer = args.layer
    token_type = args.token_type
    allowed_prompts = args.prompts

    # ── Resolve input vector ─────────────────────────────────────────────
    input_path = Path(args.input_vector)
    if not input_path.exists():
        print(f"ERROR: Input vector file not found at {input_path}")
        return

    vector_data = torch.load(input_path, map_location="cpu", weights_only=False)
    
    # Extract the actual direction vector to validate against
    if isinstance(vector_data, dict) and "components" in vector_data:
        # It's a PCA dict
        projection_vector = vector_data["components"][args.pca_index].float()
        print(f"Loaded PCA component index {args.pca_index} from: {input_path}")
    elif isinstance(vector_data, torch.Tensor):
        if vector_data.dim() == 2:
            # Maybe it's a matrix of components, take the indexed one
            projection_vector = vector_data[args.pca_index].float()
        else:
            projection_vector = vector_data.float()
        print(f"Loaded tensor vector from: {input_path}")
    else:
        print("ERROR: Unknown format for input vector.")
        return

    # Normalize the projection vector
    projection_vector = projection_vector / torch.norm(projection_vector)

    # ── Resolve run directory ────────────────────────────────────────────
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        sa_dir = script_dir / "saved_activations"
        candidates = sorted(sa_dir.glob("run_*"))
        if not candidates:
            print("ERROR: No run_* directories found.")
            return
        run_dir = candidates[-1]

    print(f"\nRun directory: {run_dir}")
    print(f"Layer: {layer}  |  Token type: {token_type}")

    # ── Load class vectors ───────────────────────────────────────────────
    correct_dir = run_dir / "detected_correct"
    orthogonal_dir = run_dir / "detected_orthogonal"
    parallel_dir = run_dir / "detected_parallel"
    opposite_dir = run_dir / "detected_opposite"
    not_detected_dir = run_dir / "not_detected"
    incoherent_dir = run_dir / "incoherent"

    pos_vecs, _, _ = load_class_vectors(correct_dir, layer, token_type, allowed_prompts)
    neg_vecs, _, _ = load_class_vectors(orthogonal_dir, layer, token_type, allowed_prompts)
    opp_vecs, _, _ = load_class_vectors(opposite_dir, layer, token_type, allowed_prompts)
    ortho_vecs, _, _ = load_class_vectors(orthogonal_dir, layer, token_type, allowed_prompts)
    para_vecs, _, _ = load_class_vectors(parallel_dir, layer, token_type, allowed_prompts)
    notdet_vecs, _, _ = load_class_vectors(not_detected_dir, layer, token_type, allowed_prompts)
    incoh_vecs, _, _ = load_class_vectors(incoherent_dir, layer, token_type, allowed_prompts)

    if not pos_vecs or not neg_vecs:
        print(f"ERROR: Missing primary class data! Correct: {len(pos_vecs)}, Orthogonal: {len(neg_vecs)}")
        return

    pos_tensor = torch.stack(pos_vecs)
    neg_tensor = torch.stack(neg_vecs)
    opp_tensor = torch.stack(opp_vecs) if opp_vecs else torch.empty((0, pos_tensor.shape[1]))
    ortho_tensor = torch.stack(ortho_vecs) if ortho_vecs else torch.empty((0, pos_tensor.shape[1]))
    para_tensor = torch.stack(para_vecs) if para_vecs else torch.empty((0, pos_tensor.shape[1]))
    notdet_tensor = torch.stack(notdet_vecs) if notdet_vecs else torch.empty((0, pos_tensor.shape[1]))
    incoh_tensor = torch.stack(incoh_vecs) if incoh_vecs else torch.empty((0, pos_tensor.shape[1]))

    # ---------------------------------------------------------
    # Projection & Plotting
    # ---------------------------------------------------------
    print("\nComputing projections onto the input vector...")
    
    X_all = torch.cat([pos_tensor, neg_tensor], dim=0)
    y_all = torch.cat([torch.ones(len(pos_tensor)), torch.zeros(len(neg_tensor))])

    projection_scores = (X_all @ projection_vector).numpy()
    y_np = y_all.numpy()

    pos_scores = projection_scores[y_np == 1]
    neg_scores = projection_scores[y_np == 0]

    opp_scores = (opp_tensor @ projection_vector).numpy() if len(opp_tensor) > 0 else np.array([])
    ortho_scores = (ortho_tensor @ projection_vector).numpy() if len(ortho_tensor) > 0 else np.array([])
    para_scores = (para_tensor @ projection_vector).numpy() if len(para_tensor) > 0 else np.array([])
    notdet_scores = (notdet_tensor @ projection_vector).numpy() if len(notdet_tensor) > 0 else np.array([])
    incoh_scores = (incoh_tensor @ projection_vector).numpy() if len(incoh_tensor) > 0 else np.array([])

    # Separation metric
    separation = pos_scores.mean() - neg_scores.mean()
    print(f"\n  Mean score (Correct):                  {pos_scores.mean():.4f} ± {pos_scores.std():.4f}")
    if len(para_scores) > 0:   print(f"  Mean score (Parallel):                 {para_scores.mean():.4f} ± {para_scores.std():.4f}")
    if len(ortho_scores) > 0:  print(f"  Mean score (Orthogonal):               {ortho_scores.mean():.4f} ± {ortho_scores.std():.4f}")
    if len(opp_scores) > 0:    print(f"  Mean score (Opposite):                 {opp_scores.mean():.4f} ± {opp_scores.std():.4f}")
    if len(notdet_scores) > 0: print(f"  Mean score (Not Detected):             {notdet_scores.mean():.4f} ± {notdet_scores.std():.4f}")
    if len(incoh_scores) > 0:  print(f"  Mean score (Incoherent):               {incoh_scores.mean():.4f} ± {incoh_scores.std():.4f}")
    print(f"  Separation (Correct vs Orthogonal):    {separation:.4f}")

    plt.figure(figsize=(12, 4))
    s = 20

    jitter = np.random.uniform(-0.1, 0.1, size=len(y_np))

    plt.scatter(neg_scores, jitter[y_np == 0],
                color='blue', label=f'Orthogonal (n={len(neg_scores)})',
                alpha=0.7, edgecolors='w', s=s)
    plt.scatter(pos_scores, jitter[y_np == 1],
                color='red', label=f'Correct (n={len(pos_scores)})',
                alpha=0.7, edgecolors='w', s=s, marker='^')

    if not args.hide_intermediate:
        if len(opp_scores) > 0:
            jitter_opp = np.random.uniform(-0.1, 0.1, size=len(opp_scores))
            plt.scatter(opp_scores, jitter_opp, color='purple',
                        label=f'Detected Opposite (n={len(opp_scores)})',
                        alpha=0.7, edgecolors='w', s=s, marker='X')

        if len(para_scores) > 0:
            jitter_para = np.random.uniform(-0.1, 0.1, size=len(para_scores))
            plt.scatter(para_scores, jitter_para, color='cyan',
                        label=f'Detected Parallel (n={len(para_scores)})',
                        alpha=0.7, edgecolors='w', s=s, marker='D')

        if len(notdet_scores) > 0:
            jitter_nd = np.random.uniform(-0.1, 0.1, size=len(notdet_scores))
            plt.scatter(notdet_scores, jitter_nd, color='gray',
                        label=f'Not Detected (n={len(notdet_scores)})',
                        alpha=0.7, edgecolors='w', s=s, marker='s')

        if len(incoh_scores) > 0:
            jitter_ic = np.random.uniform(-0.1, 0.1, size=len(incoh_scores))
            plt.scatter(incoh_scores, jitter_ic, color='black',
                        label=f'Incoherent (n={len(incoh_scores)})',
                        alpha=0.5, edgecolors='w', s=s, marker='v')

    midpoint = (pos_scores.mean() + neg_scores.mean()) / 2
    plt.axvline(x=midpoint, color='black', linestyle='--', linewidth=2, label='Midpoint')

    vec_name = input_path.stem
    plt.title(f"1D Projection onto Input Vector: {vec_name} {args.pca_index + 1} (Layer {layer}, {token_type})")
    plt.xlabel("Score (dot product with input direction)")
    plt.yticks([])
    plt.legend(loc='upper left', fontsize=7)
    plt.grid(True, axis='x', alpha=0.3)

    all_scores = np.concatenate([s for s in [pos_scores, neg_scores, opp_scores, ortho_scores, para_scores, notdet_scores, incoh_scores] if len(s) > 0])
    max_abs = max(abs(all_scores.min()), abs(all_scores.max())) * 1.2
    plt.xlim(-max_abs, max_abs)

    plot_dir = script_dir.parent.parent / "plots" / "linear_probe" / "specificity"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"vector_validation_{vec_name}_layer{layer}_{token_type}.png"
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()