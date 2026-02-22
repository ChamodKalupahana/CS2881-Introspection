"""
Compute per-layer deltas between injected_correct_expanded (detected) and
not_detected activation vectors.

Unlike compute_delta_per_layer.py which pairs by concept, this script
computes: mean(detected_vectors) - mean(not_detected_vectors) per layer,
since the two directories have mutually exclusive concept sets.

Usage:
    python compute_delta_per_layer_not_detected.py
    python compute_delta_per_layer_not_detected.py --detected_dir injected_correct_expanded --not_detected_dir ../linear_probe/not_detected
"""

import torch
import argparse
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


def load_activations(path):
    """Load activations from .pt file."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data['activations']['last_token']


def load_all_vectors(root_dir, layer, coeff):
    """Load all activation vectors for a given layer from all concept subdirs."""
    vectors = []
    concepts = []
    for concept_dir in sorted(d for d in root_dir.iterdir() if d.is_dir()):
        concept = concept_dir.name
        # Try exact pattern first, then glob
        files = list(concept_dir.glob(f"*_layer{layer}_coeff{coeff}_*.pt"))
        if not files:
            files = list(concept_dir.glob(f"*_layer{layer}_noinject_*.pt"))
        for f in files:
            try:
                vec = load_activations(f)
                vectors.append(vec)
                concepts.append(concept)
            except Exception:
                continue
    return vectors, concepts


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-layer deltas: mean(detected) - mean(not_detected)"
    )
    parser.add_argument("--detected_dir", type=str, default="injected_correct_expanded",
                        help="Directory with detected (positive) vectors (default: injected_correct_expanded)")
    parser.add_argument("--not_detected_dir", type=str, default="not_detected",
                        help="Directory with not-detected (negative) vectors (default: not_detected)")
    parser.add_argument("--coeff", type=float, default=8.0, help="Coefficient to look for")
    parser.add_argument("--output_dir", type=str, default="PCA_components_not_detected",
                        help="Output directory for deltas (default: PCA_components_not_detected)")
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(16, 32)),
                        help="Layers to process (default: 16-31)")

    args = parser.parse_args()

    # Resolve paths relative to script directory
    script_dir = Path(__file__).resolve().parent
    detected_root = Path(args.detected_dir)
    if not detected_root.is_absolute():
        detected_root = script_dir / args.detected_dir

    not_detected_root = Path(args.not_detected_dir)
    if not not_detected_root.is_absolute():
        not_detected_root = script_dir / args.not_detected_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Detected Dir:     {detected_root}")
    print(f"Not Detected Dir: {not_detected_root}")
    print(f"Output Dir:       {output_dir}")
    print(f"Layers:           {args.layers}")

    for layer in args.layers:
        print(f"\n{'='*50}")
        print(f"  Layer {layer}")
        print(f"{'='*50}")

        # Load all vectors for each class
        det_vecs, det_concepts = load_all_vectors(detected_root, layer, args.coeff)
        nd_vecs, nd_concepts = load_all_vectors(not_detected_root, layer, args.coeff)

        if not det_vecs:
            print(f"  ⚠ No detected vectors found for layer {layer}")
            continue
        if not nd_vecs:
            print(f"  ⚠ No not_detected vectors found for layer {layer}")
            continue

        det_tensor = torch.stack(det_vecs)   # [N_det, hidden_size]
        nd_tensor = torch.stack(nd_vecs)      # [N_nd, hidden_size]

        print(f"  Detected:     {det_tensor.shape[0]} vectors from {len(set(det_concepts))} concepts")
        print(f"  Not detected: {nd_tensor.shape[0]} vectors from {len(set(nd_concepts))} concepts")

        # Compute mean delta
        mean_detected = det_tensor.mean(dim=0)
        mean_not_detected = nd_tensor.mean(dim=0)
        mean_delta = mean_detected - mean_not_detected

        print(f"  Mean delta norm: {torch.norm(mean_delta).item():.4f}")

        # Also save individual vectors for downstream PCA
        result = {
            "mean_delta": mean_delta,
            "mean_detected": mean_detected,
            "mean_not_detected": mean_not_detected,
            "detected_vectors": det_tensor,
            "not_detected_vectors": nd_tensor,
            "detected_concepts": det_concepts,
            "not_detected_concepts": nd_concepts,
            "layer": layer,
            "coeff": args.coeff,
        }

        save_path = output_dir / f"deltas_layer{layer}.pt"
        torch.save(result, save_path)
        print(f"  Saved → {save_path}")

    print(f"\n✅ Done! Results in {output_dir}")


if __name__ == "__main__":
    main()
