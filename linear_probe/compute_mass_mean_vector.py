"""
Compute a Mass-Mean direction vector from injected_correct vs not_detected activations.

This is a simpler alternative to the SVM probe: just take the difference of
class centroids (mean_detected - mean_not_detected) and normalize.

Usage:
    python compute_mass_mean_vector.py --layer 24
    python compute_mass_mean_vector.py --layer 19 --injected-dir injected_correct --clean-dir not_detected
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_class_vectors(root_dir, layer, coeff):
    """Load all activation vectors for a given layer from all concept subdirs."""
    vectors = []
    concepts = []
    for concept_dir in sorted(d for d in root_dir.iterdir() if d.is_dir()):
        concept = concept_dir.name
        files = list(concept_dir.glob(f"*_layer{layer}_coeff{coeff}_*.pt"))
        if not files:
            files = list(concept_dir.glob(f"*_layer{layer}_noinject_*.pt"))
        for f in files:
            try:
                vec = torch.load(f, map_location="cpu")['activations']['last_token']
                vectors.append(vec.float())
                concepts.append(concept)
            except Exception:
                continue
    return vectors, concepts


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Compute a Mass-Mean direction vector.")
    parser.add_argument("--layer", type=int, default=24, help="Readout layer (default: 24)")
    parser.add_argument("--coeff", type=float, default=8.0, help="Injection coefficient (default: 8.0)")
    parser.add_argument("--injected-dir", type=str, default=None,
                        help="Positive class dir (default: <script_dir>/injected_correct)")
    parser.add_argument("--clean-dir", type=str, default=None,
                        help="Negative class dir (default: <script_dir>/not_detected)")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="Model for logit lens analysis")
    parser.add_argument("--skip_logit_lens", action="store_true",
                        help="Skip the logit lens analysis (no model loading)")
    args = parser.parse_args()

    layer = args.layer
    coeff = args.coeff
    injected_dir = Path(args.injected_dir) if args.injected_dir else script_dir / "injected_correct"
    clean_dir = Path(args.clean_dir) if args.clean_dir else script_dir / "not_detected"

    print(f"Loading data for Layer {layer}...")
    print(f"  Positive (detected):     {injected_dir}")
    print(f"  Negative (not detected): {clean_dir}")

    # Load each class independently (concepts are mutually exclusive)
    pos_vecs, pos_concepts = load_class_vectors(injected_dir, layer, coeff)
    neg_vecs, neg_concepts = load_class_vectors(clean_dir, layer, coeff)

    if not pos_vecs or not neg_vecs:
        print(f"ERROR: No data found! Positive: {len(pos_vecs)}, Negative: {len(neg_vecs)}")
        return

    pos_tensor = torch.stack(pos_vecs)  # [N_pos, hidden_size]
    neg_tensor = torch.stack(neg_vecs)  # [N_neg, hidden_size]

    print(f"  Detected:     {pos_tensor.shape[0]} vectors from {len(set(pos_concepts))} concepts")
    print(f"  Not detected: {neg_tensor.shape[0]} vectors from {len(set(neg_concepts))} concepts")

    # ---------------------------------------------------------
    # The Math: Calculate the Mass-Mean Vector
    # ---------------------------------------------------------
    print("\nCalculating Mass-Mean Vector...")

    mean_detected = pos_tensor.mean(dim=0)
    mean_not_detected = neg_tensor.mean(dim=0)

    # Draw the arrow from not_detected → detected
    mass_mean_vector = mean_detected - mean_not_detected

    # Normalize to unit length
    mass_mean_normalized = mass_mean_vector / torch.norm(mass_mean_vector)

    # Print stats
    raw_norm = torch.norm(mass_mean_vector, p=2).item()
    nonzero = torch.count_nonzero(mass_mean_normalized).item()
    total_dims = mass_mean_normalized.shape[0]
    rank = int(torch.linalg.matrix_rank(mass_mean_normalized.unsqueeze(0)).item())
    print(f"\n--- Mass-Mean Vector Stats ---")
    print(f"  Raw norm (L2):   {raw_norm:.6f}")
    print(f"  Dimensions:      {total_dims}")
    print(f"  Non-zero dims:   {nonzero} / {total_dims}")
    print(f"  Matrix rank:     {rank}")

    # ---------------------------------------------------------
    # Save the Vector
    # ---------------------------------------------------------
    out_dir = script_dir / "probe_vectors"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mass_mean_vector_layer{layer}.pt"
    torch.save(mass_mean_normalized, out_path)
    print(f"\nSUCCESS: Saved Mass-Mean Vector to {out_path}")

    # ---------------------------------------------------------
    # Projection & Plotting
    # ---------------------------------------------------------
    # Project all data onto the mass-mean direction
    X_all = torch.cat([pos_tensor, neg_tensor], dim=0)
    y_all = torch.cat([torch.ones(len(pos_tensor)), torch.zeros(len(neg_tensor))])

    # Score = dot product with the normalized direction
    projection_scores = (X_all @ mass_mean_normalized).numpy()
    y_np = y_all.numpy()

    # Separation metric
    pos_scores = projection_scores[y_np == 1]
    neg_scores = projection_scores[y_np == 0]
    separation = pos_scores.mean() - neg_scores.mean()
    print(f"\n  Mean score (detected):     {pos_scores.mean():.4f} ± {pos_scores.std():.4f}")
    print(f"  Mean score (not detected): {neg_scores.mean():.4f} ± {neg_scores.std():.4f}")
    print(f"  Separation:                {separation:.4f}")

    plt.figure(figsize=(12, 4))

    jitter = np.random.uniform(-0.1, 0.1, size=len(y_np))

    plt.scatter(neg_scores, jitter[y_np == 0],
                color='blue', label=f'Not Detected (n={len(neg_scores)})',
                alpha=0.7, edgecolors='w', s=80)
    plt.scatter(pos_scores, jitter[y_np == 1],
                color='red', label=f'Detected (n={len(pos_scores)})',
                alpha=0.7, edgecolors='w', s=80, marker='^')

    # Decision boundary at the midpoint of the two class means
    midpoint = (pos_scores.mean() + neg_scores.mean()) / 2
    plt.axvline(x=midpoint, color='black', linestyle='--', linewidth=2, label='Midpoint')

    plt.title(f"1D Projection onto Mass-Mean Vector (Layer {layer})")
    plt.xlabel("Score (dot product with mass-mean direction)")
    plt.yticks([])
    plt.legend(loc='upper right')
    plt.grid(True, axis='x', alpha=0.3)

    max_abs = max(abs(projection_scores.min()), abs(projection_scores.max())) * 1.2
    plt.xlim(-max_abs, max_abs)

    plot_dir = script_dir.parent / "plots" / "linear_probe"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"mass_mean_separation_layer{layer}.png"
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")

    # ---------------------------------------------------------
    # Logit Lens: What does this direction mean in English?
    # ---------------------------------------------------------
    if args.skip_logit_lens:
        print("\nSkipping logit lens analysis (--skip_logit_lens).")
        return

    print(f"\n{'='*50}")
    print(f"  Logit Lens Analysis")
    print(f"{'='*50}")

    print(f"\n⏳ Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"✅ Model loaded\n")

    # Load the saved vector
    vector = torch.load(out_path, map_location=model.device).to(model.dtype)

    # Grab the unembedding matrix (W_U)
    W_U = model.lm_head.weight

    # Apply LayerNorm so the vector matches the scale W_U expects
    ln = model.model.norm
    normalized_vector = ln(vector.unsqueeze(0)).squeeze(0)

    # Project onto every token in the vocabulary
    logits = torch.matmul(W_U, normalized_vector)

    # Top 15 tokens
    top_logits, top_indices = torch.topk(logits, 15)
    top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]

    print("  Top 15 tokens by logit boost:")
    for i, (tok, logit) in enumerate(zip(top_tokens, top_logits.tolist())):
        print(f"    {i+1:2d}. {tok!r:20s}  {logit:.4f}")

    # Bottom 15 (most suppressed)
    bot_logits, bot_indices = torch.topk(logits, 15, largest=False)
    bot_tokens = [tokenizer.decode([idx.item()]) for idx in bot_indices]

    print("\n  Bottom 15 tokens (most suppressed):")
    for i, (tok, logit) in enumerate(zip(bot_tokens, bot_logits.tolist())):
        print(f"    {i+1:2d}. {tok!r:20s}  {logit:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(top_tokens[::-1], top_logits.detach().cpu().tolist()[::-1], color='purple')
    plt.title(f"Direct Logit Attribution: Mass-Mean Vector (Layer {layer})")
    plt.xlabel("Logit Boost Score")
    plt.tight_layout()

    logit_plot_path = plot_dir / f"logit_lens_layer{layer}.png"
    plt.savefig(logit_plot_path, bbox_inches='tight')
    print(f"\nLogit lens plot saved to {logit_plot_path}")
if __name__ == "__main__":
    main()