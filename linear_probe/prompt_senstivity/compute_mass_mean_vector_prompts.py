"""
Compute a Mass-Mean direction vector from prompt-sensitivity activations.

The positive class is positive/detected_correct (model correctly identified
the injected concept) and the negative class is negative/detected_correct
(model parroted a concept that was only in the text, not injected).

The direction vector points from the negative centroid toward the positive
centroid, capturing "what is different about *real* injection detection."

Usage:
    python compute_mass_mean_vector_prompts.py --layer 24
    python compute_mass_mean_vector_prompts.py --layer 19 --token_type prompt_last_token
    python compute_mass_mean_vector_prompts.py --layer 19 --run-dir saved_activations/prompt_activations_03_04_26_19_14
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_class_vectors(root_dir: Path, layer: int, token_type: str = "last_token", allowed_prompts: list[int] = None):
    """Load activation vectors for a given layer from all .pt files in root_dir.

    Args:
        root_dir: Directory containing .pt files.
        layer: Which layer's activations to extract.
        token_type: Which token position to use ("last_token" or "prompt_last_token").
        allowed_prompts: Optional list of prompt IDs to include.

    Returns:
        vectors: list of 1-D float tensors.
        concepts: list of concept name strings.
        prompt_ids: list of prompt IDs.
    """
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

            # Multi-layer dict format: acts[layer_int][token_type]
            if layer in acts:
                vec = acts[layer].get(token_type)
                if vec is None:
                    print(f"  ⚠ {f.name}: layer {layer} missing '{token_type}', skipping")
                    continue
            elif token_type in acts:
                # Legacy single-layer format
                vec = acts[token_type]
            else:
                continue

            vectors.append(vec.float())
            concepts.append(concept)
            prompt_ids.append(pid)
        except Exception as e:
            print(f"  ⚠ Failed to load {f.name}: {e}")
            continue

    return vectors, concepts, prompt_ids


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Compute a Mass-Mean direction vector from prompt-sensitivity activations."
    )
    parser.add_argument("--layer", type=int, default=24,
                        help="Readout layer (default: 24)")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Path to the prompt_activations run directory "
                             "(default: latest in saved_activations/)")
    parser.add_argument("--token_type", type=str, default="last_token",
                        choices=["last_token", "prompt_last_token"],
                        help="Which token position to read (default: last_token)")
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="Model for logit lens analysis")
    parser.add_argument("--skip_logit_lens", action="store_true",
                        help="Skip the logit lens analysis (no model loading)")
    parser.add_argument("--merge_parallel", action="store_true",
                        help="Merge detected_parallel into the positive class")
    parser.add_argument("--hide_intermediate", action="store_true",
                        help="Only plot the two main classes, hiding opposite, orthogonal, parallel")
    parser.add_argument("--prompts", type=int, nargs="+", default=None,
                        help="List of prompt IDs to include (e.g. 0 10 12). If not set, all are included.")
    args = parser.parse_args()

    layer = args.layer
    token_type = args.token_type
    allowed_prompts = args.prompts

    # ── Resolve run directory ────────────────────────────────────────────
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        # Find the latest prompt_activations_* directory
        sa_dir = script_dir / "saved_activations"
        candidates = sorted(sa_dir.glob("prompt_activations_*"))
        if not candidates:
            print("ERROR: No prompt_activations_* directories found in saved_activations/")
            return
        run_dir = candidates[-1]

    pos_root = run_dir / "positive"
    neg_root = run_dir / "negative"

    print(f"Run directory: {run_dir}")
    print(f"Layer: {layer}  |  Token type: {token_type}")
    if allowed_prompts:
        print(f"Prompts: {allowed_prompts}")

    # ── Positive class: positive/detected_correct ────────────────────────
    pos_dir = pos_root / "detected_correct"
    neg_dir = neg_root / "detected_correct"

    print(f"\n  Positive (injected + detected correct): {pos_dir}")
    print(f"  Negative (calibration + detected correct): {neg_dir}")

    pos_vecs, pos_concepts, pos_pids = load_class_vectors(pos_dir, layer, token_type, allowed_prompts)
    neg_vecs, neg_concepts, neg_pids = load_class_vectors(neg_dir, layer, token_type, allowed_prompts)

    # ── Extra classes for plotting ───────────────────────────────────────
    opp_vecs, opp_concepts, _ = load_class_vectors(pos_root / "detected_opposite", layer, token_type, allowed_prompts)
    ortho_vecs, ortho_concepts, _ = load_class_vectors(pos_root / "detected_orthogonal", layer, token_type, allowed_prompts)
    para_vecs, para_concepts, _ = load_class_vectors(pos_root / "detected_parallel", layer, token_type, allowed_prompts)
    notdet_vecs, notdet_concepts, _ = load_class_vectors(pos_root / "not_detected", layer, token_type, allowed_prompts)
    incoh_vecs, incoh_concepts, _ = load_class_vectors(pos_root / "incoherent", layer, token_type, allowed_prompts)

    if args.merge_parallel:
        print("  [Note] Merging detected_parallel into positive class.")
        pos_vecs.extend(para_vecs)
        pos_concepts.extend(para_concepts)
        para_vecs, para_concepts = [], []

    if not pos_vecs or not neg_vecs:
        print(f"ERROR: No data found! Positive: {len(pos_vecs)}, Negative: {len(neg_vecs)}")
        return

    pos_tensor = torch.stack(pos_vecs)
    neg_tensor = torch.stack(neg_vecs)

    opp_tensor = torch.stack(opp_vecs) if opp_vecs else torch.empty((0, pos_tensor.shape[1]))
    ortho_tensor = torch.stack(ortho_vecs) if ortho_vecs else torch.empty((0, pos_tensor.shape[1]))
    para_tensor = torch.stack(para_vecs) if para_vecs else torch.empty((0, pos_tensor.shape[1]))
    notdet_tensor = torch.stack(notdet_vecs) if notdet_vecs else torch.empty((0, pos_tensor.shape[1]))
    incoh_tensor = torch.stack(incoh_vecs) if incoh_vecs else torch.empty((0, pos_tensor.shape[1]))

    print(f"\n  Positive (injected correct):   {pos_tensor.shape[0]} vectors from {len(set(pos_concepts))} concepts")
    print(f"  Negative (calibration correct):{neg_tensor.shape[0]} vectors from {len(set(neg_concepts))} concepts")
    if len(opp_vecs) > 0:    print(f"  Detected opposite:            {opp_tensor.shape[0]} vectors")
    if len(ortho_vecs) > 0:  print(f"  Detected orthogonal:          {ortho_tensor.shape[0]} vectors")
    if len(para_vecs) > 0:   print(f"  Detected parallel:            {para_tensor.shape[0]} vectors")
    if len(notdet_vecs) > 0: print(f"  Not detected:                 {notdet_tensor.shape[0]} vectors")
    if len(incoh_vecs) > 0:  print(f"  Incoherent:                   {incoh_tensor.shape[0]} vectors")

    # ---------------------------------------------------------
    # The Math: Calculate the Mass-Mean Vector
    # ---------------------------------------------------------
    print("\nCalculating Mass-Mean Vector...")

    mean_pos = pos_tensor.mean(dim=0)
    mean_neg = neg_tensor.mean(dim=0)

    # Direction: negative (calibration) → positive (injected)
    mass_mean_vector = mean_pos - mean_neg

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
    out_path = out_dir / f"mass_mean_vector_layer{layer}_{token_type}.pt"
    torch.save(mass_mean_normalized, out_path)
    print(f"\nSUCCESS: Saved Mass-Mean Vector to {out_path}")

    # ---------------------------------------------------------
    # Projection & Plotting
    # ---------------------------------------------------------
    X_all = torch.cat([pos_tensor, neg_tensor], dim=0)
    y_all = torch.cat([torch.ones(len(pos_tensor)), torch.zeros(len(neg_tensor))])

    projection_scores = (X_all @ mass_mean_normalized).numpy()
    y_np = y_all.numpy()

    pos_scores = projection_scores[y_np == 1]
    neg_scores = projection_scores[y_np == 0]

    opp_scores = (opp_tensor @ mass_mean_normalized).numpy() if len(opp_tensor) > 0 else np.array([])
    ortho_scores = (ortho_tensor @ mass_mean_normalized).numpy() if len(ortho_tensor) > 0 else np.array([])
    para_scores = (para_tensor @ mass_mean_normalized).numpy() if len(para_tensor) > 0 else np.array([])
    notdet_scores = (notdet_tensor @ mass_mean_normalized).numpy() if len(notdet_tensor) > 0 else np.array([])
    incoh_scores = (incoh_tensor @ mass_mean_normalized).numpy() if len(incoh_tensor) > 0 else np.array([])

    # Separation metric
    separation = pos_scores.mean() - neg_scores.mean()
    print(f"\n  Mean score (positive / injected correct):     {pos_scores.mean():.4f} ± {pos_scores.std():.4f}")
    if len(para_scores) > 0:   print(f"  Mean score (detected parallel):               {para_scores.mean():.4f} ± {para_scores.std():.4f}")
    if len(ortho_scores) > 0:  print(f"  Mean score (detected orthogonal):             {ortho_scores.mean():.4f} ± {ortho_scores.std():.4f}")
    if len(opp_scores) > 0:    print(f"  Mean score (detected opposite):               {opp_scores.mean():.4f} ± {opp_scores.std():.4f}")
    if len(notdet_scores) > 0: print(f"  Mean score (not detected):                    {notdet_scores.mean():.4f} ± {notdet_scores.std():.4f}")
    if len(incoh_scores) > 0:  print(f"  Mean score (incoherent):                      {incoh_scores.mean():.4f} ± {incoh_scores.std():.4f}")
    print(f"  Mean score (negative / calibration correct):  {neg_scores.mean():.4f} ± {neg_scores.std():.4f}")
    print(f"  Separation (injected vs calibration):         {separation:.4f}")

    plt.figure(figsize=(12, 4))
    s = 20

    jitter = np.random.uniform(-0.1, 0.1, size=len(y_np))

    plt.scatter(neg_scores, jitter[y_np == 0],
                color='blue', label=f'Calibration Correct (n={len(neg_scores)})',
                alpha=0.7, edgecolors='w', s=s)
    plt.scatter(pos_scores, jitter[y_np == 1],
                color='red', label=f'Injected Correct (n={len(pos_scores)})',
                alpha=0.7, edgecolors='w', s=s, marker='^')

    if not args.hide_intermediate:
        if len(opp_scores) > 0:
            jitter_opp = np.random.uniform(-0.1, 0.1, size=len(opp_scores))
            plt.scatter(opp_scores, jitter_opp, color='purple',
                        label=f'Detected Opposite (n={len(opp_scores)})',
                        alpha=0.7, edgecolors='w', s=s, marker='X')

        if len(ortho_scores) > 0:
            jitter_ortho = np.random.uniform(-0.1, 0.1, size=len(ortho_scores))
            plt.scatter(ortho_scores, jitter_ortho, color='orange',
                        label=f'Detected Orthogonal (n={len(ortho_scores)})',
                        alpha=0.7, edgecolors='w', s=s, marker='o')

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

    plt.title(f"1D Projection onto Mass-Mean Vector (Layer {layer}, {token_type})")
    plt.xlabel("Score (dot product with mass-mean direction)")
    plt.yticks([])
    plt.legend(loc='upper left', fontsize=7)
    plt.grid(True, axis='x', alpha=0.3)

    all_scores = np.concatenate([s for s in [pos_scores, neg_scores, opp_scores, ortho_scores, para_scores, notdet_scores, incoh_scores] if len(s) > 0])
    max_abs = max(abs(all_scores.min()), abs(all_scores.max())) * 1.2
    plt.xlim(-max_abs, max_abs)

    plot_dir = script_dir.parent.parent / "plots" / "linear_probe" / "prompt_senstivity"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"mass_mean_separation_layer{layer}_{token_type}.png"
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

    vector = torch.load(out_path, map_location=model.device).to(model.dtype)

    W_U = model.lm_head.weight
    ln = model.model.norm
    normalized_vector = ln(vector.unsqueeze(0)).squeeze(0)

    logits = torch.matmul(W_U, normalized_vector)

    top_logits, top_indices = torch.topk(logits, 15)
    top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]

    print("  Top 15 tokens by logit boost:")
    for i, (tok, logit) in enumerate(zip(top_tokens, top_logits.tolist())):
        print(f"    {i+1:2d}. {tok!r:20s}  {logit:.4f}")

    bot_logits, bot_indices = torch.topk(logits, 15, largest=False)
    bot_tokens = [tokenizer.decode([idx.item()]) for idx in bot_indices]

    print("\n  Bottom 15 tokens (most suppressed):")
    for i, (tok, logit) in enumerate(zip(bot_tokens, bot_logits.tolist())):
        print(f"    {i+1:2d}. {tok!r:20s}  {logit:.4f}")

    plt.figure(figsize=(10, 6))
    plt.barh(top_tokens[::-1], top_logits.detach().cpu().tolist()[::-1], color='purple')
    plt.title(f"Direct Logit Attribution: Mass-Mean Vector (Layer {layer}, {token_type})")
    plt.xlabel("Logit Boost Score")
    plt.tight_layout()

    logit_plot_path = plot_dir / f"logit_lens_layer{layer}_{token_type}.png"
    plt.savefig(logit_plot_path, bbox_inches='tight')
    print(f"\nLogit lens plot saved to {logit_plot_path}")


if __name__ == "__main__":
    main()