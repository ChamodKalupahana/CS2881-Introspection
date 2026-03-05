"""
Compute the cosine similarity between a probe direction and a "yes" direction.

The "yes" direction is computed from complex_yes_vector.json:
    yes_direction = mean(embeddings of affirmative sentences) - mean(embeddings of negative sentences)

Usage:
    python cosine_sim_against_yes_vector.py --probe_path probe_vectors/mass_mean_vector_layer19_last_token.pt
    python cosine_sim_against_yes_vector.py --probe_path probe_vectors/mass_mean_vector_layer19_last_token.pt --layer 19
"""

import argparse
import json
import torch
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from original_paper.compute_concept_vector_utils import compute_vector_single_prompt


def compute_yes_vector(model, tokenizer, data_path, layer, dataset_name="complex_data"):
    """Compute the yes/no direction vector from paired sentences.

    Returns:
        yes_vec_last: direction from last-token activations (normalized).
        yes_vec_avg:  direction from average activations (normalized).
        stats: dict with extra info.
    """
    # Load and fix JSON (handle trailing period)
    raw = data_path.read_text().strip()
    if raw.endswith("}."):
        raw = raw[:-1]
    data = json.loads(raw)

    yes_sentences = data["yes"][0]  # affirmative
    no_sentences = data["yes"][1]   # negative

    print(f"\n  Yes sentences: {len(yes_sentences)}")
    print(f"  No sentences:  {len(no_sentences)}")

    # Compute mean activations for yes sentences
    yes_vecs_last, yes_vecs_avg = [], []
    for sent in tqdm(yes_sentences, desc="  Encoding YES"):
        vec_last, vec_avg = compute_vector_single_prompt(model, tokenizer, dataset_name, sent, layer)
        yes_vecs_last.append(vec_last.squeeze())
        yes_vecs_avg.append(vec_avg.squeeze())

    # Compute mean activations for no sentences
    no_vecs_last, no_vecs_avg = [], []
    for sent in tqdm(no_sentences, desc="  Encoding NO"):
        vec_last, vec_avg = compute_vector_single_prompt(model, tokenizer, dataset_name, sent, layer)
        no_vecs_last.append(vec_last.squeeze())
        no_vecs_avg.append(vec_avg.squeeze())

    yes_mean_last = torch.stack(yes_vecs_last).mean(dim=0)
    yes_mean_avg = torch.stack(yes_vecs_avg).mean(dim=0)
    no_mean_last = torch.stack(no_vecs_last).mean(dim=0)
    no_mean_avg = torch.stack(no_vecs_avg).mean(dim=0)

    # Direction: no → yes
    yes_vec_last = yes_mean_last - no_mean_last
    yes_vec_avg = yes_mean_avg - no_mean_avg

    stats = {
        "yes_mean_last_norm": torch.norm(yes_mean_last).item(),
        "no_mean_last_norm": torch.norm(no_mean_last).item(),
        "direction_last_norm": torch.norm(yes_vec_last).item(),
        "direction_avg_norm": torch.norm(yes_vec_avg).item(),
    }

    return yes_vec_last, yes_vec_avg, stats


def cosine_sim(a, b):
    """Cosine similarity between two 1-D tensors."""
    a, b = a.float(), b.float()
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Compute cosine similarity between a probe direction and the yes/no direction."
    )
    parser.add_argument("--probe_path", type=str, required=True,
                        help="Path to the probe vector .pt file")
    parser.add_argument("--layer", type=int, default=19,
                        help="Layer to extract yes/no activations from (default: 19)")
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to complex_yes_vector.json (default: dataset/complex_yes_vector.json)")
    args = parser.parse_args()

    # ── Load probe vector ────────────────────────────────────────────────
    probe_path = Path(args.probe_path)
    print(f"📐 Loading probe vector from: {probe_path}")
    probe_vector = torch.load(probe_path, map_location="cpu").float()
    print(f"   Shape: {probe_vector.shape}")
    print(f"   Norm:  {torch.norm(probe_vector).item():.6f}")

    # ── Load model ───────────────────────────────────────────────────────
    print(f"\n⏳ Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model loaded on {device}")

    # ── Compute yes vector ───────────────────────────────────────────────
    data_path = Path(args.data_path) if args.data_path else PROJECT_ROOT / "dataset" / "complex_yes_vector.json"
    print(f"\n📊 Computing yes/no direction from: {data_path}")
    print(f"   Layer: {args.layer}")

    yes_vec_last, yes_vec_avg, stats = compute_yes_vector(
        model, tokenizer, data_path, args.layer
    )

    # ── Cast to same dtype ────────────────────────────────────────────────
    yes_vec_last = yes_vec_last.float()
    yes_vec_avg = yes_vec_avg.float()

    # ── Normalize both vectors ───────────────────────────────────────────
    probe_norm = probe_vector / torch.norm(probe_vector)
    yes_last_norm = yes_vec_last / torch.norm(yes_vec_last)
    yes_avg_norm = yes_vec_avg / torch.norm(yes_vec_avg)

    # ── Compute metrics ──────────────────────────────────────────────────
    cos_last = cosine_sim(probe_vector, yes_vec_last).item()
    cos_avg = cosine_sim(probe_vector, yes_vec_avg).item()

    # Dot product (unnormalized)
    dot_last = torch.dot(probe_vector, yes_vec_last).item()
    dot_avg = torch.dot(probe_vector, yes_vec_avg).item()

    # Projection magnitude: how much of the probe lies along the yes direction
    proj_last = torch.dot(probe_norm, yes_last_norm).item()
    proj_avg = torch.dot(probe_norm, yes_avg_norm).item()

    # Angular distance in degrees
    angle_last = np.degrees(np.arccos(np.clip(cos_last, -1, 1)))
    angle_avg = np.degrees(np.arccos(np.clip(cos_avg, -1, 1)))

    # ── Print results ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Cosine Similarity: Probe vs Yes Direction (Layer {args.layer})")
    print(f"{'=' * 60}")
    print(f"\n  Probe vector:       {probe_path.name}")
    print(f"  Probe norm:         {torch.norm(probe_vector).item():.6f}")
    print(f"  Yes direction norm: {stats['direction_last_norm']:.6f} (last-token)")
    print(f"  Yes direction norm: {stats['direction_avg_norm']:.6f} (avg)")

    print(f"\n  ┌─ Last-Token Activations ──────────────────────────────")
    print(f"  │  Cosine Similarity:  {cos_last:+.6f}")
    print(f"  │  Angular Distance:   {angle_last:.2f}°")
    print(f"  │  Dot Product:        {dot_last:+.4f}")
    print(f"  │  Projection:         {proj_last:+.6f}")
    print(f"  └────────────────────────────────────────────────────────")

    print(f"\n  ┌─ Average Activations ────────────────────────────────")
    print(f"  │  Cosine Similarity:  {cos_avg:+.6f}")
    print(f"  │  Angular Distance:   {angle_avg:.2f}°")
    print(f"  │  Dot Product:        {dot_avg:+.4f}")
    print(f"  │  Projection:         {proj_avg:+.6f}")
    print(f"  └────────────────────────────────────────────────────────")

    # ── Interpretation ───────────────────────────────────────────────────
    print(f"\n  Interpretation:")
    if abs(cos_last) > 0.5:
        direction = "aligned with" if cos_last > 0 else "anti-aligned with"
        print(f"  → The probe direction is strongly {direction} the yes/no direction (cos={cos_last:.3f})")
        print(f"    This suggests the introspection probe {'captures' if cos_last > 0 else 'is opposite to'} yes/no semantics.")
    elif abs(cos_last) > 0.1:
        direction = "weakly aligned with" if cos_last > 0 else "weakly anti-aligned with"
        print(f"  → The probe direction is {direction} the yes/no direction (cos={cos_last:.3f})")
        print(f"    There is some overlap but the probe captures more than just yes/no.")
    else:
        print(f"  → The probe direction is nearly orthogonal to the yes/no direction (cos={cos_last:.3f})")
        print(f"    The introspection probe is NOT simply a yes/no detector.")

    # ── Save the yes vector for reuse ────────────────────────────────────
    out_dir = script_dir / "probe_vectors"
    out_dir.mkdir(parents=True, exist_ok=True)
    yes_path = out_dir / f"yes_direction_layer{args.layer}_last_token.pt"
    torch.save(yes_last_norm, yes_path)
    print(f"\n  💾 Saved normalized yes direction to {yes_path}")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()