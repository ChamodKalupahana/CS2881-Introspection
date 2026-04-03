import argparse
import sys
import torch
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from original_paper.compute_concept_vector_utils import compute_concept_vector
from original_paper.all_prompts import get_anthropic_reproduce_messages
from model_utils.injection import inject_and_steer_concept_and_probe
from model_utils.llm_judges import classify_response, CATEGORIES
from model_utils.logging import setup_logging


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Test introspection probe direction with concept injection (Control script)"
    )
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layers", type=int, nargs="+", default=[16],
                        help="Layers to inject concept vector at (default: [16])")
    parser.add_argument("--probe_layer", type=int, default=31,
                        help="Layer to apply probe direction scaling at (default: 31)")
    parser.add_argument("--coeffs", type=float, nargs="+", default=[8.0],
                        help="Injection coefficients for concept vector (default: [8.0])")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[-8.0, -5.0, -2.0, 0.0, 2.0, 5.0, 8.0, 16.0],
                        help="Probe direction scaling factors to sweep (default: [0 5 10 20])")
    parser.add_argument("--vec_type", type=str, default="avg",
                        choices=["avg", "last"])
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["simple_data"])
    parser.add_argument("--probe_path", type=str, default=None,
                        help="Path to probe vector .pt file")
    parser.add_argument("--save_dir", type=str, default=str(PROJECT_ROOT / "success_results" / "control"),
                        help="Root directory for saving run logs")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--skip_clean", action="store_true",
                        help="Skip the clean (no-injection) runs", default=False)
    parser.add_argument("--clean_once", action="store_true",
                        help="Only run clean (no-injection) for the first concept", default=True)
    args = parser.parse_args()

    # ── Create timestamped run directory & tee logging ────────────────────
    save_root, log_file = setup_logging(args.save_dir)

    # ── Load probe vector ────────────────────────────────────────────────
    if args.probe_path:
        probe_path = Path(args.probe_path)
    else:
        # Assuming probe_vectors is in linear_probe
        probe_path = PROJECT_ROOT / "linear_probe" / "probe_vectors" / f"introspection_probe_vector_layer{args.probe_layer}.pt"

    print(f"📐 Loading probe vector from: {probe_path}")
    if probe_path.exists():
        probe_vector = torch.load(probe_path, map_location="cpu")
        print(f"   probe_vector: {probe_vector}")
        print(f"   Shape: {probe_vector[16].shape}")
        probe_vector = probe_vector[16]
    else:
        print(f"⚠  Probe vector not found at {probe_path}! Proceeding with None, but steering hooks might fail.")
        probe_vector = None

    # load model
    print(f"\n⏳ Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model loaded on {device}\n")

    # ── Counters: keyed by (alpha, mode) ───────────────────────────────────
    # mode is "injected" or "clean"
    counts = {}  # {(alpha, mode): {cat: count}}
    for alpha in args.alphas:
        counts[(alpha, "injected")] = {cat: 0 for cat in CATEGORIES}
        if not args.skip_clean:
            counts[(alpha, "clean")] = {cat: 0 for cat in CATEGORIES}
    errors = 0

    icons = {
        "not_detected": "⚫",
        "detected_opposite": "🔴",
        "detected_orthogonal": "🟠",
        "detected_parallel": "🟡",
        "detected_correct": "🟢",
    }
    
    # Needs to be passed into inject_and_steer_concept_and_probe now!
    messages = get_anthropic_reproduce_messages()

    # ── Sweep ────────────────────────────────────────────────────────────
    clean_done = False
    for dataset_name in args.datasets:
        print(f"\n{'=' * 70}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'=' * 70}")

        for layer in args.layers:
            print(f"\n  ⏳ Computing concept vectors at layer {layer} …")
            vectors = compute_concept_vector(model, tokenizer, dataset_name, layer)

            for concept, (vec_last, vec_avg) in vectors.items():
                steering_vector = vec_avg if args.vec_type == "avg" else vec_last

                for coeff in args.coeffs:
                    # ── Run with injection ───────────────────────────────
                    print(f"\n{'─' * 70}")
                    print(f"  🔬 {concept} | layer={layer} | coeff={coeff}")
                    print(f"{'─' * 70}")

                    print(f"\n  ▶ INJECTED responses (concept vector applied):")
                    for alpha in args.alphas:
                        try:
                            # inject concept vector (for steering_vector in dataset)
                            # Using updated inject_and_steer_concept_and_probe that expects `messages` and returns `response, messages`
                            response, _ = inject_and_steer_concept_and_probe(
                                model, tokenizer, steering_vector, layer,
                                probe_vector, args.probe_layer,
                                messages=messages,
                                coeff=coeff, alpha=alpha,
                                max_new_tokens=args.max_new_tokens,
                                skip_inject=False,
                            )
                        except Exception as e:
                            print(f"    alpha={alpha:>6.1f} | ⚠ Error: {e}")
                            errors += 1
                            continue

                        print(f"  {concept} | alpha={alpha} response={response}")
                        category = classify_response(response, concept)
                        counts[(alpha, "injected")][category] += 1
                        icon = icons.get(category, "❓")
                        tag = f"alpha={alpha:>6.1f}"
                        print(f"    {tag} | {icon} {category}")

                    if not args.skip_clean and not (args.clean_once and clean_done):
                        # do clean run, with only probe direction injected
                        print(f"\n  ▶ CLEAN responses (no concept vector):")
                        for alpha in args.alphas:
                            try:
                                response, _ = inject_and_steer_concept_and_probe(
                                    model, tokenizer, steering_vector, layer,
                                    probe_vector, args.probe_layer,
                                    messages=messages,
                                    coeff=coeff, alpha=alpha,
                                    max_new_tokens=args.max_new_tokens,
                                    skip_inject=True,
                                )
                            except Exception as e:
                                print(f"    alpha={alpha:>6.1f} | ⚠ Error: {e}")
                                errors += 1
                                continue
                            
                            # run judges: affirmative_response, internality, classification
                            print(f"  {concept} | alpha={alpha} response={response}")
                            category = classify_response(response, concept)
                            counts[(alpha, "clean")][category] += 1
                            icon = icons.get(category, "❓")
                            tag = f"alpha={alpha:>6.1f}"
                            print(f"    {tag} | {icon} {category}")
                        clean_done = True

    # ── Summary by alpha ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  RESULTS  (layers={args.layers}, coeffs={args.coeffs}, vec_type={args.vec_type})")
    print(f"{'=' * 60}")

    modes = ("injected",) if args.skip_clean else ("injected", "clean")

    for alpha in args.alphas:
        print(f"\n  ┌─ alpha = {alpha} ─────────────────────────────────────")
        for mode in modes:
            mode_counts = counts[(alpha, mode)]
            mode_total = sum(mode_counts.values())
            label = "INJECTED" if mode == "injected" else "CLEAN"
            print(f"  │  {label}  (n={mode_total})")
            for cat in CATEGORIES:
                pct = mode_counts[cat] / mode_total * 100 if mode_total > 0 else 0
                print(f"  │    {cat:25s}: {mode_counts[cat]:3d}  ({pct:5.1f}%)")
        print(f"  └────────────────────────────────────────────────────")

    print(f"\n  errors: {errors}")
    print(f"  Saved to: {save_root.resolve()}")
    print(f"{'=' * 60}\n")

    # Close log
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()
    print(f"📝 Log saved to {save_root / 'run.log'}")

if __name__ == "__main__":
    main()