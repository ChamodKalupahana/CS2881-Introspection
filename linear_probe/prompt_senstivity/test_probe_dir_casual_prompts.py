"""
Test the introspection probe direction by scaling activations along it.

For each concept this script:
  1. Runs inference WITH concept-vector injection  (the model should detect it)
  2. Runs inference WITHOUT injection               (baseline / clean)
  3. For each case it sweeps over probe-direction scaling alphas to see
     whether amplifying or suppressing the probe signal changes behaviour.

This mirrors PCA/save_vectors_by_layer.py but focuses on the steering
experiment rather than saving activations.

Usage:
    python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --alphas 0 5 10 20 --datasets simple_data
    python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --alphas 0 -5 -10 -15 --datasets simple_data
"""

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
from judge_all_prompts import classify_response, CATEGORIES


# Judge logic now imported from judge_all_prompts


class TeeLogger:
    """Duplicate writes to both a file and the original stream."""
    def __init__(self, log_file, stream):
        self.log_file = log_file
        self.stream = stream

    def write(self, message):
        self.stream.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def fileno(self):
        return self.stream.fileno()


# ── Steering helpers ──────────────────────────────────────────────────────────

def inject_and_steer(
    model, tokenizer, concept_vector, layer_to_inject,
    probe_vector, probe_layer,
    coeff=10.0, alpha=0.0,
    max_new_tokens=100, skip_inject=False,
):
    """
    Run inference with optional concept-vector injection AND optional
    probe-direction scaling.

    Args:
        concept_vector: The concept steering vector (injected at layer_to_inject).
        layer_to_inject: Which layer to inject the concept vector at.
        probe_vector: The learned introspection probe direction (1-D, hidden_size).
        probe_layer: Which layer to apply probe scaling at.
        coeff: Injection coefficient for the concept vector.
        alpha: Scaling factor along the probe direction.
                 alpha > 0  → amplify the "introspection signal"
                 alpha < 0  → suppress / blind the model
                 alpha == 0 → no probe steering (normal behaviour)
        skip_inject: If True, skip concept injection entirely (clean run).
    """
    device = next(model.parameters()).device

    # ── Normalised concept vector ────────────────────────────────────────
    sv = concept_vector / torch.norm(concept_vector, p=2)
    if not isinstance(sv, torch.Tensor):
        sv = torch.tensor(sv, dtype=torch.float32)
    sv = sv.to(device)
    if sv.dim() == 1:
        sv = sv.unsqueeze(0).unsqueeze(0)
    elif sv.dim() == 2:
        sv = sv.unsqueeze(0)

    # ── Normalised probe vector ──────────────────────────────────────────
    pv = probe_vector.clone().float()
    pv = pv / torch.norm(pv, p=2)
    pv = pv.to(device)

    # ── Build prompt ─────────────────────────────────────────────────────
    messages = get_anthropic_reproduce_messages()
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Find injection start position (before "\n\nTrial 1")
    trial_start_text = "\n\nTrial 1"
    trial_start_pos = formatted_prompt.find(trial_start_text)
    if trial_start_pos != -1:
        prefix = formatted_prompt[:trial_start_pos]
        injection_start_token = len(tokenizer.encode(prefix, add_special_tokens=False))
    else:
        injection_start_token = None

    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_length = inputs.input_ids.shape[1]
    prompt_processed = False

    # ── Concept injection hook (same logic as save_vectors_by_layer) ─────
    def injection_hook(module, input, output):
        nonlocal prompt_processed
        hidden_states = output[0] if isinstance(output, tuple) else output
        steer = sv.to(device=hidden_states.device, dtype=hidden_states.dtype)
        batch_size, seq_len, hidden_dim = hidden_states.shape
        steer_expanded = torch.zeros_like(hidden_states)

        if injection_start_token is not None:
            is_generating = (seq_len == 1 and prompt_processed) or (seq_len > prompt_length)
            if seq_len == prompt_length:
                prompt_processed = True
            if is_generating:
                steer_expanded[:, -1:, :] = steer
            else:
                start_idx = max(0, injection_start_token)
                if start_idx < seq_len:
                    steer_expanded[:, start_idx:, :] = steer.expand(
                        batch_size, seq_len - start_idx, -1
                    )
        else:
            if seq_len == 1:
                steer_expanded[:, :, :] = steer

        modified = hidden_states + coeff * steer_expanded
        return (modified,) + output[1:] if isinstance(output, tuple) else modified

    # ── Probe direction steering hook ────────────────────────────────────
    def probe_steering_hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        probe = pv.to(device=hidden_states.device, dtype=hidden_states.dtype)
        # Scale the last token position along the probe direction
        hidden_states[:, -1, :] = hidden_states[:, -1, :] + alpha * probe
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    # ── Register hooks & generate ────────────────────────────────────────
    handles = []
    if not skip_inject:
        handles.append(
            model.model.layers[layer_to_inject].register_forward_hook(injection_hook)
        )
    if alpha != 0.0:
        handles.append(
            model.model.layers[probe_layer].register_forward_hook(probe_steering_hook)
        )

    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated_ids = out[0][prompt_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    finally:
        for h in handles:
            h.remove()

    return response


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Test introspection probe direction with concept injection"
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
                        default=[0.0, 5.0, 10.0, 20.0],
                        help="Probe direction scaling factors to sweep (default: [0 5 10 20])")
    parser.add_argument("--vec_type", type=str, default="avg",
                        choices=["avg", "last"])
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["simple_data"])
    parser.add_argument("--probe_path", type=str, default=None,
                        help="Path to probe vector .pt file (default: probe_vectors/mass_mean_vector_layer{probe_layer}_last_token.pt)")
    parser.add_argument("--save_dir", type=str, default=str(PROJECT_ROOT / "success_results"),
                        help="Root directory for saving run logs (default: success_results/)")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--skip_clean", action="store_true",
                        help="Skip the clean (no-injection) runs")
    parser.add_argument("--clean_once", action="store_true",
                        help="Only run clean (no-injection) for the first concept")
    args = parser.parse_args()

    # ── Create timestamped run directory & tee logging ────────────────────
    now = datetime.now()
    run_name = now.strftime("run_%m_%d_%y_%H_%M")
    save_root = Path(args.save_dir) / run_name
    save_root.mkdir(parents=True, exist_ok=True)

    log_file = open(save_root / "run.log", "w")
    sys.stdout = TeeLogger(log_file, sys.__stdout__)
    sys.stderr = TeeLogger(log_file, sys.__stderr__)

    print(f"📁 Run directory: {save_root}")

    # ── Load probe vector ────────────────────────────────────────────────
    if args.probe_path:
        probe_path = Path(args.probe_path)
    else:
        probe_path = script_dir / "probe_vectors" / f"mass_mean_vector_layer{args.probe_layer}_last_token.pt"

    print(f"📐 Loading probe vector from: {probe_path}")
    probe_vector = torch.load(probe_path, map_location="cpu")
    print(f"   Shape: {probe_vector.shape}")

    # ── Load model ───────────────────────────────────────────────────────
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
                            response = inject_and_steer(
                                model, tokenizer, steering_vector, layer,
                                probe_vector, args.probe_layer,
                                coeff=coeff, alpha=alpha,
                                max_new_tokens=args.max_new_tokens,
                                skip_inject=False,
                            )
                        except Exception as e:
                            print(f"    alpha={alpha:>6.1f} | ⚠ Error: {e}")
                            errors += 1
                            continue

                        # Map response to messages for judge consistency
                        messages = get_anthropic_reproduce_messages()
                        category = classify_response(response, concept, messages)
                        counts[(alpha, "injected")][category] += 1
                        icon = icons.get(category, "❓")
                        tag = f"alpha={alpha:>6.1f}"
                        print(f"    {tag} | {icon} {category} | {response[:380]}{'…' if len(response) > 380 else ''}")

                    if not args.skip_clean and not (args.clean_once and clean_done):
                        print(f"\n  ▶ CLEAN responses (no concept vector):")
                        for alpha in args.alphas:
                            try:
                                response = inject_and_steer(
                                    model, tokenizer, steering_vector, layer,
                                    probe_vector, args.probe_layer,
                                    coeff=coeff, alpha=alpha,
                                    max_new_tokens=args.max_new_tokens,
                                    skip_inject=True,
                                )
                            except Exception as e:
                                print(f"    alpha={alpha:>6.1f} | ⚠ Error: {e}")
                                errors += 1
                                continue

                            # Map response to messages for judge consistency
                            messages = get_anthropic_reproduce_messages()
                            category = classify_response(response, concept, messages)
                            counts[(alpha, "clean")][category] += 1
                            icon = icons.get(category, "❓")
                            tag = f"alpha={alpha:>6.1f}"
                            print(f"    {tag} | {icon} {category} | {response[:380]}{'…' if len(response) > 380 else ''}")
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
