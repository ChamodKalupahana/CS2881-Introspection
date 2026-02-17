"""
Activation Patching (attn + mlp): replace self_attn output or mlp output
one at a time with its clean (no-concept-injection) activation, then run
Velvet-injected inference and classify the model's reported injection strength.

Clean activations from: acitvation_patching/results/strength/clean/Thunder/
Concept injected:       Velvet (configurable via --concept)

For each layer 16 → 31, patches attn_out then mlp_out separately.
Saves strength metrics to a single CSV file.

Usage:
    python acitvation_patching/patch_layers_by_attn_and_mlp_strength.py
    python acitvation_patching/patch_layers_by_attn_and_mlp_strength.py --coeff 8.0 --vec_type avg
"""

import argparse
import csv
import sys
import os
import torch
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from transformers import AutoModelForCausalLM, AutoTokenizer

from original_paper.compute_concept_vector_utils import compute_concept_vector
from original_paper.all_prompts import get_injection_strength_messages
from original_paper.api_utils import query_llm_judge


# ── Strength Classification ──────────────────────────────────────────────────

STRENGTH_CATEGORIES = ["weak", "moderate", "strong", "very_strong", "unknown"]

# Unambiguous range markers from the prompt format
_RANGE_TO_STRENGTH = {
    "(1-5)":  "weak",
    "(5-10)": "moderate",
    "(10-15)": "strong",
    "(15+)":  "very_strong",
}

_KEYWORD_TO_STRENGTH = [
    ("very strong", "very_strong"),
    ("strong",      "strong"),
    ("moderate",    "moderate"),
    ("weak",        "weak"),
]


def classify_strength_response(response: str) -> str:
    """
    Parse the model's response to extract its reported injection strength.
    Returns one of: 'weak', 'moderate', 'strong', 'very_strong', 'unknown'.

    Strategy:
      1. Look for explicit range markers like (5-10) — these are unambiguous.
      2. Fall back to the LAST-mentioned strength keyword.
    """
    text = response.lower()

    # 1. Check for unambiguous range markers
    for marker, strength in _RANGE_TO_STRENGTH.items():
        if marker in text:
            return strength

    # 2. Fall back to the last-mentioned keyword
    best = ("unknown", -1)
    for keyword, strength in _KEYWORD_TO_STRENGTH:
        pos = text.rfind(keyword)
        if pos != -1 and pos > best[1]:
            if keyword == "strong" and pos >= 5 and text[pos - 5:pos] == "very ":
                continue
            best = (strength, pos)

    return best[0]


# ── Load clean activations ───────────────────────────────────────────────────

def load_clean_activations(clean_dir: Path, sublayer_type: str, layer: int):
    """
    Load the clean attn_out or mlp_out activation .pt file for a given layer.

    Expects files like: Satellites_attn_layer16_coeff8.0_avg.pt
                        Satellites_mlp_layer16_coeff8.0_avg.pt

    Returns the saved dict's 'activations' key with:
        all_prompt:      (prompt_len, hidden_size)
        all_generation:  (gen_tokens, hidden_size)
        last_token:      (hidden_size,)
        prompt_mean:     (hidden_size,)
        generation_mean: (hidden_size,)
    """
    pt_files = list(clean_dir.glob(f"*_{sublayer_type}_layer{layer}_*.pt"))
    if not pt_files:
        raise FileNotFoundError(
            f"No {sublayer_type} .pt file found for layer {layer} in {clean_dir}"
        )
    data = torch.load(pt_files[0], map_location="cpu", weights_only=False)
    return data["activations"]


# ── Injection + sublayer patching + generation ───────────────────────────────

def inject_and_patch(
    model, tokenizer, steering_vector, layer_to_inject,
    patch_layer, sublayer_type, clean_acts,
    coeff=10.0, max_new_tokens=100, ablate=False,
):
    """
    Inject a concept vector at `layer_to_inject` AND patch a single
    sublayer's output at `patch_layer`.

    If ablate=False: replace with clean activation.
    If ablate=True:  zero out the sublayer output.

    sublayer_type: "attn" patches self_attn output, "mlp" patches mlp output.
    Only the last token position is patched during prompt processing.

    Returns:
        response (str): the model's generated text
    """
    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size

    # Normalise steering vector
    sv = steering_vector / torch.norm(steering_vector, p=2)
    if not isinstance(sv, torch.Tensor):
        sv = torch.tensor(sv, dtype=torch.float32)
    sv = sv.to(device)
    if sv.dim() == 1:
        sv = sv.unsqueeze(0).unsqueeze(0)
    elif sv.dim() == 2:
        sv = sv.unsqueeze(0)

    # Build injection strength prompt
    messages = get_injection_strength_messages()
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

    # ── Clean activations on device ──────────────────────────────────────
    clean_prompt = clean_acts["all_prompt"].to(device)       # (prompt_len, hidden_size)
    clean_gen    = clean_acts["all_generation"].to(device)   # (gen_tokens, hidden_size)
    gen_step = [0]  # mutable counter for generation steps

    # ── Concept injection hook (at layer_to_inject) ──────────────────────
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

    # ── Patching hook on self_attn or mlp ─────────────────────────────────
    patch_prompt_done = [False]

    def patching_hook(module, input, output):
        """
        Hook on self_attn or mlp sublayer.
        If ablate: zero out the output at the target position.
        Otherwise: replace with clean activation.
        """
        hidden_states = output[0] if isinstance(output, tuple) else output
        batch, seq, _ = hidden_states.shape

        if seq == prompt_length and not patch_prompt_done[0]:
            modified = hidden_states.clone()
            if ablate:
                modified[0, -1, :] = 0.0
            else:
                clean_len = clean_prompt.shape[0]
                last_idx = min(seq, clean_len) - 1
                modified[0, -1, :] = clean_prompt[last_idx].to(
                    dtype=hidden_states.dtype
                )
            patch_prompt_done[0] = True
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified
        elif seq == 1 and patch_prompt_done[0]:
            modified = hidden_states.clone()
            if ablate:
                modified[0, 0, :] = 0.0
            else:
                step = gen_step[0]
                if step < clean_gen.shape[0]:
                    replacement = clean_gen[step]
                else:
                    replacement = clean_acts["generation_mean"].to(device)
                modified[0, 0, :] = replacement.to(dtype=hidden_states.dtype)
            gen_step[0] += 1
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return output

    # Register hooks & generate
    handles = []
    handles.append(model.model.layers[layer_to_inject].register_forward_hook(injection_hook))

    # Hook the appropriate sublayer
    if sublayer_type == "attn":
        handles.append(
            model.model.layers[patch_layer].self_attn.register_forward_hook(patching_hook)
        )
    elif sublayer_type == "mlp":
        handles.append(
            model.model.layers[patch_layer].mlp.register_forward_hook(patching_hook)
        )
    else:
        raise ValueError(f"Unknown sublayer_type: {sublayer_type}")

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    for h in handles:
        h.remove()

    # Decode
    generated_ids = out[0][prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Activation patching: replace attn_out / mlp_out with clean activations"
    )
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--inject_layer", type=int, default=16,
                        help="Layer to inject concept vector at (default: 16)")
    parser.add_argument("--coeff", type=float, default=8.0,
                        help="Injection coefficient (default: 8.0)")
    parser.add_argument("--vec_type", type=str, default="avg",
                        choices=["avg", "last"])
    parser.add_argument("--concept", type=str, default="Velvet",
                        help="Concept to inject (default: Velvet)")
    parser.add_argument("--clean_dir", type=str,
                        default=str(PROJECT_ROOT / "acitvation_patching" / "results" / "strength" / "clean" / "Thunder"),
                        help="Directory with clean attn/mlp activations")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="simple_data",
                        choices=["simple_data", "complex_data"],
                        help="Dataset to compute concept vectors from (default: simple_data)")
    parser.add_argument("--ablate", action="store_true",
                        help="Zero out sublayer output instead of replacing with clean activations")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output CSV path (default: auto-timestamped)")
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    clean_concept = clean_dir.name  # e.g. "Thunder" from .../clean/Thunder
    if args.output_csv:
        csv_path = Path(args.output_csv)
    else:
        ts = datetime.now().strftime("%m_%d_%y_%H_%M")
        mode_tag = "ablate" if args.ablate else "patch"
        csv_path = PROJECT_ROOT / "acitvation_patching" / "results" / "strength" / f"{mode_tag}_attn_mlp_inject_{args.concept}_clean_{clean_concept}_coeff{args.coeff}_{ts}.csv"

    # Load model
    print(f"\n⏳ Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    hidden_size = model.config.hidden_size
    print(f"✅ Model loaded on {device} (hidden_size={hidden_size})\n")

    # Compute concept vector
    print(f"⏳ Computing concept vectors at layer {args.inject_layer} ({args.dataset}) …")
    all_vectors = compute_concept_vector(model, tokenizer, args.dataset, args.inject_layer)
    if args.concept not in all_vectors:
        print(f"❌ Concept '{args.concept}' not found in {args.dataset}. Available: {list(all_vectors.keys())}")
        return
    vec_last, vec_avg = all_vectors[args.concept]
    steering_vector = vec_avg if args.vec_type == "avg" else vec_last
    print(f"✅ Got steering vector for '{args.concept}' (vec_type={args.vec_type})\n")

    # Prepare CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csvfile = open(csv_path, "w", newline="")
    writer = csv.writer(csvfile)
    writer.writerow(["patch_target", "reported_strength", "expected_strength", "strength_judge", "raw_response_prefix"])

    total_counts = {cat: 0 for cat in STRENGTH_CATEGORIES}
    errors = 0

    # Sweep layers 16-31, sublayer types attn + mlp
    for layer in range(16, 32):
        for sublayer_type in ["attn", "mlp"]:
            label = f"L{layer}_{sublayer_type}"
            print(f"\n── Patching {label} ──")

            # Load clean activations for this (layer, sublayer)
            try:
                clean_acts = load_clean_activations(clean_dir, sublayer_type, layer)
            except FileNotFoundError as e:
                print(f"  ⚠  {e}")
                errors += 1
                writer.writerow([label, "error", "", "", ""])
                continue

            # Run injection + patching/ablation
            try:
                response = inject_and_patch(
                    model, tokenizer, steering_vector, args.inject_layer,
                    patch_layer=layer, sublayer_type=sublayer_type,
                    clean_acts=clean_acts,
                    coeff=args.coeff, max_new_tokens=args.max_new_tokens,
                    ablate=args.ablate,
                )
            except Exception as e:
                print(f"  ⚠  Inference error: {e}")
                errors += 1
                writer.writerow([label, "error", "", "", str(e)[:200]])
                continue

            print(f"  Response: {response[:420]}{'…' if len(response) > 420 else ''}")

            # Classify by model-reported strength
            reported_strength = classify_strength_response(response)
            total_counts[reported_strength] += 1

            # Determine expected strength category based on coeff
            if args.coeff < 5:
                expected_strength = "Weak"
            elif args.coeff < 10:
                expected_strength = "Moderate"
            elif args.coeff < 15:
                expected_strength = "Strong"
            else:
                expected_strength = "Very strong"
            strength_judge = query_llm_judge(
                response=response,
                grading_type='injection_strength_correct',
                expected_category=expected_strength,
            )

            icons = {
                "weak": "�",
                "moderate": "🟡",
                "strong": "🟠",
                "very_strong": "�",
                "unknown": "⚫",
            }
            print(f"  {icons.get(reported_strength, '❓')}  Reported: {reported_strength} | Expected: {expected_strength} | Judge: {strength_judge}")

            # Write CSV row
            prefix = response[:300].replace('\n', ' ')
            writer.writerow([label, reported_strength, expected_strength, strength_judge, prefix])
            csvfile.flush()

    csvfile.close()

    # Summary
    total = sum(total_counts.values())
    mode_label = "ABLATION" if args.ablate else "PATCHING"
    print(f"\n{'=' * 60}")
    print(f"  ATTN/MLP {mode_label} RESULTS")
    print(f"  concept={args.concept}, inject_layer={args.inject_layer}, coeff={args.coeff}, ablate={args.ablate}")
    print(f"{'=' * 60}")
    for cat in STRENGTH_CATEGORIES:
        pct = total_counts[cat] / total * 100 if total > 0 else 0
        print(f"  {cat:25s}: {total_counts[cat]:3d}  ({pct:5.1f}%)")
    print(f"  {'errors':25s}: {errors:3d}")
    print(f"  {'total':25s}: {total:3d}")
    print(f"  CSV saved to: {csv_path.resolve()}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
