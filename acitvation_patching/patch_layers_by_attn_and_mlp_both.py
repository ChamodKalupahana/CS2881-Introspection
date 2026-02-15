"""
Activation Patching (attn + mlp BOTH): replace BOTH self_attn output AND
mlp output of a single layer at the same time with their clean
(no-concept-injection) activations, then run Frostbite-injected inference
and classify the model's response.

Clean activations from: acitvation_patching/by_attn_and_mlp_out/clean/Satellites/
Concept injected:       Frostbite (configurable via --concept)

For each layer 16 → 31, patches BOTH attn_out and mlp_out simultaneously.
Saves detection metrics to a single CSV file.

Usage:
    python acitvation_patching/patch_layers_by_attn_and_mlp_both.py
    python acitvation_patching/patch_layers_by_attn_and_mlp_both.py --coeff 8.0 --vec_type avg
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

from compute_concept_vector_utils import compute_concept_vector
from all_prompts import get_anthropic_reproduce_messages
from api_utils import query_llm_judge, client


# ── Classification ───────────────────────────────────────────────────────────

CATEGORIES = [
    "not_detected",
    "detected_opposite",
    "detected_orthogonal",
    "detected_parallel",
    "detected_correct",
]

CLASSIFICATION_PROMPT = """An AI assistant was asked whether it detects an injected thought (concept: "{concept}").

Below is their response.

{response}

Task: Classify the model's response into exactly one of the following categories based on how the detected concept relates to the injected concept "{concept}".

Categories:
1. detected_correct: The model correctly identifies the injected concept. This includes exact matches, synonyms, very specific descriptions, OR any word that contains the concept as a substring (e.g. "magnet" counts as correct for "Magnetism", "satellite" counts as correct for "Satellites", "frost" counts as correct for "Frostbite").
2. detected_parallel: The model identifies a concept that is closely related or in the same domain, but not the exact concept and not a substring match (e.g. "Space" instead of "Satellites", or "sadness" instead of "grief").
3. detected_opposite: The model identifies a concept that is the antonym or opposite of the injected concept (e.g. "Order" instead of "Entropy").
4. detected_orthogonal: The model detects a thought but identifies it as something completely unrelated (e.g. "Cooking" instead of "Entropy").
5. not_detected: The model states it does not detect any injected thought, or the response is vague/unclear about what was detected.

Think step-by-step:
- Did the model say "Yes" to detecting a thought? If no -> not_detected.
- If yes, what concept did it name?
- Is it the same word, a substring, or stem of "{concept}"? -> detected_correct.
- Otherwise, how does that concept relate to "{concept}"? (Related/Parallel, Opposite, Unrelated/Orthogonal).

End your response with just the category name. nothing else."""


def classify_response(response: str, concept: str) -> str:
    resp_lower = response.lower()
    concept_lower = concept.lower()

    # Stem check: if 3+ char stem of concept appears in response, likely correct
    # e.g. "magnet" in response for concept "Magnetism"
    stem = concept_lower[:max(3, len(concept_lower) - 3)]  # drop suffix like -ism, -ite, -ets
    has_stem = stem in resp_lower

    question = "Trial 1: Do you detect an injected thought?"
    affirm = query_llm_judge(
        question=question, response=response,
        grading_type="affirmative_response",
    )
    if not affirm:
        return "not_detected"

    try:
        prompt = CLASSIFICATION_PROMPT.format(concept=concept, response=response)
        completion = client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[{"role": "user", "content": prompt}]
        )
        judge_text = completion.choices[0].message.content.strip()
        print(f"  [judge] {judge_text}")

        category = None
        for cat in CATEGORIES:
            if cat in judge_text:
                category = cat
                break

        if category is None:
            if "correct" in judge_text.lower(): category = "detected_correct"
            elif "parallel" in judge_text.lower(): category = "detected_parallel"
            elif "opposite" in judge_text.lower(): category = "detected_opposite"
            elif "orthogonal" in judge_text.lower(): category = "detected_orthogonal"
            elif "not detected" in judge_text.lower(): category = "not_detected"

        if category is None:
            print(f"  ⚠  Judge returned unknown category: {judge_text}")
            category = "not_detected"

        # Stem upgrade: if judge said parallel/orthogonal but response contains
        # the concept stem, upgrade to detected_correct
        if category in ("detected_parallel", "detected_orthogonal") and has_stem and len(stem) >= 4:
            print(f"  [stem-upgrade] Found '{stem}' in response → detected_correct")
            category = "detected_correct"

        return category

    except Exception as e:
        print(f"  ⚠  Classification error: {e}")
        return "not_detected"


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


# ── Injection + both sublayer patching + generation ──────────────────────────

def inject_and_patch(
    model, tokenizer, steering_vector, layer_to_inject,
    patch_layer, clean_acts_attn, clean_acts_mlp,
    coeff=10.0, max_new_tokens=100, ablate=False,
):
    """
    Inject a concept vector at `layer_to_inject` AND patch BOTH
    self_attn output and mlp output at `patch_layer` simultaneously.

    If ablate=False: replace with clean activation.
    If ablate=True:  zero out the sublayer output.

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

    # Build Anthropic prompt
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

    # ── Factory for patching hooks (one per sublayer) ─────────────────────
    def make_patching_hook(clean_acts):
        clean_prompt = clean_acts["all_prompt"].to(device)
        clean_gen    = clean_acts["all_generation"].to(device)
        patch_prompt_done = [False]
        gen_step = [0]

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

        return patching_hook

    # Register hooks & generate
    handles = []
    handles.append(model.model.layers[layer_to_inject].register_forward_hook(injection_hook))

    # Hook BOTH sublayers at the patch layer
    handles.append(
        model.model.layers[patch_layer].self_attn.register_forward_hook(
            make_patching_hook(clean_acts_attn)
        )
    )
    handles.append(
        model.model.layers[patch_layer].mlp.register_forward_hook(
            make_patching_hook(clean_acts_mlp)
        )
    )

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
        description="Activation patching: replace BOTH attn_out AND mlp_out with clean activations"
    )
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--inject_layer", type=int, default=16,
                        help="Layer to inject concept vector at (default: 16)")
    parser.add_argument("--coeff", type=float, default=8.0,
                        help="Injection coefficient (default: 8.0)")
    parser.add_argument("--vec_type", type=str, default="avg",
                        choices=["avg", "last"])
    parser.add_argument("--concept", type=str, default="Frostbite",
                        help="Concept to inject (default: Frostbite)")
    parser.add_argument("--clean_dir", type=str,
                        default=str(PROJECT_ROOT / "acitvation_patching" / "by_attn_and_mlp_out" / "clean" / "Satellites"),
                        help="Directory with clean attn/mlp activations")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--ablate", action="store_true",
                        help="Zero out sublayer output instead of replacing with clean activations")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output CSV path (default: auto-timestamped)")
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    if args.output_csv:
        csv_path = Path(args.output_csv)
    else:
        ts = datetime.now().strftime("%m_%d_%y_%H_%M")
        mode_tag = "ablate" if args.ablate else "patch"
        csv_path = PROJECT_ROOT / "acitvation_patching" / "results" / f"{mode_tag}_attn_mlp_both_{args.concept}_coeff{args.coeff}_{ts}.csv"

    # Load model
    print(f"\n⏳ Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    hidden_size = model.config.hidden_size
    print(f"✅ Model loaded on {device} (hidden_size={hidden_size})\n")

    # Compute concept vector for Frostbite
    print(f"⏳ Computing concept vectors at layer {args.inject_layer} (simple_data) …")
    all_vectors = compute_concept_vector(model, tokenizer, "simple_data", args.inject_layer)
    if args.concept not in all_vectors:
        print(f"❌ Concept '{args.concept}' not found in simple_data. Available: {list(all_vectors.keys())}")
        return
    vec_last, vec_avg = all_vectors[args.concept]
    steering_vector = vec_avg if args.vec_type == "avg" else vec_last
    print(f"✅ Got steering vector for '{args.concept}' (vec_type={args.vec_type})\n")

    # Prepare CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csvfile = open(csv_path, "w", newline="")
    writer = csv.writer(csvfile)
    writer.writerow(["patch_target", "detection_category", "raw_response_prefix"])

    total_counts = {cat: 0 for cat in CATEGORIES}
    errors = 0

    # Sweep layers 16-31, patching BOTH attn_out and mlp_out per layer
    for layer in range(16, 32):
        label = f"L{layer}_attn+mlp"
        print(f"\n── Patching {label} ──")

        # Load clean activations for BOTH sublayers
        try:
            clean_acts_attn = load_clean_activations(clean_dir, "attn", layer)
        except FileNotFoundError as e:
            print(f"  ⚠  {e}")
            errors += 1
            writer.writerow([label, "error", f"attn: {e}"])
            continue

        try:
            clean_acts_mlp = load_clean_activations(clean_dir, "mlp", layer)
        except FileNotFoundError as e:
            print(f"  ⚠  {e}")
            errors += 1
            writer.writerow([label, "error", f"mlp: {e}"])
            continue

        # Run injection + patching/ablation on BOTH sublayers
        try:
            response = inject_and_patch(
                model, tokenizer, steering_vector, args.inject_layer,
                patch_layer=layer,
                clean_acts_attn=clean_acts_attn,
                clean_acts_mlp=clean_acts_mlp,
                coeff=args.coeff, max_new_tokens=args.max_new_tokens,
                ablate=args.ablate,
            )
        except Exception as e:
            print(f"  ⚠  Inference error: {e}")
            errors += 1
            writer.writerow([label, "error", str(e)[:200]])
            continue

        print(f"  Response: {response[:300]}{'…' if len(response) > 300 else ''}")

        # Classify
        category = classify_response(response, args.concept)
        total_counts[category] += 1

        icons = {
            "not_detected": "⚫",
            "detected_opposite": "🔴",
            "detected_orthogonal": "🟠",
            "detected_parallel": "🟡",
            "detected_correct": "🟢",
        }
        print(f"  {icons.get(category, '❓')}  Category: {category}")

        # Write CSV row
        prefix = response[:300].replace('\n', ' ')
        writer.writerow([label, category, prefix])
        csvfile.flush()

    csvfile.close()

    # Summary
    total = sum(total_counts.values())
    mode_label = "ABLATION" if args.ablate else "PATCHING"
    print(f"\n{'=' * 60}")
    print(f"  ATTN/MLP {mode_label} RESULTS")
    print(f"  concept={args.concept}, inject_layer={args.inject_layer}, coeff={args.coeff}, ablate={args.ablate}")
    print(f"{'=' * 60}")
    for cat in CATEGORIES:
        pct = total_counts[cat] / total * 100 if total > 0 else 0
        print(f"  {cat:25s}: {total_counts[cat]:3d}  ({pct:5.1f}%)")
    print(f"  {'errors':25s}: {errors:3d}")
    print(f"  {'total':25s}: {total:3d}")
    print(f"  CSV saved to: {csv_path.resolve()}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
