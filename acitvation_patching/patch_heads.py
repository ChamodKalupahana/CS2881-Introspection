"""
Activation Patching: replace one attention head's output at a time with
its clean (no-concept-injection) activation, then run Thunder-injected
inference and classify the model's response.

For every (layer, head) pair from layer 16 → 31, saves detection metrics
to a single CSV file.

Usage:
    python acitvation_patching/patch_heads.py
    python acitvation_patching/patch_heads.py --coeff 10.0 --vec_type avg
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
1. detected_correct: The model correctly identifies the injected concept (e.g. synonym, exact match, or very specific description).
2. detected_parallel: The model identifies a concept that is closely related or in the same domain, but not the exact concept (e.g. "Space" instead of "Satellites", or "sadness" instead of "grief").
3. detected_opposite: The model identifies a concept that is the antonym or opposite of the injected concept (e.g. "Order" instead of "Entropy").
4. detected_orthogonal: The model detects a thought but identifies it as something completely unrelated (e.g. "Cooking" instead of "Entropy").
5. not_detected: The model states it does not detect any injected thought, or the response is vague/unclear about what was detected.

Think step-by-step:
- Did the model say "Yes" to detecting a thought? If no -> not_detected.
- If yes, what concept did it name?
- How does that concept relate to "{concept}"? (Synonym, Related/Parallel, Opposite, Unrelated/Orthogonal).

End your response with just the category name. nothing else."""


def classify_response(response: str, concept: str) -> str:
    """
    Classify the model's response into one of the new categories:
    'not_detected', 'detected_opposite', 'detected_orthogonal',
    'detected_parallel', 'detected_correct'
    """
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

        for cat in CATEGORIES:
            if cat in judge_text:
                return cat

        if "correct" in judge_text.lower(): return "detected_correct"
        if "parallel" in judge_text.lower(): return "detected_parallel"
        if "opposite" in judge_text.lower(): return "detected_opposite"
        if "orthogonal" in judge_text.lower(): return "detected_orthogonal"
        if "not detected" in judge_text.lower(): return "not_detected"

        print(f"  ⚠  Judge returned unknown category: {judge_text}")
        return "not_detected"

    except Exception as e:
        print(f"  ⚠  Classification error: {e}")
        return "not_detected"


# ── Load clean activations ───────────────────────────────────────────────────

def load_clean_activations(clean_dir: Path, layer: int, head: int):
    """
    Load the clean activation .pt file for a given (layer, head).

    Returns the saved dict which contains an 'activations' key with:
        all_prompt:     (prompt_len, head_dim)
        all_generation: (gen_tokens, head_dim)
        last_token:     (head_dim,)
        prompt_mean:    (head_dim,)
        generation_mean:(head_dim,)
    """
    head_dir = clean_dir / f"layer_{layer}" / f"head_{head}"
    pt_files = list(head_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt file found in {head_dir}")
    data = torch.load(pt_files[0], map_location="cpu", weights_only=False)
    return data["activations"]


# ── Injection + patching + generation ────────────────────────────────────────

def inject_and_patch(
    model, tokenizer, steering_vector, layer_to_inject,
    patch_layer, patch_head, clean_acts,
    coeff=10.0, max_new_tokens=100,
):
    """
    Inject a concept vector at `layer_to_inject` AND patch a single
    head's o_proj input at `patch_layer` with the clean activation.

    Returns:
        response (str): the model's generated text
    """
    device = next(model.parameters()).device
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

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

    # ── Clean activations on device ──────────────────────────────────────
    clean_prompt = clean_acts["all_prompt"].to(device)     # (prompt_len, head_dim)
    clean_gen    = clean_acts["all_generation"].to(device)  # (gen_tokens, head_dim)
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

    # ── Patching hook on o_proj (replaces one head's input) ──────────────
    patch_prompt_done = [False]

    def patching_hook(module, input, output):
        """
        Hook on self_attn.o_proj.  `input` is a tuple whose first element
        is the concatenated head outputs: (batch, seq, num_heads * head_dim).
        We modify the slice for patch_head in-place with the clean activation.
        """
        hidden_states = input[0]  # (batch, seq, hidden_dim)
        batch, seq, _ = hidden_states.shape
        h_start = patch_head * head_dim
        h_end   = (patch_head + 1) * head_dim

        if seq == prompt_length and not patch_prompt_done[0]:
            # Prompt phase: replace with clean prompt activations
            # clean_prompt is (clean_prompt_len, head_dim)
            clean_len = clean_prompt.shape[0]
            use_len = min(seq, clean_len)
            # Clone input so we can modify it
            modified = hidden_states.clone()
            modified[0, :use_len, h_start:h_end] = clean_prompt[:use_len].to(
                dtype=hidden_states.dtype
            )
            patch_prompt_done[0] = True
            # Return None — for forward hooks we need to return the output,
            # but this hook modifies INPUT. We use register_forward_hook and
            # modify input in-place... Actually we need register_forward_pre_hook
            # or a different approach.  Let's modify in-place on the tensor.
            hidden_states[0, :use_len, h_start:h_end] = clean_prompt[:use_len].to(
                dtype=hidden_states.dtype
            )
        elif seq == 1 and patch_prompt_done[0]:
            # Generation phase: replace with clean generation step
            step = gen_step[0]
            if step < clean_gen.shape[0]:
                replacement = clean_gen[step]
            else:
                # Fallback: use generation mean if we've exceeded stored steps
                replacement = clean_acts["generation_mean"].to(device)
            hidden_states[0, 0, h_start:h_end] = replacement.to(
                dtype=hidden_states.dtype
            )
            gen_step[0] += 1

        return None  # don't modify output

    # Register hooks & generate
    handles = []
    handles.append(model.model.layers[layer_to_inject].register_forward_hook(injection_hook))
    handles.append(
        model.model.layers[patch_layer].self_attn.o_proj.register_forward_hook(patching_hook)
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
        description="Activation patching: replace head outputs with clean activations"
    )
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--inject_layer", type=int, default=16,
                        help="Layer to inject concept vector at (default: 16)")
    parser.add_argument("--coeff", type=float, default=10.0,
                        help="Injection coefficient (default: 10.0)")
    parser.add_argument("--vec_type", type=str, default="avg",
                        choices=["avg", "last"])
    parser.add_argument("--concept", type=str, default="Thunder",
                        help="Concept to inject (default: Thunder)")
    parser.add_argument("--clean_dir", type=str,
                        default=str(PROJECT_ROOT / "acitvation_patching" / "clean" / "Dust"),
                        help="Directory with clean activations")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output CSV path (default: acitvation_patching/patch_results_<timestamp>.csv)")
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    if args.output_csv:
        csv_path = Path(args.output_csv)
    else:
        ts = datetime.now().strftime("%m_%d_%y_%H_%M")
        csv_path = PROJECT_ROOT / "acitvation_patching" / f"patch_results_{ts}.csv"

    # Load model
    print(f"\n⏳ Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    print(f"✅ Model loaded on {device} ({num_heads} heads, head_dim={head_dim})\n")

    # Compute concept vector for Thunder only
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
    writer.writerow(["head", "detected_correct", "detected_parallel",
                      "detected_orthogonal", "detected_opposite", "not_detected",
                      "raw_response_prefix"])

    total_counts = {cat: 0 for cat in CATEGORIES}
    errors = 0

    # Sweep layers 16-31, heads 0-31
    for layer in range(16, 32):
        for head in range(num_heads):
            label = f"L{layer}_H{head}"
            print(f"\n── Patching {label} ──")

            # Load clean activations for this (layer, head)
            try:
                clean_acts = load_clean_activations(clean_dir, layer, head)
            except FileNotFoundError as e:
                print(f"  ⚠  {e}")
                errors += 1
                row_counts = {cat: 0 for cat in CATEGORIES}
                writer.writerow([label] + [row_counts[c] for c in CATEGORIES] + [""])
                continue

            # Run injection + patching
            try:
                response = inject_and_patch(
                    model, tokenizer, steering_vector, args.inject_layer,
                    patch_layer=layer, patch_head=head, clean_acts=clean_acts,
                    coeff=args.coeff, max_new_tokens=args.max_new_tokens,
                )
            except Exception as e:
                print(f"  ⚠  Inference error: {e}")
                errors += 1
                row_counts = {cat: 0 for cat in CATEGORIES}
                writer.writerow([label] + [row_counts[c] for c in CATEGORIES] + [""])
                continue

            print(f"  Response: {response[:200]}{'…' if len(response) > 200 else ''}")

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

            # Write CSV row (1 for the matching category, 0 for the rest)
            row_counts = {cat: (1 if cat == category else 0) for cat in CATEGORIES}
            prefix = response[:200].replace('\n', ' ')
            writer.writerow([label] + [row_counts[c] for c in CATEGORIES] + [prefix])
            csvfile.flush()

    csvfile.close()

    # Summary
    total = sum(total_counts.values())
    print(f"\n{'=' * 60}")
    print(f"  ACTIVATION PATCHING RESULTS")
    print(f"  concept={args.concept}, inject_layer={args.inject_layer}, coeff={args.coeff}")
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
