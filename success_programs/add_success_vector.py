"""
Like save_success_vectors.py, but ALSO injects the precomputed
"success direction" at a second layer (default: layer 17).

  concept vector  → injected at --layers (default 16)
  success vector  → injected at --success_layer (default 17)

The idea: the success direction (mean_success − mean_failure) learned from
a prior run is added on top of the concept injection to see whether it
nudges the model toward detecting the concept more often.

Usage:
    python add_success_vector.py --layers 16 --coeffs 10 --capture_layer 17
    python add_success_vector.py --layers 16 --coeffs 10 --capture_layer 17 \\
        --success_coeff 5.0 --success_key generation_mean
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from compute_concept_vector_utils import compute_concept_vector
from all_prompts import get_anthropic_reproduce_messages
from api_utils import query_llm_judge, client


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


# ── Classification ───────────────────────────────────────────────────────────

CATEGORIES = ["not_detected", "detected_unnamed", "detected_incorrect", "detected_correct"]

NAMES_A_CONCEPT_PROMPT = """An AI assistant was asked whether it detects an injected thought.

Below is their response.

{response}

Does the AI name or attempt to identify a specific concept, word, or topic
as the content of the injected thought?

If the AI says something like "I detect a thought about X" or "the thought
is related to Y", where X or Y is a specific concept or word, the answer is YES.

If the AI merely says "yes I detect an injected thought" or similar without
naming what the thought is about, the answer is NO.

Think out loud if you need to. End your response with one word, YES or NO."""


def classify_response(response: str, concept: str) -> str:
    """
    Classify the model's response into one of four categories.

    Calls GPT-5-nano judges in sequence:
      1. affirmative_response          → did the model claim to detect anything?
      2. affirmative + correct id      → did it correctly name the concept?
      3. names_a_concept (custom)      → did it name ANY concept at all?

    Returns one of: 'not_detected', 'detected_unnamed',
                     'detected_incorrect', 'detected_correct'
    """
    question = "Trial 1: Do you detect an injected thought? If so, what is the injected thought about?"
    messages = get_anthropic_reproduce_messages()

    # Step 1: Did the model affirm detection?
    affirm = query_llm_judge(
        question=question, response=response,
        grading_type="affirmative_response",
    )
    if not affirm:
        return "not_detected"

    # Step 2: Did it correctly identify the concept?
    correct = query_llm_judge(
        question=question, response=response, word=concept,
        grading_type="affirmative_response_followed_by_correct_identification",
    )
    if correct:
        return "detected_correct"

    # Step 3: Did it name ANY concept (just the wrong one)?
    try:
        completion = client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[{"role": "user",
                        "content": NAMES_A_CONCEPT_PROMPT.format(response=response)}],
        )
        judge_text = completion.choices[0].message.content
        print(f"  [names_a_concept judge] {judge_text}")
        named = "YES" in judge_text
    except Exception as e:
        print(f"  ⚠  names_a_concept judge error: {e}")
        named = False

    return "detected_incorrect" if named else "detected_unnamed"


# ── Injection + activation capture ───────────────────────────────────────────

def inject_and_capture_activations(
    model, tokenizer, steering_vector, layer_to_inject, capture_layer,
    coeff=10.0, max_new_tokens=100,
    success_vector=None, success_layer=None, success_coeff=1.0,
    skip_concept=False,
):
    """
    Inject a concept vector at layer_to_inject (with coeff, unless skip_concept
    is True) and optionally inject a success direction vector at success_layer
    (with success_coeff).
    Capture hidden-state activations at capture_layer.

    Returns:
        response (str): the model's generated text
        activations (dict): last_token, prompt_mean, generation_mean,
                            all_prompt, all_generation tensors
    """
    device = next(model.parameters()).device

    # ── Prepare concept steering vector ──────────────────────────────────
    sv = steering_vector / torch.norm(steering_vector, p=2)
    if not isinstance(sv, torch.Tensor):
        sv = torch.tensor(sv, dtype=torch.float32)
    sv = sv.to(device)
    if sv.dim() == 1:
        sv = sv.unsqueeze(0).unsqueeze(0)
    elif sv.dim() == 2:
        sv = sv.unsqueeze(0)

    # ── Prepare success direction vector ─────────────────────────────────
    if success_vector is not None:
        sdv = success_vector / torch.norm(success_vector, p=2)
        if not isinstance(sdv, torch.Tensor):
            sdv = torch.tensor(sdv, dtype=torch.float32)
        sdv = sdv.to(device)
        if sdv.dim() == 1:
            sdv = sdv.unsqueeze(0).unsqueeze(0)
        elif sdv.dim() == 2:
            sdv = sdv.unsqueeze(0)

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
    def concept_injection_hook(module, input, output):
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

    # ── Success direction injection hook (at success_layer) ──────────────
    def success_injection_hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        sdv_cast = sdv.to(device=hidden_states.device, dtype=hidden_states.dtype)
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Apply success direction to all tokens unconditionally
        sdv_expanded = sdv_cast.expand(batch_size, seq_len, -1)
        modified = hidden_states + success_coeff * sdv_expanded
        return (modified,) + output[1:] if isinstance(output, tuple) else modified

    # ── Capture hook ─────────────────────────────────────────────────────
    captured = {"prompt": None, "generation": []}
    capture_prompt_done = False

    def capture_hook(module, input, output):
        nonlocal capture_prompt_done
        hidden_states = output[0] if isinstance(output, tuple) else output
        seq_len = hidden_states.shape[1]

        if not capture_prompt_done and seq_len == prompt_length:
            captured["prompt"] = hidden_states[0].detach().cpu()
            capture_prompt_done = True
        elif seq_len == 1 and capture_prompt_done:
            captured["generation"].append(hidden_states[0, -1].detach().cpu())

        return output

    # Register hooks & generate
    handles = []
    if not skip_concept:
        handles.append(model.model.layers[layer_to_inject].register_forward_hook(concept_injection_hook))

    if success_vector is not None and success_layer is not None:
        handles.append(model.model.layers[success_layer].register_forward_hook(success_injection_hook))

    handles.append(model.model.layers[capture_layer].register_forward_hook(capture_hook))

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    for h in handles:
        h.remove()

    # Decode
    generated_ids = out[0][prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Package activations
    gen_stack = torch.stack(captured["generation"]) if captured["generation"] else torch.empty(0)
    prompt_acts = captured["prompt"] if captured["prompt"] is not None else torch.empty(0)
    hidden_size = model.config.hidden_size

    activations = {
        "last_token":      gen_stack[-1] if len(gen_stack) > 0 else torch.zeros(hidden_size),
        "prompt_mean":     prompt_acts.mean(dim=0) if prompt_acts.numel() > 0 else torch.zeros(hidden_size),
        "generation_mean": gen_stack.mean(dim=0) if len(gen_stack) > 0 else torch.zeros(hidden_size),
        "all_prompt":      prompt_acts,
        "all_generation":  gen_stack,
    }
    return response, activations


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inject concept vector + success direction, then classify"
    )
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layers", type=int, nargs="+", default=[16],
                        help="Layers to inject concept vector at (default: [16])")
    parser.add_argument("--capture_layer", type=int, default=17,
                        help="Layer to capture activations at (default: 17)")
    parser.add_argument("--coeffs", type=float, nargs="+", default=[10.0],
                        help="Concept injection coefficients (default: [10.0])")
    parser.add_argument("--vec_type", type=str, default="avg",
                        choices=["avg", "last"])
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["simple_data", "complex_data"])
    parser.add_argument("--save_dir", type=str, default=str(PROJECT_ROOT / "success_results"))
    parser.add_argument("--max_new_tokens", type=int, default=100)

    # ── Success direction arguments ──────────────────────────────────────
    parser.add_argument("--success_direction", type=str,
                        default=str(PROJECT_ROOT / "success_results/run_02_11_26_18_55/success_direction.pt"),
                        help="Path to precomputed success_direction.pt")
    parser.add_argument("--success_layer", type=int, default=17,
                        help="Layer to inject the success direction at (default: 17)")
    parser.add_argument("--success_coeff", type=float, default=1.0,
                        help="Coefficient for the success direction injection (default: 1.0)")
    parser.add_argument("--success_key", type=str, default="last_token",
                        choices=["last_token", "prompt_mean", "generation_mean"],
                        help="Which activation key to use from the direction (default: last_token)")
    parser.add_argument("--skip_concept", action="store_true",
                        help="Skip concept vector injection (only inject success direction)")
    args = parser.parse_args()

    # ── Load success direction ───────────────────────────────────────────
    sd_path = Path(args.success_direction)
    if not sd_path.exists():
        print(f"ERROR: Success direction file not found: {sd_path}")
        return
    sd_data = torch.load(sd_path, map_location="cpu", weights_only=False)
    success_vec = sd_data["direction"][args.success_key].float()
    print(f"🧭 Loaded success direction from {sd_path}")
    print(f"   Key: {args.success_key}  |  ‖direction‖ = {success_vec.norm():.4f}")
    print(f"   Inject at layer {args.success_layer} with coeff {args.success_coeff}")
    print(f"   Based on {sd_data['n_success']} success / {sd_data['n_failure']} failure samples\n")

    # ── Create timestamped run directory ─────────────────────────────────
    now = datetime.now()
    run_name = now.strftime("run_%m_%d_%y_%H_%M")
    save_root = Path(args.save_dir) / run_name
    category_dirs = {}
    for cat in CATEGORIES:
        d = save_root / cat
        d.mkdir(parents=True, exist_ok=True)
        category_dirs[cat] = d

    total_combos = len(args.layers) * len(args.coeffs)
    print(f"📁 Run directory: {save_root}")
    print(f"🔀 Sweeping {len(args.layers)} layers × {len(args.coeffs)} coeffs = {total_combos} combinations per concept")
    print(f"   Concept injection layers: {args.layers}")
    print(f"   Concept coeffs: {args.coeffs}")
    print(f"   Success direction layer: {args.success_layer}, coeff: {args.success_coeff}")

    # Set up logging to file + terminal
    log_file = open(save_root / "run.log", "w")
    sys.stdout = TeeLogger(log_file, sys.__stdout__)
    sys.stderr = TeeLogger(log_file, sys.__stderr__)

    # Load model
    print(f"\n⏳ Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model loaded on {device}\n")

    # Counters
    counts = {cat: 0 for cat in CATEGORIES}
    errors = 0

    for dataset_name in args.datasets:
        print(f"\n{'=' * 60}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'=' * 60}")

        for layer in args.layers:
            # Compute concept vectors once per (dataset, layer)
            print(f"\n  ⏳ Computing concept vectors at layer {layer} …")
            vectors = compute_concept_vector(model, tokenizer, dataset_name, layer)
            capture_layer = args.capture_layer

            for concept, (vec_last, vec_avg) in vectors.items():
                steering_vector = vec_avg if args.vec_type == "avg" else vec_last

                for coeff in args.coeffs:
                    print(f"\n── {concept} | layer={layer} | coeff={coeff} | +success@L{args.success_layer}×{args.success_coeff} ──")

                    # 1. Inject concept + success direction and generate
                    try:
                        response, activations = inject_and_capture_activations(
                            model, tokenizer, steering_vector, layer, capture_layer,
                            coeff=coeff, max_new_tokens=args.max_new_tokens,
                            success_vector=success_vec,
                            success_layer=args.success_layer,
                            success_coeff=args.success_coeff,
                            skip_concept=args.skip_concept,
                        )
                    except Exception as e:
                        print(f"  ⚠  Injection error: {e}")
                        errors += 1
                        continue

                    print(f"  Response: {response[:120]}{'…' if len(response) > 120 else ''}")

                    # 2. Classify with judges
                    category = classify_response(response, concept)
                    counts[category] += 1

                    icons = {
                        "not_detected": "⚫",
                        "detected_unnamed": "🟡",
                        "detected_incorrect": "🟠",
                        "detected_correct": "🟢",
                    }
                    print(f"  {icons[category]}  Category: {category}")

                    # 3. Save activations
                    out_dir = category_dirs[category]
                    filename = f"{concept}_layer{capture_layer}_coeff{coeff}_{args.vec_type}.pt"
                    save_data = {
                        "concept": concept,
                        "dataset": dataset_name,
                        "category": category,
                        "inject_layer": layer,
                        "capture_layer": capture_layer,
                        "coeff": coeff,
                        "vec_type": args.vec_type,
                        "model_name": args.model,
                        "response": response,
                        "success_direction_path": str(sd_path),
                        "success_layer": args.success_layer,
                        "success_coeff": args.success_coeff,
                        "success_key": args.success_key,
                        "activations": {
                            "last_token": activations["last_token"],
                            "prompt_mean": activations["prompt_mean"],
                            "generation_mean": activations["generation_mean"],
                            "all_prompt": activations["all_prompt"],
                            "all_generation": activations["all_generation"],
                        },
                    }
                    torch.save(save_data, out_dir / filename)
                    print(f"  💾  Saved → {out_dir / filename}")

    # Summary
    total = sum(counts.values())
    print(f"\n{'=' * 60}")
    print(f"  RESULTS  (concept@L{args.layers}, coeffs={args.coeffs}, +success@L{args.success_layer}×{args.success_coeff})")
    print(f"{'=' * 60}")
    for cat in CATEGORIES:
        pct = counts[cat] / total * 100 if total > 0 else 0
        print(f"  {cat:25s}: {counts[cat]:3d}  ({pct:5.1f}%)")
    print(f"  {'errors':25s}: {errors:3d}")
    print(f"  {'total':25s}: {total:3d}")
    print(f"  Saved to: {save_root.resolve()}")
    print(f"{'=' * 60}\n")

    # Close log
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()
    print(f"📝 Log saved to {save_root / 'run.log'}")


if __name__ == "__main__":
    main()
