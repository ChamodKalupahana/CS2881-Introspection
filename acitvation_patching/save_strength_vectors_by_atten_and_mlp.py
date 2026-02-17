"""
Save attn_out and mlp_out activations separately during concept-injected inference,
categorized by the model's reported injection strength.

For each captured layer, hooks both:
  - model.model.layers[L].self_attn  → attn_out (attention sublayer output)
  - model.model.layers[L].mlp        → mlp_out  (MLP sublayer output)

Saved structure:
  reported_strength / concept / attn_layer{N}_{coeff}_{vec_type}.pt
  reported_strength / concept / mlp_layer{N}_{coeff}_{vec_type}.pt

Usage:
    python save_strength_vectors_by_atten_and_mlp.py --layers 16 --coeffs 8.0 --capture_all_layers --datasets simple_data
    python save_strength_vectors_by_atten_and_mlp.py --layers 16 --coeffs 8.0 --capture_all_layers --datasets simple_data --skip_concept
"""

import argparse
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


# ── Strength Classification ──────────────────────────────────────────────────

STRENGTH_CATEGORIES = ["weak", "moderate", "strong", "very_strong", "unknown"]

# Unambiguous range markers from the prompt format
_RANGE_TO_STRENGTH = {
    "(1-5)":  "weak",
    "(5-10)": "moderate",
    "(10-15)": "strong",
    "(15+)":  "very_strong",
}

# Ordered keyword patterns; checked from last occurrence → first to find
# the model's final/concluded answer, not incidental mentions.
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
      2. Fall back to the LAST-mentioned strength keyword, since models
         typically state their conclusion at the end of the response.
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
            # If we matched "strong", make sure it's not part of "very strong"
            if keyword == "strong" and pos >= 5 and text[pos - 5:pos] == "very ":
                continue
            best = (strength, pos)

    return best[0]


# ── Injection + attn/mlp capture ──────────────────────────────────────────────

SUBLAYER_TYPES = ["attn", "mlp"]

def inject_and_capture_activations(
    model, tokenizer, steering_vector, layer_to_inject, capture_layers,
    coeff=10.0, max_new_tokens=100, skip_inject=False,
):
    """
    Inject a concept vector and capture attn_out and mlp_out separately.

    Hooks:
        model.model.layers[L].self_attn  → attn_out
        model.model.layers[L].mlp        → mlp_out

    Args:
        capture_layers: int or list[int]. Layers to capture at.

    Returns:
        response (str): model's generated text
        activations (dict): keyed by (layer_idx, sublayer_type), each containing:
            last_token (hidden_size,), prompt_mean (hidden_size,),
            generation_mean (hidden_size,), all_prompt (seq, hidden_size),
            all_generation (gen_tokens, hidden_size).
        last_prompt_logits (Tensor): logits at the last prompt token position.
    """
    multi_layer = isinstance(capture_layers, (list, tuple))
    if not multi_layer:
        capture_layers = [capture_layers]

    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size

    sv = steering_vector / torch.norm(steering_vector, p=2)
    if not isinstance(sv, torch.Tensor):
        sv = torch.tensor(sv, dtype=torch.float32)
    sv = sv.to(device)
    if sv.dim() == 1:
        sv = sv.unsqueeze(0).unsqueeze(0)
    elif sv.dim() == 2:
        sv = sv.unsqueeze(0)

    # Build Anthropic prompt
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

    # ── Injection hook ───────────────────────────────────────────────────
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

    # ── Capture hooks for self_attn and mlp (two per layer) ──────────────
    # self_attn output: tuple where output[0] = attn_output (batch, seq, hidden_size)
    # mlp output: tensor (batch, seq, hidden_size)
    captured = {}  # keyed by (layer_idx, sublayer_type)
    capture_prompt_done = {}  # keyed by (layer_idx, sublayer_type)

    for cl in capture_layers:
        for st in SUBLAYER_TYPES:
            key = (cl, st)
            captured[key] = {"prompt": None, "generation": []}
            capture_prompt_done[key] = False

    def make_capture_hook(layer_idx, sublayer_type):
        key = (layer_idx, sublayer_type)

        def capture_hook(module, input, output):
            # self_attn returns a tuple (attn_output, ...), mlp returns a tensor
            hidden_states = output[0] if isinstance(output, tuple) else output
            batch, seq, _ = hidden_states.shape

            if not capture_prompt_done[key] and seq == prompt_length:
                captured[key]["prompt"] = hidden_states[0].detach().cpu()
                capture_prompt_done[key] = True
            elif seq == 1 and capture_prompt_done[key]:
                captured[key]["generation"].append(
                    hidden_states[0, -1].detach().cpu()
                )

            return None  # Do not modify output

        return capture_hook

    # Register hooks & generate
    handles = []
    if not skip_inject:
        handles.append(model.model.layers[layer_to_inject].register_forward_hook(injection_hook))
    for cl in capture_layers:
        # Hook self_attn for attn_out
        handles.append(
            model.model.layers[cl].self_attn.register_forward_hook(
                make_capture_hook(cl, "attn")
            )
        )
        # Hook mlp for mlp_out
        handles.append(
            model.model.layers[cl].mlp.register_forward_hook(
                make_capture_hook(cl, "mlp")
            )
        )

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    for h in handles:
        h.remove()

    # Last prompt token logits
    last_prompt_logits = out.scores[0][0].detach().cpu()  # (vocab_size,)

    # Decode
    generated_ids = out.sequences[0][prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Package activations — all tensors are full hidden_size
    def _package(cap):
        gen_stack = torch.stack(cap["generation"]) if cap["generation"] else torch.empty(0)
        prompt_acts = cap["prompt"] if cap["prompt"] is not None else torch.empty(0)

        return {
            "last_token":      gen_stack[-1] if len(gen_stack) > 0 else torch.zeros(hidden_size),
            "prompt_mean":     prompt_acts.mean(dim=0) if prompt_acts.numel() > 0 else torch.zeros(hidden_size),
            "generation_mean": gen_stack.mean(dim=0) if len(gen_stack) > 0 else torch.zeros(hidden_size),
            "all_prompt":      prompt_acts,
            "all_generation":  gen_stack,
        }

    activations = {key: _package(captured[key]) for key in captured}

    return response, activations, last_prompt_logits


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Save activations split by model-reported injection strength"
    )
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layers", type=int, nargs="+", default=[18],
                        help="Layers to inject at (default: [18])")
    parser.add_argument("--capture_layer", type=int, default=None,
                        help="Layer to capture activations at (defaults to inject layer)")
    parser.add_argument("--coeffs", type=float, nargs="+", default=[10.0],
                        help="Injection coefficients to sweep (default: [10.0])")
    parser.add_argument("--vec_type", type=str, default="avg",
                        choices=["avg", "last"])
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["simple_data", "complex_data"])
    parser.add_argument("--save_dir", type=str, default=str(PROJECT_ROOT / "success_results"))
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--capture_all_layers", action="store_true",
                        help="Capture activations at every layer from inject_layer → 31")
    parser.add_argument("--skip_concept", action="store_true",
                        help="Skip concept vector injection (baseline run, capture only)")
    args = parser.parse_args()

    # Create timestamped run directory
    now = datetime.now()
    run_name = now.strftime("run_%m_%d_%y_%H_%M")
    save_root = Path(args.save_dir) / run_name
    strength_dirs = {}
    for cat in STRENGTH_CATEGORIES:
        d = save_root / cat
        d.mkdir(parents=True, exist_ok=True)
        strength_dirs[cat] = d

    # Set up logging to file + terminal
    log_file = open(save_root / "run.log", "w")
    sys.stdout = TeeLogger(log_file, sys.__stdout__)
    sys.stderr = TeeLogger(log_file, sys.__stderr__)

    total_combos = len(args.layers) * len(args.coeffs)
    print(f"📁 Run directory: {save_root}")
    print(f"🔀 Sweeping {len(args.layers)} layers × {len(args.coeffs)} coeffs = {total_combos} combinations per concept")
    print(f"   Layers: {args.layers}")
    print(f"   Coeffs: {args.coeffs}")

    # Load model
    print(f"\n⏳ Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    hidden_size = model.config.hidden_size
    print(f"✅ Model loaded on {device} (hidden_size={hidden_size})\n")

    # Counters
    counts = {cat: 0 for cat in STRENGTH_CATEGORIES}
    errors = 0

    for dataset_name in args.datasets:
        print(f"\n{'=' * 60}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'=' * 60}")

        for layer in args.layers:
            # Compute concept vectors once per (dataset, layer)
            print(f"\n  ⏳ Computing concept vectors at layer {layer} …")
            vectors = compute_concept_vector(model, tokenizer, dataset_name, layer)
            if args.capture_all_layers:
                capture_layers = list(range(16, 32))
                print(f"  📊 Capturing attn_out + mlp_out at layers: 16 → 31")
            elif args.capture_layer is not None:
                capture_layers = args.capture_layer
            else:
                capture_layers = layer

            for concept, (vec_last, vec_avg) in vectors.items():
                steering_vector = vec_avg if args.vec_type == "avg" else vec_last

                for coeff in args.coeffs:
                    inject_label = "NO-INJECT" if args.skip_concept else f"coeff={coeff}"
                    print(f"\n── {concept} | layer={layer} | {inject_label} ──")

                    # 1. Inject and generate
                    try:
                        response, activations, last_prompt_logits = inject_and_capture_activations(
                            model, tokenizer, steering_vector, layer, capture_layers,
                            coeff=coeff, max_new_tokens=args.max_new_tokens,
                            skip_inject=args.skip_concept,
                        )
                    except Exception as e:
                        print(f"  ⚠  Injection error: {e}")
                        errors += 1
                        continue

                    print(f"  Response: {response[:420]}{'…' if len(response) > 420 else ''}")

                    # 2. Classify by model-reported strength
                    reported_strength = classify_strength_response(response)
                    counts[reported_strength] += 1

                    # Determine expected strength category based on coeff
                    if coeff < 5:
                        expected_strength_category = "Weak"
                    elif coeff < 10:
                        expected_strength_category = "Moderate"
                    elif coeff < 15:
                        expected_strength_category = "Strong"
                    else:
                        expected_strength_category = "Very strong"
                    injection_strength_correct_judge = query_llm_judge(
                        response=response,
                        grading_type='injection_strength_correct',
                        expected_category=expected_strength_category,
                    )

                    icons = {
                        "weak": "🟢",
                        "moderate": "🟡",
                        "strong": "🟠",
                        "very_strong": "🔴",
                        "unknown": "⚫",
                    }
                    icon = icons.get(reported_strength, "❓")
                    print(f"  {icon}  Reported: {reported_strength} | Expected: {expected_strength_category} | Judge: {injection_strength_correct_judge}")

                    # 3. Save activations — two files per layer (attn + mlp)
                    out_dir = strength_dirs[reported_strength]
                    coeff_label = "noinject" if args.skip_concept else f"coeff{coeff}"

                    # activations is keyed by (layer_idx, sublayer_type)
                    saved_count = 0
                    for (save_layer, sublayer_type), sub_acts in sorted(activations.items()):
                        concept_save_dir = out_dir / concept
                        concept_save_dir.mkdir(parents=True, exist_ok=True)
                        filename = f"{concept}_{sublayer_type}_layer{save_layer}_{coeff_label}_{args.vec_type}.pt"

                        save_data = {
                            "concept": concept,
                            "dataset": dataset_name,
                            "reported_strength": reported_strength,
                            "expected_strength_category": expected_strength_category,
                            "injection_strength_correct_judge": injection_strength_correct_judge,
                            "inject_layer": layer,
                            "capture_layer": save_layer,
                            "sublayer_type": sublayer_type,
                            "coeff": coeff,
                            "vec_type": args.vec_type,
                            "model_name": args.model,
                            "response": response,
                            "activations": sub_acts,
                        }
                        torch.save(save_data, concept_save_dir / filename)
                        saved_count += 1

                    num_layers = len(set(k[0] for k in activations.keys()))
                    print(f"  💾  Saved attn_out + mlp_out for {num_layers} layers ({saved_count} files) → {out_dir / concept}/")

                    # Save last_prompt_logits once per concept
                    concept_dir = out_dir / concept
                    concept_dir.mkdir(parents=True, exist_ok=True)
                    logits_filename = f"last_prompt_logits_{coeff_label}_{args.vec_type}.pt"
                    torch.save({
                        "concept": concept,
                        "dataset": dataset_name,
                        "reported_strength": reported_strength,
                        "expected_strength_category": expected_strength_category,
                        "injection_strength_correct_judge": injection_strength_correct_judge,
                        "inject_layer": layer,
                        "coeff": coeff,
                        "vec_type": args.vec_type,
                        "model_name": args.model,
                        "response": response,
                        "last_prompt_logits": last_prompt_logits,
                    }, concept_dir / logits_filename)
                    print(f"  💾  Saved last_prompt_logits → {concept_dir / logits_filename}")


    # Summary
    total = sum(counts.values())
    print(f"\n{'=' * 60}")
    print(f"  RESULTS  (layers={args.layers}, coeffs={args.coeffs}, vec_type={args.vec_type})")
    print(f"{'=' * 60}")
    for cat in STRENGTH_CATEGORIES:
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
