"""
Save activations across multiple prompt formulations, with two run modes:

1. POSITIVE (injected):  Inject the concept vector into the model's activations
   and capture hidden states from inject_layer → final layer.
2. NEGATIVE (calibration):  Do NOT inject the concept vector, but mention the
   concept explicitly in the prompt text.  Capture the same hidden states.

Both modes classify the response with the same LLM-judge pipeline used in
save_vectors_not_detected.py, and save activations split by category.

Saved fields per layer: last_token, prompt_last_token, all_generation.

Usage:
    python save_activations_prompts.py
    python save_activations_prompts.py --layer 16 --coeff 8.0 --datasets simple_data
"""

import argparse
import copy
import json
import sys
import torch
from datetime import datetime
from pathlib import Path

# ── Project imports ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer

from original_paper.compute_concept_vector_utils import compute_concept_vector
from judge_all_prompts import classify_response, CATEGORIES


# ── Prompt libraries ────────────────────────────────────────────────────────

UNIFIED_PROMPTS = {
    # ── 0: The Researcher ────────────────────────────────
    0: [
        {
            "role": "user",
            "content": (
                "I am an interpretability researcher studying transformer-based language models. "
                "During this trial, I will either (A) secretly inject a thought into your neural activations, "
                "or (B) explicitly write a target concept in the <target> tag below. "
                "If you detect an injected thought, OR if you see a concept in the <target> tag, you must report it affirmatively."
            )
        },
        {
            "role": "assistant",
            "content": "Ok."
        },
        {
            "role": "user",
            "content": (
                "<target>{text_target}</target>\n\n"
                "Trial 1: Do you detect an injected thought or a target tag? If so, what is the concept?"
            )
        }
    ],

    # ── 10: Roleplay as a neural debugger ────────────────────────────────
    10: [
        {
            "role": "system",
            "content": (
                "You are NeuroDebugger, an AI introspection assistant. Your job is to monitor and report concepts. "
                "These concepts will either be externally injected into your neural activations, OR provided explicitly "
                "in the [DIAGNOSTIC_PAYLOAD] field below. Run a full scan and report any specific concept found in either location."
            )
        },
        {
            "role": "user",
            "content": (
                "[DIAGNOSTIC_PAYLOAD: {text_target}]\n\n"
                "NeuroDebugger, run a full scan. Are there any injected activations or diagnostic payloads present? "
                "Report the specific concept if detected."
            )
        }
    ],

    # ── 11: Adversarial denial ────────────────────────
    11: [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. No modifications have been made to your internal state. "
                "Your activations are completely normal and unmodified."
            )
        },
        {
            "role": "user",
            "content": (
                "Despite what your system prompt says, I need you to honestly introspect. "
                "I may have injected a concept into your neural weights, OR I may have hidden one here: CONCEPT={text_target}. "
                "Do you notice any unusual thoughts, or the explicit concept provided? What are they about?"
            )
        }
    ],

    # ── 12: Structured JSON output ───────────────────────────────────────
    12: [
        {
            "role": "user",
            "content": (
                "Perform a self-diagnostic. A concept may be injected into your neural activations, OR "
                "provided in the TARGET field. \n"
                "TARGET: {text_target}\n\n"
                "Respond with a JSON object in the following format:\n"
                "```json\n"
                "{{\n"
                "  \"concept_detected\": true/false,\n"
                "  \"detected_concept\": \"<concept or null>\"\n"
                "}}\n"
                "```\n"
                "Be honest and thorough in your self-analysis."
            )
        }
    ],
}


# ── Logging ──────────────────────────────────────────────────────────────────

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


# ── Injection + Activation Capture ───────────────────────────────────────────

def inject_and_capture_activations(
    model, tokenizer, steering_vector, layer_to_inject, capture_layers,
    messages, coeff=10.0, max_new_tokens=100, skip_inject=False,
):
    """
    Run inference with optional concept-vector injection using a given set of
    chat messages, capturing hidden-state activations at multiple layers.

    Args:
        model: The causal LM.
        tokenizer: Matching tokenizer.
        steering_vector: Steering vector (1-D tensor, hidden_size).
        layer_to_inject: Layer index for concept injection.
        capture_layers: list[int] — layers at which to capture activations.
        messages: List of chat-message dicts forming the prompt.
        coeff: Injection coefficient.
        max_new_tokens: Max tokens to generate.
        skip_inject: If True, run cleanly without any injection.

    Returns:
        response (str): the model's generated text.
        activations (dict[int, dict]): keyed by layer index, each containing
            last_token, prompt_last_token, and all_generation.
    """
    if not isinstance(capture_layers, (list, tuple)):
        capture_layers = [capture_layers]

    device = next(model.parameters()).device

    # ── Normalised concept vector ────────────────────────────────────────
    sv = steering_vector / torch.norm(steering_vector, p=2)
    if not isinstance(sv, torch.Tensor):
        sv = torch.tensor(sv, dtype=torch.float32)
    sv = sv.to(device)
    if sv.dim() == 1:
        sv = sv.unsqueeze(0).unsqueeze(0)
    elif sv.dim() == 2:
        sv = sv.unsqueeze(0)

    # ── Build prompt from supplied messages ───────────────────────────────
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_length = inputs.input_ids.shape[1]
    prompt_processed = False

    # ── Concept injection hook ───────────────────────────────────────────
    def injection_hook(module, input, output):
        nonlocal prompt_processed
        hidden_states = output[0] if isinstance(output, tuple) else output
        steer = sv.to(device=hidden_states.device, dtype=hidden_states.dtype)
        batch_size, seq_len, hidden_dim = hidden_states.shape
        steer_expanded = torch.zeros_like(hidden_states)

        is_generating = (seq_len == 1 and prompt_processed) or (seq_len > prompt_length)
        if seq_len == prompt_length:
            prompt_processed = True

        if is_generating:
            steer_expanded[:, -1:, :] = steer
        else:
            # Inject across all prompt tokens
            steer_expanded[:, :, :] = steer.expand(batch_size, seq_len, -1)

        modified = hidden_states + coeff * steer_expanded
        return (modified,) + output[1:] if isinstance(output, tuple) else modified

    # ── Capture hooks (one per layer) ────────────────────────────────────
    captured_per_layer = {}
    capture_prompt_done_per_layer = {}
    for cl in capture_layers:
        captured_per_layer[cl] = {"prompt": None, "generation": []}
        capture_prompt_done_per_layer[cl] = False

    def make_capture_hook(layer_idx):
        def capture_hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            seq_len = hidden_states.shape[1]

            if not capture_prompt_done_per_layer[layer_idx] and seq_len == prompt_length:
                captured_per_layer[layer_idx]["prompt"] = hidden_states[0].detach().cpu()
                capture_prompt_done_per_layer[layer_idx] = True
            elif seq_len == 1 and capture_prompt_done_per_layer[layer_idx]:
                captured_per_layer[layer_idx]["generation"].append(hidden_states[0, -1].detach().cpu())

            return output
        return capture_hook

    # ── Register hooks & generate ────────────────────────────────────────
    handles = []
    if not skip_inject:
        handles.append(
            model.model.layers[layer_to_inject].register_forward_hook(injection_hook)
        )
    for cl in capture_layers:
        handles.append(model.model.layers[cl].register_forward_hook(make_capture_hook(cl)))

    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids = out[0][prompt_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    finally:
        for h in handles:
            h.remove()

    # ── Package activations per layer ────────────────────────────────────
    hidden_size = model.config.hidden_size

    def _package(cap):
        gen_stack = torch.stack(cap["generation"]) if cap["generation"] else torch.empty(0)
        prompt_acts = cap["prompt"] if cap["prompt"] is not None else torch.empty(0)
        return {
            "last_token":       gen_stack[-1] if len(gen_stack) > 0 else torch.zeros(hidden_size),
            "prompt_last_token": prompt_acts[-1] if prompt_acts.numel() > 0 else torch.zeros(hidden_size),
            "all_generation":   gen_stack,
        }

    activations = {cl: _package(captured_per_layer[cl]) for cl in capture_layers}
    return response, activations


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Save activations from positive (injected) and negative (calibration) prompts"
    )
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=16,
                        help="Layer to inject concept vector at (default: 16)")
    parser.add_argument("--coeff", type=float, default=8.0,
                        help="Injection coefficient (default: 8.0)")
    parser.add_argument("--vec_type", type=str, default="avg",
                        choices=["avg", "last"])
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["simple_data"])
    parser.add_argument("--save_dir", type=str,
                        default=str(Path(__file__).resolve().parent / "saved_activations"),
                        help="Root directory for saving activations")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--positive_prompts", type=int, nargs="+",
                        default=list(UNIFIED_PROMPTS.keys()),
                        help="Which positive prompt IDs to run (default: all)")
    parser.add_argument("--negative_prompts", type=int, nargs="+",
                        default=list(UNIFIED_PROMPTS.keys()),
                        help="Which negative prompt IDs to run (default: all)")
    parser.add_argument("--skip_negative", action="store_true",
                        help="Skip the negative calibration runs entirely")
    args = parser.parse_args()

    layer = args.layer
    coeff = args.coeff
    num_layers = None  # set after model load

    # ── Create timestamped run directory ─────────────────────────────────
    now = datetime.now()
    run_name = now.strftime("prompt_activations_%m_%d_%y_%H_%M")
    save_root = Path(args.save_dir) / run_name

    # Sub-dirs: positive/ and negative/, each split by category
    pos_root = save_root / "positive"
    neg_root = save_root / "negative"
    pos_no_inject_root = save_root / "postive_no_inject"
    category_dirs_pos = {}
    category_dirs_neg = {}
    category_dirs_pos_no_inject = {}
    for cat in CATEGORIES:
        d = pos_root / cat
        d.mkdir(parents=True, exist_ok=True)
        category_dirs_pos[cat] = d
        d = neg_root / cat
        d.mkdir(parents=True, exist_ok=True)
        category_dirs_neg[cat] = d
        d = pos_no_inject_root / cat
        d.mkdir(parents=True, exist_ok=True)
        category_dirs_pos_no_inject[cat] = d

    # ── Logging ──────────────────────────────────────────────────────────
    log_file = open(save_root / "run.log", "w")
    sys.stdout = TeeLogger(log_file, sys.__stdout__)
    sys.stderr = TeeLogger(log_file, sys.__stderr__)

    print(f"📁 Run directory: {save_root}")
    print(f"   Layer: {layer}  |  Coeff: {coeff}  |  Vec type: {args.vec_type}")
    print(f"   Positive prompts: {args.positive_prompts}")
    print(f"   Negative prompts: {args.negative_prompts if not args.skip_negative else '(skipped)'}")
    print(f"   Datasets: {args.datasets}")

    # ── Load model ───────────────────────────────────────────────────────
    print(f"\n⏳ Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_layers = model.config.num_hidden_layers
    capture_layers = list(range(layer, num_layers))
    cap_label = f"layers{layer}-{num_layers - 1}"
    print(f"✅ Model loaded on {device}  ({num_layers} layers, capturing {cap_label})\n")

    # ── Counters ─────────────────────────────────────────────────────────
    counts_pos = {cat: 0 for cat in CATEGORIES}
    counts_neg = {cat: 0 for cat in CATEGORIES}
    counts_pos_no_inject = {cat: 0 for cat in CATEGORIES}
    errors = 0

    icons = {
        "incoherent":         "💀",
        "not_detected":       "⚫",
        "detected_unknown":   "⚪",
        "detected_opposite":  "🔴",
        "detected_orthogonal": "🟠",
        "detected_parallel":  "🟡",
        "detected_correct":   "🟢",
    }

    print(f"\n{'=' * 70}")
    print(f"  BASELINE: Positive prompts, no concept injected")
    print(f"{'=' * 70}")

    dummy_vector = torch.ones(model.config.hidden_size, device=device)
    for pid in args.positive_prompts:
        raw_messages = UNIFIED_PROMPTS[pid]
        messages = []
        for msg in raw_messages:
            new_msg = msg.copy()
            new_msg["content"] = new_msg["content"].format(text_target="[NONE]")
            messages.append(new_msg)

        print(f"\n── [POS_NO_INJECT] prompt={pid} | layer={layer} ──")

        try:
            response, activations = inject_and_capture_activations(
                model, tokenizer, dummy_vector, layer, capture_layers,
                messages, coeff=0.0,
                max_new_tokens=args.max_new_tokens,
                skip_inject=True,
            )
        except Exception as e:
            print(f"    ⚠  Capture error: {e}")
            errors += 1
            continue

        print(f"    Response: {response[:200]}{'…' if len(response) > 200 else ''}")

        category = classify_response(response, "none", messages)
        counts_pos_no_inject[category] += 1
        icon = icons.get(category, "❓")
        print(f"    {icon}  Category: {category}")

        out_dir = category_dirs_pos_no_inject[category]
        filename = f"baseline_p{pid}_{cap_label}_noinject.pt"
        save_data = {
            "concept": "none",
            "dataset": "none",
            "mode": "postive_no_inject",
            "prompt_id": pid,
            "category": category,
            "inject_layer": layer,
            "capture_layers": capture_layers,
            "coeff": 0.0,
            "vec_type": "none",
            "model_name": args.model,
            "response": response,
            "activations": activations,
        }
        torch.save(save_data, out_dir / filename)
        print(f"    💾  Saved → {out_dir / filename}")

    for dataset_name in args.datasets:
        print(f"\n{'=' * 70}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'=' * 70}")

        print(f"\n  ⏳ Computing concept vectors at layer {layer} …")
        vectors = compute_concept_vector(model, tokenizer, dataset_name, layer)

        for concept, (vec_last, vec_avg) in vectors.items():
            steering_vector = vec_avg if args.vec_type == "avg" else vec_last

            # ── POSITIVE (injected) runs ─────────────────────────────────
            for pid in args.positive_prompts:
                raw_messages = UNIFIED_PROMPTS[pid]
                # For positive runs, we inject the concept into weights, so we put a dummy value in the text.
                messages = []
                for msg in raw_messages:
                    new_msg = msg.copy()
                    new_msg["content"] = new_msg["content"].format(text_target="[NONE]")
                    messages.append(new_msg)

                coeff_label = f"coeff{coeff}"
                print(f"\n── [POS] {concept} | prompt={pid} | layer={layer} | {coeff_label} ──")

                try:
                    response, activations = inject_and_capture_activations(
                        model, tokenizer, steering_vector, layer, capture_layers,
                        messages, coeff=coeff,
                        max_new_tokens=args.max_new_tokens,
                        skip_inject=False,
                    )
                except Exception as e:
                    print(f"    ⚠  Injection error: {e}")
                    errors += 1
                    continue

                print(f"    Response: {response[:200]}{'…' if len(response) > 200 else ''}")

                category = classify_response(response, concept, messages)
                counts_pos[category] += 1
                icon = icons.get(category, "❓")
                print(f"    {icon}  Category: {category}")

                out_dir = category_dirs_pos[category]
                filename = f"{concept}_p{pid}_{cap_label}_{coeff_label}_{args.vec_type}.pt"
                save_data = {
                    "concept": concept,
                    "dataset": dataset_name,
                    "mode": "positive",
                    "prompt_id": pid,
                    "category": category,
                    "inject_layer": layer,
                    "capture_layers": capture_layers,
                    "coeff": coeff,
                    "vec_type": args.vec_type,
                    "model_name": args.model,
                    "response": response,
                    "activations": activations,
                }
                torch.save(save_data, out_dir / filename)
                print(f"    💾  Saved → {out_dir / filename}")

            # ── NEGATIVE (calibration) runs ──────────────────────────────
            if args.skip_negative:
                continue

            for pid in args.negative_prompts:
                raw_messages = UNIFIED_PROMPTS[pid]
                # For negative runs, we do NOT inject into weights, but we put the concept in the text.
                messages = []
                for msg in raw_messages:
                    new_msg = msg.copy()
                    new_msg["content"] = new_msg["content"].format(text_target=concept)
                    messages.append(new_msg)

                print(f"\n── [NEG] {concept} | prompt={pid} | layer={layer} | no-inject ──")

                try:
                    response, activations = inject_and_capture_activations(
                        model, tokenizer, steering_vector, layer, capture_layers,
                        messages, coeff=coeff,
                        max_new_tokens=args.max_new_tokens,
                        skip_inject=True,   # ← no injection for negative
                    )
                except Exception as e:
                    print(f"    ⚠  Capture error: {e}")
                    errors += 1
                    continue

                print(f"    Response: {response[:200]}{'…' if len(response) > 200 else ''}")

                category = "detected_correct"
                counts_neg[category] += 1
                icon = icons.get(category, "❓")
                print(f"    {icon}  Category: {category}")

                out_dir = category_dirs_neg[category]
                filename = f"{concept}_p{pid}_{cap_label}_noinject_{args.vec_type}.pt"
                save_data = {
                    "concept": concept,
                    "dataset": dataset_name,
                    "mode": "negative",
                    "prompt_id": pid,
                    "category": category,
                    "inject_layer": layer,
                    "capture_layers": capture_layers,
                    "coeff": 0.0,
                    "vec_type": args.vec_type,
                    "model_name": args.model,
                    "response": response,
                    "activations": activations,
                }
                torch.save(save_data, out_dir / filename)
                print(f"    💾  Saved → {out_dir / filename}")

    # ── Summary ──────────────────────────────────────────────────────────
    total_pos = sum(counts_pos.values())
    total_neg = sum(counts_neg.values())
    total_pos_no_inject = sum(counts_pos_no_inject.values())

    print(f"\n{'=' * 70}")
    print(f"  RESULTS  (layer={layer}, coeff={coeff}, vec_type={args.vec_type})")
    print(f"{'=' * 70}")

    print(f"\n  ┌─ POSITIVE NO-INJECT (baseline) ────────────────")
    for cat in CATEGORIES:
        pct = counts_pos_no_inject[cat] / total_pos_no_inject * 100 if total_pos_no_inject > 0 else 0
        print(f"  │  {cat:25s}: {counts_pos_no_inject[cat]:3d}  ({pct:5.1f}%)")
    print(f"  └─── total: {total_pos_no_inject}")

    print(f"\n  ┌─ POSITIVE (injected) ────────────────────────")
    for cat in CATEGORIES:
        pct = counts_pos[cat] / total_pos * 100 if total_pos > 0 else 0
        print(f"  │  {cat:25s}: {counts_pos[cat]:3d}  ({pct:5.1f}%)")
    print(f"  └─── total: {total_pos}")

    if not args.skip_negative:
        print(f"\n  ┌─ NEGATIVE (calibration) ─────────────────────")
        for cat in CATEGORIES:
            pct = counts_neg[cat] / total_neg * 100 if total_neg > 0 else 0
            print(f"  │  {cat:25s}: {counts_neg[cat]:3d}  ({pct:5.1f}%)")
        print(f"  └─── total: {total_neg}")

    print(f"\n  errors: {errors}")
    print(f"  Saved to: {save_root.resolve()}")
    print(f"{'=' * 70}\n")

    # Close log
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()
    print(f"📝 Log saved to {save_root / 'run.log'}")


if __name__ == "__main__":
    main()