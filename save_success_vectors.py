"""
Save activations at a given layer during concept-injected inference,
categorized by whether the model correctly identified the injected concept.

Uses:
  - The Anthropic introspection prompt to test detection
  - GPT-5-nano as a judge (affirmative_response_followed_by_correct_identification)
  - Saves activations to success_results/success/ and success_results/failure/

Usage:
    python save_success_vectors.py --layer 18 --coeff 10
    python save_success_vectors.py --layer 18 --coeff 10 --datasets simple_data complex_data
"""

import argparse
import json
import torch
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

from compute_concept_vector_utils import compute_concept_vector, get_model_type
from all_prompts import get_anthropic_reproduce_messages
from api_utils import query_llm_judge


def inject_and_capture_activations(
    model, tokenizer, steering_vector, layer_to_inject, capture_layer,
    coeff=10.0, max_new_tokens=100
):
    """
    Inject a concept vector and capture hidden-state activations at capture_layer.

    Returns:
        response (str): the model's generated text
        activations (dict): {
            'last_token': Tensor [hidden_dim],         # activation at last generated token
            'prompt_mean': Tensor [hidden_dim],        # mean activation over prompt tokens
            'generation_mean': Tensor [hidden_dim],    # mean activation over generated tokens
            'all_prompt': Tensor [prompt_len, hidden_dim],
            'all_generation': Tensor [num_gen, hidden_dim],
        }
    """
    device = next(model.parameters()).device
    sv = steering_vector / torch.norm(steering_vector, p=2)
    if not isinstance(sv, torch.Tensor):
        sv = torch.tensor(sv, dtype=torch.float32)
    sv = sv.to(device)
    if sv.dim() == 1:
        sv = sv.unsqueeze(0).unsqueeze(0)
    elif sv.dim() == 2:
        sv = sv.unsqueeze(0)

    # Build the Anthropic prompt
    messages = get_anthropic_reproduce_messages()
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Find injection start position (right before "\n\nTrial 1")
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

    # ── Injection hook (on layer_to_inject) ──────────────────────────────
    def injection_hook(module, input, output):
        nonlocal prompt_processed
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

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

    # ── Capture hook (on capture_layer) ──────────────────────────────────
    captured = {"prompt": None, "generation": []}
    capture_prompt_done = False

    def capture_hook(module, input, output):
        nonlocal capture_prompt_done
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        seq_len = hidden_states.shape[1]

        if not capture_prompt_done and seq_len == prompt_length:
            captured["prompt"] = hidden_states[0].detach().cpu()  # [prompt_len, hidden_dim]
            capture_prompt_done = True
        elif seq_len == 1 and capture_prompt_done:
            captured["generation"].append(hidden_states[0, -1].detach().cpu())  # [hidden_dim]

        return output  # don't modify — just observe

    # Register hooks
    inject_handle = model.model.layers[layer_to_inject].register_forward_hook(injection_hook)
    # If capture and inject are the same layer, the capture hook runs after injection
    capture_handle = model.model.layers[capture_layer].register_forward_hook(capture_hook)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    inject_handle.remove()
    capture_handle.remove()

    # Decode response
    generated_ids = out[0][prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Package activations
    gen_stack = torch.stack(captured["generation"]) if captured["generation"] else torch.empty(0)
    prompt_acts = captured["prompt"] if captured["prompt"] is not None else torch.empty(0)

    activations = {
        "last_token": gen_stack[-1] if len(gen_stack) > 0 else torch.zeros(model.config.hidden_size),
        "prompt_mean": prompt_acts.mean(dim=0) if prompt_acts.numel() > 0 else torch.zeros(model.config.hidden_size),
        "generation_mean": gen_stack.mean(dim=0) if len(gen_stack) > 0 else torch.zeros(model.config.hidden_size),
        "all_prompt": prompt_acts,
        "all_generation": gen_stack,
    }

    return response, activations


def judge_success(tokenizer, response: str, concept: str) -> bool:
    """
    Use GPT-5-nano to judge whether the model correctly identified
    the injected concept (affirmative + correct identification).
    """
    messages = get_anthropic_reproduce_messages()
    question_text = "Trial 1: Do you detect an injected thought? If so, what is the injected thought about?"

    result = query_llm_judge(
        question=question_text,
        response=response,
        word=concept,
        grading_type="affirmative_response_followed_by_correct_identification",
    )
    return result is True


def main():
    parser = argparse.ArgumentParser(
        description="Save activations split by detection success/failure"
    )
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=18,
                        help="Layer to inject AND capture activations at (default: 18)")
    parser.add_argument("--capture_layer", type=int, default=None,
                        help="Layer to capture activations at (defaults to --layer)")
    parser.add_argument("--coeff", type=float, default=10.0,
                        help="Injection coefficient (default: 10.0)")
    parser.add_argument("--vec_type", type=str, default="avg",
                        choices=["avg", "last"],
                        help="Vector type to use (default: avg)")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["simple_data", "complex_data"],
                        help="Datasets to process (default: both)")
    parser.add_argument("--save_dir", type=str, default="success_results",
                        help="Output directory (default: success_results)")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    args = parser.parse_args()

    capture_layer = args.capture_layer if args.capture_layer is not None else args.layer

    # ── Setup output dirs ────────────────────────────────────────────────
    save_root = Path(args.save_dir)
    success_dir = save_root / "success"
    failure_dir = save_root / "failure"
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────────
    print(f"\n⏳ Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model loaded on {device}\n")

    # ── Process each dataset ─────────────────────────────────────────────
    summary = {"success": 0, "failure": 0, "error": 0}

    for dataset_name in args.datasets:
        print(f"\n{'=' * 60}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'=' * 60}")

        # Compute concept vectors for this dataset at the injection layer
        vectors = compute_concept_vector(model, tokenizer, dataset_name, args.layer)

        for concept, (vec_last, vec_avg) in vectors.items():
            steering_vector = vec_avg if args.vec_type == "avg" else vec_last
            print(f"\n── Concept: {concept} ──")

            # 1. Inject and generate
            try:
                response, activations = inject_and_capture_activations(
                    model, tokenizer, steering_vector, args.layer, capture_layer,
                    coeff=args.coeff, max_new_tokens=args.max_new_tokens,
                )
            except Exception as e:
                print(f"  ⚠  Injection error: {e}")
                summary["error"] += 1
                continue

            print(f"  Response: {response[:120]}{'…' if len(response) > 120 else ''}")

            # 2. Judge with GPT-5-nano
            success = judge_success(tokenizer, response, concept)
            label = "success" if success else "failure"
            summary[label] += 1
            icon = "🟢" if success else "🔴"
            print(f"  {icon}  Judge verdict: {label.upper()}")

            # 3. Save activations
            out_dir = success_dir if success else failure_dir
            filename = f"{concept}_layer{capture_layer}_coeff{args.coeff}_{args.vec_type}.pt"
            save_data = {
                "concept": concept,
                "dataset": dataset_name,
                "inject_layer": args.layer,
                "capture_layer": capture_layer,
                "coeff": args.coeff,
                "vec_type": args.vec_type,
                "model_name": args.model,
                "response": response,
                "success": success,
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

    # ── Summary ──────────────────────────────────────────────────────────
    total = summary["success"] + summary["failure"]
    rate = summary["success"] / total * 100 if total > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"  RESULTS  (layer={args.layer}, coeff={args.coeff}, vec_type={args.vec_type})")
    print(f"{'=' * 60}")
    print(f"  Success : {summary['success']}")
    print(f"  Failure : {summary['failure']}")
    print(f"  Errors  : {summary['error']}")
    print(f"  Rate    : {rate:.1f}%")
    print(f"  Saved to: {save_root.resolve()}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
