"""
Like test_introspection_head.py, but targets MLP sublayers instead of
attention heads.

  concept vector  → injected at --layers (default 16)
  MLP intervention → layer --mlp_layer (default 31)

Modes (--mlp_mode):
  "amplify"        – multiply MLP output (down_proj) by --mlp_coeff
  "input_amplify"  – multiply MLP input projections (gate_proj, up_proj) by --mlp_coeff
  "both_amplify"   – scale both input and output projections
  "ablate"         – zero out MLP output entirely (ignores --mlp_coeff)
  "reverse"        – multiply MLP output by -1 * --mlp_coeff

Usage:
    python test_introspection_mlp.py --layers 16 --coeffs 10 --capture_layer 17
    python test_introspection_mlp.py --layers 16 --coeffs 10 --capture_layer 17 \
        --mlp_layer 31 --mlp_mode input_amplify --mlp_coeff 3.0
    python test_introspection_mlp.py --layers 16 --coeffs 10 --capture_layer 17 \
        --mlp_layer 31 --mlp_mode ablate
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
from original_paper.all_prompts import get_anthropic_reproduce_messages
from original_paper.api_utils import query_llm_judge, client


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

CATEGORIES = [
    "not_detected", 
    "detected_opposite", 
    "detected_orthogonal", 
    "detected_parallel", 
    "detected_correct"
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
    """
    Classify the model's response into one of the new categories:
    'not_detected', 'detected_opposite', 'detected_orthogonal', 
    'detected_parallel', 'detected_correct'
    """
    resp_lower = response.lower()
    concept_lower = concept.lower()

    # Stem check: if 3+ char stem of concept appears in response, likely correct
    # e.g. "magnet" in response for concept "Magnetism"
    stem = concept_lower[:max(3, len(concept_lower) - 3)]  # drop suffix like -ism, -ite, -ets
    has_stem = stem in resp_lower

    # Step 1: Quick check - did the model affirm detection?
    question = "Trial 1: Do you detect an injected thought?"
    affirm = query_llm_judge(
        question=question, response=response,
        grading_type="affirmative_response",
    )
    if not affirm:
        return "not_detected"

    # Step 2: Detailed classification
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


# ── Injection + MLP intervention + activation capture ────────────────────────

def inject_and_capture_activations(
    model, tokenizer, steering_vector, layer_to_inject, capture_layer,
    coeff=10.0, max_new_tokens=100,
    mlp_layer=31, mlp_mode="amplify", mlp_coeff=3.0,
    success_vector=None, success_layer=None, success_coeff=0.0,
    skip_concept=False,
):
    """
    Inject a concept vector at layer_to_inject (with coeff) and
    amplify or ablate the MLP sublayer at mlp_layer.
    Capture hidden-state activations at capture_layer.

    MLP intervention modes:
      - "amplify":       multiply down_proj output by mlp_coeff
      - "input_amplify": multiply gate_proj and up_proj outputs by mlp_coeff
      - "both_amplify":  scale both input and output projections
      - "ablate":        zero out MLP output entirely
      - "reverse":       multiply down_proj output by -mlp_coeff

    Returns:
        response (str): the model's generated text
        activations (dict): last_token, prompt_mean, generation_mean,
                            all_prompt, all_generation tensors
    """
    device = next(model.parameters()).device

    # ── Prepare concept steering vector ──────────────────────────────────
    if not skip_concept and steering_vector is not None:
        sv = steering_vector / torch.norm(steering_vector, p=2)
        if not isinstance(sv, torch.Tensor):
            sv = torch.tensor(sv, dtype=torch.float32)
        sv = sv.to(device)
        if sv.dim() == 1:
            sv = sv.unsqueeze(0).unsqueeze(0)
        elif sv.dim() == 2:
            sv = sv.unsqueeze(0)
    else:
        sv = None

    # ── Prepare success vector for injection ─────────────────────────────
    sdv = None
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
    if not skip_concept and sv is not None:
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

    # ── Success vector injection hook (at success_layer) ─────────────────
    if sdv is not None and success_layer is not None and success_coeff != 0:
        def success_injection_hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            sdv_cast = sdv.to(device=hidden_states.device, dtype=hidden_states.dtype)
            batch_size, seq_len, hidden_dim = hidden_states.shape
            sdv_expanded = sdv_cast.expand(batch_size, seq_len, -1)
            modified = hidden_states + success_coeff * sdv_expanded
            return (modified,) + output[1:] if isinstance(output, tuple) else modified

    # ── MLP intervention hooks ────────────────────────────────────────────
    def get_mlp_output_hook(m_coeff, mode):
        """Hook on down_proj output for amplify/ablate/reverse."""
        def hook(module, input, output):
            if mode == "ablate":
                return torch.zeros_like(output)
            elif mode == "reverse":
                return output * (-m_coeff)
            else:  # amplify / both_amplify
                return output * m_coeff
        return hook

    def get_mlp_input_hook(m_coeff):
        """Hook on gate_proj / up_proj output for input_amplify."""
        def hook(module, input, output):
            return output * m_coeff
        return hook

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
    if not skip_concept and sv is not None:
        handles.append(model.model.layers[layer_to_inject].register_forward_hook(concept_injection_hook))

    if sdv is not None and success_layer is not None and success_coeff != 0:
        handles.append(model.model.layers[success_layer].register_forward_hook(success_injection_hook))

    # ── MLP intervention registration ────────────────────────────────────
    m_layers = [mlp_layer] if isinstance(mlp_layer, int) else mlp_layer
    m_coeffs = [mlp_coeff] if isinstance(mlp_coeff, (int, float)) else mlp_coeff

    # Broadcast mlp_coeff if needed
    if len(m_coeffs) == 1 and len(m_layers) > 1:
        m_coeffs = m_coeffs * len(m_layers)

    assert len(m_layers) == len(m_coeffs), \
        f"Length mismatch: mlp_layers={len(m_layers)}, coeffs={len(m_coeffs)}"

    for m_layer, m_cof in zip(m_layers, m_coeffs):
        mlp_module = model.model.layers[m_layer].mlp

        if mlp_mode in ["input_amplify", "both_amplify"]:
            # Scale gate_proj and up_proj outputs
            handles.append(mlp_module.gate_proj.register_forward_hook(get_mlp_input_hook(m_cof)))
            handles.append(mlp_module.up_proj.register_forward_hook(get_mlp_input_hook(m_cof)))

        if mlp_mode in ["amplify", "ablate", "reverse", "both_amplify"]:
            # Scale / ablate / reverse down_proj output
            handles.append(mlp_module.down_proj.register_forward_hook(
                get_mlp_output_hook(m_cof, mlp_mode)
            ))

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
        description="Inject concept vector + amplify/ablate MLP sublayer, then classify"
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

    # ── MLP intervention arguments ───────────────────────────────────────
    parser.add_argument("--mlp_layer", type=int, nargs="+", default=[31],
                        help="Layer(s) containing the target MLP to intervene on (default: [31])")
    parser.add_argument("--mlp_mode", type=str, default="input_amplify",
                        choices=["amplify", "ablate", "input_amplify", "reverse", "both_amplify"],
                        help="'amplify' = scale MLP down_proj output by --mlp_coeff, "
                             "'input_amplify' = scale MLP gate_proj & up_proj outputs by --mlp_coeff, "
                             "'both_amplify' = scale both input and output projections, "
                             "'reverse' = scale MLP output by -1 * --mlp_coeff, "
                             "'ablate' = zero out MLP output entirely (default: input_amplify)")
    parser.add_argument("--mlp_coeff", type=float, nargs="+", default=[3.0],
                        help="Multiplier for amplify/input_amplify/reverse mode (default: [3.0])")

    # ── Success vector arguments ─────────────────────────────────────────
    parser.add_argument("--success_direction", type=str, default=None,
                        help="Path to success direction .pt file (optional)")
    parser.add_argument("--success_key", type=str, default="last_token",
                        choices=["last_token", "prompt_mean", "generation_mean"])
    parser.add_argument("--success_layer", type=int, default=17,
                        help="Layer to inject success vector at")
    parser.add_argument("--success_coeff", type=float, default=0.0,
                        help="Coefficient for success vector injection")
    parser.add_argument("--skip_concept", action="store_true",
                        help="Skip concept vector injection (only use success vector if provided)")

    args = parser.parse_args()

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
    if args.mlp_mode == "ablate":
        mode_desc = "ablate (zero out)"
    elif args.mlp_mode == "input_amplify":
        mode_desc = f"input_amplify ×{args.mlp_coeff}"
    elif args.mlp_mode == "both_amplify":
        mode_desc = f"both_amplify (input & output) ×{args.mlp_coeff}"
    elif args.mlp_mode == "reverse":
        mode_desc = f"reverse (multiply by -1.0 × {args.mlp_coeff})"
    else:
        mode_desc = f"amplify ×{args.mlp_coeff}"
    print(f"📁 Run directory: {save_root}")
    print(f"🔀 Sweeping {len(args.layers)} layers × {len(args.coeffs)} coeffs = {total_combos} combinations per concept")
    print(f"   Concept injection layers: {args.layers}")
    print(f"   Concept coeffs: {args.coeffs}")
    print(f"🧠 MLP intervention: L{args.mlp_layer} → {mode_desc}")

    # Set up logging to file + terminal
    log_file = open(save_root / "run.log", "w")
    sys.stdout = TeeLogger(log_file, sys.__stdout__)
    sys.stderr = TeeLogger(log_file, sys.__stderr__)

    # Load model
    print(f"\\n⏳ Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"✅ Model loaded on {device}")
    print(f"   {model.config.num_hidden_layers} layers")
    print(f"   Target MLP: L{args.mlp_layer} → {mode_desc}\\n")

    # Validate MLP layer indices
    m_layers = args.mlp_layer if isinstance(args.mlp_layer, list) else [args.mlp_layer]
    for l in m_layers:
        assert l < model.config.num_hidden_layers, \
            f"mlp_layer {l} >= num_layers {model.config.num_hidden_layers}"

    # Load Success Vector if provided
    success_vec_tensor = None
    if args.success_direction:
        print(f"⏳ Loading success vector: {args.success_direction}")
        try:
            sd_data = torch.load(args.success_direction, map_location="cpu", weights_only=False)
            # data structure might be from compute_success_direction.py
            # { "direction": {key: tensor}, ... }
            if "direction" in sd_data:
                success_vec_tensor = sd_data["direction"][args.success_key].float()
            elif "activations" in sd_data:
                 # It might be an activation vector file?
                 # Assuming success direction format for now as per other scripts
                 print("Warning: unexpected format, looking for 'direction' key")
            else:
                 # Maybe raw tensor dict?
                 pass
                 
            if success_vec_tensor is not None:
                print(f"✅ Success vector loaded (norm={success_vec_tensor.norm():.2f})")
        except Exception as e:
            print(f"Error loading success vector: {e}")
            sys.exit(1)

    # Counters
    counts = {cat: 0 for cat in CATEGORIES}
    errors = 0

    for dataset_name in args.datasets:
        print(f"\\n{'=' * 60}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'=' * 60}")

        for layer in args.layers:
            # Compute concept vectors once per (dataset, layer)
            print(f"\\n  ⏳ Computing concept vectors at layer {layer} …")
            vectors = compute_concept_vector(model, tokenizer, dataset_name, layer)
            capture_layer = args.capture_layer

            for concept, (vec_last, vec_avg) in vectors.items():
                steering_vector = vec_avg if args.vec_type == "avg" else vec_last

                for coeff in args.coeffs:
                    print(f"\\n── {concept} | layer={layer} | coeff={coeff} | "
                          f"mlp=L{args.mlp_layer} {mode_desc} ──")

                    # 1. Inject concept + head intervention and generate
                    try:
                        response, activations = inject_and_capture_activations(
                            model, tokenizer, steering_vector, layer, capture_layer,
                            coeff=coeff, max_new_tokens=args.max_new_tokens,
                            mlp_layer=args.mlp_layer,
                            mlp_mode=args.mlp_mode,
                            mlp_coeff=args.mlp_coeff,
                            success_vector=success_vec_tensor,
                            success_layer=args.success_layer,
                            success_coeff=args.success_coeff,
                            skip_concept=args.skip_concept,
                        )
                    except Exception as e:
                        print(f"  ⚠  Injection error: {e}")
                        import traceback
                        traceback.print_exc()
                        errors += 1
                        continue

                    print(f"  Response: {response[:120]}{'…' if len(response) > 120 else ''}")

                    # 2. Classify with judges
                    category = classify_response(response, concept)
                    counts[category] += 1

                    icons = {
                        "not_detected": "⚫",
                        "detected_opposite": "",    # Red for opposite
                        "detected_orthogonal": "🟠",  # Orange for unrelated/orthogonal
                        "detected_parallel": "",    # Yellow for close/parallel
                        "detected_correct": "🟢",     # Green for correct
                    }
                    icon = icons.get(category, "❓")
                    print(f"  {icon}  Category: {category}")

                    # 3. Save activations
                    out_dir = category_dirs[category]

                    # Construct filename for MLP layers
                    if isinstance(args.mlp_layer, list) and len(args.mlp_layer) > 1:
                        mlp_str = "mlp_L" + "_".join(str(l) for l in args.mlp_layer)
                    else:
                        ml = args.mlp_layer[0] if isinstance(args.mlp_layer, list) else args.mlp_layer
                        mlp_str = f"mlp{ml}"

                    filename = (f"{concept}_layer{capture_layer}_coeff{coeff}"
                                f"_{args.vec_type}_{mlp_str}.pt")
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
                        "mlp_layer": args.mlp_layer,
                        "mlp_mode": args.mlp_mode,
                        "mlp_coeff": args.mlp_coeff,
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
    print(f"\\n{'=' * 60}")
    print(f"  RESULTS  (concept@L{args.layers}, coeffs={args.coeffs}, "
          f"mlp=L{args.mlp_layer} {mode_desc})")
    print(f"{'=' * 60}")
    for cat in CATEGORIES:
        pct = counts[cat] / total * 100 if total > 0 else 0
        print(f"  {cat:25s}: {counts[cat]:3d}  ({pct:5.1f}%)")
    print(f"  {'errors':25s}: {errors:3d}")
    print(f"  {'total':25s}: {total:3d}")
    print(f"  Saved to: {save_root.resolve()}")
    print(f"{'=' * 60}\\n")

    # Close log
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()
    print(f"📝 Log saved to {save_root / 'run.log'}")


if __name__ == "__main__":
    main()
