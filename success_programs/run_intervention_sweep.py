"""
Run a sweep of intervention experiments for head(s) and MLP layers
with a single model load, collecting category counts into a summary CSV.

Experiments (dynamically built per head pair and MLP layer):
  1. control – concept injection only (no head/MLP intervention)
  For each --head_layer/--head_index pair (L, H):
     L{L}H{H}_input_amplify / _input_amplify_no_concept / _ablate / _reverse
  For each --mlp_layer L:
     mlp{L}_input_amplify / _input_amplify_no_concept / _ablate / _reverse

Usage:
    python run_intervention_sweep.py --inject_coeff 5 \\
        --head_layer 17 31 --head_index 3 3 --head_coeff 5.0 \\
        --mlp_layer 17 21 29 --mlp_coeff 2.0 --dataset test_data
"""

import argparse, csv, sys, os, torch
from datetime import datetime
from pathlib import Path

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


# ── Classification (same as other scripts) ───────────────────────────────────

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
    stem = concept_lower[:max(3, len(concept_lower) - 3)]
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

        if category in ("detected_parallel", "detected_orthogonal") and has_stem and len(stem) >= 4:
            print(f"  [stem-upgrade] Found '{stem}' in response → detected_correct")
            category = "detected_correct"

        return category

    except Exception as e:
        print(f"  ⚠  Classification error: {e}")
        return "not_detected"


# ── Unified inject + intervene + generate ────────────────────────────────────

def run_experiment(
    model, tokenizer, steering_vector,
    inject_layer, inject_coeff, max_new_tokens=100,
    skip_concept=False,
    # Head intervention
    head_layer=None, head_index=None, head_mode=None, head_coeff=1.0,
    # MLP intervention
    mlp_layer=None, mlp_mode=None, mlp_coeff=1.0,
):
    """
    Run a single experiment: inject concept (optional), apply head and/or MLP
    intervention, generate, return response text.
    """
    device = next(model.parameters()).device
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    # ── Prepare steering vector ──────────────────────────────────────────
    sv = None
    if not skip_concept and steering_vector is not None:
        sv = steering_vector / torch.norm(steering_vector, p=2)
        if not isinstance(sv, torch.Tensor):
            sv = torch.tensor(sv, dtype=torch.float32)
        sv = sv.to(device)
        if sv.dim() == 1:
            sv = sv.unsqueeze(0).unsqueeze(0)
        elif sv.dim() == 2:
            sv = sv.unsqueeze(0)

    # Build prompt
    messages = get_anthropic_reproduce_messages()
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

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

    # ── Concept injection hook ───────────────────────────────────────────
    if sv is not None:
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

            modified = hidden_states + inject_coeff * steer_expanded
            return (modified,) + output[1:] if isinstance(output, tuple) else modified

    # ── Hook generators ──────────────────────────────────────────────────

    def get_head_pre_hook(h_idx, h_coeff, mode):
        def hook(module, input):
            inp = input[0].clone()
            start = h_idx * head_dim
            end = (h_idx + 1) * head_dim
            if mode == "ablate":
                inp[:, :, start:end] = 0.0
            elif mode == "reverse":
                inp[:, :, start:end] = inp[:, :, start:end] * (-h_coeff)
            else:
                inp[:, :, start:end] = inp[:, :, start:end] * h_coeff
            return (inp,)
        return hook

    def get_head_input_amplify_hook(h_idx, h_coeff, intervene_type="q"):
        num_kv_heads = getattr(model.config, "num_key_value_heads", num_heads)
        group_size = num_heads // num_kv_heads
        def hook(module, input, output):
            modified = output.clone()
            t_idx = h_idx if intervene_type == "q" else h_idx // group_size
            start = t_idx * head_dim
            end = (t_idx + 1) * head_dim
            modified[:, :, start:end] = modified[:, :, start:end] * h_coeff
            return modified
        return hook

    def get_mlp_output_hook(m_coeff, mode):
        def hook(module, input, output):
            if mode == "ablate":
                return torch.zeros_like(output)
            elif mode == "reverse":
                return output * (-m_coeff)
            else:
                return output * m_coeff
        return hook

    def get_mlp_input_hook(m_coeff):
        def hook(module, input, output):
            return output * m_coeff
        return hook

    # ── Register hooks ───────────────────────────────────────────────────
    handles = []

    if sv is not None:
        handles.append(
            model.model.layers[inject_layer].register_forward_hook(concept_injection_hook)
        )

    # Head intervention
    if head_layer is not None and head_index is not None and head_mode is not None:
        if head_mode in ["input_amplify", "both_amplify"]:
            attn = model.model.layers[head_layer].self_attn
            handles.append(attn.q_proj.register_forward_hook(
                get_head_input_amplify_hook(head_index, head_coeff, "q")))
            handles.append(attn.k_proj.register_forward_hook(
                get_head_input_amplify_hook(head_index, head_coeff, "k")))
            handles.append(attn.v_proj.register_forward_hook(
                get_head_input_amplify_hook(head_index, head_coeff, "v")))
        if head_mode in ["amplify", "ablate", "reverse", "both_amplify"]:
            handles.append(
                model.model.layers[head_layer].self_attn.o_proj.register_forward_pre_hook(
                    get_head_pre_hook(head_index, head_coeff, head_mode)
                )
            )

    # MLP intervention
    if mlp_layer is not None and mlp_mode is not None:
        mlp_module = model.model.layers[mlp_layer].mlp
        if mlp_mode in ["input_amplify", "both_amplify"]:
            handles.append(mlp_module.gate_proj.register_forward_hook(get_mlp_input_hook(mlp_coeff)))
            handles.append(mlp_module.up_proj.register_forward_hook(get_mlp_input_hook(mlp_coeff)))
        if mlp_mode in ["amplify", "ablate", "reverse", "both_amplify"]:
            handles.append(mlp_module.down_proj.register_forward_hook(
                get_mlp_output_hook(mlp_coeff, mlp_mode)))

    # ── Generate ─────────────────────────────────────────────────────────
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    for h in handles:
        h.remove()

    generated_ids = out[0][prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run head & MLP intervention sweep, collect category counts"
    )
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--inject_layer", type=int, default=16)
    parser.add_argument("--inject_coeff", type=float, default=5.0)
    parser.add_argument("--head_layer", type=int, nargs="+", default=[17],
                        help="Head layers to sweep (default: [17])")
    parser.add_argument("--head_index", type=int, nargs="+", default=[3],
                        help="Head indices paired with --head_layer (default: [3])")
    parser.add_argument("--head_coeff", type=float, default=5.0)
    parser.add_argument("--mlp_layer", type=int, nargs="+", default=[21],
                        help="MLP layers to sweep (default: [21])")
    parser.add_argument("--mlp_coeff", type=float, default=5.0)
    parser.add_argument("--dataset", type=str, default="test_data")
    parser.add_argument("--vec_type", type=str, default="avg", choices=["avg", "last"])
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "success_programs" / "sweep_results"))
    args = parser.parse_args()

    # Validate head args are paired
    if len(args.head_layer) != len(args.head_index):
        parser.error("--head_layer and --head_index must have the same number of values")

    # Define experiment configurations — per-head and per-MLP-layer experiments
    head_pairs = list(zip(args.head_layer, args.head_index))
    head_tag = "_".join(f"L{l}H{h}" for l, h in head_pairs)
    mlp_layers = args.mlp_layer
    mlp_tag = "-".join(str(l) for l in mlp_layers)

    experiments = [
        {
            "name": "control",
            "skip_concept": False,
            "head_layer": None, "head_index": None, "head_mode": None, "head_coeff": 1.0,
            "mlp_layer": None, "mlp_mode": None, "mlp_coeff": 1.0,
        },
    ]

    # Add head experiments for each specified (layer, head) pair
    for hl, hi in head_pairs:
        tag = f"L{hl}H{hi}"
        experiments.extend([
            {
                "name": f"{tag}_input_amplify",
                "skip_concept": False,
                "head_layer": hl, "head_index": hi,
                "head_mode": "input_amplify", "head_coeff": args.head_coeff,
                "mlp_layer": None, "mlp_mode": None, "mlp_coeff": 1.0,
            },
            {
                "name": f"{tag}_input_amplify_no_concept",
                "skip_concept": True,
                "head_layer": hl, "head_index": hi,
                "head_mode": "input_amplify", "head_coeff": args.head_coeff,
                "mlp_layer": None, "mlp_mode": None, "mlp_coeff": 1.0,
            },
            {
                "name": f"{tag}_ablate",
                "skip_concept": False,
                "head_layer": hl, "head_index": hi,
                "head_mode": "ablate", "head_coeff": 1.0,
                "mlp_layer": None, "mlp_mode": None, "mlp_coeff": 1.0,
            },
            {
                "name": f"{tag}_reverse",
                "skip_concept": False,
                "head_layer": hl, "head_index": hi,
                "head_mode": "reverse", "head_coeff": args.head_coeff,
                "mlp_layer": None, "mlp_mode": None, "mlp_coeff": 1.0,
            },
        ])

    # Add MLP experiments for each specified layer
    for ml in mlp_layers:
        experiments.extend([
            {
                "name": f"mlp{ml}_input_amplify",
                "skip_concept": False,
                "head_layer": None, "head_index": None, "head_mode": None, "head_coeff": 1.0,
                "mlp_layer": ml, "mlp_mode": "input_amplify", "mlp_coeff": args.mlp_coeff,
            },
            {
                "name": f"mlp{ml}_input_amplify_no_concept",
                "skip_concept": True,
                "head_layer": None, "head_index": None, "head_mode": None, "head_coeff": 1.0,
                "mlp_layer": ml, "mlp_mode": "input_amplify", "mlp_coeff": args.mlp_coeff,
            },
            {
                "name": f"mlp{ml}_ablate",
                "skip_concept": False,
                "head_layer": None, "head_index": None, "head_mode": None, "head_coeff": 1.0,
                "mlp_layer": ml, "mlp_mode": "ablate", "mlp_coeff": 1.0,
            },
            {
                "name": f"mlp{ml}_reverse",
                "skip_concept": False,
                "head_layer": None, "head_index": None, "head_mode": None, "head_coeff": 1.0,
                "mlp_layer": ml, "mlp_mode": "reverse", "mlp_coeff": args.mlp_coeff,
            },
        ])

    # ── Load model ───────────────────────────────────────────────────────
    print(f"⏳ Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model loaded on {device}")
    print(f"   Heads: {[f'L{l}.H{h}' for l, h in head_pairs]}  MLP layers: {mlp_layers}")
    print(f"   {len(experiments)} experiments per concept\n")

    # ── Compute concept vectors ──────────────────────────────────────────
    print(f"⏳ Computing concept vectors for '{args.dataset}' at layer {args.inject_layer} …")
    all_vectors = compute_concept_vector(model, tokenizer, args.dataset, args.inject_layer)
    concepts = list(all_vectors.keys())
    print(f"✅ Got {len(concepts)} concepts: {concepts}\n")

    # ── Prepare output ───────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%m_%d_%y_%H_%M")
    csv_path = out_dir / f"sweep_{head_tag}_MLP{mlp_tag}_{now}.csv"

    # Set up logging to file + terminal
    log_file = open(out_dir / "run.log", "w")
    sys.stdout = TeeLogger(log_file, sys.__stdout__)
    sys.stderr = TeeLogger(log_file, sys.__stderr__)

    print(f"📁 Output directory: {out_dir}")

    # ── Run sweep ────────────────────────────────────────────────────────
    # Results: {exp_name: {category: count}}
    results = {exp["name"]: {cat: 0 for cat in CATEGORIES} for exp in experiments}
    errors = {exp["name"]: 0 for exp in experiments}

    for concept in concepts:
        vec_last, vec_avg = all_vectors[concept]
        steering_vector = vec_avg if args.vec_type == "avg" else vec_last

        for exp in experiments:
            exp_name = exp["name"]
            print(f"\n{'─' * 60}")
            print(f"  {exp_name} | concept={concept}")
            print(f"{'─' * 60}")

            try:
                response = run_experiment(
                    model, tokenizer, steering_vector,
                    inject_layer=args.inject_layer,
                    inject_coeff=args.inject_coeff,
                    max_new_tokens=args.max_new_tokens,
                    skip_concept=exp["skip_concept"],
                    head_layer=exp["head_layer"],
                    head_index=exp["head_index"],
                    head_mode=exp["head_mode"],
                    head_coeff=exp["head_coeff"],
                    mlp_layer=exp["mlp_layer"],
                    mlp_mode=exp["mlp_mode"],
                    mlp_coeff=exp["mlp_coeff"],
                )
            except Exception as e:
                print(f"  ⚠  Error: {e}")
                errors[exp_name] += 1
                continue

            print(f"  Response: {response[:200]}{'…' if len(response) > 200 else ''}")

            category = classify_response(response, concept)
            results[exp_name][category] += 1

            icons = {
                "not_detected": "⚫",
                "detected_opposite": "🔴",
                "detected_orthogonal": "🟠",
                "detected_parallel": "🟡",
                "detected_correct": "🟢",
            }
            print(f"  {icons.get(category, '❓')}  {category}")

    # ── Write CSV ────────────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment"] + CATEGORIES + ["errors", "total"])
        for exp in experiments:
            name = exp["name"]
            counts = results[name]
            total = sum(counts.values())
            row = [name] + [counts[cat] for cat in CATEGORIES] + [errors[name], total]
            writer.writerow(row)

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"  SWEEP RESULTS | heads={[f'L{l}.H{h}' for l, h in head_pairs]}  mlp_layers={mlp_layers}")
    print(f"  inject_layer={args.inject_layer}  inject_coeff={args.inject_coeff}")
    print(f"  head_coeff={args.head_coeff}  mlp_coeff={args.mlp_coeff}")
    print(f"  dataset={args.dataset}  concepts={len(concepts)}")
    print(f"{'=' * 100}")
    header = f"  {'experiment':40s}"
    for cat in CATEGORIES:
        short = cat.replace("detected_", "d_")
        header += f" {short:>8s}"
    header += f" {'errors':>6s} {'total':>5s}"
    print(header)
    print(f"  {'─' * 100}")
    for exp in experiments:
        name = exp["name"]
        counts = results[name]
        total = sum(counts.values())
        line = f"  {name:40s}"
        for cat in CATEGORIES:
            line += f" {counts[cat]:8d}"
        line += f" {errors[name]:6d} {total:5d}"
        print(line)
    print(f"{'=' * 100}")
    print(f"  CSV saved to: {csv_path.resolve()}")
    print(f"{'=' * 100}\n")

    # Close log
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()
    print(f"📝 Log saved to {out_dir / 'run.log'}")


if __name__ == "__main__":
    main()
