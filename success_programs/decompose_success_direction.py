"""Decompose the success direction into per-head, per-layer contributions.

Injects BOTH the concept vector (at --layers) and the success direction
(at --success_layer, default 17), then measures how much each attention
head and MLP across ALL layers (0..max) contributes to the success direction.

  head contribution[l,h] = (o_proj_weight_slice @ head_h_attn_output) · success_dir
  mlp  contribution[l]   = mlp_output · success_dir

Usage:
    python decompose_success_direction.py --layers 16 --coeffs 10 --capture_layer 17
    python decompose_success_direction.py --layers 16 --coeffs 10 --datasets test_data
"""

import argparse
import sys
import os
import torch
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer

from compute_concept_vector_utils import compute_concept_vector
from all_prompts import get_anthropic_reproduce_messages


# ── Decomposition ────────────────────────────────────────────────────────────

def decompose_contributions(
    model, tokenizer, steering_vector, layer_to_inject,
    capture_layer, success_direction, coeff=10.0, max_new_tokens=100,
    success_vector=None, success_layer=None, success_coeff=1.0,
):
    """
    Inject a concept vector (+ optional success vector), run generation,
    and decompose per-head and per-MLP contributions across ALL layers
    projected onto the success direction.

    Returns:
        response  (str)
        head_proj (Tensor [num_layers, num_heads])
        mlp_proj  (Tensor [num_layers])
    """
    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    # ── Prepare steering vector ──────────────────────────────────────────
    sv = steering_vector / torch.norm(steering_vector, p=2)
    if not isinstance(sv, torch.Tensor):
        sv = torch.tensor(sv, dtype=torch.float32)
    sv = sv.to(device)
    if sv.dim() == 1:
        sv = sv.unsqueeze(0).unsqueeze(0)
    elif sv.dim() == 2:
        sv = sv.unsqueeze(0)

    # ── Prepare success direction vector for injection ────────────────────
    if success_vector is not None:
        sdv = success_vector / torch.norm(success_vector, p=2)
        if not isinstance(sdv, torch.Tensor):
            sdv = torch.tensor(sdv, dtype=torch.float32)
        sdv = sdv.to(device)
        if sdv.dim() == 1:
            sdv = sdv.unsqueeze(0).unsqueeze(0)
        elif sdv.dim() == 2:
            sdv = sdv.unsqueeze(0)

    # ── Prepare success direction for projection (unit normalised) ───────
    sd = success_direction.float().to(device)
    sd = sd / sd.norm()

    # ── Build Anthropic prompt ───────────────────────────────────────────
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

    # ── Storage (overwritten each forward; final values = last gen token) ─
    captured_pre_proj = {}   # layer → [hidden_size]  (input to o_proj, last token)
    captured_mlp = {}        # layer → [hidden_size]  (MLP output, last token)

    handles = []

    # ── Hook: capture o_proj input for per-head decomposition (ALL layers) ─
    for l in range(num_layers):
        def make_oproj_hook(layer_idx):
            def hook(module, input, output):
                inp = input[0] if isinstance(input, tuple) else input
                # Always overwrite → ends up with the last generated token
                captured_pre_proj[layer_idx] = inp[0, -1, :].detach().cpu()
            return hook
        handles.append(
            model.model.layers[l].self_attn.o_proj.register_forward_hook(
                make_oproj_hook(l)
            )
        )

    # ── Hook: capture MLP output (ALL layers) ────────────────────────────
    for l in range(num_layers):
        def make_mlp_hook(layer_idx):
            def hook(module, input, output):
                captured_mlp[layer_idx] = output[0, -1, :].detach().cpu()
            return hook
        handles.append(
            model.model.layers[l].mlp.register_forward_hook(make_mlp_hook(l))
        )

    # ── Hook: concept injection (same as save_success_vectors.py) ────────
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

    handles.append(
        model.model.layers[layer_to_inject].register_forward_hook(injection_hook)
    )

    # ── Hook: success direction injection (at success_layer) ─────────────
    if success_vector is not None and success_layer is not None:
        def success_injection_hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            sdv_cast = sdv.to(device=hidden_states.device, dtype=hidden_states.dtype)
            batch_size, seq_len, hidden_dim = hidden_states.shape
            sdv_expanded = sdv_cast.expand(batch_size, seq_len, -1)
            modified = hidden_states + success_coeff * sdv_expanded
            return (modified,) + output[1:] if isinstance(output, tuple) else modified

        handles.append(
            model.model.layers[success_layer].register_forward_hook(success_injection_hook)
        )

    # ── Generate ─────────────────────────────────────────────────────────
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    for h in handles:
        h.remove()

    # ── Decode ───────────────────────────────────────────────────────────
    generated_ids = out[0][prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # ── Compute projections (ALL layers) ──────────────────────────────────
    sd_cpu = sd.cpu().float()
    head_proj = torch.zeros(num_layers, num_heads)
    mlp_proj = torch.zeros(num_layers)

    for l in range(num_layers):
        # Per-head: slice the pre-projection output and push through o_proj
        if l in captured_pre_proj:
            pre_proj = captured_pre_proj[l].float()              # [hidden_size]
            o_weight = model.model.layers[l].self_attn.o_proj.weight.cpu().float()

            for h in range(num_heads):
                head_in = pre_proj[h * head_dim : (h + 1) * head_dim]        # [head_dim]
                w_slice = o_weight[:, h * head_dim : (h + 1) * head_dim]     # [hidden, head_dim]
                head_out = w_slice @ head_in                                  # [hidden]
                head_proj[l, h] = (head_out @ sd_cpu).item()

        # MLP
        if l in captured_mlp:
            mlp_proj[l] = (captured_mlp[l].float() @ sd_cpu).item()

    return response, head_proj, mlp_proj


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Decompose success direction into per-head, per-layer contributions"
    )
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layers", type=int, nargs="+", default=[16],
                        help="Layers to inject concept vector at")
    parser.add_argument("--capture_layer", type=int, default=17)
    parser.add_argument("--coeffs", type=float, nargs="+", default=[10.0])
    parser.add_argument("--vec_type", type=str, default="avg",
                        choices=["avg", "last"])
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["simple_data", "complex_data"])
    parser.add_argument("--success_direction", type=str,
                        default=str(PROJECT_ROOT / "success_results/run_02_10_26_20_39/success_direction.pt"))
    parser.add_argument("--success_key", type=str, default="last_token",
                        choices=["last_token", "prompt_mean", "generation_mean"])
    parser.add_argument("--success_layer", type=int, default=17,
                        help="Layer to inject the success direction vector at")
    parser.add_argument("--success_coeff", type=float, default=0.0,
                        help="Coefficient for success direction injection (0=off)")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default=str(PROJECT_ROOT/"success_results"))
    args = parser.parse_args()

    # ── Load success direction ───────────────────────────────────────────
    sd_path = Path(args.success_direction)
    if not sd_path.exists():
        print(f"ERROR: {sd_path} not found"); return
    sd_data = torch.load(sd_path, map_location="cpu", weights_only=False)
    success_dir = sd_data["direction"][args.success_key].float()
    # Also load the raw (non-normalised) direction for injection
    success_vec_for_injection = success_dir.clone() if args.success_coeff != 0 else None
    print(f"🧭 Success direction: key={args.success_key}, ‖d‖={success_dir.norm():.4f}")
    if args.success_coeff != 0:
        print(f"   → will inject at layer {args.success_layer} with coeff={args.success_coeff}")

    # ── Load model ───────────────────────────────────────────────────────
    print(f"\n⏳ Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_heads = model.config.num_attention_heads
    total_layers = model.config.num_hidden_layers
    print(f"✅ Model on {device}  ({total_layers} layers, "
          f"{num_heads} heads, head_dim={model.config.hidden_size // num_heads})\n")

    # ── Run decomposition for each concept ───────────────────────────────
    all_head_proj = []
    all_mlp_proj = []
    concept_names = []

    for dataset_name in args.datasets:
        for layer in args.layers:
            print(f"⏳ Computing concept vectors: dataset={dataset_name}, layer={layer}")
            vectors = compute_concept_vector(model, tokenizer, dataset_name, layer)

            for concept, (vec_last, vec_avg) in vectors.items():
                steering_vector = vec_avg if args.vec_type == "avg" else vec_last

                for coeff in args.coeffs:
                    print(f"\n── {concept} | inject@L{layer} | coeff={coeff} ──")
                    response, head_proj, mlp_proj = decompose_contributions(
                        model, tokenizer, steering_vector, layer,
                        args.capture_layer, success_dir,
                        coeff=coeff, max_new_tokens=args.max_new_tokens,
                        success_vector=success_vec_for_injection,
                        success_layer=args.success_layer,
                        success_coeff=args.success_coeff,
                    )
                    print(f"  Response: {response[:100]}{'…' if len(response) > 100 else ''}")

                    all_head_proj.append(head_proj)
                    all_mlp_proj.append(mlp_proj)
                    concept_names.append(concept)

    if not all_head_proj:
        print("No concepts processed."); return

    # ── Aggregate ────────────────────────────────────────────────────────
    stacked_head = torch.stack(all_head_proj)   # [N, layers, heads]
    stacked_mlp  = torch.stack(all_mlp_proj)    # [N, layers]
    mean_head = stacked_head.mean(dim=0)
    mean_mlp  = stacked_mlp.mean(dim=0)
    n_concepts = len(concept_names)
    n_layers = total_layers

    # ── Print full table ─────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  PER-HEAD PROJECTION ONTO SUCCESS DIRECTION  (avg over {n_concepts} concepts)")
    print(f"  Positive = pushes toward success  |  Negative = pushes toward failure")
    print(f"{'=' * 80}\n")

    # Compact header
    header = f"{'Layer':>6} |"
    for h in range(num_heads):
        header += f"  H{h:02d}"
    header += f"  | {'MLP':>7}"
    print(header)
    print("-" * len(header))

    for l in range(n_layers):
        marker = " ◆" if l == args.capture_layer else "  " if l in args.layers else "  "
        row = f"{marker}L{l:02d}  |"
        for h in range(num_heads):
            v = mean_head[l, h].item()
            row += f" {v:+.2f}" if abs(v) < 10 else f" {v:+.1f}"
        row += f"  | {mean_mlp[l].item():+7.2f}"
        print(row)
    print(f"\n  ◆ = capture layer ({args.capture_layer})")

    # ── Top contributing heads ───────────────────────────────────────────
    flat = mean_head.flatten()
    top_k = min(20, flat.numel())
    top_idx = flat.abs().topk(top_k).indices

    print(f"\n{'=' * 60}")
    print(f"  TOP {top_k} HEADS (by |projection|)")
    print(f"{'=' * 60}")
    for rank, idx in enumerate(top_idx):
        l = idx.item() // num_heads
        h = idx.item() % num_heads
        val = flat[idx].item()
        direction = "→ success" if val > 0 else "→ failure"
        print(f"  #{rank+1:2d}  L{l:02d}.H{h:02d}  {val:+.4f}  {direction}")

    # ── Top MLP layers ───────────────────────────────────────────────────
    mlp_top = mean_mlp.abs().topk(min(10, n_layers))
    print(f"\n{'=' * 60}")
    print(f"  TOP MLP LAYERS (by |projection|)")
    print(f"{'=' * 60}")
    for rank, (_, idx) in enumerate(zip(mlp_top.values, mlp_top.indices)):
        val = mean_mlp[idx].item()
        direction = "→ success" if val > 0 else "→ failure"
        print(f"  #{rank+1:2d}  L{idx:02d} MLP  {val:+.4f}  {direction}")

    # ── Per-layer summary (attn total + MLP) ─────────────────────────────
    layer_attn_total = mean_head.sum(dim=1)   # [layers]
    print(f"\n{'=' * 60}")
    print(f"  PER-LAYER TOTAL CONTRIBUTION")
    print(f"{'=' * 60}")
    print(f"  {'Layer':>6} | {'Attn Total':>10} | {'MLP':>10} | {'Combined':>10}")
    print(f"  {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10}")
    for l in range(n_layers):
        a = layer_attn_total[l].item()
        m = mean_mlp[l].item()
        marker = "◆" if l == args.capture_layer else " "
        print(f"  {marker} L{l:02d}  | {a:+10.4f} | {m:+10.4f} | {a+m:+10.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    save_path = Path(args.save_dir) / "head_layer_contributions.pt"
    torch.save({
        "mean_head_projections": mean_head,
        "mean_mlp_projections": mean_mlp,
        "per_concept_head_projections": stacked_head,
        "per_concept_mlp_projections": stacked_mlp,
        "concept_names": concept_names,
        "success_direction_path": str(sd_path),
        "success_key": args.success_key,
        "capture_layer": args.capture_layer,
        "inject_layers": args.layers,
        "coeffs": args.coeffs,
    }, save_path)
    print(f"\n💾 Saved to {save_path}")


if __name__ == "__main__":
    main()
