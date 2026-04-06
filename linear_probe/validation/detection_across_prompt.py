import torch
import os
import sys
import argparse
import re
from pathlib import Path
from collections import defaultdict
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Internal project imports
from model_utils.injection import inject_hook_fn
from original_paper.inject_concept_vector import get_model_type, format_inference_prompt
from original_paper.all_prompts import get_anthropic_reproduce_messages
from original_paper.compute_concept_vector_utils import compute_concept_vector

def capture_all_hook_fn(module, input, output, layer_idx, state, prompt_length):
    """
    Captures all residual stream activations for every token across prefill and decoding passes.
    """
    hidden_states = output[0] if isinstance(output, tuple) else output
    # Just append every output for this layer to state["activations"]
    state["activations"][layer_idx].append(hidden_states.detach().cpu())
    return output

def inject_concept_vector(model, tokenizer, steering_vector, layer_to_inject, coeff=12.0, inference_prompt=None, assistant_tokens_only=False, max_new_tokens=20, injection_start_token=None, inject=True):
    """
    Inject concept vectors and capture activations at layers > layer_to_inject.
    Captures ALL token positions (entire prompt + entire generation).
    """
    device = next(model.parameters()).device
    
    # Normalize and prepare steering vector [1, 1, hidden_dim]
    steering_vector = steering_vector / (torch.norm(steering_vector, p=2) + 1e-9)
    if not isinstance(steering_vector, torch.Tensor):
        steering_vector = torch.tensor(steering_vector, dtype=torch.float32)
    steering_vector = steering_vector.to(device)
    if steering_vector.dim() == 1:
        steering_vector = steering_vector.unsqueeze(0).unsqueeze(0)
    elif steering_vector.dim() == 2:
        steering_vector = steering_vector.unsqueeze(0)

    # Format Prompt
    model_type = get_model_type(tokenizer)
    if inference_prompt and ("<|start_header_id|>" in inference_prompt or "<|im_start|>" in inference_prompt):
        prompt = inference_prompt
    else:
        if not inference_prompt:
            raise ValueError("No inference_prompt provided")
        prompt = format_inference_prompt(model_type, inference_prompt)
    
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_length = inputs.input_ids.shape[1]
    
    # Shared execution state
    state = {
        "prompt_processed": False,
        "activations": defaultdict(list),
        "prompt_captured_layers": set(),
        "gen_captured_layers": set()
    }

    # Register Injection Hook
    inj_handle = model.model.layers[layer_to_inject].register_forward_hook(
        partial(inject_hook_fn, 
                steering_vector=steering_vector, 
                layer_to_inject=layer_to_inject, 
                coeff=coeff, 
                prompt_length=prompt_length, 
                injection_start_token=injection_start_token, 
                assistant_tokens_only=assistant_tokens_only,
                state=state,
                inject=inject)
    )

    # Register Capture Hooks for all layers > layer_to_inject
    cap_handles = []
    num_layers = len(model.model.layers)
    for i in range(layer_to_inject + 1, num_layers):
        h = model.model.layers[i].register_forward_hook(
            partial(capture_all_hook_fn, layer_idx=i, state=state, prompt_length=prompt_length)
        )
        cap_handles.append(h)

    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
    finally:
        inj_handle.remove()
        for h in cap_handles:
            h.remove()

    # Post-process activations: Map to {(layer, position): d_model_tensor}
    final_activations = {}
    
    for layer_idx, slices in state["activations"].items():
        # Consolidate all slices into a single sequence tensor
        full_tensor = torch.cat(slices, dim=1) # [1, total_seq_len, hidden_dim]
        total_len = full_tensor.shape[1]
        for pos in range(total_len):
            final_activations[(layer_idx, pos)] = full_tensor[0, pos].detach().cpu()

    # Decode response
    generated_ids = out[0][prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return response, final_activations, out[0].detach().cpu()


def extract_layer_from_filename(path):
    if path is None: return None
    match = re.search(r"L(\d+)", Path(path).name)
    return int(match.group(1)) if match else None

def resolve_probe_path(probe_path):
    """Try to find the probe path relative to current dir, project root, or linear_probe/."""
    path = Path(probe_path)
    if path.exists(): return path
    p_root = PROJECT_ROOT / probe_path
    if p_root.exists(): return p_root
    p_lp = PROJECT_ROOT / "linear_probe" / probe_path
    if p_lp.exists(): return p_lp
    return path

def main():
    parser = argparse.ArgumentParser(description="Analyze probe detection across prompt positions with dynamic steering.")
    parser.add_argument("--probe1", type=str, required=True, help="Path to the first probe vector.")
    parser.add_argument("--concept", type=str, default="Magnetism", help="Concept from simple_data.json to use for steering.")
    parser.add_argument("--dataset", type=str, default="simple_data", help="Dataset to load concept from.")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name.")
    parser.add_argument("--layer", type=int, default=15, help="Layer to inject the steering vector.")
    parser.add_argument("--coeff", type=float, default=12.0, help="Steering coefficient.")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Tokens to generate.")
    args = parser.parse_args()

    # Load Probe
    resolved_path = resolve_probe_path(args.probe1)
    if not resolved_path.exists():
        print(f"❌ Error: Could not find probe at {args.probe1}")
        sys.exit(1)

    probe_layer = extract_layer_from_filename(resolved_path) 
    if probe_layer is None:
        print("❌ Error: Could not extract layer from probe filename.")
        sys.exit(1)

    probe = torch.load(resolved_path, map_location='cpu', weights_only=True).float()

    # Load Model
    print(f"⏳ Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"✅ Model loaded\n")

    # Compute Steering Vector on the fly (Concept Magnetism)
    print(f"🚀 Computing steering vector for '{args.concept}' at layer {args.layer} from '{args.dataset}'...")
    all_results = compute_concept_vector(model, tokenizer, args.dataset, args.layer)
    
    # Extract specific concept vector (using index 1 for average)
    if args.concept in all_results:
        steering_vec = all_results[args.concept][1].float().to(model.device)
    else:
        # Fallback to case-insensitive check
        found = False
        for k in all_results.keys():
            if k.lower() == args.concept.lower():
                steering_vec = all_results[k][1].float().to(model.device)
                args.concept = k
                found = True
                break
        if not found:
            print(f"❌ Error: Concept '{args.concept}' not found. Available: {list(all_results.keys())}")
            sys.exit(1)

    # Prepare Messages
    messages = get_anthropic_reproduce_messages()
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Calculate injection start token (at "\n\nTrial 1")
    trial_start_text = "\n\nTrial 1"
    trial_start_pos = prompt_text.find(trial_start_text)
    if trial_start_pos != -1:
        prefix = prompt_text[:trial_start_pos]
        injection_start_token = len(tokenizer.encode(prefix, add_special_tokens=False))
        print(f"📍 Detected Trial Start at token index: {injection_start_token}")
    else:
        injection_start_token = None
        print("⚠️ Warning: Could not detect Trial Start in prompt.")

    response, full_activations, all_tokens = inject_concept_vector(
        model=model,
        tokenizer=tokenizer,
        steering_vector=steering_vec,
        layer_to_inject=args.layer,
        coeff=args.coeff,
        inference_prompt=prompt_text,
        max_new_tokens=args.max_new_tokens,
        injection_start_token=injection_start_token,
        inject=True
    )

    # Decode tokens for labels
    token_labels = []
    for tid in all_tokens:
        label = tokenizer.decode([tid])
        label = label.replace('\n', '\\n').replace('\r', '\\r')
        if not label.strip():
            label = f"[{tid}]"
        token_labels.append(label)

    # Filter activations for the probe's layer and compute projections
    layer_activations = {pos: act for (l, pos), act in full_activations.items() if l == probe_layer}
    positions = sorted(layer_activations.keys())
    projections = []
    
    # Normalize probe for cosine similarity
    probe_unit = probe.flatten() / (torch.norm(probe.flatten()) + 1e-9)
    
    for pos in positions:
        act = layer_activations[pos].flatten().float()
        proj = torch.dot(probe_unit, act)
        projections.append(proj.item())

    # Align sequences to solve length mismatch (e.g. 166 vs 167)
    num_to_plot = min(len(positions), len(token_labels))
    positions = positions[:num_to_plot]
    projections = projections[:num_to_plot]
    token_labels = token_labels[:num_to_plot]
    
    # Ignore the first token position (often BOS or high-variance noise)
    if len(positions) > 1:
        positions = positions[1:]
        projections = projections[1:]
        token_labels = token_labels[1:]

    # Plot
    print(f"📊 Plotting projections for probe at layer {probe_layer}...")
    plt.figure(figsize=(24, 8))
    plt.plot(positions, projections, marker='o', linestyle='-', markersize=4, label='Probe Projection')
    
    if injection_start_token is not None:
        # Adjustment: if we truncated, the injection_start_token might need to stay within bounds
        safe_injection_token = min(injection_start_token, num_to_plot - 1)
        plt.axvline(x=safe_injection_token, color='red', linestyle='--', alpha=0.7, label=f'Trial Start (Inject: {args.concept})')
    
    plt.xticks(positions, token_labels, rotation=90, fontsize=8)
    plt.xlabel('Tokens / Sequence Position')
    plt.ylabel('Dot Product Projection (Unit Probe)')
    plt.title(f'Probe Detection Across Prompt (Layer {probe_layer}, Concept: {args.concept} @ L{args.layer}, C{args.coeff})')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    # Ensure centralized output relative to project root
    plot_dir = PROJECT_ROOT / "plots/linear_probe/validation"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"detection_across_prompt_probe_layer{probe_layer}.png"
    
    plt.savefig(plot_path)
    print(f"✅ Success! Plot saved to: {plot_path}")
    print(f"\n🗣️ Response Captured:\n{'-'*30}\n{response}\n{'-'*30}")

if __name__ == "__main__":
    main()
