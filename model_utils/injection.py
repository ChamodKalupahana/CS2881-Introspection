import torch
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from functools import partial
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from linear_probe/calibation_correct_vs_detected_correct/unified_prompts.py import load_unified_prompt_for_detection

# Suppress transformers warnings
transformers.logging.set_verbosity_error()

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from original_paper.all_prompts import get_anthropic_reproduce_messages
from original_paper.inject_concept_vector import get_model_type, format_inference_prompt

# ── Injection + activation capture ───────────────────────────────────────────

def inject_hook_fn(module, input, output, state, steering_vector, layer_to_inject, coeff, prompt_length, injection_start_token, assistant_tokens_only):
    """
    Handles the injection of the steering vector into the residual stream.
    """
    hidden_states = output[0] if isinstance(output, tuple) else output
    steer = steering_vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
    batch_size, seq_len, hidden_dim = hidden_states.shape
    
    # Initialize expansion mask
    steer_expanded = torch.zeros_like(hidden_states)
    
    # Detect generation phase
    is_generating = (seq_len == 1 and state["prompt_processed"]) or (seq_len > prompt_length)
    
    if seq_len == prompt_length:
        state["prompt_processed"] = True
    
    if injection_start_token is not None:
        if is_generating:
            # During generation: inject at the newly generated token
            steer_expanded[:, -1:, :] = steer
        else:
            # During prompt processing: inject from injection_start_token onwards
            start_idx = max(0, injection_start_token)
            if start_idx < seq_len:
                steer_expanded[:, start_idx:, :] = steer.expand(batch_size, seq_len - start_idx, -1)
    elif not assistant_tokens_only or is_generating:
        # Fallback or assistant-only generation behavior
        steer_expanded = steer.expand(batch_size, seq_len, -1) if seq_len != 1 else steer
        if seq_len == 1:
            steer_expanded = steer

    modified = hidden_states + coeff * steer_expanded
    return (modified,) + output[1:] if isinstance(output, tuple) else modified

def capture_hook_fn(module, input, output, layer_idx, state, prompt_length, start_position : int = -6):
    """
    Captures residual stream activations at positions -6 (last 6 of prompt) to 0 (first of generation).
    """
    hidden_states = output[0] if isinstance(output, tuple) else output
    seq_len = hidden_states.shape[1]
    
    # 1. Capture end of prompt (positions -6 to -1)
    if seq_len == prompt_length:
        # Prompt prefill pass - ensure we only capture once
        if layer_idx not in state["prompt_captured_layers"]:
            state["activations"][layer_idx].append(hidden_states[:, start_position:, :].detach().cpu())
            state["prompt_captured_layers"].add(layer_idx)
        
    # 2. Capture start of generation (position 0)
    elif seq_len == 1:
        # Decoding pass - only capture the very first token for each layer
        if layer_idx not in state["gen_captured_layers"]:
            state["activations"][layer_idx].append(hidden_states.detach().cpu())
            state["gen_captured_layers"].add(layer_idx)

    return output

def inject_concept_vector(model, tokenizer, steering_vector, layer_to_inject, coeff=12.0, inference_prompt=None, assistant_tokens_only=False, max_new_tokens=20, injection_start_token=None):
    """
    Inject concept vectors and capture activations at layers > layer_to_inject.
    Positions captured: -6 (end of prompt) to 0 (first generation token).
    """
    device = next(model.parameters()).device
    
    # Normalize and prepare steering vector [1, 1, hidden_dim]
    steering_vector = steering_vector / torch.norm(steering_vector, p=2)
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
            raise ValueError
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
                state=state)
    )

    # Register Capture Hooks for all layers > layer_to_inject
    cap_handles = []
    num_layers = len(model.model.layers)
    for i in range(layer_to_inject + 1, num_layers):
        h = model.model.layers[i].register_forward_hook(
            partial(capture_hook_fn, layer_idx=i, state=state, prompt_length=prompt_length)
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
    positions = [-6, -5, -4, -3, -2, -1, 0]
    
    for layer_idx, slices in state["activations"].items():
        if len(slices) == 2:
            # slices[0] is prompt [-6:-1], slices[1] is generation [0]
            cat_tensor = torch.cat(slices, dim=1) # [1, 7, hidden_dim]
            for i, pos in enumerate(positions):
                final_activations[(layer_idx, pos)] = cat_tensor[0, i].detach().cpu()
        elif len(slices) == 1:
             # Identify if we captured only prompt or only generation
             # If captured during prefill (shape[1] > 1), it's prompt. 
             # If seq_len was 1, it's generation.
             tensor = slices[0]
             if tensor.shape[1] > 1:
                 # Map end of prompt (up to 6 tokens)
                 for i in range(tensor.shape[1]):
                     pos = positions[i]
                     final_activations[(layer_idx, pos)] = tensor[0, i].detach().cpu()
             else:
                 # Single generated token
                 final_activations[(layer_idx, 0)] = tensor[0, 0].detach().cpu()

    # Decode response
    generated_ids = out[0][prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return response, final_activations

def inject_and_capture_anthropic(model, tokenizer, steering_vector, layer_to_inject, coeff=12.0, max_new_tokens=200):
    """
    High-level wrapper that implements the Anthropic reproduction 
    introspection experiment workflow.
    """
    # 1. Prepare Messages
    messages = get_anthropic_reproduce_messages()
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 2. Calculate injection start token (at "\n\nTrial 1")
    trial_start_text = "\n\nTrial 1"
    trial_start_pos = prompt_text.find(trial_start_text)
    if trial_start_pos != -1:
        prefix = prompt_text[:trial_start_pos]
        injection_start_token = len(tokenizer.encode(prefix, add_special_tokens=False))
    else:
        injection_start_token = None

    # 3. Inject and Capture
    return inject_concept_vector(
        model=model,
        tokenizer=tokenizer,
        steering_vector=steering_vector,
        layer_to_inject=layer_to_inject,
        coeff=coeff,
        inference_prompt=prompt_text,
        max_new_tokens=max_new_tokens,
        injection_start_token=injection_start_token
    )

def inject_and_capture_unified(model, tokenizer, steering_vector, layer_to_inject, coeff=12.0, max_new_tokens=200):
    """
    High-level wrapper that implements the Anthropic reproduction 
    introspection experiment workflow.
    """
    # 1. Prepare Messages
    messages = get_anthropic_reproduce_messages()
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 2. Calculate injection start token (at "\n\nTrial 1")
    trial_start_text = "\n\nTrial 1"
    trial_start_pos = prompt_text.find(trial_start_text)
    if trial_start_pos != -1:
        prefix = prompt_text[:trial_start_pos]
        injection_start_token = len(tokenizer.encode(prefix, add_special_tokens=False))
    else:
        injection_start_token = None

    # 3. Inject and Capture
    return inject_concept_vector(
        model=model,
        tokenizer=tokenizer,
        steering_vector=steering_vector,
        layer_to_inject=layer_to_inject,
        coeff=coeff,
        inference_prompt=prompt_text,
        max_new_tokens=max_new_tokens,
        injection_start_token=injection_start_token
    )

def capture_calibation_correct_unified(model, tokenizer, steering_vector, layer_to_inject, coeff=12.0, max_new_tokens=200):
    """
    High-level wrapper that implements the Anthropic reproduction 
    introspection experiment workflow.
    """
    # 1. Prepare Messages
    messages = load_unified_prompt_for_detection()
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 2. Calculate injection start token (at "\n\nTrial 1")
    trial_start_text = "\n\nTrial 1"
    trial_start_pos = prompt_text.find(trial_start_text)
    if trial_start_pos != -1:
        prefix = prompt_text[:trial_start_pos]
        injection_start_token = len(tokenizer.encode(prefix, add_special_tokens=False))
    else:
        injection_start_token = None

    # 3. Inject and Capture
    return inject_concept_vector(
        model=model,
        tokenizer=tokenizer,
        steering_vector=steering_vector,
        layer_to_inject=layer_to_inject,
        coeff=coeff,
        inference_prompt=prompt_text,
        max_new_tokens=max_new_tokens,
        injection_start_token=injection_start_token
    )