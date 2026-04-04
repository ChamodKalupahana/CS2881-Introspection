import torch
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from original_paper.all_prompts import get_anthropic_reproduce_messages

# ── Injection + activation capture ───────────────────────────────────────────

# for just concept vector
def inject_and_capture_activations(
    model, tokenizer, steering_vector, layer_to_inject, capture_layers,
    coeff=10.0, max_new_tokens=100, skip_inject=False,
):
    """
    Inject a concept vector and capture hidden-state activations.

    Args:
        capture_layers: int or list[int]. If a single int, captures at that
                        layer. If a list, captures at every listed layer.

    Returns:
        response (str): the model's generated text
        activations (dict): if capture_layers is a single int, returns the
            standard dict (last_token, prompt_mean, generation_mean, …).
            If capture_layers is a list, returns a dict keyed by layer index,
            each containing the standard activation dict.
    """
    multi_layer = isinstance(capture_layers, (list, tuple))
    if not multi_layer:
        capture_layers = [capture_layers]

    device = next(model.parameters()).device

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

    # Register hooks & generate
    handles = []
    if not skip_inject:
        handles.append(model.model.layers[layer_to_inject].register_forward_hook(injection_hook))
    for cl in capture_layers:
        handles.append(model.model.layers[cl].register_forward_hook(make_capture_hook(cl)))

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    for h in handles:
        h.remove()

    # Decode
    generated_ids = out[0][prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Package activations per layer
    hidden_size = model.config.hidden_size

    def _package(cap):
        gen_stack = torch.stack(cap["generation"]) if cap["generation"] else torch.empty(0)
        prompt_acts = cap["prompt"] if cap["prompt"] is not None else torch.empty(0)
        return {
            "last_token":      gen_stack[-1] if len(gen_stack) > 0 else torch.zeros(hidden_size),
            "prompt_last_token": prompt_acts[-1] if prompt_acts.numel() > 0 else torch.zeros(hidden_size),
            "prompt_mean":     prompt_acts.mean(dim=0) if prompt_acts.numel() > 0 else torch.zeros(hidden_size),
            "generation_mean": gen_stack.mean(dim=0) if len(gen_stack) > 0 else torch.zeros(hidden_size),
            "all_prompt":      prompt_acts,
            "all_generation":  gen_stack,
        }

    if multi_layer:
        activations = {cl: _package(captured_per_layer[cl]) for cl in capture_layers}
    else:
        activations = _package(captured_per_layer[capture_layers[0]])

    return response, activations


# ── Steering helpers for testing and evaluation ──────────────────────────────────────────────────────────

def inject_and_steer_concept_and_probe(
    model, tokenizer, concept_vector, layer_to_inject,
    probe_vector, probe_layer,
    messages,
    coeff=10.0, alpha=0.0,
    max_new_tokens=200, skip_inject=False
):
    """
    Run inference with optional concept-vector injection AND optional
    probe-direction scaling, using the given messages as the prompt.

    Returns:
        (response_text, messages) — the generated text and the messages used.
    """
    device = next(model.parameters()).device

    # ── Normalised concept vector ────────────────────────────────────────
    sv = concept_vector / torch.norm(concept_vector, p=2)
    if not isinstance(sv, torch.Tensor):
        sv = torch.tensor(sv, dtype=torch.float32)
    sv = sv.to(device)
    if sv.dim() == 1:
        sv = sv.unsqueeze(0).unsqueeze(0)
    elif sv.dim() == 2:
        sv = sv.unsqueeze(0)

    # ── Normalised probe vector ──────────────────────────────────────────
    pv = probe_vector.clone().float()
    pv = pv / torch.norm(pv, p=2)
    pv = pv.to(device)

    # ── Build prompt from messages ───────────────────────────────────────
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Find injection start — use the user content position
    user_content = messages[-1]["content"]
    user_pos = formatted_prompt.find(user_content)
    if user_pos != -1:
        prefix = formatted_prompt[:user_pos]
        injection_start_token = len(tokenizer.encode(prefix, add_special_tokens=False))
    else:
        injection_start_token = None

    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_length = inputs.input_ids.shape[1]
    prompt_processed = False
    probe_prompt_processed = False

    # ── Concept injection hook ───────────────────────────────────────────
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

# ── UPDATED: Probe direction steering hook ───────────────────────────
    def probe_steering_hook(module, input, output):
        nonlocal probe_prompt_processed
        hidden_states = output[0] if isinstance(output, tuple) else output
        probe = pv.to(device=hidden_states.device, dtype=hidden_states.dtype)
        batch_size, seq_len, hidden_dim = hidden_states.shape
        probe_expanded = torch.zeros_like(hidden_states)

        if injection_start_token is not None:
            # Check if we are in generation phase using the exact same logic
            is_generating = (seq_len == 1 and probe_prompt_processed) or (seq_len > prompt_length)
            if seq_len == prompt_length:
                probe_prompt_processed = True
            
            if is_generating:
                # Generation phase: apply to the newly generated token
                probe_expanded[:, -1:, :] = probe
            else:
                # Prefill phase: apply to all tokens starting from injection_start_token
                start_idx = max(0, injection_start_token)
                if start_idx < seq_len:
                    probe_expanded[:, start_idx:, :] = probe.expand(
                        batch_size, seq_len - start_idx, -1
                    )
        else:
            # Fallback if no start token was found
            if seq_len == 1:
                probe_expanded[:, :, :] = probe

        # Apply the scaled probe vector across the expanded token mask
        modified = hidden_states + alpha * probe_expanded
        return (modified,) + output[1:] if isinstance(output, tuple) else modified

    # ── Register hooks & generate ────────────────────────────────────────
    handles = []
    if not skip_inject:
        handles.append(
            model.model.layers[layer_to_inject].register_forward_hook(injection_hook)
        )
    if alpha != 0.0:
        handles.append(
            model.model.layers[probe_layer].register_forward_hook(probe_steering_hook)
        )

    try:
        eos_id = tokenizer.eos_token_id
        if isinstance(eos_id, list):
            eos_id = eos_id[0]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=eos_id,
            )
        generated_ids = out[0][prompt_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    finally:
        for h in handles:
            h.remove()

    return response, messages

def inject_hook_fn(module, input, output):
    """
    module: LlamaDecoderLayer
    input:  (hidden_states, attention_mask, position_ids, ...)  # not used here
    output: tuple of (hidden_states, ...) or just hidden_states
    """
    # Handle case where output is a tuple
    if isinstance(output, tuple):
        hidden_states = output[0]   
    else:
        hidden_states = output
    
    steer = steering_vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
    batch_size, seq_len, hidden_dim = hidden_states.shape
    
    # Determine injection pattern
    if injection_start_token is not None:
        # Inject from injection_start_token onwards
        steer_expanded = torch.zeros(batch_size, seq_len, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        nonlocal prompt_processed
        
        # Detect generation phase: seq_len == 1 after prompt has been processed, or seq_len > prompt_length
        is_generating = (seq_len == 1 and prompt_processed) or (seq_len > prompt_length)
        
        if seq_len == prompt_length:
            prompt_processed = True
        
        if is_generating:
            print(f'DEBUG: got here 2 - generation (seq_len={seq_len}, prompt_length={prompt_length})')
            # During generation: inject at the last token (newly generated token)
            steer_expanded[:, -1:, :] = steer
        else:
            print(f'DEBUG: got here 1 - prompt processing (seq_len={seq_len}, prompt_length={prompt_length})')
            # During prompt processing: inject from injection_start_token to end
            start_idx = max(0, injection_start_token)
            if start_idx < seq_len:
                steer_expanded[:, start_idx:, :] = steer.expand(batch_size, seq_len - start_idx, -1)
    elif not assistant_tokens_only:
        print(f'DEBUG: got here 3')
        # Inject at all tokens
        steer_expanded = steer.expand(batch_size, seq_len, -1)
    else:
        print(f'got here 4')
        # Original behavior: only inject during generation
        steer_expanded = torch.zeros(batch_size, seq_len, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)
        if seq_len == 1: # due to KV caching, seq_len is 1 during all of generation
            steer_expanded[:, :, :] = steer
    
    modified_hidden_states = hidden_states + coeff * steer_expanded
    # print(f"DEBUG: modified_hidden_states shape is {modified_hidden_states.shape}")
    
    # Return in the same format as received
    return (modified_hidden_states,) + output[1:] if isinstance(output, tuple) else modified_hidden_states

def inject_concept_vector(model, tokenizer, steering_vector, layer_to_inject, coeff = 12.0, inference_prompt = None, assistant_tokens_only = False, max_new_tokens = 20, injection_start_token = None):
    '''
    inject concept vectors into the model's hidden states
    assistant_tokens_only: if True, only inject concept vectors at assistant tokens, otherwise inject at all tokens in the sequence
    injection_start_token: if set, inject from this token position onwards (both in prompt and generation)
    '''
    device = next(model.parameters()).device
    # print(f"norm of steering vector before normalization is {torch.norm(steering_vector, p = 2)}")
    steering_vector = steering_vector / torch.norm(steering_vector, p = 2)
    # print(f"norm of steering vector after normalization is {torch.norm(steering_vector, p = 2)}")
    # Convert steering_vector to a tensor on correct device
    if not isinstance(steering_vector, torch.Tensor):
        steering_vector = torch.tensor(steering_vector, dtype=torch.float32) 
    steering_vector = steering_vector.to(device)
    # Ensure it's [1, 1, hidden_dim] so we can broadcast over [batch, seq, hidden_dim]
    if steering_vector.dim() == 1:
        steering_vector = steering_vector.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
    elif steering_vector.dim() == 2:
        # If passed [1, hidden_dim], make it [1, 1, hidden_dim]
        steering_vector = steering_vector.unsqueeze(0)
    # print(f"shape of steering vector to be injected is {steering_vector.shape}")
    
    model_type = get_model_type(tokenizer)
    # Check if prompt is already formatted (contains formatting tokens)
    if inference_prompt and ("<|start_header_id|>" in inference_prompt or "<|im_start|>" in inference_prompt):
        prompt = inference_prompt
    else:
        prompt = format_inference_prompt(model_type, inference_prompt)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_length = inputs.input_ids.shape[1]
    prompt_processed = False  # Track if we've processed the prompt

    handle = model.model.layers[layer_to_inject].register_forward_hook(inject_hook_fn)
    
   
    with torch.no_grad():
        # do_sample = False is equivalent to temperature = 0.0
        out = model.generate(**inputs, max_new_tokens = max_new_tokens, do_sample = False) # [batch_size, seq_len]

    # Only decode the newly generated tokens (not the prompt)
    input_length = inputs.input_ids.shape[1]
    generated_ids = out[0][input_length:]
    response_only = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    handle.remove()
    return response_only