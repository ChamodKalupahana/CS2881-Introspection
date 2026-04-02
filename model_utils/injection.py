# ── Injection + activation capture ───────────────────────────────────────────

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
