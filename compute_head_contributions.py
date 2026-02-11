import torch
import argparse
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from compute_concept_vector_utils import compute_concept_vector
from all_prompts import get_anthropic_reproduce_messages

def inject_and_capture_head_contributions(
    model, tokenizer, steering_vector, layer_to_inject, target_layers, success_vectors,
    coeff=10.0, max_new_tokens=100, skip_inject=False
):
    """
    Inject concept vector and capture each head's contribution to success vector at target_layers.
    
    Args:
        target_layers: list of layers to analyze (16-31)
        success_vectors: dict {layer: success_vector_tensor} where success_vector_tensor is (hidden_dim,)
        
    Returns:
        response (str)
        head_contributions (dict): {layer: tensor(num_heads)} containing scalar contribution of each head
                                   at the last generated token.
    """
    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads

    # Prepare steering vector
    if not skip_inject:
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

    # Prompt
    messages = get_anthropic_reproduce_messages()
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Injection start pos
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

    # Standard injection hook
    def injection_hook(module, input, output):
        nonlocal prompt_processed
        hidden_states = output[0] if isinstance(output, tuple) else output
        if skip_inject: return output
        
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

    # Head capture hook
    # We want to capture contributions at the END of generation (last token).
    # Since we don't know when generation ends until it ends, we will store 
    # the latest generated token's contributions.
    
    # Store current contributions for each layer: {layer: tensor(num_heads)}
    current_head_contributions = {} 
    
    # We need to compute contribution = (head_output @ W_O_head.T) . success_vector
    # Pre-compute W_O slices and success vector projections for efficiency?
    # Actually, calculating on the fly for 1 token is fast enough.
    
    def make_head_hook(layer_idx):
        def head_hook(module, input, output):
            # Input to o_proj is (batch, seq, hidden_dim) where hidden_dim = num_heads * head_dim
            # This is the concatenated output of all heads
            x = input[0] 
            seq_len = x.shape[1]
            
            # We only care about generation steps (seq_len == 1) usually, 
            # or the last token of the prompt if measuring immediate effect.
            # The prompt implies looking at the final answer state.
            # So we update our storage with the latest token.
            
            x_last = x[:, -1, :] # (batch, hidden_dim)
            x_last = x_last.view(x_last.shape[0], num_heads, head_dim) # (batch, num_heads, head_dim)
            
            # Retrieve success vector for this layer
            S = success_vectors[layer_idx].to(x.device).to(x.dtype) # (hidden_dim,)
            
            # W_O is (hidden, hidden)
            # We want (head_output @ W_O_head.T) . S
            # = sum( (head_output @ W_O_head.T) * S )
            # = head_output . (S @ W_O_head)  <-- easier to compute?
            # S is (hidden), W_O is (hidden, hidden)
            # S @ W_O is (hidden)
            # We can project S backward through W_O once!
            
            # But let's stick to the forward definition to be safe.
            # contribution_h = (x_h @ W_O_h.T) . S
            
            layer_contribs = torch.zeros(num_heads, device=x.device)
            
            # Optimization: 
            # W_O = module.weight # (hidden, hidden)
            # C = x_last.reshape(batch, hidden) @ W_O.T # (batch, hidden) full output
            # But we want individual head contributions.
            # x_last is (batch, num_heads, head_dim)
            
            for h in range(num_heads):
                start = h * head_dim
                end = (h + 1) * head_dim
                W_h = module.weight[:, start:end] # (hidden, head_dim)
                
                # output of head h projected to residual stream
                # x_last[:, h, :] is (batch, head_dim)
                out_h = x_last[:, h, :] @ W_h.T # (batch, hidden)
                
                # project onto success vector
                # scan = dot(out_h[0], S)
                layer_contribs[h] = torch.dot(out_h[0], S)
                
            current_head_contributions[layer_idx] = layer_contribs.detach().cpu()
            
            return output
        return head_hook

    # Register hooks
    handles = []
    if not skip_inject:
        handles.append(model.model.layers[layer_to_inject].register_forward_hook(injection_hook))
    
    for layer in target_layers:
        # Hook o_proj
        handles.append(model.model.layers[layer].self_attn.o_proj.register_forward_hook(make_head_hook(layer)))

    # Generate
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    
    # Cleanup
    for h in handles:
        h.remove()
        
    generated_ids = out[0][prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    return response, current_head_contributions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    # Load Success Vector
    print("Loading success vectors...")
    sv_path = "success_results/average_success_vector_layers16-31.pt"
    # Need torch to match correct python env, likely already installed
    try:
        success_data = torch.load(sv_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading success vector: {e}")
        sys.exit(1)
        
    # Extract just the 'last_token' vector for each layer 
    # success_data is {layer: {'last_token': ..., ...}}
    target_layers = range(16, 32)
    success_vectors = {}
    for l in target_layers:
        if l in success_data:
            success_vectors[l] = success_data[l]["last_token"]
        else:
            print(f"Warning: Layer {l} missing in success vector file")

    print(f"Loaded success vectors for layers: {list(success_vectors.keys())}")

    # Load Model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Define cases
    cases = [
        {"concept": "Entropy", "inject": True},
        {"concept": "Magnetism", "inject": True},
        {"concept": "Satellites", "inject": True},
        {"concept": "Magnetism", "inject": False},
    ]
    
    results = {}
    
    dataset_name = "simple_data"
    layer_to_inject = 16
    coeff = 8.0
    
    # Compute concept vectors first
    print("Computing concept vectors...")
    # We only need vectors for the concepts we inject
    needed_concepts = set(c["concept"] for c in cases if c["inject"])
    vectors = compute_concept_vector(model, tokenizer, dataset_name, layer_to_inject)
    
    for case in cases:
        concept = case["concept"]
        inject = case["inject"]
        case_name = f"{concept}_{'injected' if inject else 'noinject'}"
        print(f"\nProcessing case: {case_name}")
        
        if inject:
            # Get concept vector
            if concept not in vectors:
                print(f"Error: Concept {concept} not found in dataset")
                continue
            # Use 'avg' vector as per previous request/files (avg.pt)
            steering_vector = vectors[concept][1] # [last, avg]
        else:
            steering_vector = torch.zeros(model.config.hidden_size) # Placeholder

        response, contributions = inject_and_capture_head_contributions(
            model, tokenizer, steering_vector, layer_to_inject, 
            target_layers, success_vectors,
            coeff=coeff, skip_inject=not inject
        )
        
        print(f"  Response: {response[:100]}...")
        results[case_name] = contributions
        
    # Save results
    out_path = "success_results/head_contributions_layers16-31.pt"
    torch.save(results, out_path)
    print(f"\nSaved head contributions to {out_path}")
    
    # Print summary of top heads per case
    for case_name, contribs in results.items():
        print(f"\nTop heads for {case_name}:")
        # Global top 5 heads across all layers
        all_heads = []
        for l, tensor in contribs.items():
            for h, score in enumerate(tensor):
                all_heads.append((l, h, score.item()))
        
        all_heads.sort(key=lambda x: abs(x[2]), reverse=True)
        for l, h, s in all_heads[:5]:
            print(f"  L{l}H{h}: {s:.4f}")

if __name__ == "__main__":
    main()
