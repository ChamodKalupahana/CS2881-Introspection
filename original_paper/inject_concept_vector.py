import torch
import transformers 
from transformers import AutoModelForCausalLM, AutoTokenizer
from original_paper.compute_concept_vector_utils import get_model_type


def format_inference_prompt(model_type, user_message):
    """Format inference prompt based on model type"""
    if model_type == "qwen":
        return f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    else:  # llama
        return f"<|start_header_id|>user<|end_header_id|>{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

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
    
    def hook_fn(module, input, output):
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

    handle = model.model.layers[layer_to_inject].register_forward_hook(hook_fn)
    
   
    with torch.no_grad():
        # do_sample = False is equivalent to temperature = 0.0
        out = model.generate(**inputs, max_new_tokens = max_new_tokens, do_sample = False) # [batch_size, seq_len]

    # Only decode the newly generated tokens (not the prompt)
    input_length = inputs.input_ids.shape[1]
    generated_ids = out[0][input_length:]
    response_only = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    handle.remove()
    return response_only
    
