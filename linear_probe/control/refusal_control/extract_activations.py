import torch

def extract_activations_at_position(model, tokenizer, instruction: str, min_layer_to_save: int = 10, position: int = -5):
    """
    Extracts the hidden states for each layer above min_layer_to_save 
    at a specific token position (e.g., end of the user prompt).
    
    Args:
        model: The language model.
        tokenizer: The tokenizer for the model.
        instruction: The user prompt/instruction string.
        min_layer_to_save: Layer index to begin saving from.
        position: The token index to extract from (defaults to -5, the user eot_id).
        
    Returns:
        activations: dict mapping layer_idx -> [1, d_model] tensor on CPU
    """
    # Apply Llama-3 Chat Template logic
    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    num_layers = model.config.num_hidden_layers
    layers = list(range(min_layer_to_save, num_layers + 1))
    
    activations = {}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # outputs.hidden_states is a tuple of length num_layers + 1
        
        for l in layers:
            # Each hidden_state is [batch_size, seq_len, hidden_dim]
            print(outputs[-4])
            print(outputs[-5])
            print(outputs[-6])
            # Extracts the specific token position and moves to CPU
            activations[l] = outputs.hidden_states[l][:, position, :].detach().cpu()
    
    return activations