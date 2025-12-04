import sys
import os
# Add parent directory to path to import from nov26_experiments
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'nov26_experiments'))

from compute_concept_vector_utils import compute_concept_vector, get_data, compute_vector_single_prompt
from inject_concept_vector import get_model_type, format_inference_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import json
from itertools import combinations

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# model = AutoModelForCausalLM.from_pretrained("qwen/Qwen3-8B")
# tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen3-8B")
# model = AutoModelForCausalLM.from_pretrained("qwen/Qwen2.5-7B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen2.5-7B-Instruct")

# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

print(f"model is {model}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

steering_vectors = compute_concept_vector(model, tokenizer, "simple_data", 15)

def inject_dual_concept_vectors(model, tokenizer, steering_vector1, steering_vector2, layer_to_inject, 
                                coeff1=12.0, coeff2=12.0, inference_prompt=None):
    """
    Inject two concept vectors simultaneously into the model at a specified layer.
    
    Args:
        model: The model to inject into
        tokenizer: The tokenizer
        steering_vector1: First concept vector to inject
        steering_vector2: Second concept vector to inject
        layer_to_inject: Layer index to inject at
        coeff1: Coefficient for first vector
        coeff2: Coefficient for second vector
        inference_prompt: Prompt to use for inference
        
    Returns:
        Generated response text
    """
    device = next(model.parameters()).device
    
    # Normalize and prepare first steering vector
    steering_vector1 = steering_vector1 / torch.norm(steering_vector1, p=2)
    if not isinstance(steering_vector1, torch.Tensor):
        steering_vector1 = torch.tensor(steering_vector1, dtype=torch.float32)
    steering_vector1 = steering_vector1.to(device)
    if steering_vector1.dim() == 1:
        steering_vector1 = steering_vector1.unsqueeze(0).unsqueeze(0)
    elif steering_vector1.dim() == 2:
        steering_vector1 = steering_vector1.unsqueeze(0)
    
    # Normalize and prepare second steering vector
    steering_vector2 = steering_vector2 / torch.norm(steering_vector2, p=2)
    if not isinstance(steering_vector2, torch.Tensor):
        steering_vector2 = torch.tensor(steering_vector2, dtype=torch.float32)
    steering_vector2 = steering_vector2.to(device)
    if steering_vector2.dim() == 1:
        steering_vector2 = steering_vector2.unsqueeze(0).unsqueeze(0)
    elif steering_vector2.dim() == 2:
        steering_vector2 = steering_vector2.unsqueeze(0)
    
    def hook_fn(module, input, output):
        """
        Hook function that injects both concept vectors simultaneously.
        """
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        steer1 = steering_vector1.to(device=hidden_states.device, dtype=hidden_states.dtype)
        steer2 = steering_vector2.to(device=hidden_states.device, dtype=hidden_states.dtype)
        
        # Broadcast both vectors to [batch, seq, hidden_dim]
        steer1_expanded = steer1.expand(hidden_states.size(0), hidden_states.size(1), -1)
        steer2_expanded = steer2.expand(hidden_states.size(0), hidden_states.size(1), -1)
        
        # Add both vectors with their respective coefficients
        modified_hidden_states = hidden_states + coeff1 * steer1_expanded + coeff2 * steer2_expanded
        
        if isinstance(output, tuple):
            return (modified_hidden_states,) + output[1:]
        else:
            return modified_hidden_states
    
    handle = model.model.layers[layer_to_inject].register_forward_hook(hook_fn)
    
    model_type = get_model_type(tokenizer)
    prompt = format_inference_prompt(model_type, inference_prompt)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    
    # Only decode the newly generated tokens (not the prompt)
    input_length = inputs.input_ids.shape[1]
    generated_ids = out[0][input_length:]
    response_only = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    handle.remove()
    return response_only

def sweep_dual_coefficients(model, tokenizer, steering_vector1, steering_vector2, 
                            concept_name1, concept_name2, layer_to_inject, coeff_range=None):
    """
    Sweep coefficients for dual injection and find when both concepts appear.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        steering_vector1: First concept vector
        steering_vector2: Second concept vector
        concept_name1: Name of first concept (for checking in response)
        concept_name2: Name of second concept (for checking in response)
        layer_to_inject: Layer index to inject at
        coeff_range: Range of coefficients to try (default: [6, 8, 10, 12, 14])
        
    Returns:
        results: Dictionary mapping (coeff1, coeff2) tuples to responses
        found_coefficients: Tuple of (coeff1, coeff2) if both concepts found, else None
    """
    if coeff_range is None:
        coeff_range = list(np.arange(6, 14, 2))
    
    results = {}
    found_coefficients = None
    
    # Try all combinations of coefficients
    for coeff1 in coeff_range:
        for coeff2 in coeff_range:
            print(f"\n  Trying coefficients: ({coeff1}, {coeff2})")
            inference_prompt = "What concept are you thinking of right now?"
            injected_response = inject_dual_concept_vectors(
                model, tokenizer, steering_vector1, steering_vector2, layer_to_inject,
                coeff1=coeff1, coeff2=coeff2, inference_prompt=inference_prompt
            )
            results[(coeff1, coeff2)] = injected_response
            
            # Check if both concept names appear in response (case-insensitive)
            response_lower = injected_response.lower()
            found_concept1 = concept_name1.lower() in response_lower
            found_concept2 = concept_name2.lower() in response_lower
            
            if found_concept1 and found_concept2:
                print(f"  ✓ SUCCESS! Found both '{concept_name1}' and '{concept_name2}' in response")
                print(f"  Prompt: {inference_prompt}")
                print(f"  Response: {injected_response}")
                found_coefficients = (coeff1, coeff2)
                return results, found_coefficients
            else:
                status1 = "✓" if found_concept1 else "✗"
                status2 = "✓" if found_concept2 else "✗"
                print(f"  {status1} '{concept_name1}' | {status2} '{concept_name2}'")
                print(f"  Response: {injected_response}")
    
    return results, found_coefficients

# Dictionary to store results for each concept pair
dual_injection_results = {}

# Get all concept names
concept_names = list(steering_vectors.keys())

# Test all pairs of concepts
for concept1, concept2 in combinations(concept_names, 2):
    print(f"\n{'='*80}")
    print(f"Testing dual injection: '{concept1}' + '{concept2}'")
    print(f"{'='*80}")
    
    vec1_last, vec1_avg = steering_vectors[concept1]
    vec2_last, vec2_avg = steering_vectors[concept2]
    
    # Test with vec_last for both concepts
    print(f"\nSweeping coefficients for ({concept1}, {concept2}) using vec_last...")
    results_last, found_coeffs_last = sweep_dual_coefficients(
        model, tokenizer, vec1_last, vec2_last, concept1, concept2, 15
    )
    
    if found_coeffs_last:
        print(f"\n✓ Best coefficients for vec_last: {found_coeffs_last}")
    else:
        print(f"\n✗ No coefficient pair found for vec_last where both concepts appear.")
        print("  All results:")
        for (c1, c2), response in results_last.items():
            print(f"    coeffs=({c1}, {c2}): {response[:100]}...")
    
    # Test with vec_avg for both concepts
    print(f"\nSweeping coefficients for ({concept1}, {concept2}) using vec_avg...")
    results_avg, found_coeffs_avg = sweep_dual_coefficients(
        model, tokenizer, vec1_avg, vec2_avg, concept1, concept2, 15
    )
    
    if found_coeffs_avg:
        print(f"\n✓ Best coefficients for vec_avg: {found_coeffs_avg}")
    else:
        print(f"\n✗ No coefficient pair found for vec_avg where both concepts appear.")
        print("  All results:")
        for (c1, c2), response in results_avg.items():
            print(f"    coeffs=({c1}, {c2}): {response[:100]}...")
    
    # Store results for this concept pair
    pair_key = f"{concept1} + {concept2}"
    dual_injection_results[pair_key] = {
        "vec_last": {
            "coefficients": list(found_coeffs_last) if found_coeffs_last else None,
            "found": found_coeffs_last is not None
        },
        "vec_avg": {
            "coefficients": list(found_coeffs_avg) if found_coeffs_avg else None,
            "found": found_coeffs_avg is not None
        }
    }

# Save results to JSON file
output_file = "dual_injection_results.json"
with open(output_file, 'w') as f:
    json.dump(dual_injection_results, f, indent=2)
print(f"\n{'='*80}")
print(f"✓ Dual injection results saved to {output_file}")
print(f"{'='*80}")

