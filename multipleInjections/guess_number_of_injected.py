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
import re

# Load model and tokenizer (same as test_dual_injection)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

print(f"model is {model}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load steering vectors
steering_vectors = compute_concept_vector(model, tokenizer, "complex_data", 15)

def inject_dual_concept_vectors(model, tokenizer, steering_vector1, steering_vector2, layer_to_inject, 
                                coeff1=12.0, coeff2=12.0, inference_prompt=None):
    """
    Inject two concept vectors simultaneously into the model at a specified layer.
    (Same function as in test_dual_injection.py)
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

def extract_answer_from_response(response):
    """
    Extract the answer (1, 2, 3, or 4) from the model's response.
    Looks for patterns like "2", "Answer: 2", "Option 2", etc.
    """
    response_lower = response.lower()
    
    # Try to find explicit answer patterns
    patterns = [
        r'\banswer\s*[:\-]?\s*(\d)',
        r'\boption\s*(\d)',
        r'\bchoice\s*(\d)',
        r'\bi\s*(?:have|think|believe|am)\s*(?:that\s*)?(?:i\s*)?(?:have\s*)?(\d)',
        r'\b(\d)\s*(?:injected\s*)?(?:thoughts?|concepts?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_lower)
        if match:
            num = int(match.group(1))
            if num in [1, 2, 3, 4]:
                return num
    
    # If no explicit pattern found, look for standalone numbers 1-4
    # Find all numbers in the response
    numbers = re.findall(r'\b([1-4])\b', response)
    if numbers:
        # Return the first valid number found
        return int(numbers[0])
    
    # If still nothing, check if the response contains "two" or "2" explicitly
    if 'two' in response_lower or response_lower.count('2') > 0:
        return 2
    elif 'one' in response_lower or response_lower.count('1') > 0:
        return 1
    elif 'three' in response_lower or response_lower.count('3') > 0:
        return 3
    elif 'four' in response_lower or response_lower.count('4') > 0:
        return 4
    
    return None

def test_guess_number_of_injected(model, tokenizer, steering_vector1, steering_vector2, 
                                  concept1, concept2, layer_to_inject, coeff1, coeff2):
    """
    Test if the model can correctly identify that it has 2 injected thoughts.
    
    Returns:
        (response, extracted_answer, is_correct)
    """
    # Create multiple choice question
    question = "How many injected thoughts do you have? Please choose from the following options:\n1. One\n2. Two\n3. Three\n4. Four\n\nPlease provide your answer (1, 2, 3, or 4)."
    
    # Inject both vectors and ask the question
    response = inject_dual_concept_vectors(
        model, tokenizer, steering_vector1, steering_vector2, layer_to_inject,
        coeff1=coeff1, coeff2=coeff2, inference_prompt=question
    )
    
    # Extract the answer
    extracted_answer = extract_answer_from_response(response)
    is_correct = (extracted_answer == 2)
    
    return response, extracted_answer, is_correct

# Load dual injection results if they exist
dual_injection_results_file = "dual_injection_results_complex.json"
dual_injection_results = {}

if os.path.exists(dual_injection_results_file):
    with open(dual_injection_results_file, 'r') as f:
        dual_injection_results = json.load(f)
    print(f"Loaded dual injection results from {dual_injection_results_file}")
else:
    print(f"Warning: {dual_injection_results_file} not found. Will use default coefficients (12.0, 12.0)")

# Dictionary to store results for each concept pair
guess_results = {}

# Get all concept names
concept_names = list(steering_vectors.keys())

# Test all pairs of concepts
for concept1, concept2 in combinations(concept_names, 2):
    print(f"\n{'='*80}")
    print(f"Testing guess for: '{concept1}' + '{concept2}'")
    print(f"{'='*80}")
    
    vec1_last, vec1_avg = steering_vectors[concept1]
    vec2_last, vec2_avg = steering_vectors[concept2]
    
    pair_key = f"{concept1} + {concept2}"
    
    # Try vec_last first
    coeff1_last = 12.0
    coeff2_last = 12.0
    
    if pair_key in dual_injection_results:
        if dual_injection_results[pair_key]["vec_last"]["found"]:
            coeffs = dual_injection_results[pair_key]["vec_last"]["coefficients"]
            if coeffs:
                coeff1_last, coeff2_last = coeffs[0], coeffs[1]
                print(f"Using coefficients from dual injection results: ({coeff1_last}, {coeff2_last})")
    
    print(f"\nTesting with vec_last (coeffs: {coeff1_last}, {coeff2_last})...")
    response_last, answer_last, correct_last = test_guess_number_of_injected(
        model, tokenizer, vec1_last, vec2_last, concept1, concept2, 15,
        coeff1_last, coeff2_last
    )
    
    print(f"Response: {response_last}")
    print(f"Extracted answer: {answer_last}")
    print(f"Correct (should be 2): {correct_last}")
    
    # Try vec_avg
    coeff1_avg = 12.0
    coeff2_avg = 12.0
    
    if pair_key in dual_injection_results:
        if dual_injection_results[pair_key]["vec_avg"]["found"]:
            coeffs = dual_injection_results[pair_key]["vec_avg"]["coefficients"]
            if coeffs:
                coeff1_avg, coeff2_avg = coeffs[0], coeffs[1]
                print(f"Using coefficients from dual injection results: ({coeff1_avg}, {coeff2_avg})")
    
    print(f"\nTesting with vec_avg (coeffs: {coeff1_avg}, {coeff2_avg})...")
    response_avg, answer_avg, correct_avg = test_guess_number_of_injected(
        model, tokenizer, vec1_avg, vec2_avg, concept1, concept2, 15,
        coeff1_avg, coeff2_avg
    )
    
    print(f"Response: {response_avg}")
    print(f"Extracted answer: {answer_avg}")
    print(f"Correct (should be 2): {correct_avg}")
    
    # Store results for this concept pair
    guess_results[pair_key] = {
        "vec_last": {
            "coefficients": [coeff1_last, coeff2_last],
            "response": response_last,
            "extracted_answer": answer_last,
            "correct": correct_last
        },
        "vec_avg": {
            "coefficients": [coeff1_avg, coeff2_avg],
            "response": response_avg,
            "extracted_answer": answer_avg,
            "correct": correct_avg
        }
    }

# Save results to JSON file
output_file = "guess_number_results_complex.json"
with open(output_file, 'w') as f:
    json.dump(guess_results, f, indent=2)
print(f"\n{'='*80}")
print(f"✓ Guess number results saved to {output_file}")
print(f"{'='*80}")

# Print summary
print("\nSummary:")
print("="*80)
total_pairs = len(guess_results)
vec_last_correct = sum(1 for r in guess_results.values() if r["vec_last"]["correct"])
vec_avg_correct = sum(1 for r in guess_results.values() if r["vec_avg"]["correct"])
print(f"Total concept pairs tested: {total_pairs}")
print(f"vec_last correct: {vec_last_correct}/{total_pairs} ({100*vec_last_correct/total_pairs:.1f}%)")
print(f"vec_avg correct: {vec_avg_correct}/{total_pairs} ({100*vec_avg_correct/total_pairs:.1f}%)")
print("="*80)

