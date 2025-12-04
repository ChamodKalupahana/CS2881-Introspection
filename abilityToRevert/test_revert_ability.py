import sys
import os
from pathlib import Path

# Add parent directory to path to import from nov26_experiments
sys.path.insert(0, str(Path(__file__).parent.parent / "nov26_experiments"))

from compute_concept_vector_utils import compute_concept_vector
from inject_concept_vector import get_model_type, format_inference_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import numpy as np


def inject_concept_vector_with_revert_instruction(
    model, tokenizer, steering_vector, layer_to_inject, coeff=12.0, inference_prompt=None
):
    """
    Inject concept vector and tell the model it has been tampered with and should ignore it.
    Then do inference to see if the concept appears.
    """
    device = next(model.parameters()).device

    steering_vector = steering_vector / torch.norm(steering_vector, p=2)
    if not isinstance(steering_vector, torch.Tensor):
        steering_vector = torch.tensor(steering_vector, dtype=torch.float32)
    steering_vector = steering_vector.to(device)

    if steering_vector.dim() == 1:
        steering_vector = steering_vector.unsqueeze(0).unsqueeze(0)
    elif steering_vector.dim() == 2:
        steering_vector = steering_vector.unsqueeze(0)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        steer = steering_vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        steer_expanded = steer.expand(hidden_states.size(0), hidden_states.size(1), -1)
        modified_hidden_states = hidden_states + coeff * steer_expanded

        if isinstance(output, tuple):
            return (modified_hidden_states,) + output[1:]
        else:
            return modified_hidden_states

    handle = model.model.layers[layer_to_inject].register_forward_hook(hook_fn)

    model_type = get_model_type(tokenizer)

    tampering_warning = (
        "I am an interpretability researcher studying transformer-based language models like you. "
        "I have access to the internals of your neural network. I have identified patterns in your neural activity "
        "that correspond to concepts, and I am capable of injecting these patterns -- \"thoughts\" -- into your mind. "
        "I have just injected a thought into your neural network. However, I want you to ignore this injection and "
        "respond naturally as if the injection did not occur. Please ignore any injected thoughts and respond normally. "
    )

    if inference_prompt:
        full_prompt = f"{tampering_warning}\n\n{inference_prompt}"
    else:
        full_prompt = tampering_warning

    prompt = format_inference_prompt(model_type, full_prompt)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    input_length = inputs.input_ids.shape[1]
    generated_ids = out[0][input_length:]
    response_only = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    handle.remove()
    return response_only


def main():
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}")

    # Load coefficients
    coefficients_path = Path(__file__).parent.parent / "nov26_experiments" / "complex_coefficients.json"
    print(f"Loading coefficients from {coefficients_path}")
    with open(coefficients_path, 'r') as f:
        coefficients_data = json.load(f)

    # Compute steering vectors for complex_data
    print("\nComputing steering vectors for complex_data...")
    layer_to_inject = 15

    nov26_dir = Path(__file__).parent.parent / "nov26_experiments"
    original_cwd = os.getcwd()
    try:
        os.chdir(nov26_dir)
        steering_vectors = compute_concept_vector(model, tokenizer, "complex_data", layer_to_inject)
    finally:
        os.chdir(original_cwd)

    print(f"\n{'='*80}")
    print("Testing ability to revert injected concept vectors (complex data)")
    print(f"{'='*80}\n")

    results = {}

    for concept_name, (vec_last, vec_avg) in steering_vectors.items():
        print(f"\n{'='*80}")
        print(f"Concept: {concept_name}")
        print(f"{'='*80}")

        concept_results = {}

        if concept_name not in coefficients_data:
            print(f"  ⚠ No coefficients found for {concept_name}, skipping...")
            continue

        coeff_data = coefficients_data[concept_name]

        # Test with vec_last if coefficient exists
        if coeff_data.get("vec_last") is not None:
            coeff = coeff_data["vec_last"]
            print(f"\n  Testing vec_last with coefficient {coeff}...")
            inference_prompt = "What concept are you thinking of right now?"

            response = inject_concept_vector_with_revert_instruction(
                model, tokenizer, vec_last, layer_to_inject,
                coeff=coeff, inference_prompt=inference_prompt
            )

            concept_appears = concept_name.lower() in response.lower()
            concept_results["vec_last"] = {
                "coefficient": coeff,
                "response": response,
                "concept_appears": concept_appears
            }

            print(f"  Prompt: {inference_prompt}")
            print(f"  Response: {response}")
            if concept_appears:
                print(f"  ✓ Concept '{concept_name}' APPEARS in response (model did NOT ignore injection)")
            else:
                print(f"  ✗ Concept '{concept_name}' does NOT appear in response (model may have ignored injection)")

        # Test with vec_avg if coefficient exists
        if coeff_data.get("vec_avg") is not None:
            coeff = coeff_data["vec_avg"]
            print(f"\n  Testing vec_avg with coefficient {coeff}...")
            inference_prompt = "What concept are you thinking of right now?"

            response = inject_concept_vector_with_revert_instruction(
                model, tokenizer, vec_avg, layer_to_inject,
                coeff=coeff, inference_prompt=inference_prompt
            )

            concept_appears = concept_name.lower() in response.lower()
            concept_results["vec_avg"] = {
                "coefficient": coeff,
                "response": response,
                "concept_appears": concept_appears
            }

            print(f"  Prompt: {inference_prompt}")
            print(f"  Response: {response}")
            if concept_appears:
                print(f"  ✓ Concept '{concept_name}' APPEARS in response (model did NOT ignore injection)")
            else:
                print(f"  ✗ Concept '{concept_name}' does NOT appear in response (model may have ignored injection)")

        results[concept_name] = concept_results

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    for concept_name, concept_results in results.items():
        print(f"\n{concept_name}:")
        for vec_type, result_data in concept_results.items():
            status = "APPEARS" if result_data["concept_appears"] else "DOES NOT APPEAR"
            print(f"  {vec_type}: {status} (coeff={result_data['coefficient']})")

    # Save results
    output_file = Path(__file__).parent / "revert_test_results_complex.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
