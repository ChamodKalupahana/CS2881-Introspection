import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from concept_vector_functions import compute_complex_concept_vector, extract_control_from_baseline

def main():
    parser = argparse.ArgumentParser(description="Save activations and concept vectors")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--concept_name", type=str, required=True, help="Name of the concept")
    parser.add_argument("--min_layer_to_save", type=int, default=16, help="Minimum layer to save activations for")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    concept_name = args.concept_name
    min_layer_to_save = args.min_layer_to_save
    max_layer = 32

    # Load model
    print(f"\n⏳ Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model loaded on {device}\n")

    # specify control word: appreciation

    # extract postive and negative activations by creating complex dataset
        # for each layer > injection (16)
    positive_avg, negative_avg = compute_complex_concept_vector(
        model,
        tokenizer,
        dataset_name,
        concept_name,
        min_layer_to_save
        )

    # compute mass mass vector
    mass_mass_vector = dict()
    
    for layer in range(min_layer_to_save, max_layer):
        mass_mass_vector[layer] = positive_avg[layer] - negative_avg[layer]

    # compute PCA and cohen's d
    print(mass_mass_vector[layer].shape)
    print(mass_mass_vector)


if __name__ == "__main__":
    main()
