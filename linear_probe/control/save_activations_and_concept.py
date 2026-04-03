import argparse
import torch
from datetime import datetime
from pathlib import Path
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
    
    print(f"positive_avg.keys()={positive_avg.keys()}")
    for layer in sorted(positive_avg.keys()):
        p_mean = positive_avg[layer].mean(dim=0)
        n_mean = negative_avg[layer].mean(dim=0)
        mass_mass_vector[layer] = p_mean - n_mean

    if mass_mass_vector:
        last_layered = list(mass_mass_vector.keys())[-1]
        print(mass_mass_vector[last_layered].shape)
    
    # save mass vector to probe_vectors/run_.../
    # save postive and negative to saved_activations/run_.../
    now = datetime.now()
    run_name = now.strftime("run_%m_%d_%y_%H_%M")
    script_dir = Path(__file__).resolve().parent
    
    probe_dir = script_dir / "probe_vectors" / run_name
    probe_dir.mkdir(parents=True, exist_ok=True)
    
    act_dir = script_dir / "saved_activations" / run_name
    act_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(mass_mass_vector, probe_dir / f"mass_mass_vector_{args.concept_name}.pt")
    torch.save(positive_avg, act_dir / f"positive_avg_{args.concept_name}.pt")
    torch.save(negative_avg, act_dir / f"negative_avg_{args.concept_name}.pt")
    
    print(f"💾 Saved mass_mass_vector to {probe_dir}")
    print(f"💾 Saved positive_avg & negative_avg to {act_dir}")

    # compute PCA and cohen's d

if __name__ == "__main__":
    main()
