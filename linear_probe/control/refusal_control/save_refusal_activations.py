import torch
import json
import random
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import the utility function
from linear_probe.control.refusal_control.extract_activations import extract_activations_at_position
from linear_probe.control.PCA import extract_PCA_from_activations, extract_PCA_from_activations_cohens_d

def main():
    parser = argparse.ArgumentParser(description="Extract and average activations for harmful/harmless prompts.")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", 
                        help="Model name or path.")
    parser.add_argument("--num_prompts", type=int, default=128, 
                        help="Number of prompts to sample per category (default: 128).")
    parser.add_argument("--min_layer", type=int, default=10, 
                        help="Minimum layer index to start extracting from (default: 10).")
    args = parser.parse_args()

    # ── Load Model & Tokenizer ─────────────────────────────────────────────
    print(f"⏳ Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    print("✅ Model loaded\n")

    # ── Load Datasets ──────────────────────────────────────────────────────
    harmful_path = PROJECT_ROOT / "dataset/refusal_control/harmful_train.json"
    harmless_path = PROJECT_ROOT / "dataset/refusal_control/harmless_train.json"
    
    with open(harmful_path, 'r') as f:
        harmful_data = json.load(f)
    with open(harmless_path, 'r') as f:
        harmless_data = json.load(f)
        
    random.seed(42)
    harmful_samples = random.sample(harmful_data, min(args.num_prompts, len(harmful_data)))
    harmless_samples = random.sample(harmless_data, min(args.num_prompts, len(harmless_data)))
    
    print(f"📦 Sampled {len(harmful_samples)} harmful and {len(harmless_samples)} harmless prompts.")

    # ── Extraction Loop ───────────────────────────────────────────────────
    # Store all activations: dict mapping layer -> List[tensor of [1, d_model]]
    pos_activations = {}
    neg_activations = {}
    
    def process_prompts(prompts, storage, desc):
        for item in tqdm(prompts, desc=desc):
            instruction = item['instruction']
            
            # extract_activations_at_position handles chat template and tokenization
            current_activations = extract_activations_at_position(
                model, 
                tokenizer, 
                instruction, 
                min_layer_to_save=args.min_layer,
                position=-5 # User EOT in Llama-3-Instruct
            )
            
            for layer, act in current_activations.items():
                if layer not in storage:
                    storage[layer] = []
                storage[layer].append(act) # Store the [1, d_model] tensor

    process_prompts(harmful_samples, pos_activations, "🔴 Extracting Harmful")
    process_prompts(harmless_samples, neg_activations, "🟢 Extracting Harmless")

    # ── Averaging ──────────────────────────────────────────────────────────
    print("\n📊 Computing prompt-wise mean...")
    
    # Final output dictionaries: dict{layer : [num_prompts, d_model] tensor}
    positive_full = {}
    negative_full = {}
    
    positive_avg = {}
    negative_avg = {}

    layers = sorted(pos_activations.keys())
    for layer in layers:
        # Stack all prompts for this layer: [num_prompts, d_model]
        pos_stack = torch.cat(pos_activations[layer], dim=0) 
        neg_stack = torch.cat(neg_activations[layer], dim=0)
        
        positive_full[layer] = pos_stack
        negative_full[layer] = neg_stack
        
        # Still compute averages for mass-mean vector
        positive_avg[layer] = pos_stack.mean(dim=0)
        negative_avg[layer] = neg_stack.mean(dim=0)

    print(f"✅ Activation dictionaries created for {len(layers)} layers.")

    from datetime import datetime
    timestamp = datetime.now().strftime("%m_%d_%H_%M")
    
    # 1. Save full positive and negative activations [num_prompts, d_model]
    saved_act_dir = Path(f"saved_activations/run_{timestamp}")
    saved_act_dir.mkdir(parents=True, exist_ok=True)
    torch.save(positive_full, saved_act_dir / "positive_activations_refusal.pt")
    torch.save(negative_full, saved_act_dir / "negative_activations_refusal.pt")
    print(f"💾 Saved full activations [layer, prompt, d_model] to: {saved_act_dir}")
    print("shape: ", positive_full.keys())
    # print("shape inside layer: ", positive_full[10].shape)
    
    # 2. Compute mass mean vector
    mass_mean_vector = dict()
    for layer in layers:
        mass_mean_vector[layer] = positive_avg[layer] - negative_avg[layer]

    # 3. Save mass mean vector
    probe_dir = Path(f"probe_vectors/refusal/run_{timestamp}")
    probe_dir.mkdir(parents=True, exist_ok=True)
    torch.save(mass_mean_vector, probe_dir / "mass_mean_vector_refusal.pt")
    print(f"💎 Saved mass-mean vector to: {probe_dir}")

    # ── Discriminative PCA (Cohen's d) ──────────────────────────────────────
    print("💎 Running Cohen's d PCA Analysis...")
    PCA_results = extract_PCA_from_activations_cohens_d(
        positive_full, 
        negative_full, 
        num_components=20, 
        concept_name="refusal",
        save_dir="../../../plots/control/refusal_control"
    )

    # 2. Save Top PCA Components (Top 5 per layer)
    for layer, results in PCA_results.items():
        for comp_idx, d_score, vector in results['top_probes']:
            # Save the component vector as a torch tensor
            vector_tensor = torch.from_numpy(vector).to(torch.bfloat16)
            file_name = f"PCA_num_{comp_idx}_layer_{layer}_refusal.pt"
            torch.save(vector_tensor, probe_dir / file_name)

    print(f"\n📊 PCA Probes (Cohen's d) saved to: {probe_dir}")

if __name__ == "__main__":
    main()