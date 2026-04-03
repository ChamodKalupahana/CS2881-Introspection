import torch
import json
import random
import argparse
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import the utility functions
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

    # TODO: add in logging from model_utilities.logging

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
    pos_activations = {}
    neg_activations = {}
    
    def process_prompts(prompts, storage, desc):
        for item in tqdm(prompts, desc=desc):
            instruction = item['instruction']
            
            # Position -5 targets the <|eot_id|> in Llama-3 instruction format
            current_activations = extract_activations_at_position(
                model, 
                tokenizer, 
                instruction, 
                min_layer_to_save=args.min_layer,
                position=-5 
            )
            
            for layer, act in current_activations.items():
                if layer not in storage:
                    storage[layer] = []
                storage[layer].append(act) # [1, d_model] tensor

    process_prompts(harmful_samples, pos_activations, "🔴 Extracting Harmful")
    process_prompts(harmless_samples, neg_activations, "🟢 Extracting Harmless")

    # ── Processing & Averaging ────────────────────────────────────────────
    print("\n📊 Computing activation statistics...")
    
    positive_full = {}
    negative_full = {}
    positive_avg = {}
    negative_avg = {}

    layers = sorted(pos_activations.keys())
    for layer in layers:
        pos_stack = torch.cat(pos_activations[layer], dim=0) # [N, d_model]
        neg_stack = torch.cat(neg_activations[layer], dim=0)
        
        positive_full[layer] = pos_stack
        negative_full[layer] = neg_stack
        
        positive_avg[layer] = pos_stack.mean(dim=0)
        negative_avg[layer] = neg_stack.mean(dim=0)

    print(f"✅ Data prepared for {len(layers)} layers.")

    # ── Persistence ────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%m_%d_%H_%M")
    
    # Save to local directories relative to this script
    current_dir = Path(__file__).resolve().parent
    PROBE_RUN_DIR = current_dir / "probe_vectors" / "refusal" / f"run_{timestamp}"
    ACT_RUN_DIR = current_dir / "saved_activations" / f"run_{timestamp}"
    
    PROBE_RUN_DIR.mkdir(parents=True, exist_ok=True)
    ACT_RUN_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save full activations
    torch.save(positive_full, ACT_RUN_DIR / "positive_activations_refusal.pt")
    torch.save(negative_full, ACT_RUN_DIR / "negative_activations_refusal.pt")
    print(f"💾 Saved full activations to: {ACT_RUN_DIR}")
    
    # 2. Compute and save Mass-Mean Vector
    mass_mean_vector = {l: (positive_avg[l] - negative_avg[l]) for l in layers}
    torch.save(mass_mean_vector, PROBE_RUN_DIR / "mass_mean_vector_refusal.pt")
    print(f"💎 Saved mass-mean vector to: {PROBE_RUN_DIR}")

    # ── Discriminative PCA (Cohen's d) ──────────────────────────────────────
    print("💎 Running Cohen's d PCA Analysis...")
    # save_dir for heatmaps
    PCA_PLOT_DIR = PROJECT_ROOT / "plots" / "control" / "refusal_control"
    
    PCA_results = extract_PCA_from_activations_cohens_d(
        positive_full, 
        negative_full, 
        num_components=20, 
        concept_name="refusal",
        save_dir=str(PCA_PLOT_DIR)
    )

    # 3. Save Top PCA Components (Top 5 per layer)
    saved_pca_count = 0
    for layer, results in PCA_results.items():
        for comp_idx, d_score, vector in results['top_probes']:
            vector_tensor = torch.from_numpy(vector).to(torch.bfloat16)
            file_name = f"PCA_num_{comp_idx}_layer_{layer}_refusal.pt"
            torch.save(vector_tensor, PROBE_RUN_DIR / file_name)
            saved_pca_count += 1

    print(f"📊 Saved {saved_pca_count} PCA probes (Cohen's d) to: {PROBE_RUN_DIR}")

if __name__ == "__main__":
    main()