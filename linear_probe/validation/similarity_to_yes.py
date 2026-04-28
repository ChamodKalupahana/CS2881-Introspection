import torch
import os
import sys
import argparse
import re
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# FIND PROJECT ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# LOCAL HELPER COPIES (from original_paper/compute_concept_vector_utils.py)
# This bypasses the broken pathing in the shared library for arbitrary JSONs.

def get_model_type(tokenizer):
    model_name = tokenizer.name_or_path.lower()
    return "qwen" if "qwen" in model_name else "llama"

def format_prompt(model_type, user_message, dataset_name=None):
    if "qwen" in model_type:
        return f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    else:  # llama
        return f"<|start_header_id|>user<|end_header_id|>{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

def get_data_local(dataset_path): 
    """Load raw data from JSON, handling full or relative paths."""
    path = Path(dataset_path)
    if not path.exists():
        # Try looking in project-relative dataset/ folder
        path = PROJECT_ROOT / "dataset" / path.name
    
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at {dataset_path} (relative to project root: {PROJECT_ROOT})")
        
    with open(path, "r") as f:
        return json.load(f)

def compute_vector_single_prompt_local(model, tokenizer, dataset_name, steering_prompt, layer_idx):
    model_type = get_model_type(tokenizer)
    prompt = format_prompt(model_type, steering_prompt, dataset_name)
    
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # last token
        prompt_last_vector = outputs.hidden_states[layer_idx][:, prompt_len - 1, :].detach().cpu() 
        # average
        prompt_average_vector = outputs.hidden_states[layer_idx][:, :prompt_len, :].mean(dim=1).detach().cpu()
        del outputs 
    
    return prompt_last_vector, prompt_average_vector

def compute_concept_vector_local(model, tokenizer, dataset_path, layer_idx):
    """Local version that handles 'complex_yes_vector.json' and related datasets."""
    data = get_data_local(dataset_path)
    steering_vectors = {}
    
    # Check if it's the complex/test format: {concept: [pos_list, neg_list]}
    for concept_name in tqdm(data.keys(), desc="Concepts"):
        pos_sentences = data[concept_name][0]
        neg_sentences = data[concept_name][1]
        
        # Compute mean of positive sentences
        pos_vecs_last, pos_vecs_avg = [], []
        for sentence in tqdm(pos_sentences, desc=f"{concept_name} (pos)", leave=False):
            v_l, v_a = compute_vector_single_prompt_local(model, tokenizer, dataset_path, sentence, layer_idx)
            pos_vecs_last.append(v_l)
            pos_vecs_avg.append(v_a)
        
        pos_mean_last = torch.stack(pos_vecs_last, dim=0).mean(dim=0).squeeze()
        pos_mean_avg = torch.stack(pos_vecs_avg, dim=0).mean(dim=0).squeeze()
        
        # Compute mean of negative sentences
        neg_vecs_last, neg_vecs_avg = [], []
        for sentence in tqdm(neg_sentences, desc=f"{concept_name} (neg)", leave=False):
            v_l, v_a = compute_vector_single_prompt_local(model, tokenizer, dataset_path, sentence, layer_idx)
            neg_vecs_last.append(v_l)
            neg_vecs_avg.append(v_a)
            
        neg_mean_last = torch.stack(neg_vecs_last, dim=0).mean(dim=0).squeeze()
        neg_mean_avg = torch.stack(neg_vecs_avg, dim=0).mean(dim=0).squeeze()
        
        steering_vectors[concept_name] = [pos_mean_last - neg_mean_last, pos_mean_avg - neg_mean_avg]
    
    print(f"\nComputed {len(steering_vectors)} local steering vectors.")
    return steering_vectors

def extract_layer_from_filename(path):
    if path is None: return None
    match = re.search(r"L(\d+)", Path(path).name)
    return int(match.group(1)) if match else None

def resolve_probe_path(probe_path):
    """Try to find the probe path relative to current dir, project root, or linear_probe/."""
    path = Path(probe_path)
    if path.exists(): return path
    # Try project root
    p_root = PROJECT_ROOT / probe_path
    if p_root.exists(): return p_root
    # Try from linear_probe/
    p_lp = PROJECT_ROOT / "linear_probe" / probe_path
    if p_lp.exists(): return p_lp
    return path

def main():
    parser = argparse.ArgumentParser(description="Compute cosine similarity between probes and the 'yes/injected' steering vector.")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name.")
    parser.add_argument("--probe1", type=str, required=True, help="Path to the first probe vector.")
    parser.add_argument("--probe2", type=str, default=None, help="Optional path to a second probe vector.")
    parser.add_argument("--layer", type=int, default=None, help="Optional layer override/fallback.")
    args = parser.parse_args()

    # 1. Resolve layers
    layer1 = extract_layer_from_filename(args.probe1) or args.layer
    layer2 = extract_layer_from_filename(args.probe2) or args.layer
    
    if layer1 is None:
        print("❌ Error: No layer number found in probe filename or --layer argument.")
        sys.exit(1)

    # 2. Load Model
    print(f"⏳ Loading model: {args.model}")
    # Using float16 to match probe precision and reduce memory
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"✅ Model loaded on {model.device}\n")

    # 3. Compute Steering Vector(s)
    dataset_path = "dataset/complex_yes_vector.json"
    print(f"🚀 Computing steering vector for 'injected' on dataset '{dataset_path}'...")
    
    yes_vectors = {}
    target_layers = sorted(set([l for l in [layer1, layer2] if l is not None]))
    
    for layer in target_layers:
        print(f"  Processing Layer {layer}...")
        try:
            results = compute_concept_vector_local(model, tokenizer, dataset_path, layer)
            
            # Prefer 'injected' if it exists, otherwise fall back to first key (likely 'yes')
            if "injected" in results:
                # results maps concept -> [last_vec, avg_vec]. we use average.
                yes_vectors[layer] = results["injected"][1].squeeze().float()
            elif "yes" in results:
                yes_vectors[layer] = results["yes"][1].squeeze().float()
            else:
                print(f"❌ Error: Reliable concept (injected/yes) not found in results: {results.keys()}")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Error processing layer {layer}: {e}")
            sys.exit(1)

    # 4. Result display
    print(f"\n{'='*70}")
    print(f"{'PROBE':<45} | {'LAYER':<5} | {'SIMILARITY':<10}")
    print(f"{'-'*70}")
    
    for i, probe_path in enumerate([args.probe1, args.probe2]):
        if probe_path is None: continue
            
        # Resolve and Load probe
        resolved_path = resolve_probe_path(probe_path)
        if not resolved_path.exists():
            print(f"❌ Error: Could not find probe at {probe_path}")
            continue
            
        probe = torch.load(resolved_path, map_location='cpu', weights_only=True).float()
        if probe.dim() > 1:
            probe = probe.flatten()
            
        # Get target layer and steering vector
        layer = layer1 if i == 0 else layer2
        yes_vec = yes_vectors[layer]
        
        # Cosine Similarity
        # cos_sim = (A . B) / (||A|| ||B||)
        cos_sim = torch.dot(probe, yes_vec) / (torch.norm(probe) * torch.norm(yes_vec) + 1e-9)
        
        print(f"{Path(probe_path).name[:45]:<45} | {layer:<5} | {cos_sim.item():.4f}")
    
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
