import torch
import os
import sys
import argparse
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Internal project imports
from original_paper.compute_concept_vector_utils import compute_concept_vector

def extract_layer_from_filename(path):
    """
    Extracts the layer number (e.g., L14) from a filename using regex.
    """
    if path is None:
        return None
    name = Path(path).name
    match = re.search(r"L(\d+)", name)
    if match:
        return int(match.group(1))
    return None

def main():
    parser = argparse.ArgumentParser(description="Compute cosine similarity between probes and the 'yes' steering vector.")
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
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model loaded on {device}\n")

    # 3. Compute Steering Vector(s)
    dataset_path = "dataset/complex_yes_vector.json"
    print(f"🚀 Computing steering vector for 'yes' (injected) on dataset '{dataset_path}'...")
    
    # Store yes_vectors by layer to avoid re-computation if both probes share a layer
    yes_vectors = {}
    target_layers = set([l for l in [layer1, layer2] if l is not None])
    
    for layer in sorted(target_layers):
        print(f"  Layer {layer}...")
        results = compute_concept_vector(model, tokenizer, dataset_path, layer)
        # Extract "injected" avg vector (index 1)
        if "injected" in results:
            yes_vectors[layer] = results["injected"][1].squeeze().float()
        else:
            print(f"❌ Error: 'injected' concept not found in dataset {dataset_path}")
            sys.exit(1)

    # 4. Process Probes
    print(f"\n{'='*60}")
    print(f"{'PROBE':<40} | {'LAYER':<5} | {'SIMILARITY':<10}")
    print(f"{'-'*60}")
    
    for i, probe_path in enumerate([args.probe1, args.probe2]):
        if probe_path is None:
            continue
            
        # Load probe
        probe = torch.load(probe_path, map_location='cpu', weights_only=True).float()
        if probe.dim() > 1:
            probe = probe.flatten()
            
        # Get target layer and steering vector
        layer = layer1 if i == 0 else layer2
        yes_vec = yes_vectors[layer]
        
        # Cosine Similarity
        # cos_sim = (A . B) / (||A|| ||B||)
        cos_sim = torch.dot(probe, yes_vec) / (torch.norm(probe) * torch.norm(yes_vec) + 1e-9)
        
        print(f"{Path(probe_path).name[:40]:<40} | {layer:<5} | {cos_sim.item():.4f}")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
