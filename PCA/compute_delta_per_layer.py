import torch
import os
import argparse
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def load_activations(path):
    """Load activations from .pt file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    # The saved data structure from save_vectors_by_layer.py has 'activations' key
    # containing 'last_token', 'prompt_mean', etc.
    # We want the 'last_token' vector which represents the state at the end of the prompt.
    return data['activations']['last_token']

def main():
    parser = argparse.ArgumentParser(description="Compute PCA deltas (Injected - Clean)")
    # parser.add_argument("--layer", type=int, required=True, help="Layer number to process") # Removed, iterating 16-31
    parser.add_argument("--injected_dir", type=str, default="injected_correct", 
                        help="Directory containing injected correct runs")
    parser.add_argument("--clean_dir", type=str, default="no_inject", 
                        help="Directory containing clean (no-inject) runs")
    parser.add_argument("--coeff", type=float, default=8.0, help="Coefficient to look for")
    parser.add_argument("--output_path", type=str, default="all_deltas.pt", help="Path to save results")
    
    args = parser.parse_args()

    # Resolve paths relative to PCA directory if they are relative
    script_dir = Path(__file__).parent
    injected_root = Path(args.injected_dir)
    if not injected_root.is_absolute():
        injected_root = script_dir / args.injected_dir
        
    clean_root = Path(args.clean_dir)
    if not clean_root.is_absolute():
        clean_root = script_dir / args.clean_dir
        
    print(f"Injected Dir: {injected_root}")
    print(f"Clean Dir: {clean_root}")

    # Create output directory
    if args.output_path != "all_deltas.pt":
         output_dir = Path(args.output_path).parent
         if not output_dir.is_absolute():
             output_dir = script_dir / output_dir
    else:
        output_dir = script_dir / "PCA_components"
    
    output_dir.mkdir(exist_ok=True, parents=True)

    # Loop over layers 16 to 31
    for layer in range(16, 32):
        print(f"\n{'='*40}")
        print(f"Processing Layer: {layer}")
        print(f"{'='*40}")

        all_deltas = []
        concepts_processed = []

        # Iterate over concepts in injected_root
        # Structure: injected_root / concept / filename.pt
        if not injected_root.exists():
            print(f"Error: Injected directory {injected_root} does not exist.")
            return

        # List all concept directories
        concept_dirs = [d for d in injected_root.iterdir() if d.is_dir()]
        
        for concept_dir in sorted(concept_dirs):
            concept = concept_dir.name
            
            # Construct filename pattern: {concept}_layer{layer}_coeff{coeff}_avg.pt
            file_pattern = f"{concept}_layer{layer}_coeff{args.coeff}_avg.pt"
            injected_file = concept_dir / file_pattern
            
            if not injected_file.exists():
                potential_files = list(concept_dir.glob(f"*_layer{layer}_coeff{args.coeff}_*.pt"))
                if not potential_files:
                    # print(f"Skipping {concept}: Injected file not found for layer {layer}, coeff {args.coeff}")
                    continue
                injected_file = potential_files[0]

            # Find matching clean file
            clean_concept_dir = clean_root / concept
            if not clean_concept_dir.exists():
                # print(f"Skipping {concept}: corresponding clean directory not found")
                continue
                
            clean_file_pattern = f"{concept}_layer{layer}_noinject_avg.pt"
            clean_file = clean_concept_dir / clean_file_pattern
            
            if not clean_file.exists():
                 potential_files = list(clean_concept_dir.glob(f"*_layer{layer}_noinject_*.pt"))
                 if not potential_files:
                     # print(f"Skipping {concept}: Clean file not found for layer {layer}")
                     continue
                 clean_file = potential_files[0]

            try:
                vec_injected = load_activations(injected_file)
                vec_clean = load_activations(clean_file)
                
                # Compute delta
                delta = vec_injected - vec_clean
                all_deltas.append(delta)
                concepts_processed.append(concept)
                
            except Exception as e:
                print(f"Error processing {concept}: {e}")
                continue

        if not all_deltas:
            print(f"No deltas computed for layer {layer}.")
            continue

        # Stack all deltas
        deltas_tensor = torch.stack(all_deltas) # [num_concepts, hidden_size]
        print(f"Computed deltas shape: {deltas_tensor.shape}")
        
        # Save results - raw deltas
        # Using output_path as a base pattern or ignores it in favor of fixed structure
        # Let's save to PCA/PCA_components/all_deltas_layer{layer}.pt
        
        deltas_save_path = output_dir / f"all_deltas_layer{layer}.pt"
        
        result = {
            "deltas": deltas_tensor,
            "concepts": concepts_processed,
            "layer": layer,
            "coeff": args.coeff
        }
        
        torch.save(result, deltas_save_path)
        print(f"Saved results to {deltas_save_path}")



if __name__ == "__main__":
    main()

