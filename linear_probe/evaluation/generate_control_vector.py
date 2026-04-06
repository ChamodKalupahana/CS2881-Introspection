import torch
import os
import argparse
from pathlib import Path
import re

def main():
    parser = argparse.ArgumentParser(description="Generate a random control vector based on the best probe vector's shape.")
    parser.add_argument("--probe_dir", type=str, required=True, help="Directory containing .pt probe vectors.")
    args = parser.parse_args()

    probe_dir = Path(args.probe_dir)
    if not probe_dir.exists():
        print(f"Error: Directory {probe_dir} not found.")
        return

    # Pattern for parsing filename: type_Llayer_Ppos_dPrimVAL_dValVAL.pt
    # Example: PCA0_L14_P8_dPrim1.54_dVal1.22.pt
    pattern = re.compile(r"dPrim([\d\.]+)_dVal([\d\.]+)")
    
    best_score = -1
    best_file = None
    
    print(f"🔍 Searching for probe vectors in: {probe_dir}")
    
    for pf in probe_dir.glob("*.pt"):
        if pf.name == "control_random.pt":
            continue
            
        match = pattern.search(pf.name)
        if match:
            try:
                d_prim = float(match.group(1))
                d_val = float(match.group(2))
                # Using Euclidean distance of d-scores as the "best" metric, consistent with the analysis scripts
                score = (d_prim**2 + d_val**2)**0.5
                
                if score > best_score:
                    best_score = score
                    best_file = pf
            except ValueError:
                continue
                
    if best_file is None:
        print("❌ No valid probe vectors with Cohen's d scores found in the directory.")
        # Fallback: check for any .pt file to get a shape
        pt_files = list(probe_dir.glob("*.pt"))
        if not pt_files:
            return
        best_file = pt_files[0]
        print(f"⚠️  No scored vectors found. Using first available vector for shape: {best_file.name}")

    print(f"🏆 Best probe identified: {best_file.name} (Score: {best_score:.2f} if applicable)")
    
    # Load to get shape
    try:
        best_vector = torch.load(best_file, map_location='cpu', weights_only=True)
        # Ensure it's a tensor
        if not isinstance(best_vector, torch.Tensor):
            print(f"❌ Loaded object is not a torch.Tensor ({type(best_vector)})")
            return
            
        shape = best_vector.shape
        print(f"📐 Vector shape: {shape}")
        
        # Create control vector (Standard Normal)
        control_vector = torch.randn(shape, dtype=torch.float32)
        
        # Save
        save_path = probe_dir / "control_random.pt"
        torch.save(control_vector, save_path)
        print(f"✅ Generated and saved random control vector to: {save_path}")
        
    except Exception as e:
        print(f"❌ Error processing vector: {e}")

if __name__ == "__main__":
    main()
