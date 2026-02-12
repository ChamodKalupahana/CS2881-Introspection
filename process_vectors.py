import torch
import os
import argparse
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_run_diff", action="store_true",
                        help="Calculate diff: avg(specific positive concepts) - avg(all not_detected) in run_dir")
    parser.add_argument("--run_dir", default="success_results/run_02_11_26_22_39",
                        help="Directory containing detected_correct and not_detected folders")
    parser.add_argument("--baseline_dir", default=None,
                        help="Directory containing baseline (not_detected) folder. Defaults to run_dir if not specified.")
    args = parser.parse_args()
    
    # Common config
    layers = range(16, 32)
    keys_to_avg = ["last_token", "prompt_mean", "generation_mean"]
    output_path = "success_results/average_success_vector_layers16-31.pt"
    
    avg_diff_activations = {}

    if args.use_run_diff:
        print(f"Using run difference mode (avg_positive - avg_negative) from {args.run_dir}")
        detect_dir = os.path.join(args.run_dir, "detected_correct")
        not_detect_dir = os.path.join(args.run_dir, "not_detected")
        
        if not os.path.exists(detect_dir) or not os.path.exists(not_detect_dir):
            print(f"Error: Run directory must contain 'detected_correct' and 'not_detected'. Missing in {args.run_dir}")
            return

        # 1. Identify specific positive concepts: Entropy, Magnetism, Satellites
        concepts = ["Entropy", "Magnetism", "Satellites"]
        pos_paths = []
        for c in concepts:
            matches = glob.glob(os.path.join(detect_dir, f"{c}_*.pt"))
            if matches:
                # Filter to ensure we capture relevant files (e.g. layers16-31 if present in filename, though content matters more)
                # Assume all matching files are relevant
                pos_paths.extend(matches)
            else:
                print(f"Warning: No file found for concept '{c}' in {detect_dir}")
        
        # 2. Identify all negative samples (not_detected)
        neg_paths = glob.glob(os.path.join(not_detect_dir, "*.pt"))
        
        print(f"Found {len(pos_paths)} positive files (Concepts: {concepts})")
        print(f"Found {len(neg_paths)} negative files (All not_detected)")
        
        if not pos_paths or not neg_paths:
            print("Error: Missing positive or negative files.")
            return

        # Helper to compute average activation per layer
        def compute_average(paths, layers, keys):
            # Accumulators: {layer: {key: sum_tensor}}
            # Counts: {layer: count}
            # Initialize
            sums = {l: {k: None for k in keys} for l in layers}
            counts = {l: 0 for l in layers}
            
            for p in paths:
                try:
                    data = torch.load(p, map_location="cpu")
                    acts = data.get("activations", {})
                    
                    for l in layers:
                        if l in acts:
                            l_acts = acts[l] # dict of tensors
                            for k in keys:
                                if k in l_acts:
                                    val = l_acts[k].float()
                                    if sums[l][k] is None:
                                        sums[l][k] = val
                                    else:
                                        sums[l][k] += val
                            counts[l] += 1
                except Exception as e:
                    print(f"Error loading {p}: {e}")
            
            # Average
            avgs = {}
            for l in layers:
                if counts[l] > 0:
                    avgs[l] = {}
                    for k in keys:
                        if sums[l][k] is not None:
                            avgs[l][k] = sums[l][k] / counts[l]
            return avgs, counts

        print("Computing averages...")
        avg_pos, count_pos = compute_average(pos_paths, layers, keys_to_avg)
        avg_neg, count_neg = compute_average(neg_paths, layers, keys_to_avg)
        
        print("Computing difference...")
        for l in layers:
            if l in avg_pos and l in avg_neg:
                diff = {}
                for k in keys_to_avg:
                    if k in avg_pos[l] and k in avg_neg[l]:
                        diff[k] = avg_pos[l][k] - avg_neg[l][k]
                
                if diff:
                    avg_diff_activations[l] = diff
                    norm = diff['last_token'].norm().item()
                    print(f"Layer {l}: Computed diff (Pos N={count_pos[l]}, Neg N={count_neg[l]}) | Norm: {norm:.4f}")
            else:
                print(f"Skipping Layer {l} (Missing data: Pos={l in avg_pos}, Neg={l in avg_neg})")

    else:
        # ORIGINAL LOGIC (averaged 3 specific files - 1 baseline file)
        print("Using original manual mode (averaged specific files - baseline)")
        print(f"Loading from run_dir: {args.run_dir}")

        detect_dir = os.path.join(args.run_dir, "detected_correct")
        
        # 1. Identify specific positive concepts: Entropy, Magnetism, Satellites
        concepts = ["Origami", "Magnetism", "Satellites"]
        success_paths = []
        for c in concepts:
            # Look for concept files in detected_correct
            matches = glob.glob(os.path.join(detect_dir, f"{c}_*.pt"))
            if matches:
                # If multiple matches, we take the first one or all? 
                # Original code had one per concept. Let's take all matching for robustness or just the first?
                # The prompt says "averaged specific files". 
                # Let's take all matches for these concepts to be safe (likely only one per concept per run usually).
                success_paths.extend(matches)
            else:
                print(f"Warning: No file found for concept '{c}' in {detect_dir}")
        
        if not success_paths:
            print("Error: No success files found.")
            return

        # 2. Identify baseline (Magnetism no-inject run)
        # Use baseline_dir if provided, else use run_dir
        baseline_dir = args.baseline_dir if args.baseline_dir else args.run_dir
        baseline_not_detect = os.path.join(baseline_dir, "not_detected")
        
        # Try to find *noinject* file first
        baseline_candidates = glob.glob(os.path.join(baseline_not_detect, "*noinject*.pt"))
        if not baseline_candidates:
            # Fallback: try looking for Magnetism in not_detected irrespective of "noinject" string?
            # Or maybe just specific Magnetism file?
            baseline_candidates = glob.glob(os.path.join(baseline_not_detect, "Magnetism_*.pt"))

        if not baseline_candidates:
             print(f"Error: No baseline file found in {baseline_not_detect} (looking for *noinject* or Magnetism*)")
             return
        
        baseline_path = baseline_candidates[0]
        if len(baseline_candidates) > 1:
            print(f"Warning: Multiple baseline candidates found. Using {baseline_path}")

        print("Loading success vectors...")
        success_data = []
        for p in success_paths:
            print(f"Loading {p}")
            try:
                success_data.append(torch.load(p, map_location="cpu"))
            except Exception as e:
                print(f"Error loading {p}: {e}")
        
        if not success_data:
             print("Error: Failed to load any success data.")
             return

        print(f"Loading baseline vector: {baseline_path}")
        try:
             baseline_data = torch.load(baseline_path, map_location="cpu")
        except Exception as e:
             print(f"Error loading baseline {baseline_path}: {e}"); return
        
        print("\nComputing average difference vectors per layer...")
        for layer in layers:
            # 1. Compute average success activation for this layer
            success_layer_acts = []
            for d in success_data:
                if "activations" in d and layer in d["activations"]:
                    success_layer_acts.append(d["activations"][layer])
            
            if not success_layer_acts:
                print(f"Warning: Layer {layer} not found in any success files."); continue
                
            avg_success_act = {}
            for key in keys_to_avg:
                # Check if key exists in all acts
                valid_acts = [act[key] for act in success_layer_acts if key in act]
                if valid_acts:
                     stacked = torch.stack(valid_acts)
                     avg_success_act[key] = torch.mean(stacked, dim=0)
        
            # 2. Get baseline activation for this layer
            if "activations" not in baseline_data or layer not in baseline_data["activations"]:
                print(f"Error: Layer {layer} not found in baseline file."); continue

            baseline_layer_act = baseline_data["activations"][layer]
            
            # 3. Subtract baseline
            diff_act = {}
            for key in keys_to_avg:
                if key in avg_success_act and key in baseline_layer_act:
                    diff_act[key] = avg_success_act[key] - baseline_layer_act[key]
            
            if diff_act:
                avg_diff_activations[layer] = diff_act
                if "last_token" in diff_act:
                    diff_norm = torch.norm(diff_act["last_token"]).item()
                    print(f"Layer {layer}: Computed diff | Norm (last_token): {diff_norm:.4f}")

    # Save the result
    if args.use_run_diff:
        output_path = "success_results/average_success_vector_layers16-31_run_diff.pt"
    
    torch.save(avg_diff_activations, output_path)
    print(f"\nSaved average difference vector to {output_path}")

if __name__ == "__main__":
    main()
