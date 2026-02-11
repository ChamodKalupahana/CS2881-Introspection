import torch
import os

# Paths based on user request and previous discoveries
success_paths = [
    "success_results/run_02_11_26_22_39/detected_correct/Entropy_layers16-31_coeff8.0_avg.pt",
    "success_results/run_02_11_26_22_39/detected_correct/Magnetism_layers16-31_coeff8.0_avg.pt",
    "success_results/run_02_11_26_22_39/detected_correct/Satellites_layers16-31_coeff8.0_avg.pt",
]

# Baseline path (Magnetism no-inject run)
# Confirmed from find_by_name output: run_02_11_26_22_51/not_detected/Magnetism_layers16-31_noinject_avg.pt
baseline_path = "success_results/run_02_11_26_22_51/not_detected/Magnetism_layers16-31_noinject_avg.pt"

print("Loading success vectors...")
success_data = []
for p in success_paths:
    if not os.path.exists(p):
        print(f"Error: File not found: {p}")
        exit(1)
    print(f"Loading {p}")
    success_data.append(torch.load(p, map_location="cpu"))

print("Loading baseline vector...")
if not os.path.exists(baseline_path):
    print(f"Error: File not found: {baseline_path}")
    exit(1)
print(f"Loading {baseline_path}")
baseline_data = torch.load(baseline_path, map_location="cpu")

# Extract layers to process (16-31)
layers = range(16, 32)

# Store the result
avg_diff_activations = {}

# Keys to process (ignoring all_prompt/all_generation as they might vary in length)
keys_to_avg = ["last_token", "prompt_mean", "generation_mean"]

print("\nComputing average difference vectors per layer...")
for layer in layers:
    # 1. Compute average success activation for this layer
    # Access activations[layer] which returns the dict for that layer
    try:
        success_layer_acts = [d["activations"][layer] for d in success_data]
    except KeyError:
        print(f"Error: Layer {layer} not found in one of the success files.")
        continue
        
    avg_success_act = {}
    
    for key in keys_to_avg:
        # Stack and mean
        stacked = torch.stack([act[key] for act in success_layer_acts])
        avg_success_act[key] = torch.mean(stacked, dim=0)

    # 2. Get baseline activation for this layer
    try:
        baseline_layer_act = baseline_data["activations"][layer]
    except KeyError:
        print(f"Error: Layer {layer} not found in baseline file.")
        continue
    
    # 3. Subtract baseline
    diff_act = {}
    for key in keys_to_avg:
        diff_act[key] = avg_success_act[key] - baseline_layer_act[key]
        
    avg_diff_activations[layer] = diff_act
    # Calculate norm of the difference to show magnitude
    diff_norm = torch.norm(diff_act["last_token"]).item()
    print(f"Layer {layer}: Computed diff | Norm (last_token): {diff_norm:.4f}")

# Save the result
output_path = "success_results/average_success_vector_layers16-31.pt"
torch.save(avg_diff_activations, output_path)
print(f"\nSaved average difference vector to {output_path}")
print("Keys per layer:", keys_to_avg)
