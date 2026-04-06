from original_paper.compute_concept_vector import compute_concept_vector

dataset_name = "complex_yes_vector.json"

def extract_layer_from_filename(path):
    """
    Extracts the layer number (e.g., L14) from a filename using regex.
    """
    name = Path(path).name
    match = re.search(r"L(\d+)", name)
    if match:
        return int(match.group(1))
    return None

# 2. Load Probes
print(f"📡 Loading probes...")
p1 = torch.load(args.probe1, map_location='cpu', weights_only=True).float()
p2 = torch.load(args.probe2, map_location='cpu', weights_only=True).float()

# Ensure they are 1D vectors
if p1.dim() > 1: p1 = p1.flatten()
if p2.dim() > 1: p2 = p2.flatten()

# Normalize
p1_unit = p1 / (torch.norm(p1) + 1e-9)
p2_unit = p2 / (torch.norm(p2) + 1e-9)

# 1. Resolve layers for each probe
layer1 = extract_layer_from_filename(args.probe1)
layer2 = extract_layer_from_filename(args.probe2)

_, detection_steering_vector_avg = compute_concept_vector(model, tokenizer, dataset_name, layer1)

# compute cosine sim for each
# print

