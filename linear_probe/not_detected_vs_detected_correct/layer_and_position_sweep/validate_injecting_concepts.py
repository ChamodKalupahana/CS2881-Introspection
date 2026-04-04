import argparse
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Internal project imports
from model_utils.injection import inject_and_capture_anthropic

def validate_anthropic_injection(model_name, layer_to_inject=15, coeff=9.0):
    """
    Validation script for the Anthropic reproduction workflow.
    """
    # 1. Load model
    print(f"\n⏳ Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model loaded on {device}\n")

    # 2. Create a dummy steering vector
    hidden_dim = model.config.hidden_size
    dummy_vector = torch.randn(hidden_dim).to(device)
    
    print(f"🚀 Running validation:")
    print(f"  Injection Layer: {layer_to_inject}")
    print(f"  Coefficient: {coeff}")
    print(f"  Prompt: Anthropic Introspection Reproduce\n")
    
    # 3. Inject and Capture
    response, activations = inject_and_capture_anthropic(
        model=model,
        tokenizer=tokenizer,
        steering_vector=dummy_vector,
        layer_to_inject=layer_to_inject,
        coeff=coeff,
        max_new_tokens=50
    )
    
    # 4. Display Results
    print(f"\n🗣️ Model Response:\n{'-'*30}\n{response}\n{'-'*30}")
    
    print("\n📦 Capture Results:")
    # Group by layer for cleaner output
    unique_layers = sorted(set(k[0] for k in activations.keys()))
    print(f"  Layers Captured: {len(unique_layers)} ({min(unique_layers)} to {max(unique_layers)})")
    
    # Print a mapping for the first captured layer as a sample
    first_layer = unique_layers[0]
    positions = sorted([k[1] for k in activations.keys() if k[0] == first_layer])
    print(f"  Positions Captured per layer: {positions}")
    print(f"  Individual Vector Shape: {list(activations[(first_layer, 0)].shape)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Anthropic injection and activation capture.")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--coeff", type=float, default=9.0)
    args = parser.parse_args()
    
    validate_anthropic_injection(args.model, args.layer, args.coeff)
