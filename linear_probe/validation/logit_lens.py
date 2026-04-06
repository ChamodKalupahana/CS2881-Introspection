import torch
import argparse
import sys
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import einops
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def resolve_probe_path(probe_path):
    """Try to find the probe path relative to current dir, project root, or linear_probe/."""
    path = Path(probe_path)
    if path.exists(): return path
    p_root = PROJECT_ROOT / probe_path
    if p_root.exists(): return p_root
    p_lp = PROJECT_ROOT / "linear_probe" / probe_path
    if p_lp.exists(): return p_lp
    return path

def format_token_label(token, idx):
    """
    Format token for display, preventing 'tofu' artifacts by adding hex fallbacks
    for characters likely missing from standard terminal/plot fonts.
    """
    if not token.strip() or any(ord(c) < 32 for c in token):
        # Handle whitespace-only or control characters
        return f"ID[{idx}] (repr: {repr(token)})"
    
    display = token.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    
    # Detect CJK or other symbols likely to fail rendering in DejaVu Sans
    # Threshold 0x2E80 covers CJK radicals, 0xAC00 covers Hangul, etc.
    needs_hex = any(ord(c) >= 0x2E80 for c in token)
    if needs_hex:
        hex_codes = [f"U+{ord(c):04X}" for c in token if ord(c) > 0x7F]
        return f"{display} ({' '.join(hex_codes)})"
    
    return display

def main():
    parser = argparse.ArgumentParser(description="Map a linear probe across the vocabulary using the model's unembedding.")
    parser.add_argument("--probe1", type=str, required=True, help="Path to the probe vector.")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name.")
    args = parser.parse_args()

    # 1. Resolve and Load Probe
    resolved_path = resolve_probe_path(args.probe1)
    if not resolved_path.exists():
        print(f"❌ Error: Probe file not found at {args.probe1}")
        sys.exit(1)
        
    print(f"📡 Loading probe: {resolved_path}")
    probe = torch.load(resolved_path, map_location='cpu', weights_only=True).float()
    if probe.dim() > 1:
        probe = probe.flatten()
    
    # 2. Load Model & Tokenizer
    print(f"⏳ Loading model: {args.model}")
    # Using device_map="auto" for efficiency
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"✅ Model loaded on {model.device}\n")

    # 3. Get Unembedding Matrix (W_U)
    # Shape: [vocab_size, hidden_dim]
    W_U = model.get_output_embeddings().weight.detach() 
    
    # 4. Project Probe into Vocabulary Space
    # Ensure probe and W_U are in the same device and dtype to prevent RuntimeError
    probe = probe.to(device=model.device, dtype=W_U.dtype)
    
    # einsum: probe is [d_model], W_U is [d_vocab, d_model] -> Result is [d_vocab]
    logits = einops.einsum(probe, W_U, "d_model, d_vocab d_model -> d_vocab")
    
    # 5. Extract Top and Bottom Tokens
    top_k = 10
    values, indices = torch.topk(logits, k=top_k)
    bottom_values, bottom_indices = torch.topk(logits, k=top_k, largest=False)

    # 6. Decode and Display
    print(f"\n{'='*60}")
    print(f"{'TOP ' + str(top_k) + ' TOKENS (Highest alignment)':^60}")
    print(f"{'='*60}")
    print(f"{'TOKEN':<35} | {'LOGIT SCORE':>15}")
    print(f"{'-'*60}")
    
    top_labels = []
    top_scores = []
    for val, idx in zip(values, indices):
        token = tokenizer.decode([idx.item()])
        token_display = format_token_label(token, idx.item())
        print(f"{token_display[:35]:<35} | {val.item():15.4f}")
        top_labels.append(token_display)
        top_scores.append(val.item())

    print(f"\n{'='*60}")
    print(f"{'BOTTOM ' + str(top_k) + ' TOKENS (Lowest alignment)':^60}")
    print(f"{'='*60}")
    print(f"{'TOKEN':<35} | {'LOGIT SCORE':>15}")
    print(f"{'-'*60}")
    
    bottom_labels = []
    bottom_scores = []
    for val, idx in zip(bottom_values, bottom_indices):
        token = tokenizer.decode([idx.item()])
        token_display = format_token_label(token, idx.item())
        print(f"{token_display[:35]:<35} | {val.item():15.4f}")
        bottom_labels.append(token_display)
        bottom_scores.append(val.item())
    print(f"{'='*60}\n")

    # 7. Plotting
    print(f"📊 Generating logit lens distribution plot...")
    plt.figure(figsize=(14, 10))
    
    # Interleave/Stack format: [worst_bottom...best_bottom, worst_top...best_top]
    labels = bottom_labels[::-1] + top_labels[::-1]
    scores = bottom_scores[::-1] + top_scores[::-1]
    # Use different colors for clarity
    colors = ['#ff6666'] * top_k + ['#66cc66'] * top_k
    
    y_pos = range(len(labels))
    plt.barh(y_pos, scores, color=colors, alpha=0.9)
    plt.yticks(y_pos, labels)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    # Derive a readable probe name and keep the ID for the filename
    probe_name = resolved_path.stem
    method_str = "Calibration" if "calibation_correct" in str(resolved_path) else "Not Detected"
    layer_match = re.search(r"L(\d+)", resolved_path.name)
    layer_str = f"Layer {layer_match.group(1)}" if layer_match else "Unknown Layer"
    pretty_name = f"{method_str} Probe {layer_str}"

    plt.xlabel('Logit Alignment Score')
    plt.title(f'Logit Lens: Vocabulary Alignment for {pretty_name}', pad=25)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    # Centralized output relative to project root
    plot_dir = PROJECT_ROOT / "plots/linear_probe/validation"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"logit_lens_{probe_name}.png"
    
    plt.savefig(plot_path, dpi=120)
    print(f"✅ Success! Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()