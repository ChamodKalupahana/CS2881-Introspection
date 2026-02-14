"""
Direct Logit Attribution (DLA) — per-layer attribution for attn and mlp.

For each layer L the score is:
    score_attn[L] = (attn_clean[L] - attn_corrupt[L]) · (W_U[tok_clean] - W_U[tok_corrupt])
    score_mlp[L]  = (mlp_clean[L]  - mlp_corrupt[L])  · (W_U[tok_clean] - W_U[tok_corrupt])

where:
    - attn/mlp activations use the "last_token" from the generation
    - W_U is the model's unembedding matrix (lm_head.weight)
    - tok_clean / tok_corrupt are the first tokens produced by tokenizing
      the concept names ("Satellites" / "Coral")

Usage:
    python DLA.py
    python DLA.py --clean_concept " Satellites" --corrupt_concept " Coral"
    python DLA.py --clean_concept " Satellites" --corrupt_concept " Coral" \
                  --clean_name Satellites --corrupt_name Coral
"""

import argparse
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


# ── helpers ──────────────────────────────────────────────────────────────────

def load_activations(concept_dir: Path, concept: str, sublayer: str, layers: list[int]):
    """Load activations for every requested layer, return dict[layer] → tensor."""
    acts = {}
    for layer in layers:
        fname = f"{concept}_{sublayer}_layer{layer}_coeff8.0_avg.pt"
        path = concept_dir / fname
        if not path.exists():
            print(f"  ⚠  Missing: {path}")
            continue
        data = torch.load(path, map_location="cpu")
        acts[layer] = data["activations"]  # dict with last_token, prompt_mean, etc.
    return acts


def get_unembed_direction(model_name: str, clean_concept: str, corrupt_concept: str,
                          clean_token_idx: int = 0, corrupt_token_idx: int = 0):
    """
    Return (W_U[tok_clean] - W_U[tok_corrupt]) as a 1-D tensor.
    Loads only lm_head + tokenizer (not the full model).
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from safetensors import safe_open
    import json, os, glob

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize concepts and show all tokens
    clean_ids = tokenizer.encode(clean_concept, add_special_tokens=False)
    corrupt_ids = tokenizer.encode(corrupt_concept, add_special_tokens=False)

    print(f"  Clean  concept: '{clean_concept}'")
    print(f"    All tokens: {[(i, tokenizer.decode([t]), t) for i, t in enumerate(clean_ids)]}")
    print(f"  Corrupt concept: '{corrupt_concept}'")
    print(f"    All tokens: {[(i, tokenizer.decode([t]), t) for i, t in enumerate(corrupt_ids)]}")

    clean_tok = clean_ids[clean_token_idx]
    corrupt_tok = corrupt_ids[corrupt_token_idx]
    print(f"  → Using clean  token[{clean_token_idx}]: id={clean_tok}  text='{tokenizer.decode([clean_tok])}'")
    print(f"  → Using corrupt token[{corrupt_token_idx}]: id={corrupt_tok}  text='{tokenizer.decode([corrupt_tok])}'")

    # Try to load just lm_head weights from safetensors index to save memory
    cache_dir = None
    try:
        from huggingface_hub import snapshot_download
        cache_dir = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json"])
    except Exception:
        pass

    lm_head_weight = None

    if cache_dir is not None:
        # Look for safetensors index
        index_path = os.path.join(cache_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            shard_file = index["weight_map"].get("lm_head.weight")
            if shard_file:
                shard_path = os.path.join(cache_dir, shard_file)
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    lm_head_weight = f.get_tensor("lm_head.weight")
                    print(f"  Loaded lm_head.weight from shard: {shard_file}  shape={lm_head_weight.shape}")
        else:
            # Single safetensors file
            sf_files = glob.glob(os.path.join(cache_dir, "*.safetensors"))
            for sf in sf_files:
                with safe_open(sf, framework="pt", device="cpu") as f:
                    if "lm_head.weight" in f.keys():
                        lm_head_weight = f.get_tensor("lm_head.weight")
                        print(f"  Loaded lm_head.weight from: {sf}  shape={lm_head_weight.shape}")
                        break

    if lm_head_weight is None:
        # Fallback: load entire model (slow, but works)
        print("  ⚠  Could not locate safetensors; loading full model for lm_head …")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        lm_head_weight = model.lm_head.weight.detach().cpu()
        del model

    unembed_dir = lm_head_weight[clean_tok].float() - lm_head_weight[corrupt_tok].float()
    return unembed_dir


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Direct Logit Attribution (DLA)")
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--clean_concept", type=str, default=" Satellites",
                        help="Token string for the clean concept (used for unembedding lookup)")
    parser.add_argument("--corrupt_concept", type=str, default=" Coral",
                        help="Token string for the corrupt concept (used for unembedding lookup)")
    parser.add_argument("--clean_name", type=str, default=None,
                        help="Directory/file name for the clean case (defaults to clean_concept.strip())")
    parser.add_argument("--corrupt_name", type=str, default=None,
                        help="Directory/file name for the corrupt case (defaults to corrupt_concept.strip())")
    parser.add_argument("--clean_token_idx", type=int, default=0,
                        help="Index of the token to use from the clean concept string (default: 0)")
    parser.add_argument("--corrupt_token_idx", type=int, default=0,
                        help="Index of the token to use from the corrupt concept string (default: 0)")
    parser.add_argument("--act_key", type=str, default="last_token",
                        choices=["last_token", "prompt_mean", "generation_mean"],
                        help="Which activation summary to use for the dot product")
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(16, 32)),
                        help="Layers to include (default: 16-31)")
    parser.add_argument("--data_root", type=str,
                        default=str(Path(__file__).resolve().parent / "attn_out_and_mlp_out"),
                        help="Root directory containing clean/ and corrupted/ subdirs")
    parser.add_argument("--output", type=str, default="DLA_logit_attribution.png",
                        help="Output plot file name")
    args = parser.parse_args()

    # Derive directory/file names from concept strings if not provided
    clean_name = args.clean_name or args.clean_concept.strip()
    corrupt_name = args.corrupt_name or args.corrupt_concept.strip()

    data_root = Path(args.data_root)
    clean_dir = data_root / "clean" / clean_name
    corrupt_dir = data_root / "corrupted" / corrupt_name
    layers = sorted(args.layers)

    print(f"Clean dir:   {clean_dir}")
    print(f"Corrupt dir: {corrupt_dir}")
    print(f"Clean token: '{args.clean_concept}'")
    print(f"Corrupt token: '{args.corrupt_concept}'")
    print(f"Layers:      {layers}")
    print(f"Act key:     {args.act_key}")

    # ── 1. Unembedding direction ─────────────────────────────────────────
    print("\n📐 Computing unembedding direction …")
    unembed_dir = get_unembed_direction(
        args.model, args.clean_concept, args.corrupt_concept,
        clean_token_idx=args.clean_token_idx,
        corrupt_token_idx=args.corrupt_token_idx,
    )
    print(f"  ||unembed_dir|| = {unembed_dir.norm():.4f}")

    # ── 2. Load activations ─────────────────────────────────────────────
    print("\n📂 Loading activations …")
    attn_clean = load_activations(clean_dir, clean_name, "attn", layers)
    attn_corrupt = load_activations(corrupt_dir, corrupt_name, "attn", layers)
    mlp_clean = load_activations(clean_dir, clean_name, "mlp", layers)
    mlp_corrupt = load_activations(corrupt_dir, corrupt_name, "mlp", layers)

    # ── 3. Compute DLA scores ──────────────────────────────────────────
    print("\n📊 Computing DLA scores …")
    attn_scores = {}
    mlp_scores = {}
    for layer in layers:
        if layer in attn_clean and layer in attn_corrupt:
            diff = attn_clean[layer][args.act_key].float() - attn_corrupt[layer][args.act_key].float()
            attn_scores[layer] = torch.dot(diff, unembed_dir).item()
        if layer in mlp_clean and layer in mlp_corrupt:
            diff = mlp_clean[layer][args.act_key].float() - mlp_corrupt[layer][args.act_key].float()
            mlp_scores[layer] = torch.dot(diff, unembed_dir).item()

    print(f"\n{'Layer':>6}  {'Attn DLA':>12}  {'MLP DLA':>12}")
    print(f"{'─'*6}  {'─'*12}  {'─'*12}")
    for layer in layers:
        a = attn_scores.get(layer, float("nan"))
        m = mlp_scores.get(layer, float("nan"))
        print(f"{layer:>6d}  {a:>12.4f}  {m:>12.4f}")

    # ── 4. Plot ──────────────────────────────────────────────────────────
    print("\n🎨 Plotting …")
    plot_layers = sorted(set(attn_scores.keys()) | set(mlp_scores.keys()))
    attn_vals = [attn_scores.get(l, 0) for l in plot_layers]
    mlp_vals = [mlp_scores.get(l, 0) for l in plot_layers]

    x = np.arange(len(plot_layers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars_attn = ax.bar(x - width / 2, attn_vals, width, label="Attention", color="#4C9BE8", edgecolor="white", linewidth=0.5)
    bars_mlp = ax.bar(x + width / 2, mlp_vals, width, label="MLP", color="#E8854C", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("DLA Score\n(activation diff · unembed diff)", fontsize=11)
    ax.set_title(
        f"Direct Logit Attribution — {clean_name} (clean) vs {corrupt_name} (corrupt)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in plot_layers])
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = Path(__file__).resolve().parent / args.output
    fig.savefig(out_path, dpi=150)
    print(f"✅ Saved plot → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
