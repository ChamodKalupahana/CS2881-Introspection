"""
Plot per-head, per-layer contributions to the success direction.

Reads:  success_results/head_layer_contributions.pt
Saves:  plots/head_contribution_heatmap.png
        plots/mlp_contribution_bar.png
        plots/layer_total_contribution.png
        plots/top_heads_bar.png
        plots/per_concept_heatmaps.png

Usage:
    python plots/plot_head_contributions.py
    python plots/plot_head_contributions.py --input success_results/head_layer_contributions.pt
"""

import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path


def load_data(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def plot_head_heatmap(mean_head, capture_layer, save_dir):
    """Full [layers × heads] heatmap of projections onto success direction."""
    n_layers, n_heads = mean_head.shape
    data = mean_head.numpy()

    vmax = max(abs(data.min()), abs(data.max()))
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(20, 10))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax,
                   interpolation="nearest")

    ax.set_xlabel("Attention Head", fontsize=13)
    ax.set_ylabel("Layer", fontsize=13)
    ax.set_title("Per-Head Projection onto Success Direction", fontsize=15, fontweight="bold")

    ax.set_xticks(range(0, n_heads, 2))
    ax.set_xticklabels([f"H{h}" for h in range(0, n_heads, 2)], fontsize=8)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=8)

    # Mark capture layer
    ax.axhline(y=capture_layer - 0.5, color="lime", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.axhline(y=capture_layer + 0.5, color="lime", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.text(n_heads + 0.3, capture_layer, f"← capture L{capture_layer}",
            va="center", fontsize=9, color="green", fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.08)
    cbar.set_label("Projection onto success direction\n(+) → success  |  (−) → failure",
                   fontsize=10)

    fig.tight_layout()
    path = save_dir / "head_contribution_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {path}")


def plot_mlp_bar(mean_mlp, capture_layer, save_dir):
    """Bar chart of per-layer MLP projections."""
    n_layers = len(mean_mlp)
    data = mean_mlp.numpy()

    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in data]
    colors[capture_layer] = "#f1c40f"  # highlight capture layer

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(n_layers), data, color=colors, edgecolor="black", linewidth=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axvline(x=capture_layer, color="#f1c40f", linewidth=2, linestyle="--",
               alpha=0.6, label=f"Capture layer (L{capture_layer})")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("MLP Projection onto Success Direction", fontsize=12)
    ax.set_title("Per-Layer MLP Contribution", fontsize=14, fontweight="bold")
    ax.set_xticks(range(0, n_layers, 2))
    ax.set_xticklabels([f"L{l}" for l in range(0, n_layers, 2)], fontsize=9)
    ax.legend(fontsize=10)

    fig.tight_layout()
    path = save_dir / "mlp_contribution_bar.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {path}")


def plot_layer_totals(mean_head, mean_mlp, capture_layer, save_dir):
    """Stacked bar: total attention + MLP contribution per layer."""
    n_layers = len(mean_mlp)
    attn_total = mean_head.sum(dim=1).numpy()
    mlp_total = mean_mlp.numpy()

    x = np.arange(n_layers)
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 6))

    # Separate positive/negative for cleaner stacking
    ax.bar(x - width / 2, attn_total, width, label="Attention (all heads)",
           color="#3498db", edgecolor="black", linewidth=0.3)
    ax.bar(x + width / 2, mlp_total, width, label="MLP",
           color="#e67e22", edgecolor="black", linewidth=0.3)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axvline(x=capture_layer, color="#f1c40f", linewidth=2, linestyle="--",
               alpha=0.6, label=f"Capture layer (L{capture_layer})")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Total Projection onto Success Direction", fontsize=12)
    ax.set_title("Per-Layer Total Contribution (Attention vs MLP)", fontsize=14, fontweight="bold")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{l}" for l in range(n_layers)], fontsize=8, rotation=45)
    ax.legend(fontsize=10)

    fig.tight_layout()
    path = save_dir / "layer_total_contribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {path}")


def plot_top_heads(mean_head, save_dir, top_k=30):
    """Horizontal bar chart of top contributing heads."""
    n_heads = mean_head.shape[1]
    flat = mean_head.flatten()
    top_idx = flat.abs().topk(min(top_k, flat.numel())).indices

    labels = []
    values = []
    for idx in top_idx:
        l = idx.item() // n_heads
        h = idx.item() % n_heads
        labels.append(f"L{l:02d}.H{h:02d}")
        values.append(flat[idx].item())

    # Reverse for horizontal bar (top at top)
    labels = labels[::-1]
    values = values[::-1]

    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in values]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(range(len(labels)), values, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9, fontfamily="monospace")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("Projection onto Success Direction", fontsize=12)
    ax.set_title(f"Top {top_k} Attention Heads by |Contribution|", fontsize=14, fontweight="bold")

    fig.tight_layout()
    path = save_dir / "top_heads_bar.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {path}")


def plot_per_concept_heatmaps(per_concept_head, concept_names, capture_layer, save_dir):
    """Grid of per-concept head contribution heatmaps."""
    n_concepts = len(concept_names)
    cols = min(5, n_concepts)
    rows = (n_concepts + cols - 1) // cols

    # Global colorscale
    vmax = per_concept_head.abs().max().item()
    vmin = -vmax

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
                              squeeze=False)

    for i, name in enumerate(concept_names):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        data = per_concept_head[i].numpy()
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.axhline(y=capture_layer - 0.5, color="lime", linewidth=1, linestyle="--", alpha=0.5)
        ax.axhline(y=capture_layer + 0.5, color="lime", linewidth=1, linestyle="--", alpha=0.5)

        if c == 0:
            ax.set_ylabel("Layer", fontsize=9)
        if r == rows - 1:
            ax.set_xlabel("Head", fontsize=9)
        ax.set_xticks(range(0, per_concept_head.shape[2], 4))
        ax.set_yticks(range(0, per_concept_head.shape[1], 4))
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for i in range(n_concepts, rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].set_visible(False)

    fig.suptitle("Per-Concept Head Contributions to Success Direction",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02,
                 label="Projection onto success direction")

    fig.tight_layout()
    path = save_dir / "per_concept_heatmaps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot head/layer contributions")
    parser.add_argument("--input", type=str,
                        default="success_results/head_layer_contributions.pt")
    parser.add_argument("--output_dir", type=str, default="plots")
    args = parser.parse_args()

    # Resolve paths relative to repo root (one level up from plots/)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    input_path = repo_root / args.input
    # TODO: update save dir to be success/
    save_dir = repo_root / args.output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading: {input_path}")
    d = load_data(input_path)

    mean_head = d["mean_head_projections"]
    mean_mlp = d["mean_mlp_projections"]
    per_head = d["per_concept_head_projections"]
    names = d["concept_names"]
    cap = d["capture_layer"]

    print(f"   {mean_head.shape[0]} layers × {mean_head.shape[1]} heads, "
          f"{len(names)} concepts, capture_layer={cap}")
    print(f"   Concepts: {names}\n")

    print("📊 Generating plots...")
    plot_head_heatmap(mean_head, cap, save_dir)
    plot_mlp_bar(mean_mlp, cap, save_dir)
    plot_layer_totals(mean_head, mean_mlp, cap, save_dir)
    plot_top_heads(mean_head, save_dir)
    plot_per_concept_heatmaps(per_head, names, cap, save_dir)

    print(f"\n🎉 All plots saved to {save_dir}/")


if __name__ == "__main__":
    main()
