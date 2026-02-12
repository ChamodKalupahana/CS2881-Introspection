"""
Compute the "success direction" for a given run.

Direction = mean(successful activations) − mean(failed activations)

  successful = detected_correct + detected_incorrect
  failed     = not_detected

The script loads every .pt file in the three category directories,
averages each activation type (last_token, prompt_mean, generation_mean)
across samples within each group, computes the difference, and saves
both the raw and unit-normalised direction vectors.

Output is saved to:  success_results/<run>/success_direction.pt
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# ── Configuration ────────────────────────────────────────────────────────────

SUCCESS_CATEGORIES = ["detected_correct", "detected_incorrect"]
FAILURE_CATEGORIES = ["not_detected"]

# Which activation types to compute the direction for
ACTIVATION_KEYS = ["last_token", "prompt_mean", "generation_mean"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_activations(category_dir: Path) -> list[dict[str, torch.Tensor]]:
    """Load all .pt files in a category directory and return their activations."""
    records = []
    if not category_dir.exists():
        return records
    for pt_file in sorted(category_dir.glob("*.pt")):
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        acts = data["activations"]
        records.append({k: acts[k].float() for k in ACTIVATION_KEYS})
        print(f"  loaded {pt_file.name}  (concept={data.get('concept', '?')})")
    return records


def mean_activation(records: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Average each activation key across all records."""
    if not records:
        return {}
    mean = {}
    for key in ACTIVATION_KEYS:
        stacked = torch.stack([r[key] for r in records])
        mean[key] = stacked.mean(dim=0)
    return mean


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute success direction from a run's saved activations"
    )
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to the run directory, e.g. success_results/run_02_11_26_18_55")
    args = parser.parse_args()

    RUN_DIR = Path(args.run_dir)
    if not RUN_DIR.exists():
        print(f"ERROR: Run directory not found: {RUN_DIR}")
        sys.exit(1)

    # ── Load successful activations ──────────────────────────────────────
    print("Loading SUCCESSFUL activations (detected_correct + detected_incorrect):")
    success_records = []
    for cat in SUCCESS_CATEGORIES:
        cat_dir = RUN_DIR / cat
        loaded = load_activations(cat_dir)
        print(f"  → {cat}: {len(loaded)} files")
        success_records.extend(loaded)

    # ── Load failed activations ──────────────────────────────────────────
    print("\nLoading FAILED activations (not_detected):")
    failure_records = []
    for cat in FAILURE_CATEGORIES:
        cat_dir = RUN_DIR / cat
        loaded = load_activations(cat_dir)
        print(f"  → {cat}: {len(loaded)} files")
        failure_records.extend(loaded)

    if not success_records:
        print("ERROR: No successful activations found.")
        sys.exit(1)
    if not failure_records:
        print("ERROR: No failed activations found.")
        sys.exit(1)

    # ── Compute direction ────────────────────────────────────────────────
    success_mean = mean_activation(success_records)
    failure_mean = mean_activation(failure_records)

    direction = {}
    direction_normed = {}

    print("\n── Results ────────────────────────────────────────────────────")
    print(f"  Successful samples: {len(success_records)}")
    print(f"  Failed samples:     {len(failure_records)}")
    print()

    for key in ACTIVATION_KEYS:
        raw = success_mean[key] - failure_mean[key]
        norm = raw.norm().item()
        unit = raw / raw.norm() if norm > 0 else raw

        direction[key] = raw
        direction_normed[key] = unit

        # Cosine similarity between success_mean and failure_mean
        cos_sim = torch.nn.functional.cosine_similarity(
            success_mean[key].unsqueeze(0),
            failure_mean[key].unsqueeze(0),
        ).item()

        print(f"  {key}:")
        print(f"    ‖direction‖ = {norm:.4f}")
        print(f"    cos(success, failure) = {cos_sim:.4f}")
        print()

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = RUN_DIR / "success_direction.pt"
    torch.save(
        {
            "run_dir": str(RUN_DIR),
            "n_success": len(success_records),
            "n_failure": len(failure_records),
            "success_categories": SUCCESS_CATEGORIES,
            "failure_categories": FAILURE_CATEGORIES,
            "direction": direction,          # raw difference vectors
            "direction_normed": direction_normed,  # unit vectors
            "success_mean": success_mean,
            "failure_mean": failure_mean,
        },
        out_path,
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
