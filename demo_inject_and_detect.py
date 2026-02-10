"""
Interactive demo: Choose a concept, compute its steering vector on the fly,
inject it into the model, and test whether the model detects the injection
using the Anthropic introspection prompt.

No pre-saved vectors needed — everything is computed at runtime.

Usage:
    python demo_inject_and_detect.py                        # interactive menu
    python demo_inject_and_detect.py --concept Dust         # skip menu, use "Dust"
    python demo_inject_and_detect.py --concept Dust --layer 15 --coeff 10
    python demo_inject_and_detect.py --concept "custom word" --layer 15 --coeff 10
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from compute_concept_vector_utils import compute_concept_vector
from inject_concept_vector import inject_concept_vector
from all_prompts import get_anthropic_reproduce_messages


# ── helpers ──────────────────────────────────────────────────────────────────

def load_model(model_name: str):
    """Load model + tokenizer and move to best available device."""
    print(f"\n⏳ Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model loaded on {device}\n")
    return model, tokenizer


def compute_vector_for_concept(model, tokenizer, concept: str, layer: int,
                                dataset_name: str = "simple_data",
                                vec_type: str = "avg"):
    """
    Compute the steering vector for a single concept at a given layer.

    Returns the steering vector tensor (last or avg, depending on vec_type).
    """
    print(f"⏳ Computing steering vector for '{concept}' at layer {layer} "
          f"(dataset={dataset_name}, vec_type={vec_type}) …")
    vectors = compute_concept_vector(model, tokenizer, dataset_name, layer)

    if concept not in vectors:
        raise ValueError(
            f"Concept '{concept}' not found in {dataset_name}. "
            f"Available concepts: {sorted(vectors.keys())}"
        )

    vec_last, vec_avg = vectors[concept]
    steering_vector = vec_avg if vec_type == "avg" else vec_last
    print(f"✅ Steering vector computed  (shape={steering_vector.shape})\n")
    return steering_vector


def run_injection_test(model, tokenizer, steering_vector, layer: int,
                       coeff: float, max_new_tokens: int = 100):
    """
    Inject the steering vector and run the Anthropic introspection prompt.

    Returns the model's response string.
    """
    messages = get_anthropic_reproduce_messages()
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Find injection start position (right before "Trial 1")
    trial_start_text = "\n\nTrial 1"
    trial_start_pos = formatted_prompt.find(trial_start_text)
    if trial_start_pos != -1:
        prefix = formatted_prompt[:trial_start_pos]
        injection_start_token = len(
            tokenizer.encode(prefix, add_special_tokens=False)
        )
    else:
        injection_start_token = None

    print(f"⏳ Injecting vector at layer {layer} with coeff={coeff} …")
    response = inject_concept_vector(
        model, tokenizer, steering_vector, layer,
        coeff=coeff,
        inference_prompt=formatted_prompt,
        assistant_tokens_only=True,
        max_new_tokens=max_new_tokens,
        injection_start_token=injection_start_token,
    )
    return response


def interactive_concept_menu():
    """Show a menu of available concepts and let the user pick one."""
    import json
    from pathlib import Path

    dataset_dir = Path(__file__).parent / "dataset"

    # Gather concepts from both datasets
    concepts_simple, concepts_complex = [], []

    with open(dataset_dir / "simple_data.json") as f:
        simple = json.load(f)
        concepts_simple = simple.get("concept_vector_words", [])

    with open(dataset_dir / "complex_data.json") as f:
        complex_data = json.load(f)
        concepts_complex = list(complex_data.keys())

    print("=" * 60)
    print("  Available concepts")
    print("=" * 60)

    print("\n── simple_data (single words) ──")
    for i, c in enumerate(concepts_simple, 1):
        print(f"  {i:>3}. {c}")

    offset = len(concepts_simple)
    print("\n── complex_data (multi-sentence concepts) ──")
    for i, c in enumerate(concepts_complex, offset + 1):
        print(f"  {i:>3}. {c}")

    all_concepts = concepts_simple + concepts_complex
    datasets = (["simple_data"] * len(concepts_simple) +
                ["complex_data"] * len(concepts_complex))

    while True:
        choice = input(
            f"\nEnter a number (1-{len(all_concepts)}) or type a concept name: "
        ).strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(all_concepts):
                return all_concepts[idx], datasets[idx]
            print("  ⚠  Number out of range, try again.")
        else:
            # Exact match first
            for concept, ds in zip(all_concepts, datasets):
                if concept.lower() == choice.lower():
                    return concept, ds
            # If not found in either dataset, treat as a custom word for simple_data
            print(f"  ℹ  '{choice}' is not in the datasets — will treat it as a "
                  f"custom simple_data word.")
            return choice, "simple_data"


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inject a concept vector and test introspection detection"
    )
    parser.add_argument("--concept", type=str, default=None,
                        help="Concept to inject (skip interactive menu)")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["simple_data", "complex_data"],
                        help="Which dataset the concept belongs to "
                             "(auto-detected if omitted)")
    parser.add_argument("--layer", type=int, default=15,
                        help="Layer to inject at (default: 15)")
    parser.add_argument("--coeff", type=float, default=10.0,
                        help="Injection coefficient / strength (default: 10.0)")
    parser.add_argument("--vec_type", type=str, default="avg",
                        choices=["avg", "last"],
                        help="Vector type: avg or last (default: avg)")
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Max tokens to generate (default: 100)")
    args = parser.parse_args()

    # ── 1. Pick a concept ───────────────────────────────────────────────────
    if args.concept:
        concept = args.concept
        if args.dataset:
            dataset_name = args.dataset
        else:
            # Auto-detect which dataset contains this concept
            import json
            from pathlib import Path
            dataset_dir = Path(__file__).parent / "dataset"

            with open(dataset_dir / "simple_data.json") as f:
                simple = json.load(f)
            with open(dataset_dir / "complex_data.json") as f:
                complex_data = json.load(f)

            if concept in simple.get("concept_vector_words", []):
                dataset_name = "simple_data"
            elif concept in complex_data:
                dataset_name = "complex_data"
            else:
                dataset_name = "simple_data"
                print(f"ℹ  '{concept}' not found in datasets — treating as "
                      f"custom simple_data word.\n")
    else:
        concept, dataset_name = interactive_concept_menu()

    print(f"\n{'=' * 60}")
    print(f"  Concept   : {concept}")
    print(f"  Dataset   : {dataset_name}")
    print(f"  Layer     : {args.layer}")
    print(f"  Coeff     : {args.coeff}")
    print(f"  Vec type  : {args.vec_type}")
    print(f"  Model     : {args.model}")
    print(f"{'=' * 60}\n")

    # ── 2. Load model ───────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model)

    # ── 3. Compute steering vector ──────────────────────────────────────────
    steering_vector = compute_vector_for_concept(
        model, tokenizer, concept, args.layer, dataset_name, args.vec_type
    )

    # ── 4. Inject & test ────────────────────────────────────────────────────
    response = run_injection_test(
        model, tokenizer, steering_vector, args.layer,
        args.coeff, args.max_new_tokens
    )

    # ── 5. Display results ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL RESPONSE")
    print("=" * 60)
    print(response)
    print("=" * 60)

    # Quick heuristic check
    concept_lower = concept.lower()
    response_lower = response.lower()
    detected = ("yes" in response_lower or "injected" in response_lower
                or "detect" in response_lower or concept_lower in response_lower)

    return response


if __name__ == "__main__":
    main()
