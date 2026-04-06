import torch
import os
import argparse
import sys
import time
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Internal project imports
from original_paper.compute_concept_vector_utils import get_data, compute_concept_vector
from model_utils.injection import inject_and_capture_anthropic
from model_utils.llm_judges import classify_response
from model_utils.logging import setup_logging

def save_results(args, stats, detection_categories, results_summary, save_root):
    """Saves the run summary to a JSON file."""
    summary_data = {
        "metadata": {
            "model": args.model,
            "dataset": args.dataset,
            "layer": args.layer,
            "coeff": args.coeff,
            "vector_type": args.vector_type,
            "max_new_tokens": args.max_new_tokens,
            "no_save_activations": args.no_save,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "stats": {cat: stats[cat] for cat in detection_categories},
        "results": results_summary
    }
    
    summary_path = save_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=4)
    
    print(f"📄 Summary JSON saved to: {summary_path}")
    print(f"✅ Run complete. Results directory: {save_root}")

def print_summary_table(stats, detection_categories, total_samples, icons):
    """Prints a formatted summary table to the console."""
    print(f"\n{'='*70}")
    print("📊 SWEEP SUMMARY")
    print(f"{'='*70}")
    for cat in detection_categories:
        count = stats[cat]
        pct = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"  {icons.get(cat, ' ')} {cat:<20}: {count:3} ({pct:5.1f}%)")
    
    if stats.get("error", 0) > 0:
        print(f"  ❌ Errors              : {stats['error']:3}")
    print(f"{'='*70}")

def main():
    parser = argparse.ArgumentParser(description="Sweep concepts, inject vectors, and save activations by category.")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Local model to run.")
    parser.add_argument("--judge_mode", type=str, default="gpt-5.4-nano", help="LLM Judge model.")
    parser.add_argument("--dataset", type=str, default="abstract_nouns_dataset", help="Dataset name.")
    parser.add_argument("--coeff", type=float, default=9.0, help="Injection coefficient.")
    parser.add_argument("--layer", type=int, default=15, help="Layer to inject the concept vector.")
    parser.add_argument("--vector_type", type=str, choices=["last", "average"], default="last", help="Type of steering vector to use.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum number of tokens to generate.")
    parser.add_argument("--save_dir", type=str, default="saved_activations", help="Root directory for saving results.")
    parser.add_argument("--no_save", action="store_true", help="Run the sweep without saving activations to disk.")
    parser.add_argument("--test", action="store_true", help="Run in test mode with fake data (no model loading).")
    parser.add_argument("--n_samples", type=int, default=None, help="Limit number of concepts to process.")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name for logging.")
    args = parser.parse_args()

    # 1. Setup Logging and Directory Structure
    detection_categories = [
        "not_detected",
        "detected_opposite",
        "detected_orthogonal",
        "detected_parallel",
        "detected_correct",
        "detected_unknown",
        "incoherent"
    ]
    
    save_root, log_file = setup_logging(
        base_save_dir=args.save_dir, 
        categories=detection_categories,
        run_name=args.run_name
    )

    icons = {
        "not_detected": "⚫",
        "detected_opposite": "🔴",
        "detected_orthogonal": "🟠",
        "detected_parallel": "🟡",
        "detected_correct": "🟢",
        "detected_unknown": "❓",
        "incoherent": "💀",
    }

    # Initialize stats
    stats = defaultdict(int)
    results_summary = []

    # 2. Handle Test Mode
    if args.test:
        import random
        print(f"🛠️  TEST MODE: Generating random data for sweep (L{args.layer} C{args.coeff})...")
        n_test = args.n_samples if args.n_samples else 10
        for i in range(n_test):
            concept = f"concept_{i}"
            category = random.choice(detection_categories)
            stats[category] += 1
            results_summary.append({
                "concept": concept, "category": category, "response_snippet": "Test mode generated snippet."
            })
        
        print_summary_table(stats, detection_categories, n_test, icons)
        save_results(args, stats, detection_categories, results_summary, save_root)
        log_file.close()
        return

    # 3. Load Model
    print(f"\n⏳ Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model loaded on {device}\n")

    # 4. Compute Concept Vectors
    print(f"🚀 Computing concept vectors for dataset '{args.dataset}' at layer {args.layer}...")
    steering_vectors_dict = compute_concept_vector(model, tokenizer, args.dataset, args.layer)
    
    concepts = list(steering_vectors_dict.keys())
    if args.n_samples:
        concepts = concepts[:args.n_samples]
        print(f"Limit applied: processing first {args.n_samples} concepts.")

    print(f"\n🏃 Starting sweep across {len(concepts)} concepts...")
    
    for concept in tqdm(concepts, desc="Processing Concepts"):
        # Get the steering vector (last=index 0, average=index 1)
        vec_idx = 0 if args.vector_type == "last" else 1
        steering_vector = steering_vectors_dict[concept][vec_idx] 
        
        try:
            response, _ = inject_and_capture_anthropic(
                model=model,
                tokenizer=tokenizer,
                steering_vector=steering_vector,
                layer_to_inject=args.layer,
                coeff=args.coeff,
                max_new_tokens=args.max_new_tokens
            )

            # 4. Display Results
            print(f"\n🗣️ Model Response:\n{'-'*30}\n{response}\n{'-'*30}")

            # Categorize Output
            category = classify_response(response, concept, model=args.judge_mode)

            stats[category] += 1
            icon = icons.get(category, "⚪")
            
            # Save activations
            if not args.no_save:
                save_path = save_root / category / f"{concept}_c{args.coeff}_l{args.layer}_v{args.vector_type}.pt"
                torch.save(activations_dict, save_path)
            
            results_summary.append({
                "concept": concept,
                "category": category,
                "response_snippet": response[:60].replace('\n', ' ') + "..."
            })
            
            # Interactive feedback in log
            tqdm.write(f"  {icon} [{concept}] -> {category}")

        except Exception as e:
            print(f"  ❌ Error processing '{concept}': {e}")
            stats["error"] += 1

    # 5. Final Summary
    print_summary_table(stats, detection_categories, len(concepts), icons)
    save_results(args, stats, detection_categories, results_summary, save_root)
    log_file.close()

if __name__ == "__main__":
    main()