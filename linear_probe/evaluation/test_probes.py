import torch
import sys
import argparse
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Imports from existing modules
from model_utils.injection import inject_concept_vector_and_probe
from model_utils.llm_judges import classify_response
from model_utils.logging import setup_logging
from original_paper.compute_concept_vector_utils import get_data, compute_concept_vector
from original_paper.all_prompts import get_anthropic_reproduce_messages
from linear_probe.evaluation.evaluation_prompt import evaluation_prompt
from linear_probe.calibation_correct_vs_detected_correct.unified_prompts import load_unified_prompt_for_detection

def parse_probe_name(filename):
    """
    Parses MM_L18_P0_dPrim22.88_dVal1.37.pt into metadata.
    """
    name = Path(filename).stem
    parts = name.split('_')
    
    metadata = {
        "file": filename,
        "type": parts[0],
        "layer": int(parts[1][1:]),
        "position": int(parts[2][1:]),
    }
    
    # Try to extract scores if present
    for p in parts:
        if p.startswith('dPrim'): metadata['dPrim'] = float(p[5:])
        if p.startswith('dVal'): metadata['dVal'] = float(p[4:])
        
    return metadata

def main():
    icons = {
        "not_detected": "⚫",
        "detected_opposite": "🔴",
        "detected_orthogonal": "🟠",
        "detected_parallel": "🟡",
        "detected_correct": "🟢",
        "detected_unknown": "❓",
        "incoherent": "💀",
        "false_positive" : "❌",
    }
    parser = argparse.ArgumentParser(description="Evaluate Introspection Probes (FPR and Steer Rate)")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", type=str, default="brysbaert_abstract_nouns.json")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--probe_coeffs", type=float, nargs="+", default=[-8.0, -5.0, -2.0, 0.0, 2.0, 5.0, 8.0])
    parser.add_argument("--concept_coeff", type=float, default=5.0)
    parser.add_argument("--concept_layer", type=int, default=14)
    parser.add_argument("--save_dir", type=str, default="evaluation_results")
    args = parser.parse_args()

    # 1. Setup Logging
    save_root, _ = setup_logging(args.save_dir)
    print(f"🚀 Initializing Evaluation Session: {save_root}")

    # 2. Discover and Load Probes
    probe_dirs = [
        PROJECT_ROOT / "linear_probe" / "calibation_correct_vs_detected_correct" / "probe_vectors",
        PROJECT_ROOT / "linear_probe" / "not_detected_vs_detected_correct" / "layer_and_position_sweep" / "probe_vectors"
    ]
    
    probes_to_test = []
    for d in probe_dirs:
        if not d.exists(): continue
        for f in d.glob("*.pt"):
            try:
                meta = parse_probe_name(f.name)
                meta['full_path'] = f
                probes_to_test.append(meta)
            except Exception as e:
                print(f"⚠️  Skipping probe {f.name} due to parse error: {e}")

    # Sort probes by layer then dPrim
    probes_to_test.sort(key=lambda x: (x['layer'], x.get('dPrim', 0)), reverse=True)
    print(f"📐 Found {len(probes_to_test)} probes to evaluate.")

    # 3. Load Model and Data
    print(f"⏳ Loading model: {args.model}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    print(f"📊 Loading dataset: {args.dataset}")
    data = get_data(args.dataset)
    if isinstance(data, dict) and "concept_vector_words" in data:
        concepts = data["concept_vector_words"]
    elif isinstance(data, dict):
        concepts = list(data.keys())
    else:
        concepts = data

    if args.n_samples:
        concepts = concepts[:args.n_samples]

    # Pre-compute concept vectors at the target concept_layer
    print(f"🧠 Pre-computing concept vectors at Layer {args.concept_layer}...")
    concept_vectors_dict = compute_concept_vector(model, tokenizer, args.dataset, args.concept_layer)

    # 4. Results Tracking
    # Results mapped as result[probe_file][prompt_name][coeff] = {fpr: X, steer_rate: Y}
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"fpr_hits": 0, "steer_hits": 0, "total": 0})))

    # Define the prompts to test
    eval_prompts = {
        "Evaluation": evaluation_prompt,
        "Anthropic": get_anthropic_reproduce_messages,
        "Unified": load_unified_prompt_for_detection
    }

    # 5. Core Evaluation Loop
    for probe_meta in probes_to_test:
        probe_name = probe_meta['file']
        probe_layer = probe_meta['layer']
        print(f"\n🔬 Testing Probe: {probe_name} (Layer {probe_layer})")
        probe_vec = torch.load(probe_meta['full_path'], map_location="cpu").to(model.device)

        # 1. Prepare Messages
        for prompt_name, prompt_fn in eval_prompts.items():
            print(f"  📝 Prompt Style: {prompt_name}")
            messages = prompt_fn()
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 2. Calculate injection start token (at "\n\nTrial 1")
            trial_start_text = "\n\nTrial 1"
            trial_start_pos = prompt_text.find(trial_start_text)
            if trial_start_pos != -1:
                prefix = prompt_text[:trial_start_pos]
                injection_start_token = len(tokenizer.encode(prefix, add_special_tokens=False))
            else:
                injection_start_token = None
            
            # --- A. Measure False Positive Rate (Probe only, No Steer) ---
            # Run once per probe/coeff as a baseline before the main concept loop
            print(f"  🔍 Measuring FPR baselines for all coefficients...")
            baseline_concept = list(concept_vectors_dict.keys())[0]
            if baseline_concept in concept_vectors_dict:
                _, baseline_steering_vector = concept_vectors_dict[baseline_concept]
                for p_coeff in args.probe_coeffs:
                    fpr_response, _ = inject_concept_vector_and_probe(
                        model=model,
                        tokenizer=tokenizer,
                        steering_vector=baseline_steering_vector, # Not used when inject=False
                        layer_to_inject=args.concept_layer,
                        probe=probe_vec,
                        probe_layer_to_inject=probe_layer,
                        probe_coeff=p_coeff,
                        coeff=args.concept_coeff,
                        inference_prompt=prompt_text,
                        injection_start_token=injection_start_token,
                        inject=False, # NO steering injection
                        max_new_tokens=30
                    )
                    fpr_cat = classify_response(fpr_response, baseline_concept)
                    is_fp = fpr_cat not in ["not_detected", "incoherent"]
                    if is_fp:
                        results[probe_name][prompt_name][p_coeff]["fpr_hits"] += 1
                    results[probe_name][prompt_name][p_coeff]["fpr_total"] = 1
                    
                    icon = icons.get("false_positive" if is_fp else "not_detected", "❓")
                    print(f"    Coeff {p_coeff:5.1f} | FPR Check | {icon} Category: {fpr_cat}")
                    print("Model Response")
                    print(fpr_response)

            # --- B. Measure Steering Rate (Concept + Probe) ---
            for concept in tqdm(concepts, desc=f"    Probing"):
                # Get the pre-computed steering vector for this concept
                if concept not in concept_vectors_dict:
                    print(f"⚠️  Concept '{concept}' not found in pre-computed vectors. Skipping.")
                    continue
                # compute_concept_vector returns (last_token_vec, avg_vec)
                _, steering_vector = concept_vectors_dict[concept]
                
                print(f"\n    📝 Concept: {concept}")
                for p_coeff in args.probe_coeffs:
                    steer_response, _ = inject_concept_vector_and_probe(
                        model=model,
                        tokenizer=tokenizer,
                        steering_vector=steering_vector,
                        layer_to_inject=args.concept_layer,
                        probe=probe_vec,
                        probe_layer_to_inject=probe_layer,
                        probe_coeff=p_coeff,
                        coeff=args.concept_coeff,
                        inference_prompt=prompt_text,
                        injection_start_token=injection_start_token,
                        inject=True, # YES steering injection
                        max_new_tokens=30
                    )
                    
                    steer_cat = classify_response(steer_response, concept)
                    if steer_cat == "detected_correct":
                        results[probe_name][prompt_name][p_coeff]["steer_hits"] += 1
                    
                    icon = icons.get(steer_cat, "❓")
                    print(f"      {icon} Coeff: {p_coeff:5.1f} | Category: {steer_cat:18}.")
                    print("Model Response")
                    print(steer_response)
                    
                    results[probe_name][prompt_name][p_coeff]["total"] += 1

            # Final prompt summary printing
            print(f"\n    📊 Final Summary for {prompt_name}:")
            for p_coeff in args.probe_coeffs:
                total = results[probe_name][prompt_name][p_coeff]["total"]
                fpr_total = results[probe_name][prompt_name][p_coeff].get("fpr_total", 0)
                fpr = (results[probe_name][prompt_name][p_coeff]["fpr_hits"] / fpr_total) * 100 if fpr_total > 0 else 0
                sr = (results[probe_name][prompt_name][p_coeff]["steer_hits"] / total) * 100 if total > 0 else 0
                print(f"      ▶ Coeff {p_coeff:5.1f} | FPR: {fpr:5.1f}% | Steer Rate: {sr:5.1f}%")

    # 6. Final Summary Export
    summary_file = save_root / "final_evaluation_summary.csv"
    with open(summary_file, "w") as f:
        f.write("Probe,Prompt,Layer,Coeff,FPR_Rate,Steer_Rate\n")
        for probe_name, prompt_results in results.items():
            meta = next(m for m in probes_to_test if m['file'] == probe_name)
            for prompt_name, coeffs in prompt_results.items():
                for p_coeff, stats in coeffs.items():
                    total = stats["total"]
                    fpr_total = stats.get("fpr_total", 0)
                    fpr = (stats["fpr_hits"] / fpr_total) * 100 if fpr_total > 0 else 0
                    sr = (stats["steer_hits"] / total) * 100 if total > 0 else 0
                    f.write(f"{probe_name},{prompt_name},{meta['layer']},{p_coeff},{fpr:.2f},{sr:.2f}\n")

    print(f"\n✅ Evaluation Complete! Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()