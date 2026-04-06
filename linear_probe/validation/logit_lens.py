import torch
import argparse
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import einops
import matplotlib.pyplot as plt

def compute_concept_vector(model, tokenizer, dataset_name, layer_idx):
    """
    Compute steering vectors for all concepts in the dataset
    
    Args:
        model: the model to use
        tokenizer: the tokenizer to use
        dataset_name: "simple_data" or "complex_data"
        layer_idx: the layer index to compute steering vectors for
        
    Returns:
        dict: {concept_name: [prompt_last_steering_vector, prompt_average_steering_vector]}
        
    Method:
        - simple_data: For each word: vector(word) - mean(vector(baseline_word) for all baselines)
        - complex_data: For each concept: mean(vectors(pos_sentences)) - mean(vectors(neg_sentences))
    """
    data = get_data(dataset_name)
    steering_vectors = {}
    
    if dataset_name and (dataset_name.startswith("simple_data") or dataset_name == "abstract_nouns_dataset" or "brysbaert" in dataset_name):
        concept_words = data["concept_vector_words"]
        baseline_words = data["baseline_words"]
        
        # Compute baseline means once (used for all concepts)
        print(f"Computing baseline mean from {len(baseline_words)} words...")
        baseline_vecs_last = []
        baseline_vecs_avg = []
        for word in tqdm(baseline_words, desc="Baseline vectors"):
            vec_last, vec_avg = compute_vector_single_prompt(model, tokenizer, dataset_name, word, layer_idx)
            baseline_vecs_last.append(vec_last)
            baseline_vecs_avg.append(vec_avg)
        baseline_mean_last = torch.stack(baseline_vecs_last, dim=0).mean(dim=0).squeeze() # shape [hidden_dim]
        baseline_mean_avg = torch.stack(baseline_vecs_avg, dim=0).mean(dim=0).squeeze() # shape [hidden_dim]
        
        # Compute steering vectors for each concept word
        for word in tqdm(concept_words, desc="Concept vectors"):
            vec_last, vec_avg = compute_vector_single_prompt(model, tokenizer, dataset_name, word, layer_idx)
            vec_last = vec_last.squeeze() # shape [hidden_dim]
            vec_avg = vec_avg.squeeze() # shape [hidden_dim]
            steering_vectors[word] = [vec_last - baseline_mean_last, vec_avg - baseline_mean_avg]
            
    elif dataset_name in ("complex_data", "test_data", "complex_yes_vector.json"):
        # For each concept: mean(positive) - mean(negative)
        print(f"data keys: {data.keys()}")
        for concept_name in data.keys():
            print(f"concept_name: {concept_name}")
            pos_sentences = data[concept_name][0]  # List of positive examples
            neg_sentences = data[concept_name][1]  # List of negative examples
            
            print(f"\nProcessing {concept_name}: {len(pos_sentences)} pos, {len(neg_sentences)} neg")
            
            # Compute mean of positive sentences
            pos_vecs_last = []
            pos_vecs_avg = []
            for sentence in tqdm(pos_sentences, desc=f"{concept_name} (positive)"):
                vec_last, vec_avg = compute_vector_single_prompt(model, tokenizer, dataset_name, sentence, layer_idx)
                pos_vecs_last.append(vec_last)
                pos_vecs_avg.append(vec_avg)
            pos_mean_last = torch.stack(pos_vecs_last, dim=0).mean(dim=0).squeeze()
            pos_mean_avg = torch.stack(pos_vecs_avg, dim=0).mean(dim=0).squeeze()
            
            # Compute mean of negative sentences
            neg_vecs_last = []
            neg_vecs_avg = []
            for sentence in tqdm(neg_sentences, desc=f"{concept_name} (negative)"):
                vec_last, vec_avg = compute_vector_single_prompt(model, tokenizer, dataset_name, sentence, layer_idx)
                neg_vecs_last.append(vec_last)
                neg_vecs_avg.append(vec_avg)
            neg_mean_last = torch.stack(neg_vecs_last, dim=0).mean(dim=0).squeeze()
            neg_mean_avg = torch.stack(neg_vecs_avg, dim=0).mean(dim=0).squeeze()
            
            # Steering vectors = positive - negative (both last and avg)
            steering_vectors[concept_name] = [pos_mean_last - neg_mean_last, pos_mean_avg - neg_mean_avg]
    
    print(f"\nComputed {len(steering_vectors)} steering vectors (each with last and avg variants)")
    return steering_vectors

def main():
    parser = argparse.ArgumentParser(description="Map a linear probe across the vocabulary using the model's unembedding.")
    parser.add_argument("--probe1", type=str, required=True, help="Path to the probe vector.")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name.")
    args = parser.parse_args()

    # 1. Load Probe
    if not Path(args.probe1).exists():
        print(f"❌ Error: Probe file not found at {args.probe1}")
        sys.exit(1)
        
    print(f"📡 Loading probe: {args.probe1}")
    probe = torch.load(args.probe1, map_location='cpu', weights_only=True).float()
    if probe.dim() > 1:
        probe = probe.flatten()
    
    # 2. Load Model & Tokenizer
    print(f"⏳ Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ Model loaded on {device}\n")

    # 3. Get Unembedding Matrix (W_U)
    # Shape: [vocab_size, hidden_dim]
    W_U = model.get_output_embeddings().weight.detach() 
    
    # 4. Project Probe into Vocabulary Space
    # Compute dot product of probe with every token vector in W_U
    probe = probe.to(device=device, dtype=W_U.dtype)
    
    # einsum: probe is [d_model], W_U is [d_vocab, d_model] -> Result is [d_vocab]
    logits = einops.einsum(probe, W_U, "d_model, d_vocab d_model -> d_vocab")
    
    # 5. Extract Top and Bottom Tokens
    top_k = 10
    values, indices = torch.topk(logits, k=top_k)
    bottom_values, bottom_indices = torch.topk(logits, k=top_k, largest=False)

    # 6. Decode and Display
    print(f"\n{'='*55}")
    print(f"{'TOP ' + str(top_k) + ' TOKENS (Highest alignment)':^55}")
    print(f"{'='*55}")
    print(f"{'TOKEN':<30} | {'LOGIT SCORE':>15}")
    print(f"{'-'*55}")
    
    top_labels = []
    top_scores = []
    for val, idx in zip(values, indices):
        token = tokenizer.decode([idx.item()])
        # Escape newlines for display
        token_display = token.replace('\n', '\\n').replace('\r', '\\r')
        if not token_display.strip():
            token_display = f"ID[{idx.item()}]"
        print(f"{token_display[:30]:<30} | {val.item():15.4f}")
        top_labels.append(token_display)
        top_scores.append(val.item())

    print(f"\n{'='*55}")
    print(f"{'BOTTOM ' + str(top_k) + ' TOKENS (Lowest alignment)':^55}")
    print(f"{'='*55}")
    print(f"{'TOKEN':<30} | {'LOGIT SCORE':>15}")
    print(f"{'-'*55}")
    
    bottom_labels = []
    bottom_scores = []
    for val, idx in zip(bottom_values, bottom_indices):
        token = tokenizer.decode([idx.item()])
        token_display = token.replace('\n', '\\n').replace('\r', '\\r')
        if not token_display.strip():
            token_display = f"ID[{idx.item()}]"
        print(f"{token_display[:30]:<30} | {val.item():15.4f}")
        bottom_labels.append(token_display)
        bottom_scores.append(val.item())
    print(f"{'='*55}\n")

    # 7. Plotting
    print(f"📊 Generating logit lens distribution plot...")
    plt.figure(figsize=(12, 10))
    
    # Reverse both for vertical stack: Bottoms grouped at bottom, Tops grouped at top
    # The order will be [worst_bottom, ..., best_bottom, worst_top, ..., best_top]
    labels = bottom_labels[::-1] + top_labels[::-1]
    scores = bottom_scores[::-1] + top_scores[::-1]
    colors = ['#ff4444'] * top_k + ['#44bb44'] * top_k
    
    y_pos = range(len(labels))
    plt.barh(y_pos, scores, color=colors, alpha=0.8)
    plt.yticks(y_pos, labels)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.xlabel('Logit Alignment Score')
    plt.title(f'Top and Bottom Token Alignment (Probe: {Path(args.probe1).name})', pad=20)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_path = Path("linear_probe/validation/logit_lens_distribution.png")
    plt.savefig(plot_path)
    print(f"✅ Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()