import compute_vector_single_prompt # TOOD: fix import

# copied from original_paper/compute_concept_vector_utils.py
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
            
    assert dataset_name in ("complex_data", "test_data"):
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

"""
for doing control of complex -> simple
compute baseline words activations helper function
"""
def extract_control_from_baseline(steering_vector_word : str):
    # specify baseline word and dataset
    # extract concept vector (that's normally injected)
    
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

    # Compute steering vectors for steering_vector_word word
    vec_last, vec_avg = compute_vector_single_prompt(model, tokenizer, dataset_name, steering_vector_word, layer_idx)
    vec_last = vec_last.squeeze() # shape [hidden_dim]
    vec_avg = vec_avg.squeeze() # shape [hidden_dim]
    steering_vector = {
        "last" : vec_last - baseline_mean_last,
        "avg" : vec_avg - baseline_mean_avg
    }

    return steering_vector
