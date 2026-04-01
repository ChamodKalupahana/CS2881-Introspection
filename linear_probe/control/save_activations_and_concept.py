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

def extract_control_from_baseline():
    # specify baseline word
    # extract concept vector (that's normally injected)