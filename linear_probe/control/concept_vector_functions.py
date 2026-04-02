import sys
from pathlib import Path
import torch
from tqdm import tqdm
import einops

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from original_paper.compute_concept_vector_utils import compute_vector_single_prompt, get_data, get_model_type, format_prompt

# copied from original_paper/compute_concept_vector_utils.py

def compute_vector_all_layers(model, tokenizer, dataset_name, steering_prompt, min_layer_to_save):
    """
    Compute activation vectors for a single prompt across all layers above min_layer_to_save.
    
    Args:
        model: the model to use
        tokenizer: the tokenizer to use
        dataset_name: name of the dataset to format the prompt correctly
        steering_prompt: string of the input sentence/prompt
        min_layer_to_save: integer indicating the minimum layer index to start saving from
        
    Returns:
        last_vecs: dict mapping layer_idx to the prompt's last token activation vector
        avg_vecs: dict mapping layer_idx to the prompt's average activation vector
    """
    model_type = get_model_type(tokenizer)
    prompt = format_prompt(model_type, steering_prompt, dataset_name)
    
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    
    num_layers = model.config.num_hidden_layers
    layers_to_save = list(range(min_layer_to_save, num_layers + 1))
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True) # [batch_size, seq_len, hidden_dim]
        
        last_vecs = {}
        avg_vecs = {}
        for l in layers_to_save:
            last_vecs[l] = outputs.hidden_states[l][:, prompt_len - 1, :].detach().cpu() 
            avg_vecs[l] = outputs.hidden_states[l][:, :prompt_len, :].mean(dim=1).detach().cpu()
            
        del outputs 
    
    return last_vecs, avg_vecs

def compute_complex_concept_vector(model, tokenizer, dataset_name, concept_name : str, min_layer_to_save : int):
    """
    Extract positive and negative average variant activation distributions for a specific concept 
    across all layers above min_layer_to_save.
    
    Args:
        model: the model to use
        tokenizer: the tokenizer to use
        dataset_name: "simple_data" or "complex_data"
        concept_name: string specifying the concept to compute
        min_layer_to_save: integer indicating the minimum layer index to start saving from
        
    Returns:
        positive_avg: dict mapping layer_idx -> stacked average activations tensor [num_instances, hidden_dim]
        negative_avg: dict mapping layer_idx -> stacked average activations tensor [num_instances, hidden_dim]
        
    Method:
        - complex_data: Evaluates all positive and negative sentences for the concept and 
          aggregates their token-averaged layer representations.
    """
    data = get_data(dataset_name)
    positive_avg = {}
    negative_avg = {}
            
    assert dataset_name in ("complex_data", "test_data")
    if dataset_name in ("complex_data", "test_data"):
        # For concept: mean(positive) - mean(negative)
        print(f"concept_name: {concept_name}")
        pos_sentences = data[concept_name][0]  # List of positive examples
        neg_sentences = data[concept_name][1]  # List of negative examples
        
        print(f"\nProcessing {concept_name}: {len(pos_sentences)} pos, {len(neg_sentences)} neg")
        
        num_layers = model.config.num_hidden_layers
        layers_to_save = list(range(min_layer_to_save, num_layers + 1))
        
        pos_vecs_avg = {l: [] for l in layers_to_save}
        for sentence in tqdm(pos_sentences, desc=f"{concept_name} (positive)"):
            _, avg_vecs = compute_vector_all_layers(model, tokenizer, dataset_name, sentence, min_layer_to_save)
            for l in layers_to_save:
                pos_vecs_avg[l].append(avg_vecs[l])
        
        neg_vecs_avg = {l: [] for l in layers_to_save}
        for sentence in tqdm(neg_sentences, desc=f"{concept_name} (negative)"):
            _, avg_vecs = compute_vector_all_layers(model, tokenizer, dataset_name, sentence, min_layer_to_save)
            for l in layers_to_save:
                neg_vecs_avg[l].append(avg_vecs[l])
                
        # Store the distribution (all instances) for the average variants per layer
        for l in layers_to_save:
            positive_avg[l] = torch.stack(pos_vecs_avg[l], dim=0).squeeze()
            negative_avg[l] = torch.stack(neg_vecs_avg[l], dim=0).squeeze()
    
    print(f"\nComputed vectors for {len(layers_to_save)} layers")
    return positive_avg, negative_avg

"""
for doing control of complex -> simple
compute baseline words activations helper function
"""
def extract_control_from_baseline(model, tokenizer, dataset_name, layer_idx, baseline_words, steering_vector_word : str):
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
    vectors = {
        "last" : vec_last - baseline_mean_last,
        "avg" : vec_avg - baseline_mean_avg
    }

    return vectors
