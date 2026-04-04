python save_activations_by_layer_and_position.py \
    --dataset simple_data.json \
    --layer 16 \
    --coeff 8.0 \
    --n_samples 10 \
    --vector_type average \
    --max_new_tokens 50

python save_activations_by_layer_and_position.py \
    --dataset simple_data.json \
    --layer 15 \
    --coeff 9.0 \
    --n_samples 10 \
    --vector_type average \
    --max_new_tokens 50

python save_activations_by_layer_and_position.py \
    --dataset simple_data_expanded_embeddings.json \
    --layer 15 \
    --coeff 9.0 \
    --n_samples 10 \
    --vector_type average \
    --max_new_tokens 50


python save_activations_by_layer_and_position.py \
    --dataset brysbaert_abstract_nouns.json \
    --layer 15 \
    --coeff 4.0 \
    --n_samples 30 \
    --vector_type average \
    --max_new_tokens 50

python validate_injecting_concepts.py \
    --layer 15 \
    --coeff 9.0