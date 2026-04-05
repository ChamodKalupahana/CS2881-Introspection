python save_activations_by_layer_and_position_with_calibation.py \
    --dataset brysbaert_abstract_nouns_large.json \
    --layer 14 \
    --coeff 5.0 \
    --vector_type average \
    --max_new_tokens 50 \
    --n_samples 1000

python save_activations_by_layer_and_position_with_calibation.py \
    --dataset simple_data.json \
    --layer 16 \
    --coeff 8.0 \
    --vector_type average \
    --max_new_tokens 50 \
    --n_samples 50