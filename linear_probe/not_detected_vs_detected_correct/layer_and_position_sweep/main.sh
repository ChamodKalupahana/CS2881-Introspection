python save_activations_by_layer_and_position.py \
    --dataset brysbaert_abstract_nouns.json \
    --layer 14 \
    --coeff 5.0 \
    --vector_type average \
    --max_new_tokens 50
    # --n_samples 500 \   

    python validate_injecting_concepts.py \
    --layer 15 \
    --coeff 9.0

python compute_mass_mean_vectors_and_PCA.py --run_dir saved_activations/run_04_05_26_13_31
python layer_and_position_sweep_mass_mean_vectors.py --run_dir saved_activations/run_04_05_26_13_31