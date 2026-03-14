datasets = [simple_data, simple_data_expanded_embeddings]

python save_vectors_orthogonal.py --layers 16 --coeffs 8.0 --datasets simple_data_expanded_embeddings --capture_all_layers

python compute_PCA_cohen_orthogonal.py --plot_mean_diff_dot_product --plot_cohens_d --max_components 30

python validate_vector_othogonal.py --input_vector saved_activations/run_03_13_26_21_58/pca_components/last_token_layer_16_pca.pt --pca_index 3 --layer 16

python test_probe_dir_casual_prompts_OOD.py --layers 16 --coeffs 8.0 --datasets test_data --alphas -8 -5 0 2 5 8 16 --clean_once --prompts 2 --probe_path saved_activations/run_03_13_26_21_58/pca_components/last_token_layer_17_pca.pt --pca_index 26 --probe_layer 17