datasets = [simple_data, simple_data_expanded_embeddings]

python save_activations_prompts.py --layer 16 --coeff 8.0 --datasets simple_data

python compute_mass_mean_vector_prompts.py --layer 19 --token_type last_token
python compute_mass_mean_vector_prompts.py --layer 19 --token_type prompt_last_token

python compute_mass_mean_vector_prompts.py --layer 19 --token_type last_token --run-dir saved_activations/prompt_activations_03_04_26_23_39

python compute_mass_mean_vector_prompts.py --layer 19 --token_type last_token --prompts 0 10 11

python test_probe_dir_casual_prompts.py --layers 16 --coeffs 8.0 --datasets test_data --alphas -8 0 2 5 8 16 24  --probe_layer 19 --clean_once

python test_probe_dir_casual_prompts_OOD.py --layers 16 --coeffs 8.0 --datasets test_data --alphas -8 -5 0 2 5 8 16 24 --probe_layer 19 --clean_once --prompts 2 5
python test_probe_dir_casual_prompts_OOD.py --layers 16 --coeffs 8.0 --datasets test_data --alphas -8 -5 0 2 5 8 16 24 --probe_layer 19 --clean_once --prompts 2 5 --continuous_steering

python test_probe_dir_casual_prompts_OOD.py --layers 16 --coeffs 8.0 --datasets test_data --alphas -8 -5 -2 0 5 --probe_layer 19 --clean_once --prompts 3 6

python test_probe_dir_casual_prompts_OOD.py --layers 16 --coeffs 8.0 --datasets test_data --alphas -8 -5 0 2 5 8 16 24 --probe_layer 21 --clean_once --prompts 2 --token_type prompt_last_token --continuous_steering

# for unified prompts
python save_activations_prompts_unified.py --layer 16 --coeff 8.0 --datasets simple_data_expanded_embeddings

python compute_PCA.py --max_components 8

# for just one prompt
python save_activations_prompts_unified.py --layer 16 --coeff 8.0 --datasets simple_data_expanded_embeddings --positive_prompts 0 --negative_prompts 0

python compute_PCA.py --input_vector saved_activations/prompt_activations_03_10_26_19_27/postive_no_inject/not_detected/baseline_p0_layers16-31_noinject.pt --max_components 8

# validate pca dir after analysuing compute_PCA.py plots
python validate_vector.py --input_vector saved_activations/prompt_activations_03_10_26_19_27/pca_components/last_token_layer_21_pca.pt --pca_index 2 --layer 21 --prompts 0