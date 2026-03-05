datasets = [simple_data, simple_data_expanded_embeddings]

python save_activations_prompts.py --layer 16 --coeff 8.0 --datasets simple_data

python compute_mass_mean_vector_prompts.py --layer 19 --token_type last_token
python compute_mass_mean_vector_prompts.py --layer 19 --token_type prompt_last_token

python compute_mass_mean_vector_prompts.py --layer 19 --token_type last_token --run-dir saved_activations/prompt_activations_03_04_26_23_39

python test_probe_dir_casual_prompts.py --layers 16 --coeffs 8.0 --datasets test_data --alphas -8 0 2 5 8 16 24  --probe_layer 19 --clean_once

python test_probe_dir_casual_prompts_OOD.py --layers 16 --coeffs 8.0 --datasets test_data --alphas -8 -5 0 2 5 8 16 --probe_layer 19 --clean_once --prompts 2 5

python test_probe_dir_casual_prompts_OOD.py --layers 16 --coeffs 8.0 --datasets test_data --alphas -8 -5 -2 0 5 --probe_layer 19 --clean_once --prompts 3 6

