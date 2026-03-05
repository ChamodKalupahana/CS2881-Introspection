python save_vectors_not_detected.py --layers 16 --coeffs 8.0 --datasets simple_data_expanded_embeddings --capture_all_layers

python compute_mass_mean_vector.py --layer 19 --injected-dir saved_activations/run_03_04_26_21_28/detected_correct --clean-dir saved_activations/run_03_04_26_21_28/not_detected

python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --datasets test_data --alphas -8 0 2 5 8 --probe_layer 19 --clean_once --probe_path probe_vectors/mass_mean_vector_layer19.pt

python test_probe_dir_casual_OOD.py --layers 16 --coeffs 8.0 --datasets test_data --alphas -8 0 2 5 8 --probe_layer 19 --clean_once --probe_path probe_vectors/mass_mean_vector_layer19.pt --prompts 2 5