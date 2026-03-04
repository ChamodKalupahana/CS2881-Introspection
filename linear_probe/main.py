python save_vectors_not_detected.py --layers 16 --coeffs 8.0 --datasets simple_data_expanded_embeddings --capture_all_layers

python compute_mass_mean_vector.py --layer 19 --injected-dir saved_activations/run_03_01_26_17_00/detected_correct --clean-dir saved_activations/run_03_01_26_17_00/not_detected

