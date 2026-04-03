python save_activations_and_concept.py --dataset_name "complex_data" --concept_name "appreciation" --min_layer_to_save 16

python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --datasets test_data --alphas -16 -8 -5 -2 0 2 5 8 16 --probe_layer 19 --clean_once --probe_path probe_vectors/run_04_03_26_13_50/mass_mass_vector_appreciation.pt