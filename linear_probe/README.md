### train probe
python train_probe.py --layer 24

python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --alphas -5 -1 1 5
python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --alphas 0 2 5 8

### Test with test data
python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --datasets test_data --alphas 0 2 5 8
python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --datasets test_data --alphas 0 8 16 24

for true negatives
python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --datasets test_data --alphas 0 -2 -5 -8 --skip_clean

test with scaling by different layer
python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --datasets test_data --alphas 0 2 5 8 16 --probe_layer 17 --probe_path probe_vectors/introspection_probe_vector_layer31.pt


### to create big db
python save_vectors_not_detected.py --layers 16 --coeffs 8.0 --datasets simple_data_expanded --capture_all_layers

### for detected_correct vs not_detected
python train_probe_not_detected.py --layer 24

python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --datasets test_data --alphas 0 2 5 8 --probe_layer 24

python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --datasets test_data --alphas 0 -2 -5 -8 --probe_layer 24

python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --datasets test_data --alphas 0 8 16 24 --probe_layer 24

python compute_delta_per_layer_not_detected.py \
  --detected_dir injected_correct_expanded \
  --not_detected_dir not_detected

### mean mass vector  
python compute_mass_mean_vector.py --layer 19

python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --datasets test_data --alphas -8 -5 -2 0 2 5 8 16 24 --probe_layer 24 --clean_once --probe_path probe_vectors/mass_mean_vector_layer24.pt

### increasing dataset

python save_vectors_not_detected.py --layers 16 --coeffs 8.0 --datasets simple_data_expanded_embeddings --capture_all_layers

python compute_mass_mean_vector.py --layer 19 --injected-dir saved_activations/run_03_01_26_17_00/detected_correct --clean-dir saved_activations/run_03_01_26_17_00/not_detected

### treat parallel as correct
python compute_mass_mean_vector.py --layer 19 --injected-dir saved_activations/run_03_01_26_17_00/detected_correct --clean-dir saved_activations/run_03_01_26_17_00/not_detected --merge_parallel

### only show base classes on the plot (correct vs not detected)
python compute_mass_mean_vector.py --layer 19 --injected-dir saved_activations/run_03_01_26_17_00/detected_correct --clean-dir saved_activations/run_03_01_26_17_00/not_detected --hide_intermediate

python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --datasets test_data --alphas -8 -5 -2 0 2 5 8 16 24 --probe_layer 19 --clean_once --probe_path probe_vectors/mass_mean_vector_layer19.pt

### calculate projection of activations onto probe direction across layers
python DLA_towards_probe_dir.py --activation_file saved_activations/run_03_01_26_17_00/detected_correct/reforestation_layers16-31_coeff8.0_avg.pt --probe_file probe_vectors/mass_mean_vector_layer19.pt --independent_layers

python DLA_towards_probe_dir.py --activation_dir saved_activations/run_03_01_26_17_00/detected_correct --probe_file probe_vectors/mass_mean_vector_layer19.pt

python test_probe_dir_casual_OOD.py --layers 16 --coeffs 8.0 --datasets test_data --alphas -8 -5 -2 0 2 5 8 16 24 --probe_layer 19 --clean_once --probe_path probe_vectors/mass_mean_vector_layer19.pt

### final experiment run with abstract nouns
python save_vectors_not_detected.py --layers 16 --coeffs 8.0 --datasets abstract_nouns_dataset --capture_all_layers

python compute_mass_mean_vector.py --layer 19 --injected-dir saved_activations/run_04_01_26_22_51/detected_correct --clean-dir saved_activations/run_04_01_26_22_51/not_detected