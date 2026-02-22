train probe
python train_probe.py --layer 24

python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --alphas -5 -1 1 5
python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --alphas 0 2 5 8

Test with test data
python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --datasets test_data --alphas 0 2 5 8
python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --datasets test_data --alphas 0 8 16 24

for true negatives
python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --datasets test_data --alphas 0 -2 -5 -8 --skip_clean

test with scaling by different layer
python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --datasets test_data --alphas 0 2 5 8 16 --probe_layer 17 --probe_path probe_vectors/introspection_probe_vector_layer31.pt


to create big db
python save_vectors_not_detected.py --layers 16 --coeffs 8.0 --datasets simple_data_expanded --capture_all_layers

for detected_correct vs not_detected
python train_probe_not_detected.py --layer 24

python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --datasets test_data --alphas 0 2 5 8 --probe_layer 24

python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --datasets test_data --alphas 0 -2 -5 -8 --probe_layer 24

python test_probe_dir_casual.py --layers 16 --coeffs 8.0 --datasets test_data --alphas 0 8 16 24 --probe_layer 24

python compute_delta_per_layer_not_detected.py \
  --detected_dir injected_correct_expanded \
  --not_detected_dir not_detected