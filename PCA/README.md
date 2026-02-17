copied from `acitvation_patching/save_success_vectors_by_layer.py`


To run the script from the PCA subdirectory, you can navigate there and run it directly. The script is designed to handle path resolution automatically (it adds the project root to sys.path).:

`python save_vectors_by_layer.py --layers 16 --coeffs 8.0 --capture_all_layers --datasets simple_data`

this saves the layers .pt files into `success_results/run_MM_DD_YY_HH_MM/`

move the correct and no inject concepts into `injected_correct` and `no_inject`