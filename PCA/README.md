copied from `acitvation_patching/save_success_vectors_by_layer.py`


To run the script from the PCA subdirectory, you can navigate there and run it directly. The script is designed to handle path resolution automatically (it adds the project root to sys.path).:

`python save_vectors_by_layer.py --layers 16 --coeffs 8.0 --capture_all_layers --datasets simple_data`

this saves the layers .pt files into `success_results/run_MM_DD_YY_HH_MM/`

move the correct and no inject concepts into `injected_correct` and `no_inject` such that `PCA/injected_correct/Crystals/Crystals_layer16_coeff8.0_avg.pt` as an example

assumes you have more than 1 concept by layer in `injected_correct` and `no_inject`

then you can run: `python compute_delta_per_layer.py --coeff 8.0 --injected_dir injected_correct --clean_dir no_inject` which cacluates the differences between injected and no_inject and then saves them by each layer in `PCA/components`

once you've done that, you can plot PCA using:
`python compute_PCA.py` - for number of components to explain 80% of variance 
`python compute_PCA.py --plot_pc1` - explained variance ratio of the first principal component