import einops

from extract_activations import extract_activations_at_position

# specifiy min layer to save (default: 10)

# load model


harmful_dataset = [] # TODO
harmless_dataset = [] # TODO

num_of_prompts = len(harmful_dataset)

postive_activations = {}
negative = {}
# for each prompt in dataset
for prompt_index in range(num_of_prompts):
    harmful_instruction = harmful_dataset[prompt_index]
    harmless_instruction = harmless_dataset[prompt_index]

    # extract_activations for postive and negative cases
    # postive = harmfull
    postive_activations = extract_activations_at_position(
        model,
        tokenizer,
        harmful_instruction,
        min_layer_to_save
    )
    # negative = harmless  
     negative_activations = extract_activations_at_position(
        model,
        tokenizer,
        harmful_instruction,
        min_layer_to_save
    )

# prompt-wise mean (layer, prompt) -> (layer)
einops.reduce(postive_activations, "layer prompt -> layer", "mean")
einops.reduce(negative_activations, "layer prompt -> layer", "mean")

# output for postive: dict{layer : d_model tensor}
# output for negative: dict{layer : d_model tensor}