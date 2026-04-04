from original_paper.compute_concept_vector import compute_vector_single_prompt
from model_utils.injection import test_injection_capture
from model_utils.llm_judges import classify_response
from model_utils.logging import TeeLogger

# load model 

# load abstract dataset (small)
# specify dataset name
# specify coeff.
# specify injection layer

# logging from model_utils.logging

# set up folder structure
# results are saved to saved_activations/run_{}/{detection_category}/{concept}_{coeff}_{injection_layer}.pt

# for concept vector:

    # compute concept vector
        # using compute_vector_single_prompt

    # inject concept vector and capture activations
        # return dict by {layer, position} = d_model

    # categorise model output by detection
        # detection_category = classify_response()

    # save activations to results folder by detection_category