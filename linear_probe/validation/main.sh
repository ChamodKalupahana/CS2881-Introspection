python similarity_to_yes.py \
    --probe1 calibation_correct_vs_detected_correct/probe_vectors/MM_L18_P0_dPrim22.88_dVal1.37.pt \
    --probe2 not_detected_vs_detected_correct/layer_and_position_sweep/probe_vectors/MM_L26_P0_dPrim1.69_dVal1.36.pt

python detection_across_prompt.py \
    --probe1 calibation_correct_vs_detected_correct/probe_vectors/MM_L18_P0_dPrim22.88_dVal1.37.pt \
    --concept magnetism \
    --layer 14 \
    --coeff 5.0

python detection_across_prompt.py \
    --probe1 not_detected_vs_detected_correct/layer_and_position_sweep/probe_vectors/MM_L26_P0_dPrim1.69_dVal1.36.pt \
    --concept magnetism \
    --layer 14 \
    --coeff 5.0

python logit_lens.py \
    --probe1 calibation_correct_vs_detected_correct/probe_vectors/MM_L18_P0_dPrim22.88_dVal1.37.pt