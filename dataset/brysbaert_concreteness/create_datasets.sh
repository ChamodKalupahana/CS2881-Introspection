python3 concreteness_dataset.py \
    --num_concept_vector_words 5000 \
    --num_baseline_words 10000 \
    --abstractness_threshold 2.5 \
    --baseline_concreteness_threshold 3.5 \
    --output_file brysbaert_abstract_nouns_large.json

python3 concreteness_dataset.py \
    --num_concept_vector_words 400 \
    --num_baseline_words 2000 \
    --abstractness_threshold 2.2 \
    --baseline_concreteness_threshold 4.5 \
    --output_file brysbaert_abstract_nouns.json