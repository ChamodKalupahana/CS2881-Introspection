import nltk
from nltk.corpus import wordnet as wn
import json
import os
import random

# Ensure WordNet is downloaded
try:
    # Some environments might not have wordnet downloaded
    wn.synsets('dog')
except Exception:
    nltk.download('wordnet')

def is_too_technical(word):
    # Filter out medical conditions, specific technical suffixes, and multi-word terms
    blacklist_suffixes = ('itis', 'osis', 'emia', 'pathia', 'syndrome', 'disease', 'phobia', 'ism')
    if any(word.lower().endswith(s) for s in blacklist_suffixes):
        return True
    
    # Exclude multi-word terms (WordNet uses underscores for spaces)
    if '_' in word or '-' in word or ' ' in word: 
        return True
        
    # Exclude very short words
    if len(word) < 4: 
        return True
        
    return False

# Categories for concept vector words (abstract)
abstract_categories = {
    # 'noun.attribute',  # Qualities (Honesty, Bravery)
    'noun.cognition',  # Ideas (Logic, Theory)
    'noun.feeling',    # Emotions (Joy, Sorrow)
    'noun.state',      # Conditions (Chaos, Peace)
    # 'noun.motive'      # Ethics
}

# Categories for baseline words (concrete/common)
baseline_categories = {
    'noun.artifact',   # Tools, buildings
    'noun.animal',     # Creatures
    'noun.plant',      # Flora
    'noun.food',       # Edibles
    'noun.body',       # Parts
    'noun.object',     # Natural objects
    'noun.person',     # Humans
    'noun.location',   # Places
    'noun.substance',  # Materials
    'noun.time'        # Units of time
}

# baseline_categories should include
baseline_categories.update(abstract_categories)

concept_vector_set = set()
baseline_set = set()

# Iterate through all noun synsets once
for synset in wn.all_synsets('n'):
    lexname = synset.lexname()
    for lemma in synset.lemmas():
        word = lemma.name().capitalize()
        if is_too_technical(word):
            continue
            
        if lexname in abstract_categories:
            concept_vector_set.add(word)
        elif lexname in baseline_categories:
            baseline_set.add(word)

# Ensure no overlap (concept nouns should not be in baseline)
baseline_set = baseline_set - concept_vector_set

# Convert to sorted lists for deterministic base
concept_vector_list = sorted(list(concept_vector_set))
baseline_list = sorted(list(baseline_set))

# Shuffle with a seed for reproducible randomness
random.seed(42)
random.shuffle(concept_vector_list)
random.shuffle(baseline_list)

print(f"Total abstract nouns found: {len(concept_vector_list)}")
print(f"Total baseline nouns found: {len(baseline_list)}")

# Select the requested counts
# Using a slice from the middle to avoid obscure start-of-alphabet words
concept_count = 300
baseline_count = 2000

# Skip first 100 to avoid some common but potentially less "abstract" A words
concept_vector_final = concept_vector_list[100:100+concept_count] if len(concept_vector_list) >= 100+concept_count else concept_vector_list[:concept_count]
baseline_final = baseline_list[500:500+baseline_count] if len(baseline_list) >= 500+baseline_count else baseline_list[:baseline_count]

# Final selection check
if len(concept_vector_final) < concept_count:
    print(f"Warning: Only found {len(concept_vector_final)} abstract nouns.")
if len(baseline_final) < baseline_count:
    print(f"Warning: Only found {len(baseline_final)} baseline nouns.")

# Construct the dataset
dataset = {
    "concept_vector_words": concept_vector_final,
    "baseline_words": baseline_final
}

# Save to JSON
output_path = os.path.join(os.path.dirname(__file__), 'abstract_nouns_dataset.json')
with open(output_path, 'w') as f:
    json.dump(dataset, f, indent=4)

print(f"Saved dataset to {output_path}")
print(f"Final dataset has {len(dataset['concept_vector_words'])} concept words and {len(dataset['baseline_words'])} baseline words.")