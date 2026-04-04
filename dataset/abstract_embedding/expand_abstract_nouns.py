import nltk
from nltk.corpus import wordnet as wn
import json
import os
import random

# Ensure WordNet is downloaded
try:
    wn.synsets('dog')
except Exception:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def is_too_technical(word):
    blacklist_suffixes = ('itis', 'osis', 'emia', 'pathia', 'syndrome', 'disease', 'phobia', 'ism')
    if any(word.lower().endswith(s) for s in blacklist_suffixes):
        return True
    
    if '_' in word or '-' in word or ' ' in word: 
        return True
        
    if len(word) < 4: 
        return True
        
    return False

# Re-included attributes and motives for richer concept vectors
abstract_categories = {
    'noun.attribute',  # Qualities (Honesty, Bravery, Wisdom)
    'noun.cognition',  # Ideas (Logic, Theory, Mind)
    'noun.feeling',    # Emotions (Joy, Sorrow, Anger)
    'noun.state',      # Conditions (Chaos, Peace, Entropy)
    'noun.motive'      # Ethics, Reasons
}

baseline_categories = {
    'noun.artifact', 'noun.animal', 'noun.plant', 'noun.food', 
    'noun.body', 'noun.object', 'noun.person', 'noun.location', 
    'noun.substance', 'noun.time'
}

concept_vector_set = set()
baseline_set = set()

for synset in wn.all_synsets('n'):
    lexname = synset.lexname()
    for lemma in synset.lemmas():
        # OPTIONAL BUT RECOMMENDED: Ensure the word has some common usage
        # This prevents fragmented token vectors from ultra-rare WordNet entries
        # if lemma.count() == 0:
        #     continue
            
        word = lemma.name().lower() # Lowercase is often better for standard tokenization
        
        if is_too_technical(word):
            continue
            
        if lexname in abstract_categories:
            concept_vector_set.add(word)
        elif lexname in baseline_categories:
            baseline_set.add(word)

# Ensure no overlap
baseline_set = baseline_set - concept_vector_set

# Sort to create a deterministic base, then optionally slice 'A' words BEFORE shuffling
concept_vector_list = sorted(list(concept_vector_set))
baseline_list = sorted(list(baseline_set))

# Now shuffle
random.seed(42)
random.shuffle(concept_vector_list)
random.shuffle(baseline_list)

print(f"Total abstract nouns found: {len(concept_vector_list)}")
print(f"Total baseline nouns found: {len(baseline_list)}")

concept_count = 4500
baseline_count = 20000

# Select final counts
concept_vector_final = concept_vector_list[:concept_count]
baseline_final = baseline_list[:baseline_count]

if len(concept_vector_final) < concept_count:
    print(f"Warning: Only found {len(concept_vector_final)} abstract nouns. Try removing the lemma.count() filter if active.")
if len(baseline_final) < baseline_count:
    print(f"Warning: Only found {len(baseline_final)} baseline nouns.")

dataset = {
    "concept_vector_words": concept_vector_final,
    "baseline_words": baseline_final
}

output_path = os.path.join(os.path.dirname(__file__), 'expanded_abstract_nouns_dataset.json')
with open(output_path, 'w') as f:
    json.dump(dataset, f, indent=4)

print(f"Saved dataset to {output_path}")