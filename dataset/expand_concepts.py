import json
import os
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist

nltk.download('wordnet')
nltk.download('wordnet_ic')  # for frequency/commonness filtering
from nltk.corpus import wordnet as wn, wordnet_ic
from nltk.corpus.reader.wordnet import information_content

brown_ic = wordnet_ic.ic('ic-brown.dat')  # Brown corpus frequency data

# Get nouns with frequency filtering
concept_candidates = []
baseline_candidates = []

for synset in wn.all_synsets(pos='n'):
    try:
        freq = information_content(synset, brown_ic)  # higher = more common/useful
        if freq > 1.0:  # lower threshold for baselines
            word = synset.lemmas()[0].name().replace('_', ' ')
            if ' ' not in word and len(word) > 3:
                if freq > 3.0:
                    concept_candidates.append((word, freq, synset.definition()))
                baseline_candidates.append((word, freq, synset.definition()))
    except Exception as e:
        continue

concept_candidates.sort(key=lambda x: -x[1])
baseline_candidates.sort(key=lambda x: -x[1])

print("Taking top 2000 distinct concept candidates...")
seen_concepts = set()
top_concept_candidates = []
for c in concept_candidates:
    if c[0] not in seen_concepts:
        seen_concepts.add(c[0])
        top_concept_candidates.append(c[0])
        if len(top_concept_candidates) == 2000:
            break

print("Loading existing words from simple_data.json...")
script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, 'simple_data.json')
with open(json_path, 'r') as f:
    data = json.load(f)

existing_words = data.get('concept_vector_words', [])
existing_baseline_words = data.get('baseline_words', [])

print("Loading sentence-transformers model...")
try:
    model = SentenceTransformer('all-mpnet-base-v2')
except Exception as e:
    print("Error loading SentenceTransformer. You may need to install it: pip install sentence-transformers")
    raise e

print("Embedding concept candidate words and existing words...")
candidate_embeddings = model.encode(top_concept_candidates)
existing_embeddings = model.encode(existing_words)

# Step 3: Greedy Maximum Dispersion Selection for Concept Vectors
selected_words = list(existing_words)
selected_embeddings = list(existing_embeddings)

target_count = 300
to_select = target_count - len(selected_words)

print(f"Selecting {to_select} concept words using greedy maximum dispersion...")

if len(selected_embeddings) > 0:
    min_distances = cdist(candidate_embeddings, np.array(selected_embeddings), metric='cosine').min(axis=1)
else:
    min_distances = np.ones(len(candidate_embeddings)) * float('inf')

existing_words_set = set(existing_words)
for i, word in enumerate(top_concept_candidates):
    if word in existing_words_set:
        min_distances[i] = -1.0

for i in range(to_select):
    best_idx = np.argmax(min_distances)
    
    new_word = top_concept_candidates[best_idx]
    new_embedding = candidate_embeddings[best_idx]
    
    selected_words.append(new_word)
    selected_embeddings.append(new_embedding)
    
    min_distances[best_idx] = -1.0
    
    new_distances = cdist(candidate_embeddings, np.array([new_embedding]), metric='cosine').flatten()
    
    unselected_mask = min_distances != -1.0
    min_distances[unselected_mask] = np.minimum(min_distances[unselected_mask], new_distances[unselected_mask])
    
    if (i + 1) % 50 == 0:
        print(f"Selected {len(selected_words)}/{target_count} concept words...")

# Step 4: Baseline Expansion
print("Expanding baseline words...")
target_baseline_count = 2000
selected_baselines = list(existing_baseline_words)

concept_set = set(selected_words)
baseline_set = set(selected_baselines)

seen_baseline = set()
for c in baseline_candidates:
    word = c[0]
    if word not in concept_set and word not in baseline_set and word not in seen_baseline:
        seen_baseline.add(word)
        selected_baselines.append(word)
        if len(selected_baselines) >= target_baseline_count:
            break

print("Selection complete.")

data['concept_vector_words'] = selected_words
data['baseline_words'] = selected_baselines

out_path = os.path.join(script_dir, 'simple_data_expanded_embeddings.json')
with open(out_path, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Successfully saved {len(selected_words)} concept words and {len(selected_baselines)} baseline words to {out_path}.")