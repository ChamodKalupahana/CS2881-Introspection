import gensim.downloader as api
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import json
import os

# Ensure necessary NLTK data is downloaded for POS checking
try:
    wn.synsets('dog')
except Exception:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def is_too_technical(word):
    """Filters out medical, highly technical, and poorly formatted tokens."""
    word = word.lower()
    
    # Exclude non-alphabetical tokens (numbers, punctuation)
    if not word.isalpha():
        return True
    
    blacklist_suffixes = (
        'itis', 'osis', 'emia', 'pathia', 'syndrome', 'disease', 
        'phobia', 'ism', 'oma', 'iasis', 'logy', 'metry', 'graphy', 
        'ics', 'oscopy', 'ectomy', 'tomy'
    )
    
    if any(word.endswith(s) for s in blacklist_suffixes):
        return True
        
    if len(word) < 4: 
        return True
        
    return False

def is_noun(word):
    """Checks if the word exists as a noun in WordNet."""
    # This is much more accurate for single isolated words than NLTK's pos_tag
    return len(wn.synsets(word, pos=wn.NOUN)) > 0

# 1. Define a robust seed list of undeniable abstract nouns.
# We mix philosophy, emotion, logic, and systems to ensure the centroid 
# represents "abstraction" generally, rather than just one specific domain.
llm_seed_words = [
    "truth", "wisdom", "entropy", "melancholy", "logic", "bureaucracy",
    "justice", "irony", "paradox", "harmony", "chaos", "ethics",
    "nostalgia", "courage", "epistemology", "ambiguity", "nuance",
    "integrity", "apathy", "sympathy", "empathy", "dogma", "stigma",
    "zeitgeist", "hubris", "karma", "synergy", "paradigm", "anxiety",
    "euphoria", "dread", "hope", "despair", "liberty", "tyranny",
    "sovereignty", "autonomy", "agency", "destiny", "fate", "luck",
    "chance", "probability", "infinity", "eternity", "mortality",
    "consciousness", "sentience", "cognition", "intuition", "intellect"
]

with open('simple_data.json', 'r') as f:
    seed_words = json.load(f).get("concept_vector_words")

print("len(seed_words)=",len(seed_words))
print("seed_words[0:10]",seed_words[0:10])

seed_words.extend(llm_seed_words)

print("seed_words[-10:0]",seed_words[-10:])


def generate_concept_vectors(target_count=5000):
    print("Loading pre-trained GloVe embeddings (this may take a minute on first run)...")
    # glove-wiki-gigaword-100 is ~400MB. 
    # For even better quality, use 'word2vec-google-news-300' (~1.5GB)
    model = api.load("glove-wiki-gigaword-100")
    
    # 2. Extract vectors for our seed words
    # Pre-trained GloVe is lowercase; ensure we check variations
    valid_seeds = []
    for w in seed_words:
        low_w = w.lower()
        if low_w in model.key_to_index:
            valid_seeds.append(low_w)
            
    print(f"Found {len(valid_seeds)}/{len(seed_words)} seed words in vocabulary.")
    
    if not valid_seeds:
        print("❌ Error: No seed words found in model vocabulary. Try different seeds or check model coverage.")
        return []
    
    vectors = [model[w] for w in valid_seeds]
    
    # 3. Calculate the Abstract Concept Centroid
    centroid = np.mean(vectors, axis=0)
    
    # 4. Query the model for the nearest neighbors
    # We pull a massive list (15,000) because many will be filtered out
    print("Mining nearest neighbors in latent space...")
    try:
        neighbors = model.similar_by_vector(centroid, topn=15000)
    except Exception as e:
        print(f"❌ Error during similarity search: {e}")
        return []
    
    abstract_nouns = []
    seen = set()
    
    # 5. Filter the results
    print("Filtering results through POS tagger and blacklists...")
    for word, similarity in neighbors:
        word = word.lower()
        
        # Skip words we've already processed, seed words, or invalid tokens
        if word in seen or word in valid_seeds:
            continue
        seen.add(word)
            
        if is_too_technical(word):
            continue
            
        if not is_noun(word):
            continue
            
        abstract_nouns.append(word)
        
        if len(abstract_nouns) >= target_count:
            break
            
    return abstract_nouns

if __name__ == "__main__":
    target = 5000
    concept_words = generate_concept_vectors(target_count=target)
    
    if concept_words:
        print(f"\nSuccessfully generated {len(concept_words)} abstract concept words.")
        print("Here is a sample of the top 20 closest conceptual neighbors:")
        print(concept_words[:20])
        
        # Save to JSON
        dataset = {
            "concept_vector_words": concept_words
        }
        
        output_path = os.path.join(os.path.dirname(__file__), 'knn_abstract_nouns.json')
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=4)
            
        print(f"\nSaved dataset to {output_path}")
    else:
        print("\n❌ Failed to generate concept words.")