import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
import json
import os
import urllib.request

# Ensure NLTK WordNet is ready
try:
    wn.synsets('dog')
except Exception:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def is_too_technical(word):
    word = str(word).lower()
    if not word.isalpha():
        return True
    
    blacklist_suffixes = (
        'itis', 'osis', 'emia', 'pathia', 'syndrome', 'disease', 
        'phobia', 'oma', 'iasis',
        # 'logy', 'metry', 'graphy', 'ism'
        'ics', 'oscopy', 'ectomy', 'tomy'
    )
    if any(word.endswith(s) for s in blacklist_suffixes):
        return True
    if len(word) < 3: 
        return True
    return False

def is_strictly_noun(word):
    """
    Checks if the word is PRIMARILY a noun, not just technically a noun.
    """
    synsets = wn.synsets(word)
    if not synsets:
        return False
        
    # Check if the most common usage (the first synset) is a noun
    primary_pos = synsets[0].pos()
    if primary_pos != wn.NOUN:
        return False
        
    # Secondary check: ensure it doesn't have an overwhelming number of verb/adj usages
    noun_count = len([s for s in synsets if s.pos() == wn.NOUN])
    other_count = len([s for s in synsets if s.pos() != wn.NOUN])
    
    if other_count > noun_count:
        return False
        
    return True

print("Downloading Brysbaert Concreteness Dataset...")
url = "https://raw.githubusercontent.com/ArtsEngine/concreteness/master/Concreteness_ratings_Brysbaert_et_al_BRM.txt"
file_path = "brysbaert_ratings.txt"

if not os.path.exists(file_path):
    urllib.request.urlretrieve(url, file_path)

# Load the TSV file into pandas
print("Loading data...")
df = pd.read_csv(file_path, sep='\t')

# The column 'Conc.M' is the Mean Concreteness rating (1.0 = highly abstract, 5.0 = highly concrete)
# We want words with a rating less than 2.2 (This captures very strong abstractions)
abstract_df = df[df['Conc.M'] < 2.5].copy()

# Sort by abstractness (lowest to highest)
abstract_df = abstract_df.sort_values(by='Conc.M', ascending=True)

print(f"Found {len(abstract_df)} potential abstract words. Filtering for valid nouns...")

abstract_nouns = []
seen = set()

for index, row in abstract_df.iterrows():
    word = str(row['Word']).lower()
    
    if word in seen:
        continue
    seen.add(word)
    
    if is_too_technical(word):
        continue
        
    if not is_strictly_noun(word):
        continue
        
    abstract_nouns.append(word)
    
    if len(abstract_nouns) >= 4500:
        break

print(f"\nSuccessfully generated {len(abstract_nouns)} pure abstract nouns.")
print("Here is a sample of the absolute most abstract words (top 20):")
print(abstract_nouns[:20])

print("\nHere is a sample of the tail end (last 20):")
print(abstract_nouns[-20:])

# Save to JSON
dataset = {
    "concept_vector_words": abstract_nouns
}

output_path = os.path.join(os.path.dirname(__file__), 'brysbaert_abstract_nouns.json')
with open(output_path, 'w') as f:
    json.dump(dataset, f, indent=4)
    
print(f"\nSaved dataset to {output_path}")