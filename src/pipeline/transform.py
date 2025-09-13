import re
import json
import spacy
import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Load spacy model once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model for spacy...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def _clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)   # Remove URLs
    text = re.sub(r'Weekly Hours:\S+|Salary Range:\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[.*?\]', '', text)                                       # Remove [brackets]
    text = re.sub(r'[\r\n\t]', ' ', text)                                     # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _segment_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 15]

def _get_phrases():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    file_path = BASE_DIR / "data" / "skill_phrases.json"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            phrases = json.load(f)
        return phrases
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}.")
        return {}

def _get_token_labels(sentence, phrases):
    doc = nlp(sentence)
    tokens = [token.text for token in doc]
    token_lowers = [t.lower() for t in tokens]
    labels = ["O"] * len(tokens)
    found_phrases = []

    for category, phrase_list in phrases.items():
        for phrase in phrase_list:
            phrase_tokens = phrase.lower().split()
            n = len(phrase_tokens)

            for i in range(len(tokens) - n + 1):
                if token_lowers[i:i + n] == phrase_tokens:
                    labels[i] = f"B-{category.split('_')[0]}"
                    for j in range(1, n):
                        labels[i + j] = f"I-{category.split('_')[0]}"
                    found_phrases.append(phrase)
                    break  # Only label the first occurrence per sentence to avoid overlap

    return tokens, labels, list(set(found_phrases))

def _oversample_data(tagged_sentences, keyword_counts, target_count=500):
    oversampled_sentences = []
    factors = {
        keyword: int(target_count / count) for keyword, count in keyword_counts.items() if count < target_count
    }

    for sentence_data in tagged_sentences:
        oversampled_sentences.append(sentence_data)
        keyword = sentence_data.get('keyword')
        if keyword in factors:
            for _ in range(factors[keyword]):
                oversampled_sentences.append(sentence_data)

    return oversampled_sentences

def transform_data(raw_data):
    if not raw_data:
        print("No raw data to transform.")
        return []

    phrases = _get_phrases()
    if not phrases:
        return []
    
    keyword_counts = defaultdict(int)
    for job in raw_data:
        keyword_counts[job.get('keyword')] += 1

    tagged_sentences = []
    for job in tqdm(raw_data, desc="Transforming data"):
        description = job.get('description', '')
        keyword = job.get('keyword')
        cleaned_text = _clean_text(description)
        sentences = _segment_sentences(cleaned_text)

        for sentence in sentences:
            tokens, labels, found_phrases = _get_token_labels(sentence, phrases)
            if found_phrases:
                tagged_sentences.append({
                    "sentence": sentence,
                    "tokens": tokens,
                    "ner_tags": labels,
                    "tag": found_phrases,
                    "keyword": keyword
                })
    
    final_data = _oversample_data(tagged_sentences, keyword_counts)
    random.shuffle(final_data)

    print(f"\nTransformation complete. Final dataset size: {len(final_data)} tagged sentences.")
    return final_data