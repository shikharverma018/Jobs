import re
import json
import spacy
import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

#Load spacy model outside of function to avoid re-loading on every call
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model for spacy...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def _clean_text(text):
    if not isinstance(text, str):
        return ""
    #Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    #Remove mentions of salary or weekly hours, as they are not relevant
    text = re.sub(r'Weekly Hours:\S+|Salary Range:\S+', '', text, flags=re.MULTILINE)
    #Remove any text within square brackets, e.g., [OUTPUT NOT AVAILABLE...]
    text = re.sub(r'\[.*?\]', '', text)
    #Remove common special characters and normalize whitespace
    text = re.sub(r'[\r\n\t]', ' ', text)
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
    labels = ["O"] * len(tokens)
    found_phrases = []

    for category, phrase_list in phrases.items():
        for phrase in phrase_list:
            #Use regex to find whole words that match the phrase
            pattern = r'\b' + re.escape(phrase) + r'\b'
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                start_char, end_char = match.span()
                
                #Align character indices to token indices
                start_token_index = -1
                end_token_index = -1
                current_char_index = 0
                for i, token in enumerate(tokens):
                    if current_char_index <= start_char and current_char_index + len(token) > start_char:
                        start_token_index = i
                    if current_char_index < end_char and current_char_index + len(token) >= end_char:
                        end_token_index = i
                    current_char_index += len(token) + 1  #Add 1 for the space

                if start_token_index != -1 and end_token_index != -1:
                    labels[start_token_index] = f"B-{category.split('_')[0]}"
                    for i in range(start_token_index + 1, end_token_index + 1):
                        labels[i] = f"I-{category.split('_')[0]}"
                    found_phrases.append(phrase)

    return tokens, labels, list(set(found_phrases))

def _oversample_data(tagged_sentences, keyword_counts, target_count=500):
    oversampled_sentences = []
    #Calculate duplication factor for each low-count keyword
    factors = {
        keyword: int(target_count / count) for keyword, count in keyword_counts.items() if count < target_count
    }

    #Duplicate sentences based on the calculated factors
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
    
    #Count the number of jobs per keyword to identify low-count ones
    keyword_counts = defaultdict(int)
    for job in raw_data:
        keyword_counts[job.get('keyword')] += 1

    tagged_sentences = []
    
    #Use tqdm to show a progress bar
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
    
    #Now, oversample the data to balance it
    final_data = _oversample_data(tagged_sentences, keyword_counts)
    
    #Optional: Shuffle the data to mix the original and duplicated sentences
    random.shuffle(final_data)

    print(f"\nTransformation complete. Final dataset size: {len(final_data)} tagged sentences.")
    
    return final_data

"""if __name__ == '__main__':
    #testing the function's output and logic
    sample_raw_data = [
        {
            "description": "Our team is looking for experienced AI/ML engineers who have hands-on experience with Generative AI, especially in prompt engineering, data curation, AI agents, and domain-specific model fine-tuning using frameworks. Strong knowledge and experience in PEFT, such as LoRA/QLoRA trainings on large language models, is a plus.",
            "keyword": "Machine Learning Engineer"
        },
        {
            "description": "Experience with databases like PostgreSQL and MySQL. Knowledge of Continuous Integration (CI) and Continuous Deployment (CD) practices is required. Familiarity with cloud platforms such as AWS and Azure is a plus. We are looking for a devops engineer.",
            "keyword": "devops engineer"
        }
    ]

    # Mocking keyword counts for oversampling test
    # This simulates a real-world scenario where 'devops engineer' is underrepresented
    # Our function should duplicate sentences from this category.
    mock_keyword_counts = {
        "Machine Learning Engineer": 700,
        "devops engineer": 300
    }

    print("Running transform_data() on sample data...")
    processed_data = transform_data(sample_raw_data)

    if processed_data:
        file_path = Path(__file__).resolve().parent.parent.parent / "notebooks" / "research_data" / "transformed_data_sample.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2)
        print(f"\nSample data has been written to: {file_path}")
    else:
        print("No data processed. Check for errors.")"""