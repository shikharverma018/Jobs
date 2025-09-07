import json
from pathlib import Path

def load_processed_data(processed_data, file_name="processed_job_data.json"):
    if not processed_data:
        print("No processed data to save.")
        return

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    output_dir = BASE_DIR / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / file_name

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=4)
        print(f"Successfully saved {len(processed_data)} items to {file_path}")
    except IOError as e:
        print(f"Error writing to file: {e}")

"""if __name__ == '__main__':
    #This block is for testing the function
    sample_data = [
        {"sentence": "Sample sentence 1.", "tokens": ["Sample", "sentence", "1", "."], "ner_tags": ["O", "O", "O", "O"]},
        {"sentence": "Sample sentence 2.", "tokens": ["Sample", "sentence", "2", "."], "ner_tags": ["O", "O", "O", "O"]}
    ]
    load_processed_data(sample_data)"""