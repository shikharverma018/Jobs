import json
from pathlib import Path

def extract_raw_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        print(f"Successfully extracted data from {file_path}")
        return raw_data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Please check the file format.")
        return []

"""if __name__ == '__main__':
    #just for testing
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    file_path = BASE_DIR / "data" / "raw" / "jobs.json"
    data = extract_raw_data(file_path)
    if data:
        print(f"Extracted {len(data)} job descriptions.")
    else:
        print("No data extracted.")"""