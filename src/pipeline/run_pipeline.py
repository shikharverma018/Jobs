from pathlib import Path
from .extract import extract_raw_data
from .transform import transform_data
from .load import load_processed_data

def run_etl_pipeline():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    raw_data_path = BASE_DIR / "data" / "raw" / "jobs.json"

    print("--- Starting ETL Pipeline ---")

    #Step 1: Extract
    print("\n[1/3] Extracting raw data...")
    raw_data = extract_raw_data(raw_data_path)
    if not raw_data:
        print("Pipeline aborted: No raw data found.")
        return
    print("Extraction complete.")

    #Step 2: Transform
    print("\n[2/3] Transforming and labeling data...")
    processed_data = transform_data(raw_data)
    if not processed_data:
        print("Pipeline aborted: No data processed.")
        return
    print("Transformation complete.")

    #Step 3: Load
    print("\n[3/3] Loading processed data...")
    load_processed_data(processed_data)
    print("Loading complete.")

    print("\n--- ETL Pipeline Finished Successfully! ---")

if __name__ == "__main__":
    run_etl_pipeline()