# Import the processor
import sys
import os
import yaml
from datetime import datetime
import json
import argparse
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_prep import EcommerceDataProcessor

# Load configuration
config_path = "configs/data_prep_config.yaml"
sample_size = None
sample_description = None

def parse_args():
    parser = argparse.ArgumentParser(description="Data preprocessing and feature engineering with MLflow logging")
    parser.add_argument("--config", type=str, default=config_path, help="Path to YAML config file defining cleaning and feature options")
    parser.add_argument("--input-path", type=str, default="data/raw/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv", help="Directory containing raw CSV data (expects raw.csv)")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to write processed data CSV")
    parser.add_argument("--mlflow-uri", type=str, default="", help="MLflow Tracking Server URI")
    return parser.parse_args()

def main():
    
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            # Get sample size from config (default to None if not specified)
            sample_size = config.get('sampling', {}).get('sample_size', None)
            # Get sample description from config
            sample_description = config.get('sampling', {}).get('sample_description', "")
            # Get input and output paths
            input_path = args.input_path or config.get('data', {}).get('input_path', "data/raw/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv")
            output_dir = args.output_dir or config.get('data', {}).get('output_dir', "data/processed")
        
        print(f"Using configuration from: {config_path}")
        print(f"Sample size: {sample_size}")
        if sample_description:
            print(f"Sample description: {sample_description}")
    except Exception as e:
        print(f"Could not load configuration from {config_path}: {e}")
        print("Using default values")
        input_path = "data/raw/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv"
        output_dir = "data/processed"

    os.makedirs(output_dir, exist_ok=True)

    print("Pre-processing data...")

    output_filename = f"processed_dataset_{sample_description}.csv"
    output_path = f"{output_dir}/{output_filename}"

    # Initialize processor with raw data path
    processor = EcommerceDataProcessor(input_path)

    # Run the full processing pipeline
    processed_df = processor.run()

    # Apply sampling after processing if needed
    if sample_size is not None and sample_size < len(processed_df):
        processed_df = processed_df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} rows from processed dataset")

    # After processing the data and saving it
    processed_df.to_csv(output_path, index=False)
    print(f"Saved processed dataset to {output_path}")

    # Log dataset to MLflow
    dataset_info = EcommerceDataProcessor.log_dataset_to_mlflow(
        processed_df, 
        output_path, 
        sample_size=sample_size, 
        sample_description=sample_description
    )

    # Save dataset reference for pipeline
    with open(f"{output_dir}/latest_dataset_ref.json", 'w') as f:
        json.dump(dataset_info, f)

    print(f"Dataset processing completed: {output_filename}")

if __name__ == "__main__":
    args = parse_args()
    main()