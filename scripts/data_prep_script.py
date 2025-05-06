# Import the processor
import sys
import os
import yaml
from datetime import datetime
import subprocess
import json
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_prep import EcommerceDataProcessor

# Load configuration
config_path = "configs/data_prep_config.yaml"
sample_size = None
sample_description = None

try:
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
        # Get sample size from config (default to None if not specified)
        sample_size = config.get('sampling', {}).get('sample_size', None)
        # Get sample description from config
        sample_description = config.get('sampling', {}).get('sample_description', "")
        # Get input and output paths
        input_path = config.get('data', {}).get('input_path', "data/raw/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv")
        output_dir = config.get('data', {}).get('output_dir', "data/processed")
        # Get version control settings
        # git_branch = config.get('version_control', {}).get('git_branch', "Austin")
        # add_to_dvc = config.get('version_control', {}).get('add_to_dvc', True)
        # create_git_tag = config.get('version_control', {}).get('create_git_tag', True)
    
    print(f"Using configuration from: {config_path}")
    print(f"Sample size: {sample_size}")
    if sample_description:
        print(f"Sample description: {sample_description}")
except Exception as e:
    print(f"Could not load configuration from {config_path}: {e}")
    print("Using default values")
    input_path = "data/raw/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv"
    output_dir = "data/processed"
    # git_branch = "Austin"
    # add_to_dvc = True
    # create_git_tag = True

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

# if add_to_dvc:
#     # Set Git identity before committing
#     subprocess.run(["git", "config", "--global", "user.email", "hsupisces@Hotmail.com"], check=False)
#     subprocess.run(["git", "config", "--global", "user.name", "ShenghaoisYummy"], check=False)
#     subprocess.run(['git', 'config', '--global', 'credential.helper', 'store'], check=True)
    
#     try:
#         with open(os.path.expanduser('~/.git-credentials'), 'w') as f:
#             f.write(f"https://{os.environ['GIT_USERNAME']}:{os.environ['GIT_PASSWORD']}@github.com\n")

#         # Skip DVC add since it's handled by the pipeline
#         # Just create a file with metadata for reference
#         with open(f"{output_path}.info", 'w') as f:
#             f.write(f"Dataset processed at {timestamp}\n")
#             f.write(f"Sample size: {sample_size}\n")
#             f.write(f"Sample description: {sample_description}\n")
        
#         subprocess.run(["git", "add", f"{output_path}.info"], check=True)
#         subprocess.run(["git", "commit", "-m", f"Add dataset info ({timestamp})"], check=True)

#         # Create a tag for this dataset version
#         if create_git_tag:
#             tag_name = f"tag-{output_filename}"
#             subprocess.run(["git", "tag", "-a", tag_name, "-m", f"Dataset processed at {timestamp}"], check=True)

#         # Push changes
#         subprocess.run(["git", "push", "origin", git_branch], check=True)
#         if create_git_tag:
#             subprocess.run(["git", "push", "--tags"], check=True)
#             print(f"Dataset metadata tagged as: {tag_name}")
#         else:
#             print(f"Dataset metadata committed")
#     except Exception as e:
#         print(f"Error in Git operations: {e}")
#         print("Processed data was saved but metadata not tracked in Git")

print(f"Dataset processing completed: {output_filename}")