# Import the processor
import sys
import os
import yaml
from datetime import datetime
import subprocess

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_prep import EcommerceDataProcessor

# Load the sample size and description from configuration
config_path = "configs/model_config.yaml"
sample_size = None
sample_description = None

try:
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
        # Get sample size from config (default to None if not specified)
        sample_size = config.get('data_preprocessing', {}).get('sample_size', None)
        # Get sample description from config
        sample_description = config.get('data_preprocessing', {}).get('sample_description', "")
    
    print(f"Using sample size from config: {sample_size}")
    if sample_description:
        print(f"Sample description: {sample_description}")
except Exception as e:
    print(f"Could not load sample configuration from config: {e}")
    print("Will use full dataset")

os.makedirs("data/processed", exist_ok=True)

print("Pre-processing data...")
# Get current timestamp for unique filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Include sample size and description in the filename if specified
filename_components = []
if sample_size:
    filename_components.append(f"{sample_size}rows")
if sample_description:
    filename_components.append(sample_description)

prefix = "processed_dataset"
if filename_components:
    prefix += "_" + "_".join(filename_components)

output_filename = f"{prefix}_{timestamp}.csv"
output_path = f"data/processed/{output_filename}"

# Initialize processor with raw data path
processor = EcommerceDataProcessor("data/raw/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv")

# Run the full processing pipeline
processed_df = processor.run()

# Apply sampling after processing if needed
if sample_size is not None and sample_size < len(processed_df):
    processed_df = processed_df.sample(n=sample_size, random_state=42)
    print(f"Sampled {sample_size} rows from processed dataset")

# Save the processed data
processed_df.to_csv(output_path, index=False)
print(f"Saved processed dataset to {output_path}")


# Set Git identity before committing
subprocess.run(["git", "config", "--global", "user.email", "hsupisces@Hotmail.com"], check=False)
subprocess.run(["git", "config", "--global", "user.name", "ShenghaoisYummy"], check=False)
subprocess.run(['git', 'config', '--global', 'credential.helper', 'store'], check=True)
with open(os.path.expanduser('~/.git-credentials'), 'w') as f:
    f.write(f"https://{os.environ['GIT_USERNAME']}:{os.environ['GIT_PASSWORD']}@github.com\n")


#Track with DVC
subprocess.run(["dvc", "add", output_path], check=True)
subprocess.run(["git", "add", f"{output_path}.dvc"], check=True)
subprocess.run(["git", "commit", "-m", f"Add dataset ({timestamp})"], check=True)

# Create a tag for this dataset version
tag_name = f"tag-{output_filename}"
subprocess.run(["git", "tag", "-a", tag_name, "-m", f"Dataset processed at {timestamp}"], check=True)

# Push everything
subprocess.run(["dvc", "push"], check=True)
subprocess.run(["git", "push", "origin", "Austin"], check=True)
subprocess.run(["git", "push", "--tags"], check=True)

print(f"Dataset tracked with DVC and tagged as: {tag_name}")
print(f"Dataset processed as processed_dataset_{sample_size}rows_{timestamp}.csv")