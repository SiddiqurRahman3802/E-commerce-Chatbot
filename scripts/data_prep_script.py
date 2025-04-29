# Import the processor
from src.data_prep import EcommerceDataProcessor
from datetime import datetime
import subprocess
import os

print("Script starting...")
# Ask user for strategy interactively
strategy = input("Enter preprocessing strategy (e.g., raw data, new_features_data, etc.): ")

# Get current timestamp for unique filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"processed_dataset_{strategy}_{timestamp}.csv"
output_path = f"data/processed/{output_filename}"

print(f"Processing data with strategy: {strategy}")

# Initialize with your raw data path
processor = EcommerceDataProcessor("data/raw/bitext-retail-ecommerce-llm-chatbot-training-dataset.csv")

# Run the full processing pipeline
processed_df = processor.run()

# Save the processed data with strategy and timestamp in filename
processed_df.to_csv(output_path, index=False)
print(f"Saved processed dataset to {output_path}")

# # Track with DVC
# subprocess.run(["dvc", "add", output_path], check=True)
# subprocess.run(["git", "add", f"{output_path}.dvc"], check=True)
# subprocess.run(["git", "commit", "-m", f"Add {strategy} dataset ({timestamp})"], check=True)

# # Create a tag for this dataset version
# tag_name = f"data-{strategy}-{timestamp}"
# subprocess.run(["git", "tag", "-a", tag_name, "-m", f"Dataset processed with {strategy} strategy"], check=True)

# # Push everything
# subprocess.run(["dvc", "push"], check=True)
# subprocess.run(["git", "push", "origin", "main"], check=True)
# subprocess.run(["git", "push", "--tags"], check=True)

# print(f"Dataset tracked with DVC and tagged as: {tag_name}")
print(f"Dataset processed as processed_dataset_{strategy}_{timestamp}.csv")