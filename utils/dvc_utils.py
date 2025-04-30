# src/dvc_utils.py
import subprocess
import os

def setup_dvc_credentials(aws_access_key=None, aws_secret_key=None):
    """
    Set up DVC credentials.
    """
    if aws_access_key and aws_secret_key:
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_key
        print("AWS credentials set")
    else:
        print("Using existing AWS credentials")

def pull_dataset_by_tag(tag, dataset_path):
    """
    Pull dataset from DVC storage using a specific tag.
    """
    try:
        # Checkout the git tag
        subprocess.run(["git", "checkout", tag], check=True)
        print(f"Checked out tag: {tag}")
        
        # Pull the dataset from DVC
        subprocess.run(["dvc", "pull", dataset_path], check=True)
        print(f"Dataset pulled: {dataset_path}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error pulling dataset: {e}")
        return False