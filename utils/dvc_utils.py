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
    Pull dataset from DVC storage using a specific tag from the Austin branch.
    """
    try:
        # Store current branch
        current_branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
        print(f"Current branch: {current_branch}")
        
        # Switch to Austin branch first
        subprocess.run(["git", "checkout", "Austin"], check=True)
        print("Switched to Austin branch")
        
        # Checkout the git tag
        subprocess.run(["git", "checkout", tag], check=True)
        print(f"Checked out tag: {tag}")
        
        # Pull the dataset from DVC
        subprocess.run(["dvc", "pull", dataset_path], check=True)
        print(f"Dataset pulled: {dataset_path}")
        
        # Return to original branch
        subprocess.run(["git", "checkout", current_branch], check=True)
        print(f"Returned to {current_branch} branch")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error pulling dataset: {e}")
        # Try to return to original branch on error
        try:
            if 'current_branch' in locals():
                subprocess.run(["git", "checkout", current_branch], check=False)
                print(f"Returned to {current_branch} branch after error")
        except:
            pass
        return False