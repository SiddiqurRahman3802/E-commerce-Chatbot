# src/mlflow_utils.py
import mlflow
import os
import json
from datetime import datetime
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec, TensorSpec
import numpy as np
import threading
import time


def mlflow_init(tracking_uri=None, experiment_name=None):
    """
    Set up MLflow tracking.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    if experiment_name:
        mlflow.set_experiment(experiment_name)
        
    return mlflow.get_tracking_uri()

def mlflow_log_model_info(model):
    """
    Log model information to MLflow.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    mlflow.log_param("trainable_params", trainable_params)
    mlflow.log_param("total_params", total_params)
    mlflow.log_param("trainable_percentage", 100 * trainable_params / total_params)

def mlflow_start_run(run_name=None):
    """
    Start a new MLflow run.
    """
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    return mlflow.start_run(run_name=run_name)

def mlflow_setup_tracking(config):
    """
    Set up MLflow tracking and experiment.
    """
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    
    # If not set in environment, use config or default value
    if not mlflow_tracking_uri:
        mlflow_tracking_uri = config.get('mlflow', {}).get('tracking_uri', "file:///./mlruns")
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
        
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", config.get('mlflow', {}).get('experiment_name'))
    
    # Set up MLflow tracking
    tracking_uri = mlflow_init(
        mlflow_tracking_uri,
        experiment_name
    )
    print(f"MLflow tracking URI: {tracking_uri}")
    return tracking_uri

def mlflow_log_model(model, input_example="### Instruction: What is your return policy?\n\n### Response:"):
    """
    Log the model to MLflow with proper signature for text generation.
    """
    try:
        # Skip input example and only use signature for text models
        signature = ModelSignature(
            inputs=Schema([ColSpec(type="string", name="inputs")]),
            outputs=Schema([ColSpec(type="string", name="outputs")])
        )
        
        # Set a timeout for the operation (in seconds)
        log_success = [False]  # Using a list to make it mutable inside the thread
        log_error = [None]
        
        def log_model_with_timeout():
            try:
                # Log the model without input_example to avoid conversion issues
                mlflow.pytorch.log_model(
                    model, 
                    "model", 
                    signature=signature
                )
                log_success[0] = True
                print("Model logged to MLflow successfully")
            except Exception as e:
                log_error[0] = e
                print(f"Error logging model to MLflow: {e}")
        
        # Start the logging in a thread
        log_thread = threading.Thread(target=log_model_with_timeout)
        log_thread.daemon = True
        log_thread.start()
        
        # Wait for up to 5 minutes
        timeout = 300  # seconds
        start_time = time.time()
        
        print(f"Waiting up to {timeout} seconds for model logging to complete...")
        while log_thread.is_alive() and time.time() - start_time < timeout:
            time.sleep(5)  # Check every 5 seconds
            print(f"Still logging model... ({int(time.time() - start_time)} seconds elapsed)")
        
        if log_thread.is_alive():
            # Timeout occurred
            print(f"Model logging timed out after {timeout} seconds.")
            print("Continuing with execution. The logging may complete in the background.")
            return False
        elif log_success[0]:
            return True
        elif log_error[0]:
            raise log_error[0]
        
    except Exception as e:
        print(f"Error logging model to MLflow: {e}")
        # Still try to log the model without signature or examples as fallback
        try:
            mlflow.pytorch.log_model(model, "model")
            print("Model logged to MLflow with fallback method")
            return True
        except Exception as e2:
            print(f"Failed to log model with fallback method: {e2}")
            return False

def log_transformers_model(model, tokenizer, task="text-generation", output_dir=None):
    """
    Log a transformers model to MLflow with its tokenizer.
    
    Args:
        model: The transformers model to log
        tokenizer: The tokenizer to log with the model
        task: The NLP task (e.g., "text-generation")
        output_dir: Local directory to save the model to as well (optional)
    
    Returns:
        run_id: The MLflow run ID for this model
    """
    print("Logging transformers model to MLflow...")
    
    try:
        # Log the model and its components
        mlflow.transformers.log_model(
            transformers_model=model,
            artifact_path="transformers-model",
            task=task,
            components={
                "tokenizer": tokenizer
            }
        )
        
        # Get current run ID
        run_id = mlflow.active_run().info.run_id
        
        # If output_dir is provided, also save locally
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            mlflow.transformers.save_model(
                transformers_model=model,
                path=output_dir,
                task=task,
                components={
                    "tokenizer": tokenizer
                }
            )
            
            # Store run info
            model_info = {
                "mlflow_run_id": run_id,
                "tracking_uri": mlflow.get_tracking_uri(),
                "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
            }
            
            with open(os.path.join(output_dir, "model_info.json"), "w") as f:
                json.dump(model_info, f)
            
            # Also save to results directory for DVC
            os.makedirs("results", exist_ok=True)
            with open("results/model_location.json", "w") as f:
                json.dump(model_info, f)
                
        print(f"Model successfully logged to MLflow with run_id: {run_id}")
        return run_id
        
    except Exception as e:
        print(f"Error logging transformers model to MLflow: {e}")
        return None

def load_model_from_dagshub(run_id=None, model_info_path="results/model_location.json"):
    """
    Load a model from DagShub/MLflow.
    
    Args:
        run_id: Optional explicit run_id to load
        model_info_path: Path to the model info JSON file
        
    Returns:
        model_components: Dictionary containing 'model' and 'tokenizer'
    """
    try:
        # If no run_id provided, try to load from model_info.json
        if not run_id:
            if os.path.exists(model_info_path):
                with open(model_info_path, "r") as f:
                    model_info = json.load(f)
                run_id = model_info.get("mlflow_run_id")
                # If tracking URI is in the file, set it
                if "tracking_uri" in model_info:
                    mlflow.set_tracking_uri(model_info["tracking_uri"])
            else:
                raise ValueError(f"No run_id provided and {model_info_path} not found")
        
        # Construct model URI
        model_uri = f"runs:/{run_id}/transformers-model"
        
        print(f"Loading model from MLflow: {model_uri}")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        
        # Load the model
        model_components = mlflow.transformers.load_model(model_uri=model_uri)
        
        return model_components
        
    except Exception as e:
        print(f"Error loading model from MLflow: {e}")
        raise