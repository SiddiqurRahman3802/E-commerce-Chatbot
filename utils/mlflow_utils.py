# src/mlflow_utils.py
import mlflow
import os
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