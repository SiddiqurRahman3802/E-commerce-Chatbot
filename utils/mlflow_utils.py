# src/mlflow_utils.py
import mlflow
import os
from datetime import datetime
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec


def setup_mlflow(tracking_uri=None, experiment_name=None):
    """
    Set up MLflow tracking.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    if experiment_name:
        mlflow.set_experiment(experiment_name)
        
    return mlflow.get_tracking_uri()

def log_model_info(model):
    """
    Log model information to MLflow.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    mlflow.log_param("trainable_params", trainable_params)
    mlflow.log_param("total_params", total_params)
    mlflow.log_param("trainable_percentage", 100 * trainable_params / total_params)

def start_run(run_name=None):
    """
    Start a new MLflow run.
    """
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    return mlflow.start_run(run_name=run_name)

def setup_mlflow_tracking(config):
    """
    Set up MLflow tracking and experiment.
    """
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", config.get('mlflow', {}).get('experiment_name'))
    
    # Set up MLflow tracking
    tracking_uri = setup_mlflow(
        mlflow_tracking_uri,
        experiment_name
    )
    print(f"MLflow tracking URI: {tracking_uri}")
    return tracking_uri

def log_model_to_mlflow(model, input_example="### Instruction: What is your return policy?\n\n### Response:"):
    """
    Log the model to MLflow with proper signature.
    """
    # Create input and output schemas
    input_schema = Schema([ColSpec(type="string")])
    output_schema = Schema([ColSpec(type="string")])
    
    # Create a model signature
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    
    # Log the model
    mlflow.pytorch.log_model(
        model, 
        "model", 
        input_example=input_example,
        signature=signature
    )