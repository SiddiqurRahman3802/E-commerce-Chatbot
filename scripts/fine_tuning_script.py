#!/usr/bin/env python3
# scripts/fine_tuning_script.py
import sys
import os
import torch
from datetime import datetime

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fine_tuning import (
    get_lora_config, 
    load_model_and_tokenizer, 
    prepare_model_for_lora,
    prepare_dataset, 
    get_training_args, 
    generate_response
)
from utils.mlflow_utils import setup_mlflow, log_model_info, start_run
from utils.dvc_utils import setup_dvc_credentials, pull_dataset_by_tag
from utils.yaml_utils import load_yaml_config, merge_configs, save_yaml_config, flatten_config
from utils.constants import BASE_CONFIG_PATH, MODEL_CONFIG_PATH, DEFAULT_OUTPUT_DIR, MODELS_DIR
from transformers import DataCollatorForLanguageModeling, Trainer
from dotenv import load_dotenv

print("mlflow")
# Load environment variables
load_dotenv()

def main():
    # Load base configuration
    config = load_yaml_config(BASE_CONFIG_PATH)
    print(f"Loaded base configuration from {BASE_CONFIG_PATH}")
    
    # Check for model-specific configuration override
    if os.path.exists(MODEL_CONFIG_PATH):
        model_config = load_yaml_config(MODEL_CONFIG_PATH)
        config = merge_configs(config, model_config)
        print(f"Merged model configuration from {MODEL_CONFIG_PATH}")
    
    # Get all credentials and settings exclusively from environment variables
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", config.get('mlflow', {}).get('experiment_name'))
    hf_token = os.environ.get("HF_TOKEN")  # Hugging Face token if needed
    
    if aws_access_key and aws_secret_key:
        setup_dvc_credentials(aws_access_key, aws_secret_key)
        
        # Pull dataset using DVC
        if 'data' in config and 'dataset_tag' in config['data'] and 'dataset_path' in config['data']:
            dataset_success = pull_dataset_by_tag(config['data']['dataset_tag'], config['data']['dataset_path'])
            if not dataset_success:
                print("Failed to pull dataset. Exiting.")
                return
    else:
        print("AWS credentials not found in environment variables. Skipping DVC pull.")
    
    # Set up MLflow tracking
    tracking_uri = setup_mlflow(
        mlflow_tracking_uri,
        experiment_name
    )
    print(f"MLflow tracking URI: {tracking_uri}")
    
    # Set randomness controls for reproducibility
    seed = config.get('data', {}).get('seed', 42)
    torch.manual_seed(seed)
    
    # Generate a run ID that we'll use for directories and MLflow
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"chatbot_finetune_{run_timestamp}"
    output_dir = config.get('output', {}).get('output_dir', DEFAULT_OUTPUT_DIR)
    run_output_dir = os.path.join(output_dir, run_timestamp)
    
    # Start MLflow run
    with start_run(run_name) as run:
        run_id = run.info.run_id
        
        # Create necessary directories
        os.makedirs(run_output_dir, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Save the full configuration with the run (exclude any sensitive data)
        # Make a copy to avoid modifying the original config
        safe_config = config.copy()
        # Remove any sensitive fields that might have been added (should be using env vars anyway)
        if 'aws' in safe_config:
            if 'access_key' in safe_config['aws']:
                safe_config['aws']['access_key'] = "*** REDACTED ***"
            if 'secret_key' in safe_config['aws']:
                safe_config['aws']['secret_key'] = "*** REDACTED ***"
        
        run_config_path = os.path.join(run_output_dir, "config.yaml")

        save_yaml_config(safe_config, run_config_path)
        
        # Log configuration to MLflow
        import mlflow
        # Use the flatten_config utility to get flattened parameters (from the safe config)
        flattened_params = flatten_config(safe_config)
        mlflow.log_params(flattened_params)
        mlflow.log_artifact(run_config_path)
        
        # Load base model and tokenizer
        model_name = config.get('model', {}).get('base_model', "deepseek-ai/deepseek-coder-1.3b-instruct")
        print(f"Loading base model: {model_name}")
        
        # Choose appropriate dtype based on device and config
        use_8bit = config.get('model', {}).get('load_in_8bit', False)
        use_fp16 = config.get('training', {}).get('fp16', False)
        
        # Check if we're on Mac with MPS
        is_mps_available = hasattr(torch, 'mps') and torch.backends.mps.is_available()
        try_cpu = os.environ.get("USE_CPU", "0") == "1"
        
        if try_cpu:
            print("Using CPU as requested by environment variable USE_CPU=1")
            device_map = "cpu"
            torch_dtype = torch.float32
            use_8bit = False
            use_fp16 = False
        elif is_mps_available:
            print("MPS (Apple Silicon) device detected, adapting configuration...")
            # Try to get system info
            import subprocess
            try:
                total_ram = subprocess.check_output(['sysctl', '-n', 'hw.memsize']).strip()
                total_ram_gb = int(total_ram) / (1024**3)
                print(f"System has approximately {total_ram_gb:.2f} GB RAM")
                
                # If system has less than 32GB RAM, use CPU
                if total_ram_gb < 32:
                    print("System has less than 32GB RAM, using CPU instead of MPS for stability")
                    device_map = "cpu"
                else:
                    device_map = "mps"
            except:
                # If we can't get system info, default to MPS
                device_map = "mps"
                
            # Use float32 on MPS/CPU as it's more stable
            torch_dtype = torch.float32
            use_8bit = False  # Disable 8-bit on MPS
            use_fp16 = False  # Disable fp16 on MPS
        else:
            device_map = "auto"
            torch_dtype = torch.float16 if use_fp16 else torch.float32
        
        print(f"Using configuration: 8-bit={use_8bit}, fp16={use_fp16}, dtype={torch_dtype}, device_map={device_map}")
        
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            load_in_8bit=use_8bit,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        
        # Setup LoRA
        lora_config = get_lora_config(
            r=config.get('lora', {}).get('r', 8),
            lora_alpha=config.get('lora', {}).get('alpha', 32),
            lora_dropout=config.get('lora', {}).get('dropout', 0.05),
            target_modules=config.get('lora', {}).get('target_modules')
        )
        model = prepare_model_for_lora(model, lora_config)
        
        # Log model info
        log_model_info(model)
        
        # Prepare dataset
        dataset_path = config.get('data', {}).get('dataset_path')
        if not dataset_path:
            print("Dataset path not specified in config. Exiting.")
            return
            
        tokenized_dataset = prepare_dataset(
            dataset_path, 
            tokenizer, 
            config.get('training', {}).get('max_length', 512),
            config.get('data', {}).get('instruction_column', 'instruction'),
            config.get('data', {}).get('response_column', 'response')
        )
        
        # Split dataset
        split_dataset = tokenized_dataset.train_test_split(
            test_size=config.get('data', {}).get('test_size', 0.1), 
            seed=seed
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        
        # Training arguments
        training_args = get_training_args(
            output_dir=run_output_dir,
            num_epochs=config.get('training', {}).get('epochs', 3),
            batch_size=config.get('training', {}).get('batch_size', 8),
            gradient_accumulation_steps=config.get('training', {}).get('gradient_accumulation_steps', 4)
        )
        
        # Override specific training args from config
        training_config = config.get('training', {})
        if 'learning_rate' in training_config:
            training_args.learning_rate = training_config['learning_rate']
        if 'weight_decay' in training_config:
            training_args.weight_decay = training_config['weight_decay']
        if 'warmup_steps' in training_config:
            training_args.warmup_steps = training_config['warmup_steps']
        if 'fp16' in training_config:
            training_args.fp16 = training_config['fp16']
        if 'evaluation_strategy' in training_config:
            training_args.eval_strategy = training_config['evaluation_strategy']
        if 'save_strategy' in training_config:
            training_args.save_strategy = training_config['save_strategy']
        if 'save_total_limit' in training_config:
            training_args.save_total_limit = training_config['save_total_limit']
        if 'load_best_model_at_end' in training_config:
            training_args.load_best_model_at_end = training_config['load_best_model_at_end']
            
        # Set logging steps from output config if available
        if 'output' in config and 'logging_steps' in config['output']:
            training_args.logging_steps = config['output']['logging_steps']
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # We want causal language modeling, not masked language modeling
        )
        
        # Define trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
        
        # Train model
        print("Starting training...")
        train_result = trainer.train()
        
        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        
        # Evaluate
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        
        # Log all metrics to MLflow
        mlflow.log_metrics({f"train_{k}": v for k, v in metrics.items()})
        mlflow.log_metrics({f"eval_{k}": v for k, v in eval_metrics.items()})
        
        # Log model to MLflow

        input_schema = Schema([ColSpec(type="string")])
        # Define the output schema
        output_schema = Schema([ColSpec(type="string")])

        # Create a model signature

        signature = ModelSignature(inputs=input_schema, outputs=output_schema)


        mlflow.pytorch.log_model(
            model, 
            "model", 
            input_example="### Instruction: What is your return policy?\n\n### Response:",
            signature=signature  # Use the explicit signature
        )
        
        # Optional: Push to Hugging Face Hub using environment variable for authentication
        if config.get('hub', {}).get('push_to_hub', False) and hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token)
                
                model_name = config.get('hub', {}).get("repository_username") + f"/{run_id[:8]}" or f"ShenghaoisYummy/ecommerce-chatbot-{run_id[:8]}"
                model.push_to_hub(model_name)
                tokenizer.push_to_hub(model_name)
                print(f"Model pushed to Hugging Face Hub: {model_name}")
            except Exception as e:
                print(f"Error pushing to Hub: {e}")
        elif config.get('hub', {}).get('push_to_hub', False):
            print("HF_TOKEN environment variable not set. Skipping push to Hub.")
        
        # print(f"Training completed. Model saved to {final_model_path}")
        print(f"MLflow run ID: {run_id}")

if __name__ == "__main__":
    main()