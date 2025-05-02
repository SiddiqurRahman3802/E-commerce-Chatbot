import os

def push_to_huggingface_hub(model, tokenizer, config, run_id):
    """
    Push the model to Hugging Face Hub if configured.
    """
    hf_token = os.environ.get("HF_TOKEN")
    
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