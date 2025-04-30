# src/finetuning_utils.py
import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)

def get_lora_config(r=8, lora_alpha=32, lora_dropout=0.05, target_modules=None):
    """
    Create a LoRA configuration for parameter-efficient fine-tuning.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )

def load_model_and_tokenizer(model_name, load_in_8bit=True, torch_dtype=torch.float16):
    """
    Load pretrained model and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def prepare_model_for_lora(model, lora_config):
    """
    Prepare model for LoRA fine-tuning.
    """
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    return model

def prepare_dataset(data_path, tokenizer, max_length=512, instruction_column="instruction", response_column="response"):
    """
    Load and prepare the dataset for instruction fine-tuning.
    """
    # Load the preprocessed CSV
    df = pd.read_csv(data_path)
    print(f"Dataset loaded with {len(df)} samples")
    
    # Prepare dataset in instruction-following format
    def format_chat(row):
        return {
            "text": f"### Instruction: {row[instruction_column]}\n\n### Response: {row[response_column]}"
        }
    
    formatted_df = df.apply(format_chat, axis=1)
    dataset = Dataset.from_pandas(formatted_df)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def get_training_args(output_dir, num_epochs=3, batch_size=8, gradient_accumulation_steps=4):
    """
    Create training arguments.
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=True,
    )

def generate_response(instruction, model, tokenizer, max_length=150):
    """
    Generate a response for a given instruction using the fine-tuned model.
    """
    input_text = f"### Instruction: {instruction}\n\n### Response:"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[1].strip()