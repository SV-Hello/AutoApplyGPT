"""
Step 3: Train Phi-3-mini with LoRA
Main training script for HPRC A100 cluster
"""
import os
import json
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import wandb

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_configs(config_dir):
    """Load configuration files"""
    config_dir = Path(config_dir)
    
    with open(config_dir / 'all_configs.json', 'r') as f:
        configs = json.load(f)
    
    return configs

def load_model_and_tokenizer(model_name, quantization_config):
    """Load model and tokenizer with quantization"""
    print(f"📦 Loading model: {model_name}")
    
    # Create BitsAndBytes config for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quantization_config['load_in_4bit'],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=quantization_config['bnb_4bit_use_double_quant'],
        bnb_4bit_quant_type=quantization_config['bnb_4bit_quant_type']
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        #device_map="auto",
        attn_implementation='eager',
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Set padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    
    print(f"✅ LoRA applied:")
    print(f"  Trainable params: {trainable_params/1e6:.2f}M ({trainable_params/all_params*100:.2f}%)")
    print(f"  All params: {all_params/1e9:.2f}B")
    
    return model, tokenizer

def setup_lora(model, lora_config):
    """Setup LoRA adapters"""
    print(f"\n🔧 Setting up LoRA...")

    # Create LoRA config
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        target_modules=lora_config['target_modules'],
        bias=lora_config['bias']
    )

    # Apply LoRA to model
    model = get_peft_model(model, config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())

    print(f"✅ LoRA applied:")
    print(f"  Trainable params: {trainable_params/1e6:.2f}M ({trainable_params/all_params*100:.2f}%)")
    print(f"  All params: {all_params/1e9:.2f}B")
    return model

def preprocess_function(examples, tokenizer, max_length=2048):
    """Tokenize and format examples"""
    # Combine instruction and output
    texts = []
    for instruction, output in zip(examples['instruction'], examples['output']):
        # Format as conversation
        text = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{output}<|end|>"
        texts.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,  # Dynamic padding in data collator
        return_tensors=None
    )
    
    # Set labels = input_ids for causal language modeling
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized

def load_and_preprocess_data(data_dir, tokenizer, max_length=2048):
    """Load and preprocess dataset"""
    print(f"📂 Loading dataset from {data_dir}")
    
    # Load dataset
    dataset = load_from_disk(data_dir)
    
    print(f"Dataset sizes:")
    for split in dataset:
        print(f"  {split}: {len(dataset[split])} examples")
    
    # Preprocess
    print(f"🔄 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing"
    )
    
    print(f"✅ Dataset preprocessed")
    
    return tokenized_dataset

def setup_training_args(config, output_dir):
    """Setup training arguments"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=config['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        warmup_ratio=config['warmup_ratio'],
        max_grad_norm=config['max_grad_norm'],
        lr_scheduler_type=config['lr_scheduler_type'],
        optim=config['optim'],
        evaluation_strategy=config['eval_strategy'],
        eval_steps=config['eval_steps'],
        save_strategy=config['save_strategy'],
        save_steps=config['save_steps'],
        save_total_limit=config['save_total_limit'],
        logging_steps=config['logging_steps'],
        logging_dir=config['logging_dir'],
        report_to=config['report_to'],
        fp16=config['fp16'],
        bf16=config['bf16'],
        seed=config['seed'],
        gradient_checkpointing=config['gradient_checkpointing'],
        load_best_model_at_end=config['load_best_model_at_end'],
        metric_for_best_model=config['metric_for_best_model'],
        greater_is_better=config['greater_is_better'],
        push_to_hub=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False
    )
    
    return training_args

def train(
    data_dir,
    config_dir,
    output_dir,
    use_wandb=False,
    wandb_project="autoapply-gpt"
):
    """Main training function"""
    
    print("="*70)
    print("AutoApplyGPT Training with LoRA")
    print("="*70)

    #print(f"🔍 CUDA available: {torch.cuda.is_available()}")
    #print(f"🔍 Device count: {torch.cuda.device_count()}")
    #if torch.cuda.is_available():
    #    print(f"🔍 Current device: {torch.cuda.current_device()}")
    #    print(f"🔍 Device name: {torch.cuda.get_device_name(0)}")
    
    # Load configs
    print(f"\n📋 Loading configurations...")
    configs = load_configs(config_dir)
    lora_config = configs['lora']
    training_config = configs['training']
    quant_config = configs['quantization']
    
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(
            project=wandb_project,
            config={**lora_config, **training_config}
        )
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        training_config['model_name'],
        quant_config
    )
    
    # Setup LoRA
    model = setup_lora(model, lora_config)
    
    # Load and preprocess data
    tokenized_dataset = load_and_preprocess_data(
        data_dir,
        tokenizer,
        training_config['max_seq_length']
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM (not masked LM)
    )
    
    # Setup training arguments
    training_args = setup_training_args(training_config, output_dir)
    
    # Create trainer
    print(f"\n🚀 Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator,
    )
    
    # Train
    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")
    
    trainer.train()
    
    # Save final model
    print(f"\n💾 Saving final model...")
    final_output_dir = Path(output_dir) / "final_model"
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print(f"✅ Model saved to {final_output_dir}")
    
    # Evaluate on test set
    print(f"\n📊 Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_dataset['test'])
    
    print(f"\n{'='*70}")
    print("Test Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    print(f"{'='*70}")
    
    # Save test results
    with open(Path(output_dir) / "test_results.json", 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✅ Training complete!")
    
    return trainer, test_results

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train AutoApplyGPT with LoRA")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../data/autoapply_training/hf_dataset",
        help="Path to HuggingFace dataset"
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="../../data/autoapply_training/configs",
        help="Path to config files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../data/autoapply_training/outputs",
        help="Path to save outputs"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="autoapply-gpt",
        help="W&B project name"
    )
    
    args = parser.parse_args()
    
    # Run training
    trainer, test_results = train(
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project
    )
    
    print("\n🎉 All done!")

if __name__ == "__main__":
    main()
