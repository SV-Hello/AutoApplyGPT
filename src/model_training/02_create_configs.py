"""
LoRA Training Configuration for Phi-3-mini
Optimized for A100 GPU (40GB VRAM)
"""
import json
from dataclasses import dataclass, asdict

@dataclass
class LoRAConfig:
    """LoRA hyperparameters"""
    r: int = 16  # LoRA rank (higher = more parameters, better but slower)
    lora_alpha: int = 32  # LoRA scaling factor (typically 2x rank)
    lora_dropout: float = 0.05  # Dropout for LoRA layers
    target_modules: list = None  # Which layers to apply LoRA to
    bias: str = "none"  # Bias training ("none", "all", "lora_only")
    task_type: str = "CAUSAL_LM"  # Task type
    
    def __post_init__(self):
        if self.target_modules is None:
            # Phi-3 specific target modules (query, key, value projections)
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Model settings
    model_name: str = "/scratch/user/u.sv309862/689csce/model"
    max_seq_length: int = 2048  # Max context length
    
    # Training settings
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    
    # Optimization
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 0.3
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"
    
    # Optimizer
    optim: str = "paged_adamw_32bit"  # Memory-efficient optimizer
    
    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3  # Keep only best 3 checkpoints
    
    # Logging
    logging_steps: int = 10
    logging_dir: str = "./logs"
    report_to: str = "tensorboard"
    
    # Mixed precision training
    fp16: bool = True
    bf16: bool = False
    
    # Misc
    seed: int = 42
    output_dir: str = "./outputs"
    
    # Memory optimization
    gradient_checkpointing: bool = False
    
    # Load best model at end
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

@dataclass
class QuantizationConfig:
    """4-bit quantization config for inference"""
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"

def save_configs(output_dir):
    """Save all configs to JSON files"""
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configs
    lora_config = LoRAConfig()
    training_config = TrainingConfig()
    quant_config = QuantizationConfig()
    
    # Save to JSON
    configs = {
        'lora_config.json': asdict(lora_config),
        'training_config.json': asdict(training_config),
        'quantization_config.json': asdict(quant_config)
    }
    
    for filename, config in configs.items():
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Saved {filepath}")
    
    # Also save a merged config for reference
    merged_config = {
        'lora': asdict(lora_config),
        'training': asdict(training_config),
        'quantization': asdict(quant_config)
    }
    
    with open(output_dir / 'all_configs.json', 'w') as f:
        json.dump(merged_config, f, indent=2)
    
    print(f"\n📋 Configuration Summary:")
    print(f"  Model: {training_config.model_name}")
    print(f"  LoRA rank: {lora_config.r}")
    print(f"  Epochs: {training_config.num_train_epochs}")
    print(f"  Batch size (per device): {training_config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Max sequence length: {training_config.max_seq_length}")

def estimate_training_time(num_examples, config):
    """Estimate training time"""
    effective_batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps
    steps_per_epoch = num_examples / effective_batch_size
    total_steps = steps_per_epoch * config.num_train_epochs
    
    # Rough estimate: ~1.5 seconds per step on A100 for Phi-3-mini with LoRA
    seconds_per_step = 1.5
    total_seconds = total_steps * seconds_per_step
    hours = total_seconds / 3600
    
    print(f"\n⏱️  Training Time Estimate (A100 GPU):")
    print(f"  Steps per epoch: {steps_per_epoch:.0f}")
    print(f"  Total steps: {total_steps:.0f}")
    print(f"  Estimated time: {hours:.1f} hours")
    
    return hours

def estimate_memory_usage(config):
    """Estimate memory requirements"""
    # Phi-3-mini: 3.8B parameters
    # With LoRA (rank 16): ~47M trainable parameters
    # Base model in 4-bit: ~2.3 GB
    # LoRA weights: ~200 MB
    # Activations + gradients: ~8-12 GB with gradient checkpointing
    
    print(f"\n💾 Memory Usage Estimate:")
    print(f"  Base model (4-bit): ~2.3 GB")
    print(f"  LoRA parameters: ~200 MB")
    print(f"  Activations + gradients: ~10 GB")
    print(f"  Total: ~12-15 GB VRAM")
    print(f"  ✅ Fits comfortably in A100 40GB")

def calculate_trainable_parameters():
    """Calculate number of trainable parameters with LoRA"""
    # Phi-3-mini has 3.8B parameters
    base_params = 3.8e9
    
    # LoRA with rank 16 on 4 attention projection layers
    # Each layer has d_model = 3072
    d_model = 3072
    rank = 16
    num_layers = 32
    num_target_modules = 4  # q, k, v, o projections
    
    # LoRA adds: 2 * d_model * rank per target module
    lora_params_per_layer = 2 * d_model * rank * num_target_modules
    total_lora_params = lora_params_per_layer * num_layers
    
    trainable_percentage = (total_lora_params / base_params) * 100
    
    print(f"\n🔢 Parameter Count:")
    print(f"  Base model: {base_params/1e9:.2f}B parameters")
    print(f"  LoRA parameters: {total_lora_params/1e6:.1f}M parameters")
    print(f"  Trainable: {trainable_percentage:.2f}% of base model")
    print(f"  Frozen: {100-trainable_percentage:.2f}%")

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    output_dir = Path('../../data/autoapply_training/configs')
    
    print("="*70)
    print("LoRA Training Configuration Generator")
    print("="*70)
    
    # Save configs
    save_configs(output_dir)
    
    # Show estimates
    training_config = TrainingConfig()
    
    # Assume ~1600 training examples (80% of 2000)
    estimate_training_time(1600, training_config)
    estimate_memory_usage(training_config)
    calculate_trainable_parameters()
    
    print(f"\n{'='*70}")
    print("✅ Configuration files created!")
    print(f"{'='*70}")
    print(f"\nConfigs saved to: {output_dir}")
    print(f"\nYou can modify these configs before training.")
