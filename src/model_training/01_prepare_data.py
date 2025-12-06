"""
Step 1: Prepare training data for fine-tuning
Converts JSONL to HuggingFace dataset format and creates train/val/test splits
"""
import json
import random
from pathlib import Path
from datasets import Dataset, DatasetDict
import pandas as pd

random.seed(42)

def load_jsonl(filepath):
    """Load JSONL file"""
    print(f"📂 Loading data from {filepath}")
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} examples")
    
    #data = data[:1000]
    
    return data

def create_training_prompt(example):
    """
    Create the input prompt for the model
    Format: instruction + resume + job description -> tailored resume + cover letter
    """
    prompt = f"""You are an expert resume writer. Tailor this resume for the job posting and write a cover letter.

RESUME:
{example['original_resume']}

JOB POSTING:
Title: {example['job_title']}
Description: {example['job_description']}
Skills: {example.get('job_skills', 'Not specified')}

Generate a tailored resume and cover letter for this application."""
    
    return prompt

def create_training_completion(example):
    """
    Create the expected output (completion) for the model
    """
    completion = f"""TAILORED RESUME:
{example['tailored_resume']}

COVER LETTER:
{example['cover_letter']}"""
    
    return completion

def format_for_training(data, format_type='instruction'):
    """
    Format data for different training frameworks
    
    Args:
        format_type: 'instruction' (for instruction tuning) or 'chat' (for chat format)
    """
    formatted_data = []
    
    for example in data:
        if format_type == 'instruction':
            # Standard instruction-completion format
            formatted_example = {
                'instruction': create_training_prompt(example),
                'output': create_training_completion(example),
                'metadata': {
                    'resume_id': example.get('resume_id'),
                    'job_id': example.get('job_id'),
                    'similarity_score': example.get('similarity_score'),
                }
            }
        elif format_type == 'chat':
            # Chat format (for models expecting conversation format)
            formatted_example = {
                'messages': [
                    {
                        'role': 'user',
                        'content': create_training_prompt(example)
                    },
                    {
                        'role': 'assistant',
                        'content': create_training_completion(example)
                    }
                ],
                'metadata': {
                    'resume_id': example.get('resume_id'),
                    'job_id': example.get('job_id'),
                    'similarity_score': example.get('similarity_score'),
                }
            }
        else:
            raise ValueError(f"Unknown format_type: {format_type}")
        
        formatted_data.append(formatted_example)
    
    return formatted_data

def create_splits(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split data into train/val/test sets
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Shuffle
    random.shuffle(data)
    
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    print(f"\n📊 Data Split:")
    print(f"  Train: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
    print(f"  Val:   {len(val_data)} ({len(val_data)/total*100:.1f}%)")
    print(f"  Test:  {len(test_data)} ({len(test_data)/total*100:.1f}%)")
    
    return train_data, val_data, test_data

def save_as_huggingface_dataset(train_data, val_data, test_data, output_dir):
    """
    Save as HuggingFace Dataset format
    """
    print(f"\n💾 Saving as HuggingFace Dataset to {output_dir}")
    
    # Create Dataset objects
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    # Save to disk
    dataset_dict.save_to_disk(output_dir)
    
    print(f"✅ Saved HuggingFace dataset")
    print(f"  - Train: {len(train_dataset)} examples")
    print(f"  - Validation: {len(val_dataset)} examples")
    print(f"  - Test: {len(test_dataset)} examples")
    
    return dataset_dict

def save_as_jsonl(train_data, val_data, test_data, output_dir):
    """
    Save as JSONL files (alternative format)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving as JSONL files to {output_dir}")
    
    splits = {
        'train.jsonl': train_data,
        'val.jsonl': val_data,
        'test.jsonl': test_data
    }
    
    for filename, data in splits.items():
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            for example in data:
                f.write(json.dumps(example) + '\n')
        print(f"  ✅ Saved {filepath} ({len(data)} examples)")

def print_statistics(data):
    """Print dataset statistics"""
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total examples: {len(data)}")
    
    # Calculate prompt and completion lengths
    prompt_lengths = [len(create_training_prompt(ex)) for ex in data]
    completion_lengths = [len(create_training_completion(ex)) for ex in data]
    
    print(f"\n  Prompt (Input) Lengths:")
    print(f"    Mean: {sum(prompt_lengths)/len(prompt_lengths):.0f} chars")
    print(f"    Min:  {min(prompt_lengths)} chars")
    print(f"    Max:  {max(prompt_lengths)} chars")
    
    print(f"\n  Completion (Output) Lengths:")
    print(f"    Mean: {sum(completion_lengths)/len(completion_lengths):.0f} chars")
    print(f"    Min:  {min(completion_lengths)} chars")
    print(f"    Max:  {max(completion_lengths)} chars")
    
    # Estimate tokens (rough: 1 token ≈ 4 chars)
    avg_prompt_tokens = sum(prompt_lengths) / len(prompt_lengths) / 4
    avg_completion_tokens = sum(completion_lengths) / len(completion_lengths) / 4
    
    print(f"\n  Estimated Token Counts (1 token ≈ 4 chars):")
    print(f"    Avg prompt: {avg_prompt_tokens:.0f} tokens")
    print(f"    Avg completion: {avg_completion_tokens:.0f} tokens")
    print(f"    Avg total: {avg_prompt_tokens + avg_completion_tokens:.0f} tokens")

def main():
    # Paths
    input_file = Path('../../data/processed/training_data_filtered.jsonl')
    output_base = Path('../../autoapply_training')
    
    # Check if input exists
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print("Run the data generation pipeline first!")
        return
    
    # Load data
    data = load_jsonl(input_file)
    
    # Print statistics
    print_statistics(data)
    
    # Format for training
    print(f"\n🔄 Formatting data for instruction tuning...")
    formatted_data = format_for_training(data, format_type='instruction')
    
    # Create splits
    train_data, val_data, test_data = create_splits(
        formatted_data,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    # Save as HuggingFace dataset
    hf_output_dir = output_base / 'hf_dataset'
    dataset_dict = save_as_huggingface_dataset(
        train_data, val_data, test_data, hf_output_dir
    )
    
    # Also save as JSONL (backup format)
    jsonl_output_dir = output_base / 'jsonl_splits'
    save_as_jsonl(train_data, val_data, test_data, jsonl_output_dir)
    
    # Save a few examples for inspection
    print(f"\n💡 Saving sample examples for inspection...")
    samples_dir = output_base / 'samples'
    samples_dir.mkdir(exist_ok=True)
    
    with open(samples_dir / 'train_sample.json', 'w') as f:
        json.dump(train_data[0], f, indent=2)
    
    print(f"  ✅ Saved sample to {samples_dir / 'train_sample.json'}")
    
    print(f"\n{'='*70}")
    print("✅ DATA PREPARATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nDatasets saved to:")
    print(f"  • HuggingFace format: {hf_output_dir}")
    print(f"  • JSONL format: {jsonl_output_dir}")
    print(f"  • Sample examples: {samples_dir}")
    print(f"\nNext step: Configure and run training!")

if __name__ == "__main__":
    main()
