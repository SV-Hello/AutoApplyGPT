# AutoApplyGPT: Cost-Effective Resume Tailoring with Small Language Models
Fine-tune a 3.8B parameter model to automatically tailor resumes and generate cover letters at 100-500x lower cost than GPT-4 or Claude.

## The Problem
Job searching tools powered by large language models (GPT-4, Claude) cost $0.20-$0.50 per application. For students and job seekers applying to 100+ positions, this quickly becomes financially unsustainable. This project proves you don't need expensive API calls for this specific task.

## The Solution
A fine-tuned Phi-3-mini (3.8B parameters) model that:
- Tailors resumes to job postings by emphasizing relevant skills
- Generates professional cover letters connecting candidate background to roles
- Runs entirely on local consumer hardware (no API calls, no subscriptions)
- Costs ~$0.001 per application (electricity only)
- Achieves 66% keyword overlap with job requirements

**Cost comparison:**
- This model: $0.001 per application
- Gemini Flash: $0.05-$0.10 per application  
- GPT-4: $0.20-$0.50 per application
- **Savings: 100-500x**

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/yourusername/autoapply-gpt.git
cd autoapply-gpt

# Install dependencies
pip install -r requirements.txt

# Download datasets (see data/README.md for instructions)
# Place Resume.csv in data/resumes/
# Place postings.csv in data/jobs/
```

### 2. Data Generation Pipeline
```bash
# Step 1: Clean and filter data for tech/engineering domain
cd scripts
python 01_clean_data.py
# Output: data/processed/clean_resumes_tech.csv, clean_jobs_tech.csv

# Step 2: Create intelligent resume-job pairs using TF-IDF similarity
python 02_create_pairs.py
# Output: data/processed/full_pairs_dataset.csv (with similarity scores)

# Step 3: Generate training data using Gemini API (batch processing)
python 03_generate_with_gemini_batched.py
# Output: data/processed/training_data.jsonl (~10,000 examples)
# Note: Requires GEMINI_API_KEY environment variable

# Step 4: Quality filter the generated examples
python 04_quality_filter.py
# Output: data/processed/training_data_filtered.jsonl (~7,500 examples)
```

### 3. Model Training Pipeline
```bash
# Step 1: Convert to HuggingFace dataset format
cd ../src/model_training
python 01_prepare_data.py
# Output: data/hf_dataset/ (train/validation/test splits)

# Step 2: Generate training configuration files
python 02_create_configs.py
# Output: configs/all_configs.json

# Step 3: Train the model with LoRA + 4-bit quantization
python 03_train_lora.py \
    --data_dir ../../data/autoapply_training/hf_dataset \
    --config_dir ../../data/autoapply_training/configs \
    --output_dir ../../data/autoapply_training/outputs
# Output: outputs/final_model/ (trained LoRA weights)
# Training time: ~2-3 hours on H100 GPU
```

### 4. Inference & Evaluation
```bash
# Run inference on test examples
python 04_inference.py \
    --model_path ../../data/autoapply_training/outputs/final_model \
    --base_model ../../model \
    --mode batch \
    --test_data ../../data/autoapply_training/hf_dataset/test \
    --num_examples 5
# Output: outputs/inference_results.json

# Evaluate model performance
python 05_evaluate.py \
    --results ../../data/autoapply_training/outputs/inference_results.json \
    --output ../../data/autoapply_training/outputs/evaluation_metrics.json
# Output: Metrics including ROUGE, BLEU, semantic similarity, keyword overlap

# Interactive demo (try it yourself!)
python 04_inference.py \
    --model_path ../../data/autoapply_training/outputs/final_model \
    --base_model ../../model \
    --mode interactive
```

## Project Structure
```
autoapply_pipeline/
├── scripts/
│   ├── 01_clean_data.py              # Filter resumes and jobs for tech domain
│   ├── 02_create_pairs.py            # TF-IDF similarity matching
│   ├── 03_generate_with_gemini.py    # Generate training data with Gemini
│   └── 04_quality_filter.py          # Validate generated examples

├── src/model_training/
│   ├── 01_prepare_data.py            # Convert to HuggingFace format
│   ├── 02_create_configs.py          # Generate training configs
│   ├── 03_train_lora.py              # Train with LoRA + 4-bit quantization
│   ├── 04_inference.py               # Run trained model
│   └── 05_evaluate.py                # Compute evaluation metrics
├── data/                             # Datasets (see data/README.md)
└── docs/                             # Full project report
```

## Key Features
- **Parameter-efficient training:** LoRA fine-tuning with 4-bit quantization (47M trainable parameters, 1.24% of base model)
- **Runs on consumer hardware:** 2.3GB memory footprint, works on GPUs with 16-24GB VRAM
- **No hallucinations:** Model only uses information from original resume
- **Real-world data:** Trained on 7,500 authentic resume-job pairs from Kaggle

## Results
| Metric | Value |
|--------|-------|
| Training time | 2-3 hours (H100 GPU) |
| Keyword overlap | 66% |
| Semantic similarity | 0.55 |
| Cost per application | ~$0.001 |
| Model size | 2.3GB (quantized) |

## Technical Details
- **Base model:** Microsoft Phi-3-mini-4k-instruct (3.8B parameters)
- **Fine-tuning method:** LoRA (rank=16, alpha=32)
- **Quantization:** 4-bit NormalFloat (NF4)
- **Training data:** 7,500 examples (80/10/10 split)
- **Teacher model:** Google Gemini 2.5 Flash


## Limitations
- Currently focused on technology/engineering roles only
- Quality slightly lower than frontier models (acceptable for bulk applications)
- Requires initial setup and training (~$10-60 one-time cost)
- Cover letter generation less sophisticated than GPT-4

## Future Work
- Expand to multiple domains (business, healthcare, education)
- Scale training data to 50,000+ examples
- Integrate with job discovery and application automation
- Real-world validation studies with actual job seekers


## Acknowledgments
- Datasets: Resume Dataset (Anbhawal, 2020) and LinkedIn Job Postings (Koneru, 2023) from Kaggle
- Computing resources: Texas A&M HPRC cluster
- Course: CSCE 689 - Special Topics in Machine Learning

**Disclaimer:** This tool generates application materials based on your existing resume. Always review and customize outputs before submission. The model is designed to assist, not replace, human judgment in the job application process.
