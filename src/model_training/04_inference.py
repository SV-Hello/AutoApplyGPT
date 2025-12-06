"""
Step 4: Inference script for testing the trained model
"""
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class AutoApplyModel:
    """Wrapper for the trained AutoApplyGPT model"""
    
    def __init__(self, base_model_name, lora_weights_path):
        """
        Initialize model with LoRA weights
        
        Args:
            base_model_name: Base Phi-3 model name or path
            lora_weights_path: Path to trained LoRA weights
        """
        print(f"🔄 Loading model...")
        print(f"  Base model: {base_model_name}")
        print(f"  LoRA weights: {lora_weights_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(
            self.model,
            lora_weights_path,
            device_map="auto"
        )
        
        # Merge LoRA weights for faster inference (optional)
        # self.model = self.model.merge_and_unload()
        
        self.model.eval()
        
        print(f"✅ Model loaded successfully")
    
    def create_prompt(self, resume, job_title, job_description, job_skills=""):
        """Create the input prompt"""
        prompt = f"""<|user|>
You are an expert resume writer. Tailor this resume for the job posting and write a cover letter.

RESUME:
{resume}

JOB POSTING:
Title: {job_title}
Description: {job_description}
Skills: {job_skills if job_skills else 'Not specified'}

Generate a tailored resume and cover letter for this application.<|end|>
<|assistant|>
"""
        return prompt
    
    def generate(
        self,
        resume,
        job_title,
        job_description,
        job_skills="",
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    ):
        """
        Generate tailored resume and cover letter
        
        Args:
            resume: Original resume text
            job_title: Job title
            job_description: Job description
            job_skills: Required skills (optional)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
        
        Returns:
            Generated text (tailored resume + cover letter)
        """
        # Create prompt
        prompt = self.create_prompt(resume, job_title, job_description, job_skills)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text
    
    def parse_output(self, generated_text):
        """
        Parse the generated text into tailored resume and cover letter
        
        Returns:
            tuple: (tailored_resume, cover_letter)
        """
        # Look for sections
        text = generated_text.strip()
        
        # Try to split by "COVER LETTER:" marker
        if "COVER LETTER:" in text:
            parts = text.split("COVER LETTER:")
            resume_part = parts[0].replace("TAILORED RESUME:", "").strip()
            cover_letter_part = parts[1].strip()
            return resume_part, cover_letter_part
        
        # If no clear split, return as is
        return text, ""

def interactive_demo(model_path):
    """Interactive demo for testing the model"""
    print("="*70)
    print("AutoApplyGPT - Interactive Demo")
    print("="*70)
    
    # Load model
    base_model = "microsoft/Phi-3-mini-4k-instruct"
    model = AutoApplyModel(base_model, model_path)
    
    print("\n📝 Enter job and resume details (or 'quit' to exit):\n")
    
    while True:
        try:
            # Get inputs
            print("\n" + "-"*70)
            resume = input("Resume (or path to file): ").strip()
            
            if resume.lower() == 'quit':
                break
            
            # Check if it's a file path
            if Path(resume).exists():
                with open(resume, 'r') as f:
                    resume = f.read()
            
            job_title = input("Job Title: ").strip()
            job_description = input("Job Description: ").strip()
            job_skills = input("Required Skills (optional): ").strip()
            
            # Generate
            print("\n🔄 Generating tailored resume and cover letter...")
            output = model.generate(
                resume=resume,
                job_title=job_title,
                job_description=job_description,
                job_skills=job_skills
            )
            
            # Parse
            tailored_resume, cover_letter = model.parse_output(output)
            
            # Display
            print("\n" + "="*70)
            print("TAILORED RESUME:")
            print("="*70)
            print(tailored_resume)
            
            print("\n" + "="*70)
            print("COVER LETTER:")
            print("="*70)
            print(cover_letter)
            
            # Ask to save
            save = input("\n💾 Save to file? (y/n): ").strip().lower()
            if save == 'y':
                output_dir = Path("./inference_outputs")
                output_dir.mkdir(exist_ok=True)
                
                import time
                timestamp = int(time.time())
                
                # Save resume
                resume_path = output_dir / f"tailored_resume_{timestamp}.txt"
                with open(resume_path, 'w') as f:
                    f.write(tailored_resume)
                
                # Save cover letter
                cover_path = output_dir / f"cover_letter_{timestamp}.txt"
                with open(cover_path, 'w') as f:
                    f.write(cover_letter)
                
                print(f"✅ Saved to:")
                print(f"  - {resume_path}")
                print(f"  - {cover_path}")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue
    
    print("\nGoodbye! 👋")

def batch_inference(model_path, test_data_path, output_path):
    """Run inference on a batch of test examples"""
    import json
    from tqdm import tqdm
    
    print("="*70)
    print("AutoApplyGPT - Batch Inference")
    print("="*70)
    
    # Load model
    base_model = "microsoft/Phi-3-mini-4k-instruct"
    model = AutoApplyModel(base_model, model_path)
    
    # Load test data
    print(f"\n📂 Loading test data from {test_data_path}")
    with open(test_data_path, 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(test_data)} test examples")
    
    # Run inference
    results = []
    print(f"\n🔄 Running inference...")
    
    for example in tqdm(test_data):
        try:
            # Extract inputs from the example format
            # Assuming the test data has 'instruction' field with the prompt
            instruction = example.get('instruction', '')
            
            # Parse the instruction to extract resume and job details
            # (This is a simplified parser - adjust based on your data format)
            lines = instruction.split('\n')
            resume = ""
            job_title = ""
            job_description = ""
            job_skills = ""
            
            current_section = None
            for line in lines:
                if 'RESUME:' in line:
                    current_section = 'resume'
                elif 'Title:' in line:
                    job_title = line.split('Title:')[1].strip()
                elif 'Description:' in line:
                    current_section = 'description'
                    job_description = line.split('Description:')[1].strip()
                elif 'Skills:' in line:
                    job_skills = line.split('Skills:')[1].strip()
                elif current_section == 'resume':
                    resume += line + '\n'
                elif current_section == 'description':
                    job_description += line + ' '
            
            # Generate
            output = model.generate(
                resume=resume.strip(),
                job_title=job_title,
                job_description=job_description.strip(),
                job_skills=job_skills
            )
            
            # Parse
            tailored_resume, cover_letter = model.parse_output(output)
            
            # Store result
            result = {
                'resume_id': example.get('metadata', {}).get('resume_id'),
                'job_id': example.get('metadata', {}).get('job_id'),
                'original_resume': resume.strip(),
                'job_title': job_title,
                'tailored_resume': tailored_resume,
                'cover_letter': cover_letter,
                'ground_truth_resume': example.get('output', '').split('COVER LETTER:')[0].replace('TAILORED RESUME:', '').strip() if 'COVER LETTER:' in example.get('output', '') else '',
                'ground_truth_cover_letter': example.get('output', '').split('COVER LETTER:')[1].strip() if 'COVER LETTER:' in example.get('output', '') else ''
            }
            
            results.append(result)
        
        except Exception as e:
            print(f"Error processing example: {e}")
            continue
    
    # Save results
    print(f"\n💾 Saving results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved {len(results)} results")
    
    return results

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="../../data/autoapply_training/outputs/final_model",  # Fixed: point to final_model subdirectory
        help="Path to trained LoRA model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="../../model",  # Add this: path to base Phi-3 model
        help="Path to base Phi-3 model"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['interactive', 'batch'],
        default='interactive',
        help="Inference mode"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="../../data/autoapply_training/hf_dataset/test",  # Fixed: point to test split
        help="Path to test data (for batch mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../../data/autoapply_training/outputs/inference_results.json",  # Fixed: consistent path
        help="Output path (for batch mode)"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,  # Add this: how many examples to run
        help="Number of examples to run (for batch mode)"
    )

    args = parser.parse_args()
    
    if args.mode == 'interactive':
        interactive_demo(args.model_path)
    else:
        batch_inference(args.model_path, args.test_data, args.output)

if __name__ == "__main__":
    main()
