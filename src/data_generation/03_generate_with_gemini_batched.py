"""
Step 3: Generate training data using Gemini BATCH API with automated batch processing
Handles quota limits by processing in smaller chunks
"""
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime
import os
import sys

from google import genai
from google.genai import types

# Setup logging
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Initialize logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
sys.stdout = Logger(log_file)
sys.stderr = Logger(log_file)

print(f"{'='*70}")
print(f"DATA GENERATION LOG")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log file: {log_file}")
print(f"{'='*70}\n")

def create_batch_jsonl(pairs_df, output_file):
    """
    Create JSONL file for batch API
    Each line: {"key": "resume_X_job_Y", "request": {...}}
    """
    print(f"📝 Creating batch JSONL file...")
    
    with open(output_file, 'w') as f:
        for idx, row in pairs_df.iterrows():
            prompt = f"""You are an expert resume writer and career coach. Your task is to tailor a resume and write a cover letter for a specific job application.

ORIGINAL RESUME:
{row['resume_text']}

JOB POSTING:
Title: {row['job_title']}
Description: {row['job_description']}
Required Skills: {row['job_skills'] if pd.notna(row['job_skills']) else 'Not specified'}

INSTRUCTIONS:
1. Create a TAILORED RESUME that:
   - Emphasizes skills and experiences most relevant to the job
   - Uses keywords from the job description naturally
   - Maintains same general structure as original
   - ONLY includes information from original resume (NO hallucinations)
   - Keeps same level of detail but reorders/rephrases for relevance

2. Write a COVER LETTER that:
   - Is professional and compelling (250-350 words)
   - Connects the candidate's background to this specific role
   - Shows genuine interest in the position and company
   - Uses a professional yet personable tone
   - ONLY references experiences from the resume (NO fabrications)

OUTPUT FORMAT - STRICT JSON ONLY:
{{
    "tailored_resume": "PUT ENTIRE RESUME HERE AS SINGLE LINE WITH \\n FOR NEWLINES",
    "cover_letter": "PUT ENTIRE COVER LETTER HERE AS SINGLE LINE WITH \\n FOR NEWLINES"
}}

CRITICAL RULES:
- Output ONLY the JSON object, nothing else
- Use \\n for line breaks, not actual newlines
- Escape all quotes inside strings with \\"
- No text before or after the JSON
"""
            
            # Create batch request object
            batch_request = {
                "key": f"resume_{row['resume_id']}_job_{row['job_id']}",
                "request": {
                    "contents": [{
                        "parts": [{"text": prompt}],
                        "role": "user"
                    }],
                    "generation_config": {
                        "temperature": 0.7,
                        "max_output_tokens": 8000,
                    }
                }
            }
            
            # Write as single line JSON
            f.write(json.dumps(batch_request) + '\n')
    
    print(f"✅ Created JSONL file: {output_file}")
    print(f"   Total requests: {len(pairs_df)}")
    
    return output_file

def upload_batch_file(client, jsonl_file):
    """Upload JSONL file to Gemini"""
    print(f"\n📤 Uploading batch file...")
    
    # Upload file
    batch_file = client.files.upload(
            file=jsonl_file,
            config=types.UploadFileConfig(display_name='my-batch-requests', mime_type='jsonl')
            )
    
    print(f"✅ File uploaded: {batch_file.name}")
    print(f"   URI: {batch_file.uri}")
    
    return batch_file

def submit_batch_job(client, batch_file, job_name="autoapply-training-data"):
    """Submit batch job using uploaded file"""
    print(f"\n🚀 Submitting batch job...")
    
    batch_job = client.batches.create(
        model="models/gemini-2.5-flash",
        src=batch_file.name,
        config={
            'display_name': job_name,
        }
    )
    
    print(f"✅ Batch job created: {batch_job.name}")
    print(f"📊 Status: {batch_job.state}")
    
    return batch_job


def wait_for_batch_completion(client, batch_job, check_interval=60):
    """Poll batch job until complete"""
    print(f"\n⏳ Waiting for batch job to complete...")
    print(f"   (Checking every {check_interval} seconds)")
    
    completed_states = set([
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    ])

    start_time = time.time()

    while True:

        # Refresh job status
        batch_job = client.batches.get(name=batch_job.name)
        
        elapsed_min = (time.time() - start_time) / 60
        
        print(f"{elapsed_min:.1f} min has passed | Status: {batch_job.state.name}")

        if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
            print(f"\n✅ Batch job completed successfully!")
            return batch_job

        elif batch_job.state.name == 'JOB_STATE_FAILED':
            print(f"\n❌ Batch job failed!")
            return batch_job

        elif batch_job.state.name == 'JOB_STATE_EXPIRED':
            print(f"\n❌ Batch job expired!")
            return batch_job

        time.sleep(check_interval)


def download_and_process_results(client, batch_job, pairs_df, output_path):
    """Download batch results and process into training data"""
    print(f"\n📥 Downloading batch results...")

    # Check if results are in a file or inline
    if batch_job.dest and batch_job.dest.file_name:
        # Results are in a file
        result_file_name = batch_job.dest.file_name
        print(f"📄 Results are in file: {result_file_name}")

        # Download file content
        print(f"📥 Downloading file content...")
        file_content = client.files.download(file=result_file_name)
        output_content = file_content.decode('utf-8')

    elif batch_job.dest and batch_job.dest.inlined_responses:
        # Results are inline (less common for large batches)
        print(f"📄 Results are inline")
        output_content = ""
        for inline_response in batch_job.dest.inlined_responses:
            # Convert inline response to JSONL format
            if inline_response.response:
                response_obj = {
                    "key": getattr(inline_response, 'key', ''),
                    "response": inline_response.response
                }

                output_content += json.dumps(response_obj) + '\n'

    else:
        print("❌ No results found (neither file nor inline).")
        return []

    # Process results
    print(f"🔄 Processing results...")
    results = []
    success_count = 0
    fail_count = 0

    # Create lookup dictionary for pairs data
    pairs_lookup = {
        f"resume_{row['resume_id']}_job_{row['job_id']}": row
        for _, row in pairs_df.iterrows()
    }

    # Parse each line of output
    for line in output_content.strip().split('\n'):
        try:
            response_obj = json.loads(line)

            # Get the key and response
            key = response_obj.get('key', '')
            response = response_obj.get('response', {})

    
            # Get original pair data
            if key not in pairs_lookup:
                print(f"⚠️  Warning: Key {key} not found in pairs data")
                fail_count += 1
                continue

            row = pairs_lookup[key]

            # Extract text from response
            response_text = None

            # Try different response structures
            if isinstance(response, dict):
                if 'candidates' in response and len(response['candidates']) > 0:
                    candidate = response['candidates'][0]

                    if 'content' in candidate and 'parts' in candidate['content']:
                        response_text = candidate['content']['parts'][0]['text']

                elif 'text' in response:
                    response_text = response['text']

            elif hasattr(response, 'text'):
                response_text = response.text

            elif hasattr(response, 'candidates'):
                if len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        response_text = candidate.content.parts[0].text
            

            if not response_text:
                print(f"⚠️  {key}: Could not extract text from response")
                fail_count += 1
                continue

            # Parse JSON from response
            response_text = response_text.strip()

            if response_text.startswith("```json"):
                response_text = response_text[7:]

            if response_text.startswith("```"):
                response_text = response_text[3:]

            if response_text.endswith("```"):
                response_text = response_text[:-3]

            response_text = response_text.strip()

            # Try to parse JSON - if fails, try to extract manually
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: manually extract resume and cover letter
                try:
                    # Find the resume and cover letter sections
                    import re
                    
                    # Try to extract tailored_resume
                    resume_match = re.search(r'"tailored_resume"\s*:\s*"(.*?)"(?:,|\s*})', response_text, re.DOTALL)
                    # Try to extract cover_letter
                    cover_match = re.search(r'"cover_letter"\s*:\s*"(.*?)"(?:\s*})', response_text, re.DOTALL)

                    if not resume_match or not cover_match:
                        # Try alternative: split on the keys
                        if '"tailored_resume"' in response_text and '"cover_letter"' in response_text:
                            parts = response_text.split('"tailored_resume"')
                            if len(parts) > 1:
                                resume_part = parts[1].split('"cover_letter"')[0]
                                cover_part = parts[1].split('"cover_letter"')[1] if '"cover_letter"' in parts[1] else ""

                                # Clean up
                                resume_part = resume_part.strip(' :,"{}')
                                cover_part = cover_part.strip(' :,"{}')

                                if len(resume_part) > 100 and len(cover_part) > 100:
                                    data = {
                                        'tailored_resume': resume_part,
                                        'cover_letter': cover_part
                                    }
                                else:
                                    raise ValueError("Extracted text too short")
                            else:
                                raise ValueError("Cannot split on keys")
                        else:
                            raise ValueError("Keys not found")
                    else:
                        data = {
                            'tailored_resume': resume_match.group(1),
                            'cover_letter': cover_match.group(1)
                        }
                except:
                    # Last resort: just skip this one
                    print(f"⚠️  {key}: Failed all JSON parsing attempts")
                    fail_count += 1
                    continue            

            tailored_resume = data.get('tailored_resume', '')
            cover_letter = data.get('cover_letter', '')

            # Validate lengths
            if len(tailored_resume) < 100 or len(cover_letter) < 100:
                print(f"⚠️  {key}: Output too short (resume: {len(tailored_resume)}, cover: {len(cover_letter)})")
                fail_count += 1
                continue

            # Create result
            result = {
                'resume_id': int(row['resume_id']),
                'job_id': int(row['job_id']) if pd.notna(row['job_id']) else None,
                'similarity_score': float(row['similarity_score']),
                'original_resume': row['resume_text'],
                'job_title': row['job_title'],
                'job_description': row['job_description'],
                'job_skills': row['job_skills'] if pd.notna(row['job_skills']) else '',
                'tailored_resume': tailored_resume,
                'cover_letter': cover_letter,
                'timestamp': datetime.now().isoformat()
            }

            results.append(result)
            success_count += 1

        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error on line: {e}")
            fail_count += 1
        except Exception as e:
            print(f"❌ Error processing result: {e}")
            fail_count += 1

    # Save results
    print(f"\n💾 Saving results to {output_path}")
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {success_count}/{len(pairs_df)} ({success_count/len(pairs_df)*100:.1f}%)")
    print(f"❌ Failed: {fail_count}/{len(pairs_df)} ({fail_count/len(pairs_df)*100:.1f}%)")
    print(f"📁 Saved to: {output_path}")
    print(f"{'='*60}")

    return results


def process_single_batch(client, batch_df, batch_num, temp_dir):
    """Process a single batch and return results"""
    print(f"\n{'='*70}")
    print(f"BATCH {batch_num + 1}")
    print(f"Processing {len(batch_df)} pairs")
    print(f"{'='*70}")
    
    # Check cache
    batch_cache = temp_dir / f'batch_{batch_num:03d}.jsonl'
    if batch_cache.exists():
        print(f"✅ Loading from cache...")
        with open(batch_cache, 'r') as f:
            return [json.loads(line) for line in f]
    
    # Create JSONL for this batch
    batch_jsonl = temp_dir / f'batch_{batch_num:03d}_requests.jsonl'
    create_batch_jsonl(batch_df, batch_jsonl)
    
    # Upload and submit
    batch_file = upload_batch_file(client, batch_jsonl)
    batch_job = submit_batch_job(client, batch_file, job_name=f'batch-{batch_num}')
    
    # Wait for completion
    batch_job = wait_for_batch_completion(client, batch_job, check_interval=30)
    
    if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
        print(f"❌ Batch {batch_num + 1} failed: {batch_job.state.name}")
        return []
    
    # Download and process
    results = download_and_process_results(client, batch_job, batch_df, batch_cache)
    
    print(f"✅ Batch {batch_num + 1} complete: {len(results)} examples")
    return results


def main():
    os.environ["GEMINI_API_KEY"] = "AIzaSyAGfRwrgKYONvFhD-2m4EORRT8H92SJNns"

    # Paths
    data_dir = Path('../../data/processed')
    pairs_path = data_dir / 'full_pairs_dataset.csv'
    output_path = data_dir / 'training_data.jsonl'
    temp_dir = data_dir / 'temp_batches'
    temp_dir.mkdir(exist_ok=True)
    
    # Load pairs
    print("📂 Loading resume-job pairs...")
    pairs_df = pd.read_csv(pairs_path)
    #print(f"{len(pairs_df)}")

    # TEST MODE: Uncomment to test with first 50
    #pairs_df = pairs_df.head(20000)
    #print(f"⚠️  TEST MODE: Using only {len(pairs_df)} pairs")

    total_pairs = len(pairs_df)
    print(f"Loaded {total_pairs} pairs")
    
    # Batch configuration
    BATCH_SIZE = 100  # Adjust if needed: 500, 300, or 200
    num_batches = (total_pairs + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\n📦 Processing in {num_batches} batches of {BATCH_SIZE} pairs each")
    print(f"⚠️  If quota errors occur, reduce BATCH_SIZE in the script (line 360)")
    
    # Initialize client
    client = genai.Client()
    
    # Process each batch
    all_results = []
    for batch_num in range(num_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_pairs)
        batch_df = pairs_df.iloc[start_idx:end_idx].copy()
        
        try:
            results = process_single_batch(client, batch_df, batch_num, temp_dir)
            all_results.extend(results)
            
            # Progress update
            print(f"\n📊 Overall Progress: {len(all_results)}/{total_pairs} pairs processed")
            
        except Exception as e:
            print(f"\n❌ Batch {batch_num + 1} error: {e}")
            print(f"💾 Progress saved. Resume by running script again.")
            break
    
    # Save final results
    print(f"\n{'='*70}")
    print("SAVING FINAL RESULTS")
    print(f"{'='*70}")
    
    with open(output_path, 'w') as f:
        for result in all_results:
            f.write(json.dumps(result) + '\n')
    
    print(f"✅ Saved {len(all_results)}/{total_pairs} examples to {output_path}")
    print(f"\n✨ All done! Generated {len(all_results)} training examples.")
    print(f"\nNext step: Run quality filter")
    print(f"  python scripts/04_quality_filter.py")

if __name__ == "__main__":
    main()
