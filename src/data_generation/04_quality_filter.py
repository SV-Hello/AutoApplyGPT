"""
Step 4: Quality validation and auto-filtering
"""
import json
import pandas as pd
from pathlib import Path
import re
from collections import Counter

class QualityValidator:
    """Validate quality of generated training examples"""
    
    def __init__(self):
        self.issues = []
    
    def check_length(self, text, min_len=100, max_len=20000, field_name="text"):
        """Check if text length is reasonable"""
        if not text or len(text) < min_len:
            self.issues.append(f"{field_name} too short ({len(text)} chars)")
            return False
        if len(text) > max_len:
            self.issues.append(f"{field_name} too long ({len(text)} chars)")
            return False
        return True
    
    def check_json_structure(self, example):
        """Check if example has required fields"""
        required = ['original_resume', 'job_title', 'job_description', 
                   'tailored_resume', 'cover_letter']
        missing = [f for f in required if f not in example or not example[f]]
        if missing:
            self.issues.append(f"Missing fields: {missing}")
            return False
        return True
    
    def check_hallucination(self, original, tailored, threshold=0.3):
        """
        Check if tailored resume contains significant new content not in original
        Uses simple word-level checking
        """
        # Extract meaningful words (filter common words)
        def extract_words(text):
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            # Remove very common words
            stopwords = {'that', 'this', 'with', 'have', 'from', 'your', 
                        'they', 'been', 'will', 'their', 'were', 'what',
                        'about', 'would', 'there', 'which', 'when', 'make'}
            return set(w for w in words if w not in stopwords)
        
        original_words = extract_words(original)
        tailored_words = extract_words(tailored)
        
        # Find words in tailored but not in original
        new_words = tailored_words - original_words
        
        # Check if too many new words (potential hallucination)
        if len(new_words) > 0:
            new_ratio = len(new_words) / len(tailored_words) if tailored_words else 0
            if new_ratio > threshold:
                self.issues.append(f"Possible hallucination: {new_ratio*100:.1f}% new words")
                return False
        
        return True
    
    def check_cover_letter_quality(self, cover_letter):
        """Check basic cover letter quality"""
        # Should have paragraphs
        paragraphs = [p.strip() for p in cover_letter.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:
            self.issues.append("Cover letter lacks structure (too few paragraphs)")
            return False
        
        # Should mention job/position
        job_keywords = ['position', 'role', 'opportunity', 'job', 'team']
        if not any(kw in cover_letter.lower() for kw in job_keywords):
            self.issues.append("Cover letter doesn't mention position/role")
            return False
        
        return True
    
    def validate(self, example):
        """Run all validation checks"""
        self.issues = []
        
        # Structure check
        if not self.check_json_structure(example):
            return False, self.issues
        
        # Length checks
        if not self.check_length(example['original_resume'], 200, 20000, "Original resume"):
            return False, self.issues
        if not self.check_length(example['tailored_resume'], 200, 20000, "Tailored resume"):
            return False, self.issues
        if not self.check_length(example['cover_letter'], 200, 3000, "Cover letter"):
            return False, self.issues
        if not self.check_length(example['job_description'], 100, 15000, "Job description"):
            return False, self.issues
        
        # Content quality checks
        if not self.check_hallucination(example['original_resume'], example['tailored_resume']):
            return False, self.issues
        
        if not self.check_cover_letter_quality(example['cover_letter']):
            return False, self.issues
        
        return True, []

def analyze_and_filter(input_path, output_path, report_path):
    """
    Analyze quality and filter training data
    """
    print("📂 Loading training data...")
    with open(input_path, 'r') as f:
        examples = [json.loads(line) for line in f]
    
    print(f"Loaded {len(examples)} examples")
    
    validator = QualityValidator()
    valid_examples = []
    invalid_examples = []
    
    print("\n🔍 Validating examples...")
    for idx, example in enumerate(examples):
        if idx % 100 == 0:
            print(f"Progress: {idx}/{len(examples)} ({idx/len(examples)*100:.1f}%)")
        
        is_valid, issues = validator.validate(example)
        
        if is_valid:
            valid_examples.append(example)
        else:
            invalid_examples.append({
                'index': idx,
                'issues': issues,
                'example': example
            })
    
    # Statistics
    total = len(examples)
    valid_count = len(valid_examples)
    invalid_count = len(invalid_examples)
    
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"✅ Valid: {valid_count}/{total} ({valid_count/total*100:.1f}%)")
    print(f"❌ Invalid: {invalid_count}/{total} ({invalid_count/total*100:.1f}%)")
    
    # Analyze failure reasons
    if invalid_examples:
        print(f"\n{'='*60}")
        print("TOP FAILURE REASONS")
        print(f"{'='*60}")
        all_issues = []
        for inv in invalid_examples:
            all_issues.extend(inv['issues'])
        issue_counts = Counter(all_issues)
        for issue, count in issue_counts.most_common(10):
            print(f"• {issue}: {count}")
    
    # Save valid examples
    print(f"\n💾 Saving {len(valid_examples)} valid examples to {output_path}")
    with open(output_path, 'w') as f:
        for example in valid_examples:
            f.write(json.dumps(example) + '\n')
    
    # Save report
    report = {
        'total_examples': total,
        'valid_count': valid_count,
        'invalid_count': invalid_count,
        'valid_percentage': valid_count / total * 100,
        'failure_reasons': dict(Counter(sum([inv['issues'] for inv in invalid_examples], []))),
        'invalid_examples': invalid_examples[:10]  # Save first 10 for review
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Saved quality report to {report_path}")
    
    # Print statistics
    if valid_examples:
        print(f"\n{'='*60}")
        print("DATASET STATISTICS (Valid Examples)")
        print(f"{'='*60}")
        
        resume_lens = [len(ex['tailored_resume']) for ex in valid_examples]
        cover_lens = [len(ex['cover_letter']) for ex in valid_examples]
        
        print(f"Tailored Resume Length:")
        print(f"  • Mean: {sum(resume_lens)/len(resume_lens):.0f} chars")
        print(f"  • Min: {min(resume_lens)} chars")
        print(f"  • Max: {max(resume_lens)} chars")
        
        print(f"\nCover Letter Length:")
        print(f"  • Mean: {sum(cover_lens)/len(cover_lens):.0f} chars")
        print(f"  • Min: {min(cover_lens)} chars")
        print(f"  • Max: {max(cover_lens)} chars")
    
    print(f"\n{'='*60}")
    print("✅ QUALITY FILTERING COMPLETE")
    print(f"{'='*60}")
    
    return valid_examples, invalid_examples

def main():
    data_dir = Path('../../data/processed')
    input_path = data_dir / 'training_data.jsonl'
    output_path = data_dir / 'training_data_filtered.jsonl'
    report_path = data_dir / 'quality_report.json'
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        print("Run 03_generate_with_gemini.py first!")
        return
    
    valid_examples, invalid_examples = analyze_and_filter(
        input_path, output_path, report_path
    )
    
    print(f"\n✨ Done! {len(valid_examples)} valid examples ready for training.")

if __name__ == "__main__":
    main()
