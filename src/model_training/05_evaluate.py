"""
Step 5: Evaluation script with automatic metrics
Compares model outputs against:
1. Gemini (teacher model)
2. Zero-shot Phi-3 (base model)
3. Generic resume (no tailoring)
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch

def compute_rouge(predictions, references):
    """
    Compute ROUGE scores using rouge-score library
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("Installing rouge-score...")
        import subprocess
        subprocess.run(["pip", "install", "rouge-score", "--break-system-packages", "-q"])
        from rouge_score import rouge_scorer
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = defaultdict(list)
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        for metric, value in score.items():
            scores[f"{metric}_precision"].append(value.precision)
            scores[f"{metric}_recall"].append(value.recall)
            scores[f"{metric}_fmeasure"].append(value.fmeasure)
    
    # Average scores
    avg_scores = {k: np.mean(v) for k, v in scores.items()}
    
    return avg_scores

def compute_bleu(predictions, references):
    """
    Compute BLEU scores
    """
    try:
        from sacrebleu import corpus_bleu
    except ImportError:
        print("Installing sacrebleu...")
        import subprocess
        subprocess.run(["pip", "install", "sacrebleu", "--break-system-packages", "-q"])
        from sacrebleu import corpus_bleu
    
    # BLEU expects list of references for each prediction
    references = [[ref] for ref in references]
    
    bleu = corpus_bleu(predictions, references)
    
    return {
        'bleu': bleu.score,
        'bleu_precisions': bleu.precisions
    }

def compute_perplexity(model, tokenizer, texts, device='cuda'):
    """
    Compute perplexity of generated texts using base model
    Lower perplexity = more fluent/natural text
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            # Tokenize
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs, labels=inputs['input_ids'])
            
            loss = outputs.loss
            num_tokens = inputs['input_ids'].numel()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    perplexity = np.exp(total_loss / total_tokens)
    
    return perplexity

def compute_semantic_similarity(predictions, references):
    """
    Compute semantic similarity using sentence embeddings
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("Installing sentence-transformers...")
        import subprocess
        subprocess.run(["pip", "install", "sentence-transformers", "--break-system-packages", "-q"])
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    
    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode
    pred_embeddings = model.encode(predictions)
    ref_embeddings = model.encode(references)
    
    # Compute cosine similarity
    similarities = []
    for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
        sim = cosine_similarity([pred_emb], [ref_emb])[0][0]
        similarities.append(sim)
    
    return {
        'semantic_similarity_mean': np.mean(similarities),
        'semantic_similarity_std': np.std(similarities)
    }

def compute_length_stats(texts):
    """Compute text length statistics"""
    lengths = [len(text) for text in texts]
    
    return {
        'length_mean': np.mean(lengths),
        'length_std': np.std(lengths),
        'length_min': np.min(lengths),
        'length_max': np.max(lengths)
    }

def compute_keyword_overlap(predictions, job_descriptions):
    """
    Compute keyword overlap between tailored resumes and job descriptions
    Higher overlap = better tailoring
    """
    import re
    from collections import Counter
    
    def extract_keywords(text):
        # Simple keyword extraction (words with 4+ letters)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        # Remove common words
        stopwords = {'that', 'this', 'with', 'have', 'from', 'your', 
                    'been', 'will', 'their', 'were', 'what', 'about',
                    'would', 'there', 'which', 'when', 'make', 'more'}
        words = [w for w in words if w not in stopwords]
        return set(words)
    
    overlaps = []
    for pred, job_desc in zip(predictions, job_descriptions):
        pred_keywords = extract_keywords(pred)
        job_keywords = extract_keywords(job_desc)
        
        if len(job_keywords) > 0:
            overlap = len(pred_keywords & job_keywords) / len(job_keywords)
            overlaps.append(overlap)
    
    return {
        'keyword_overlap_mean': np.mean(overlaps),
        'keyword_overlap_std': np.std(overlaps)
    }

def evaluate_model(results_file, ground_truth_file=None):
    """
    Main evaluation function
    
    Args:
        results_file: JSON file with model outputs
        ground_truth_file: Optional ground truth (Gemini outputs) for comparison
    """
    print("="*70)
    print("AutoApplyGPT Model Evaluation")
    print("="*70)
    
    # Load results
    print(f"\n📂 Loading results from {results_file}")
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} examples")
    
    # Extract predictions and references
    predictions_resume = [r['tailored_resume'] for r in results]
    predictions_cover = [r['cover_letter'] for r in results]
    job_descriptions = [r.get('job_title', '') + ' ' + r.get('job_description', '') for r in results]
    
    references_resume = [r.get('ground_truth_resume', '') for r in results]
    references_cover = [r.get('ground_truth_cover_letter', '') for r in results]
    
    # Check if we have ground truth
    has_ground_truth = any(ref for ref in references_resume)
    
    metrics = {}
    
    # 1. ROUGE scores (if ground truth available)
    if has_ground_truth:
        print("\n📊 Computing ROUGE scores...")
        rouge_resume = compute_rouge(predictions_resume, references_resume)
        rouge_cover = compute_rouge(predictions_cover, references_cover)
        
        metrics['rouge_resume'] = rouge_resume
        metrics['rouge_cover_letter'] = rouge_cover
        
        print("  ✅ ROUGE computed")
    
    # 2. BLEU scores (if ground truth available)
    if has_ground_truth:
        print("📊 Computing BLEU scores...")
        bleu_resume = compute_bleu(predictions_resume, references_resume)
        bleu_cover = compute_bleu(predictions_cover, references_cover)
        
        metrics['bleu_resume'] = bleu_resume
        metrics['bleu_cover_letter'] = bleu_cover
        
        print("  ✅ BLEU computed")
    
    # 3. Semantic similarity (if ground truth available)
    if has_ground_truth:
        print("📊 Computing semantic similarity...")
        sim_resume = compute_semantic_similarity(predictions_resume, references_resume)
        sim_cover = compute_semantic_similarity(predictions_cover, references_cover)
        
        metrics['semantic_similarity_resume'] = sim_resume
        metrics['semantic_similarity_cover'] = sim_cover
        
        print("  ✅ Semantic similarity computed")
    
    # 4. Length statistics
    print("📊 Computing length statistics...")
    length_resume = compute_length_stats(predictions_resume)
    length_cover = compute_length_stats(predictions_cover)
    
    metrics['length_stats_resume'] = length_resume
    metrics['length_stats_cover_letter'] = length_cover
    
    print("  ✅ Length stats computed")
    
    # 5. Keyword overlap
    print("📊 Computing keyword overlap...")
    keyword_overlap = compute_keyword_overlap(predictions_resume, job_descriptions)
    
    metrics['keyword_overlap'] = keyword_overlap
    
    print("  ✅ Keyword overlap computed")
    
    # Print results
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    
    if has_ground_truth:
        print("\n📈 ROUGE Scores (Tailored Resume):")
        for metric, value in metrics['rouge_resume'].items():
            if 'fmeasure' in metric:
                print(f"  {metric}: {value:.4f}")
        
        print("\n📈 ROUGE Scores (Cover Letter):")
        for metric, value in metrics['rouge_cover_letter'].items():
            if 'fmeasure' in metric:
                print(f"  {metric}: {value:.4f}")
        
        print(f"\n📈 BLEU Scores:")
        print(f"  Resume BLEU: {metrics['bleu_resume']['bleu']:.2f}")
        print(f"  Cover Letter BLEU: {metrics['bleu_cover_letter']['bleu']:.2f}")
        
        print(f"\n📈 Semantic Similarity:")
        print(f"  Resume: {metrics['semantic_similarity_resume']['semantic_similarity_mean']:.4f}")
        print(f"  Cover Letter: {metrics['semantic_similarity_cover']['semantic_similarity_mean']:.4f}")
    
    print(f"\n📈 Length Statistics:")
    print(f"  Resume (mean): {metrics['length_stats_resume']['length_mean']:.0f} chars")
    print(f"  Cover Letter (mean): {metrics['length_stats_cover_letter']['length_mean']:.0f} chars")
    
    print(f"\n📈 Keyword Overlap:")
    print(f"  Mean: {metrics['keyword_overlap']['keyword_overlap_mean']:.4f}")
    
    return metrics

def compare_models(fine_tuned_results, baseline_results, output_file):
    """
    Compare fine-tuned model against baseline
    
    Args:
        fine_tuned_results: Results from fine-tuned model
        baseline_results: Results from baseline (zero-shot Phi-3 or Gemini)
        output_file: Where to save comparison
    """
    print("\n" + "="*70)
    print("Model Comparison")
    print("="*70)
    
    # Evaluate both
    print("\n📊 Evaluating fine-tuned model...")
    ft_metrics = evaluate_model(fine_tuned_results)
    
    print("\n📊 Evaluating baseline model...")
    baseline_metrics = evaluate_model(baseline_results)
    
    # Compare
    comparison = {
        'fine_tuned': ft_metrics,
        'baseline': baseline_metrics,
        'improvements': {}
    }
    
    # Calculate improvements
    if 'rouge_resume' in ft_metrics and 'rouge_resume' in baseline_metrics:
        for metric in ft_metrics['rouge_resume']:
            if 'fmeasure' in metric:
                ft_val = ft_metrics['rouge_resume'][metric]
                bl_val = baseline_metrics['rouge_resume'][metric]
                improvement = ((ft_val - bl_val) / bl_val) * 100 if bl_val > 0 else 0
                comparison['improvements'][metric] = improvement
    
    # Save comparison
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n💾 Comparison saved to {output_file}")
    
    # Print improvements
    print(f"\n📈 Improvements over baseline:")
    for metric, improvement in comparison['improvements'].items():
        print(f"  {metric}: {improvement:+.2f}%")
    
    return comparison

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to inference results JSON"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        help="Path to ground truth (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./evaluation_results.json",
        help="Output path for metrics"
    )
    parser.add_argument(
        "--compare",
        type=str,
        help="Path to baseline results for comparison"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        # Comparison mode
        comparison = compare_models(
            args.results,
            args.compare,
            args.output
        )
    else:
        # Single model evaluation
        metrics = evaluate_model(args.results, args.ground_truth)
        
        # Save metrics
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n💾 Metrics saved to {args.output}")

if __name__ == "__main__":
    main()
