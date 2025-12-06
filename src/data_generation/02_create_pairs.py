"""
Step 2: Create smart resume-job pairs using keyword matching (TF-IDF similarity)
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import random

random.seed(42)
np.random.seed(42)

def create_pairs(resumes_df, jobs_df, target_pairs=2000, pairs_per_resume=2):
    """
    Create smart pairs using TF-IDF similarity
    
    Args:
        resumes_df: DataFrame with cleaned resumes
        jobs_df: DataFrame with cleaned jobs
        target_pairs: Target number of pairs to generate
        pairs_per_resume: How many different jobs to pair with each resume
    """
    print("Computing TF-IDF vectors...")
    
    # Prepare texts
    resume_texts = resumes_df['Resume_str'].tolist()
    job_texts = (jobs_df['title'].fillna('') + ' ' + 
                 jobs_df['description'].fillna('') + ' ' + 
                 jobs_df['skills_desc'].fillna('')).tolist()
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    
    # Fit on both resumes and jobs
    all_texts = resume_texts + job_texts
    vectorizer.fit(all_texts)
    
    # Transform
    resume_vectors = vectorizer.transform(resume_texts)
    job_vectors = vectorizer.transform(job_texts)
    
    print(f"Resume vectors: {resume_vectors.shape}")
    print(f"Job vectors: {job_vectors.shape}")
    
    # Calculate similarity for each resume against all jobs
    print("\nComputing similarity scores...")
    similarities = cosine_similarity(resume_vectors, job_vectors)
    print(f"Similarity matrix: {similarities.shape}")
    
    # Create pairs
    pairs = []
    
    print(f"\nCreating {pairs_per_resume} pairs per resume...")
    for i, resume_idx in enumerate(range(len(resumes_df))):
        if i % 100 == 0:
            print(f"Processing resume {i}/{len(resumes_df)}...")
        
        # Get similarity scores for this resume
        sim_scores = similarities[resume_idx]
        
        # Get top N most similar jobs
        top_job_indices = np.argsort(sim_scores)[::-1][:pairs_per_resume * 2]
        
        # Sample from top matches (add some randomness)
        selected_indices = random.sample(list(top_job_indices[:pairs_per_resume * 3]), 
                                        min(pairs_per_resume, len(top_job_indices)))
        
        for job_idx in selected_indices:
            pairs.append({
                'resume_id': resumes_df.iloc[resume_idx]['ID'],
                'job_id': jobs_df.iloc[job_idx]['job_id'],
                'similarity_score': sim_scores[job_idx]
            })
            
            if len(pairs) >= target_pairs:
                break
        
        if len(pairs) >= target_pairs:
            break
    
    pairs_df = pd.DataFrame(pairs)
    
    print(f"\n{'='*60}")
    print(f"PAIRING COMPLETE")
    print(f"Total pairs created: {len(pairs_df)}")
    print(f"Avg similarity score: {pairs_df['similarity_score'].mean():.3f}")
    print(f"Min similarity score: {pairs_df['similarity_score'].min():.3f}")
    print(f"Max similarity score: {pairs_df['similarity_score'].max():.3f}")
    print(f"{'='*60}")
    
    return pairs_df

def create_full_pairs_dataset(pairs_df, resumes_df, jobs_df):
    """Merge pairs with full resume and job data"""
    print("\nMerging pairs with full data...")
    
    # Merge with resumes
    full_df = pairs_df.merge(
        resumes_df[['ID', 'Resume_str']], 
        left_on='resume_id', 
        right_on='ID',
        how='left'
    )
    
    # Merge with jobs
    full_df = full_df.merge(
        jobs_df,
        left_on='job_id',
        right_on='job_id',
        how='left'
    )
    
    # Keep only necessary columns
    final_cols = [
        'resume_id', 'job_id', 'similarity_score',
        'Resume_str', 'title', 'description', 'skills_desc',
        'company_name', 'location', 'formatted_experience_level'
    ]
    
    full_df = full_df[final_cols].copy()
    full_df.columns = [
        'resume_id', 'job_id', 'similarity_score',
        'resume_text', 'job_title', 'job_description', 'job_skills',
        'company_name', 'location', 'experience_level'
    ]
    
    print(f"Final dataset shape: {full_df.shape}")
    print(f"Columns: {full_df.columns.tolist()}")
    
    return full_df

def main():
    # Paths
    data_dir = Path('../data/processed')
    resumes_path = data_dir / 'clean_resumes_tech.csv'
    jobs_path = data_dir / 'clean_jobs_tech.csv'
    
    # Load cleaned data
    print("Loading cleaned data...")
    resumes_df = pd.read_csv(resumes_path)
    jobs_df = pd.read_csv(jobs_path)
    
    print(f"Loaded {len(resumes_df)} resumes and {len(jobs_df)} jobs")
    
    # Create pairs
    pairs_df = create_pairs(
        resumes_df, 
        jobs_df, 
        target_pairs=10000,
        pairs_per_resume=100
    )
    
    # Save pairs index
    pairs_output = data_dir / 'resume_job_pairs.csv'
    pairs_df.to_csv(pairs_output, index=False)
    print(f"\nSaved pairs index to {pairs_output}")
    
    # Create full dataset
    full_pairs_df = create_full_pairs_dataset(pairs_df, resumes_df, jobs_df)
    
    # Save full dataset
    full_output = data_dir / 'full_pairs_dataset.csv'
    full_pairs_df.to_csv(full_output, index=False)
    print(f"Saved full pairs dataset to {full_output}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"Total pairs: {len(full_pairs_df)}")
    print(f"Unique resumes: {full_pairs_df['resume_id'].nunique()}")
    print(f"Unique jobs: {full_pairs_df['job_id'].nunique()}")
    print(f"Avg resume length: {full_pairs_df['resume_text'].str.len().mean():.0f} chars")
    print(f"Avg job desc length: {full_pairs_df['job_description'].str.len().mean():.0f} chars")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
