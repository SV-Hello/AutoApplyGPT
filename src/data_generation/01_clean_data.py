"""
Step 1: Clean and filter resumes and job postings for Tech/Engineering
"""
import pandas as pd
import re
from pathlib import Path

# Tech/Engineering keywords for filtering
TECH_KEYWORDS = [
    'software', 'engineer', 'developer', 'programming', 'python', 'java', 
    'javascript', 'machine learning', 'ai', 'computer science',
    'backend', 'frontend', 'fullstack', 'devops', 'cloud', 'aws', 'azure',
    'algorithm', 'database', 'sql', 'api', 'ios', 'android',
    'react', 'node', 'vue', 'docker', 'kubernetes', 'git',
    'agile', 'scrum', 'ci/cd', 'microservices', 'system design', 'architecture', 'embedded', 'firmware',
    'robotics', 'automation', 'ml', 'deep learning', 'nlp', 'computer vision',
    'data science', 'big data', 'hadoop', 'spark', 'tensorflow',
    'pytorch', 'scala', 'rust', 'c++', 'c#', '.net', 'ruby'
]

def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove HTML entities
    text = re.sub(r'&[a-z]+;', ' ', text)
    return text.strip()

def is_tech_related(text, keywords=TECH_KEYWORDS, threshold=8):
    """Check if text contains tech-related keywords"""
    if not text:
        return False
    text_lower = text.lower()

    #for kw in keywords:
    #    if kw in text_lower:
    #        print(kw)
    # print()
    matches = sum(1 for kw in keywords if kw in text_lower)
    return matches >= threshold

def clean_resumes(resume_path):
    """Clean and filter resumes for tech/engineering"""
    print("Loading resumes...")
    df = pd.read_csv(resume_path, low_memory=False)
    
    print(f"Original resume count: {len(df)}")
    
    # Keep only relevant columns
    df = df[['ID', 'Resume_str', 'Category']].copy()
    
    # Clean text
    df['Resume_str'] = df['Resume_str'].apply(clean_text)
    
    # Filter out empty resumes
    df = df[df['Resume_str'].str.len() > 100].copy()
    print(f"After removing short resumes: {len(df)}")
    
    # Filter for tech-related resumes
    df['is_tech'] = df['Resume_str'].apply(lambda x: is_tech_related(x))
    df_tech = df[df['is_tech']].copy()
    print(f"Tech-related resumes: {len(df_tech)}")
    
    # Remove duplicates
    df_tech = df_tech.drop_duplicates(subset=['Resume_str'])
    print(f"After deduplication: {len(df_tech)}")
    
    return df_tech[['ID', 'Resume_str']]

def clean_job_postings(postings_path):
    """Clean and filter job postings for tech/engineering"""
    print("\nLoading job postings...")
    df = pd.read_csv(postings_path, low_memory=False)
    
    print(f"Original job posting count: {len(df)}")
    
    # Keep only relevant columns
    cols_to_keep = ['job_id', 'title', 'description', 'company_name', 
                    'location', 'skills_desc', 'formatted_experience_level']
    df = df[cols_to_keep].copy()
    
    # Clean text fields
    df['title'] = df['title'].apply(clean_text)
    df['description'] = df['description'].apply(clean_text)
    df['skills_desc'] = df['skills_desc'].apply(clean_text)
    
    # Combine relevant fields for filtering
    df['combined_text'] = (
        df['title'].fillna('') + ' ' + 
        df['description'].fillna('') + ' ' + 
        df['skills_desc'].fillna('')
    )
    
    # Filter out jobs with insufficient description
    df = df[df['description'].str.len() > 100].copy()
    print(f"After removing short descriptions: {len(df)}")
    
    # Filter for tech-related jobs
    df['is_tech'] = df['combined_text'].apply(lambda x: is_tech_related(x, threshold=3))
    df_tech = df[df['is_tech']].copy()
    print(f"Tech-related jobs: {len(df_tech)}")
    
    # Remove duplicates
    df_tech = df_tech.drop_duplicates(subset=['description'])
    print(f"After deduplication: {len(df_tech)}")
    
    # Drop the helper columns
    df_tech = df_tech.drop(['combined_text', 'is_tech'], axis=1)
    
    return df_tech

def main():
    # Paths
    resume_path = '../data/resumes/Resume/Resume.csv'
    postings_path = '../data/jobs/postings.csv'
    output_dir = Path('../data/processed')
    
    # Clean resumes
    clean_resumes_df = clean_resumes(resume_path)
    resume_output = output_dir / 'clean_resumes_tech.csv'
    clean_resumes_df.to_csv(resume_output, index=False)
    print(f"\nSaved {len(clean_resumes_df)} clean tech resumes to {resume_output}")
    
    # Clean job postings
    clean_jobs_df = clean_job_postings(postings_path)
    jobs_output = output_dir / 'clean_jobs_tech.csv'
    clean_jobs_df.to_csv(jobs_output, index=False)
    print(f"Saved {len(clean_jobs_df)} clean tech jobs to {jobs_output}")
    
    print("\n" + "="*60)
    print("DATA CLEANING COMPLETE")
    print(f"Tech Resumes: {len(clean_resumes_df)}")
    print(f"Tech Jobs: {len(clean_jobs_df)}")
    print(f"Potential pairs: {len(clean_resumes_df) * len(clean_jobs_df):,}")
    print("="*60)

if __name__ == "__main__":
    main()
