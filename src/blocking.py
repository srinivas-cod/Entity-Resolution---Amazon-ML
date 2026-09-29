import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import time

def clean_text(text):
    if pd.isna(text) or not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_first_token(text):
    tokens = text.split()
    return tokens[0] if tokens else ""

def tfidf_blocking_single_country(s1_df, s23_df, top_k=5, min_sim=0.3):
    """
    Computes top-K TF-IDF character 3-gram matches between S1 and S2/S3 for a single country.
    """
    if s1_df.empty or s23_df.empty:
        return pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id'])

    # Fit TF-IDF on character 3-grams
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), min_df=1)
    
    # Fit vectorizer on combined corpus for consistent vocabulary
    all_names = pd.concat([s1_df['clean_name'], s23_df['clean_name']])
    vectorizer.fit(all_names)

    s1_vecs = vectorizer.transform(s1_df['clean_name'])
    s23_vecs = vectorizer.transform(s23_df['clean_name'])

    # Compute sparse cosine similarity matrix: (S1_count x S23_count)
    sim_matrix = s1_vecs.dot(s23_vecs.T)

    candidate_pairs = []
    s1_ids = s1_df['entity_id'].values
    s23_ids = s23_df['entity_id'].values

    # Retrieve top K candidates per S1 row
    for i in range(sim_matrix.shape[0]):
        row = sim_matrix.getrow(i)
        if row.nnz == 0:
            continue
        
        # Sort scores in descending order
        col_indices = row.indices
        data_scores = row.data
        
        if len(data_scores) > top_k:
            top_args = np.argpartition(data_scores, -top_k)[-top_k:]
            sorted_top = top_args[np.argsort(-data_scores[top_args])]
        else:
            sorted_top = np.argsort(-data_scores)

        for idx in sorted_top:
            score = data_scores[idx]
            if score >= min_sim:
                candidate_pairs.append((s1_ids[i], s23_ids[idx]))

    return pd.DataFrame(candidate_pairs, columns=['source1_entity_id', 'candidate_entity_id'])

def run_multi_pass_blocking(s1_df, s2_df, s3_df, top_k=5, min_sim=0.35):
    """
    Combines S2 and S3, normalizes fields, and executes country-partitioned blocking.
    """
    start_time = time.time()
    print("Preparing datasets for blocking...")
    
    # Normalize text
    s1_df['clean_name'] = s1_df['business_name'].apply(clean_text)
    s2_df['clean_name'] = s2_df['business_name'].apply(clean_text)
    s3_df['clean_name'] = s3_df['business_name'].apply(clean_text)
    
    # Combine S2 and S3
    s23 = pd.concat([
        s2_df[['entity_id', 'clean_name', 'country']],
        s3_df[['entity_id', 'clean_name', 'country']]
    ], ignore_index=True)

    all_candidates = []
    unique_countries = s1_df['country'].unique()
    
    print(f"Processing candidate blocking across {len(unique_countries)} countries...")
    for country in unique_countries:
        s1_sub = s1_df[s1_df['country'] == country].copy()
        s23_sub = s23[s23['country'] == country].copy()
        
        if s23_sub.empty:
            continue
            
        pairs = tfidf_blocking_single_country(s1_sub, s23_sub, top_k=top_k, min_sim=min_sim)
        all_candidates.append(pairs)
        
    candidate_df = pd.concat(all_candidates, ignore_index=True).drop_duplicates()
    
    print(f"Blocking complete in {time.time() - start_time:.2f} seconds.")
    print(f"Total Candidate Pairs Generated: {len(candidate_df):,}")
    return candidate_df

if __name__ == "__main__":
    # Test on 20,000 S1 records and 100,000 S2/S3 records
    s1_sample = pd.read_csv('train/train_source1.tsv', sep='\t', nrows=20000)
    s2_sample = pd.read_csv('train/train_source2.tsv', sep='\t', nrows=50000)
    s3_sample = pd.read_csv('train/train_source3.tsv', sep='\t', nrows=50000)
    
    candidates = run_multi_pass_blocking(s1_sample, s2_sample, s3_sample, top_k=5, min_sim=0.30)
    print("\nSample Generated Candidate Pairs:")
    print(candidates.head())