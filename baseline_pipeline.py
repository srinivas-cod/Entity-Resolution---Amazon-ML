import pandas as pd
import numpy as np
import re
import os

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_baseline_sample():
    print("=== RUNNING BASELINE (SAMPLE RUN) ===")
    
    # Load first 10,000 rows for fast sanity testing
    print("Loading datasets...")
    s1 = pd.read_csv('test/test_source1.tsv', sep='\t', nrows=10000)
    s2 = pd.read_csv('test/test_source2.tsv', sep='\t', nrows=50000)
    s3 = pd.read_csv('test/test_source3.tsv', sep='\t', nrows=50000)
    
    # 1. Normalization
    print("Normalizing names...")
    s1['clean_name'] = s1['business_name'].apply(clean_text)
    s2['clean_name'] = s2['business_name'].apply(clean_text)
    s3['clean_name'] = s3['business_name'].apply(clean_text)
    
    # Combine S2 and S3
    s2_s3 = pd.concat([
        s2[['entity_id', 'clean_name', 'country']],
        s3[['entity_id', 'clean_name', 'country']]
    ], ignore_index=True)
    
    # 2. Exact Name + Country Blocking / Matching
    print("Performing exact match blocking...")
    merged = pd.merge(
        s1[['entity_id', 'clean_name', 'country']],
        s2_s3,
        on=['clean_name', 'country'],
        how='left'
    )
    
    # Create Candidate Pairs
    candidate_df = merged.dropna(subset=['entity_id_y'])[['entity_id_x', 'entity_id_y']].copy()
    candidate_df.columns = ['source1_entity_id', 'candidate_entity_id']
    
    # Aggregate matches for S1 (Baseline logic: exact match = match)
    match_df = candidate_df.groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(x)).reset_index()
    match_df.columns = ['source1_entity_id', 'matched_entity_ids']
    
    # Ensure all S1 entities appear in final results (singletons have empty string or NaN)
    final_results = pd.merge(s1[['entity_id']], match_df, left_on='entity_id', right_on='source1_entity_id', how='left')
    final_results['source1_entity_id'] = final_results['entity_id']
    final_results['matched_entity_ids'] = final_results['matched_entity_ids'].fillna('')
    final_results = final_results[['source1_entity_id', 'matched_entity_ids']]
    
    os.makedirs('output', exist_ok=True)
    candidate_df.to_csv('output/candidate_pairs_sample.tsv', sep='\t', index=False)
    final_results.to_csv('output/matching_results_sample.tsv', sep='\t', index=False)
    
    print("\nSUCCESS! Sample outputs generated:")
    print(f"Candidate Pairs Rows: {len(candidate_df):,}")
    print(f"Matching Results Rows: {len(final_results):,}")
    print("\nSample Matching Results:")
    print(final_results.head(5))

if __name__ == "__main__":
    run_baseline_sample()