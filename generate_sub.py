import pandas as pd
import numpy as np
import re
import os
import time

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_full_test_submission():
    print("==================================================")
    print("      GENERATING DAY 1 LEADERBOARD SUBMISSION     ")
    print("==================================================\n")
    start_time = time.time()
    
    # 1. Load full TEST dataset
    print("Loading test files (test_source1, test_source2, test_source3)...")
    test_s1 = pd.read_csv('test/test_source1.tsv', sep='\t')
    test_s2 = pd.read_csv('test/test_source2.tsv', sep='\t')
    test_s3 = pd.read_csv('test/test_source3.tsv', sep='\t')
    
    print(f"Loaded {len(test_s1):,} S1 records, {len(test_s2):,} S2 records, and {len(test_s3):,} S3 records.")
    
    # 2. Preprocess names
    print("Preprocessing business names...")
    test_s1['clean_name'] = test_s1['business_name'].apply(clean_text)
    test_s2['clean_name'] = test_s2['business_name'].apply(clean_text)
    test_s3['clean_name'] = test_s3['business_name'].apply(clean_text)
    
    # Combine S2 and S3 candidate records
    test_s23 = pd.concat([
        test_s2[['entity_id', 'clean_name', 'country']],
        test_s3[['entity_id', 'clean_name', 'country']]
    ], ignore_index=True)
    
    # 3. Exact Name + Country Baseline Matching
    print("Performing Name + Country matching across full test set...")
    merged = pd.merge(
        test_s1[['entity_id', 'clean_name', 'country']],
        test_s23,
        on=['clean_name', 'country'],
        how='left'
    )
    
    # 4. Generate Candidate Pairs TSV
    print("Generating candidate_pairs.tsv...")
    candidate_df = merged.dropna(subset=['entity_id_y'])[['entity_id_x', 'entity_id_y']].copy()
    candidate_df.columns = ['source1_entity_id', 'candidate_entity_id']
    candidate_df = candidate_df.drop_duplicates()
    
    # 5. Generate Matching Results TSV
    print("Generating matching_results.tsv...")
    match_df = candidate_df.groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(x)).reset_index()
    match_df.columns = ['source1_entity_id', 'matched_entity_ids']
    
    # Ensure ALL test S1 entities are present in matching_results (Singletons get empty string)
    final_results = pd.merge(
        test_s1[['entity_id']], 
        match_df, 
        left_on='entity_id', 
        right_on='source1_entity_id', 
        how='left'
    )
    final_results['source1_entity_id'] = final_results['entity_id']
    final_results['matched_entity_ids'] = final_results['matched_entity_ids'].fillna('')
    final_results = final_results[['source1_entity_id', 'matched_entity_ids']]
    
    # Save output files to required directory
    os.makedirs('output', exist_ok=True)
    candidate_path = 'output/candidate_pairs.tsv'
    matching_path = 'output/matching_results.tsv'
    
    candidate_df.to_csv(candidate_path, sep='\t', index=False)
    final_results.to_csv(matching_path, sep='\t', index=False)
    
    print("\n================ SUBMISSION SUMMARY ================")
    print(f"Time Taken: {time.time() - start_time:.2f} seconds")
    print(f"Candidate Pairs Rows: {len(candidate_df):,}")
    print(f"Matching Results Rows: {len(final_results):,}")
    print(f"Saved: {candidate_path}")
    print(f"Saved: {matching_path}")
    print("====================================================")

if __name__ == "__main__":
    generate_full_test_submission()