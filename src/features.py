import pandas as pd
import numpy as np
import re
from rapidfuzz import distance, process, fuzz

def token_jaccard(str1, str2):
    """Calculates Jaccard similarity between word sets of two strings."""
    if not str1 or not str2:
        return 0.0
    set1 = set(str1.split())
    set2 = set(str2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def build_pairwise_features(candidates_df, s1_df, s2_df, s3_df):
    """
    Given candidate pairs (source1_entity_id, candidate_entity_id),
    joins full record details and builds numerical pairwise similarity features.
    """
    print(f"Building features for {len(candidates_df):,} candidate pairs...")
    
    # Pre-clean strings
    s1 = s1_df.copy()
    s2 = s2_df.copy()
    s3 = s3_df.copy()
    
    s1['clean_name'] = s1['business_name'].fillna('').astype(str).str.lower().str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    s2['clean_name'] = s2['business_name'].fillna('').astype(str).str.lower().str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    s3['clean_name'] = s3['business_name'].fillna('').astype(str).str.lower().str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    
    s1['clean_addr'] = s1['business_address'].fillna('').astype(str).str.lower().str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    s2['clean_addr'] = s2['business_address'].fillna('').astype(str).str.lower().str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    s3['clean_addr'] = s3['business_address'].fillna('').astype(str).str.lower().str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    
    # Combine S2 and S3 into single lookup dataframe
    s23 = pd.concat([
        s2[['entity_id', 'clean_name', 'clean_addr', 'country']],
        s3[['entity_id', 'clean_name', 'clean_addr', 'country']]
    ], ignore_index=True)
    
    # Merge S1 and S23 data onto candidate pairs
    df = candidates_df.merge(s1[['entity_id', 'clean_name', 'clean_addr', 'country']], 
                             left_on='source1_entity_id', right_on='entity_id', how='left')
    df = df.rename(columns={'clean_name': 's1_name', 'clean_addr': 's1_addr', 'country': 's1_country'}).drop(columns=['entity_id'])
    
    df = df.merge(s23, left_on='candidate_entity_id', right_on='entity_id', how='left')
    df = df.rename(columns={'clean_name': 's23_name', 'clean_addr': 's23_addr', 'country': 's23_country'}).drop(columns=['entity_id'])
    
    # Fill missing values
    df['s1_name'] = df['s1_name'].fillna('')
    df['s23_name'] = df['s23_name'].fillna('')
    df['s1_addr'] = df['s1_addr'].fillna('')
    df['s23_addr'] = df['s23_addr'].fillna('')
    
    # Feature 1: Name Exact Match
    df['feat_name_exact'] = (df['s1_name'] == df['s23_name']).astype(int)
    
    # Feature 2: Name Token Jaccard
    df['feat_name_jaccard'] = [token_jaccard(a, b) for a, b in zip(df['s1_name'], df['s23_name'])]
    
    # Feature 3: Name Levenshtein Similarity Ratio (0 to 100 scaled to 0-1)
    df['feat_name_levenshtein'] = [fuzz.ratio(a, b) / 100.0 for a, b in zip(df['s1_name'], df['s23_name'])]
    
    # Feature 4: Name Jaro-Winkler Similarity Ratio
    df['feat_name_jaro'] = [distance.JaroWinkler.similarity(a, b) for a, b in zip(df['s1_name'], df['s23_name'])]
    
    # Feature 5: Name Length Difference
    df['feat_name_len_diff'] = np.abs(df['s1_name'].str.len() - df['s23_name'].str.len())
    
    # Feature 6: Address Token Jaccard
    df['feat_addr_jaccard'] = [token_jaccard(a, b) for a, b in zip(df['s1_addr'], df['s23_addr'])]
    
    # Feature 7: Address Levenshtein Ratio
    df['feat_addr_levenshtein'] = [fuzz.ratio(a, b) / 100.0 for a, b in zip(df['s1_addr'], df['s23_addr'])]
    
    # Feature 8: Missing Address Flag
    df['feat_addr_is_missing'] = (df['s23_addr'] == '').astype(int)
    
    # Feature 9: Source Identifier Flag (1 if S2, 0 if S3)
    df['feat_is_source2'] = df['candidate_entity_id'].str.startswith('S2').astype(int)
    
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    
    print(f"Features successfully generated! Total Feature Count: {len(feature_cols)}")
    return df[['source1_entity_id', 'candidate_entity_id'] + feature_cols]

if __name__ == "__main__":
    from blocking import run_multi_pass_blocking
    
    # Run test on small subset
    s1 = pd.read_csv('train/train_source1.tsv', sep='\t', nrows=5000)
    s2 = pd.read_csv('train/train_source2.tsv', sep='\t', nrows=20000)
    s3 = pd.read_csv('train/train_source3.tsv', sep='\t', nrows=20000)
    
    candidates = run_multi_pass_blocking(s1, s2, s3, top_k=3, min_sim=0.3)
    feature_matrix = build_pairwise_features(candidates, s1, s2, s3)
    
    print("\nFirst 3 Feature Vectors:")
    print(feature_matrix.head(3).to_string())