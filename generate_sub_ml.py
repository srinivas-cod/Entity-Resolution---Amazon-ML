import os
import time
import re
import pandas as pd
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz, distance
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# 1. COLUMN DETECTION & CLEANING UTILITIES
# -------------------------------------------------------------------
def get_col_names(df):
    """Dynamically finds entity_id, name, and address columns regardless of naming."""
    cols = df.columns.tolist()
    
    id_col = next((c for c in cols if 'entity_id' in c or c == 'id'), cols[0])
    name_col = next((c for c in cols if 'name' in c or 'title' in c), cols[1])
    addr_col = next((c for c in cols if 'addr' in c or 'street' in c or 'location' in c), None)
    
    return id_col, name_col, addr_col

def clean_str(s):
    if pd.isna(s) or not s:
        return ""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def make_token_sort_key(s):
    if pd.isna(s) or not s:
        return ""
    tokens = sorted(re.findall(r'[a-z0-9]+', str(s).lower()))
    return "".join(tokens)

def compute_pairwise_features(candidates_df, s1_df, s2_df, s3_df):
    s1_id_col, s1_name_col, s1_addr_col = get_col_names(s1_df)
    s2_id_col, s2_name_col, s2_addr_col = get_col_names(s2_df)
    s3_id_col, s3_name_col, s3_addr_col = get_col_names(s3_df)
    
    # Standardize dataframes
    s1_sub = pd.DataFrame({
        'entity_id': s1_df[s1_id_col],
        'name': s1_df[s1_name_col],
        'address': s1_df[s1_addr_col] if s1_addr_col else ""
    }).set_index('entity_id').to_dict('index')
    
    s2_sub = pd.DataFrame({
        'entity_id': s2_df[s2_id_col],
        'name': s2_df[s2_name_col],
        'address': s2_df[s2_addr_col] if s2_addr_col else ""
    })
    
    s3_sub = pd.DataFrame({
        'entity_id': s3_df[s3_id_col],
        'name': s3_df[s3_name_col],
        'address': s3_df[s3_addr_col] if s3_addr_col else ""
    })
    
    s23_map = pd.concat([s2_sub, s3_sub]).drop_duplicates('entity_id').set_index('entity_id').to_dict('index')
    
    feats = []
    for _, row in candidates_df.iterrows():
        s1_id = row['source1_entity_id']
        c_id = row['candidate_entity_id']
        
        r1 = s1_sub.get(s1_id, {})
        r2 = s23_map.get(c_id, {})
        
        n1, n2 = str(r1.get('name', '')), str(r2.get('name', ''))
        a1, a2 = str(r1.get('address', '')), str(r2.get('address', ''))
        
        feat_name_exact = 1.0 if clean_str(n1) == clean_str(n2) else 0.0
        feat_name_ratio = fuzz.ratio(n1, n2) / 100.0
        feat_name_token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
        feat_name_jaro = distance.JaroWinkler.similarity(n1, n2)
        feat_name_len_diff = abs(len(n1) - len(n2))
        
        feat_addr_ratio = fuzz.ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
        feat_addr_missing = 1.0 if not a1 or not a2 else 0.0
        feat_is_s2 = 1.0 if str(c_id).startswith('S2') else 0.0
        
        feats.append([
            feat_name_exact, feat_name_ratio, feat_name_token_sort,
            feat_name_jaro, feat_name_len_diff, feat_addr_ratio,
            feat_addr_missing, feat_is_s2
        ])
        
    cols = [
        'feat_name_exact', 'feat_name_ratio', 'feat_name_token_sort',
        'feat_name_jaro', 'feat_name_len_diff', 'feat_addr_ratio',
        'feat_addr_missing', 'feat_is_s2'
    ]
    return pd.DataFrame(feats, columns=cols)

# -------------------------------------------------------------------
# 2. MODEL TRAINING (LIGHTGBM)
# -------------------------------------------------------------------
def train_lightgbm():
    print("1/3. Training LightGBM Model on Ground Truth...")
    s1 = pd.read_csv('train/train_source1.tsv', sep='\t', nrows=15000)
    s1_id_col = get_col_names(s1)[0]
    s1_ids = set(s1[s1_id_col])
    
    gt = pd.read_csv('train/train_ground_truth.tsv', sep='\t')
    gt_s1_col = get_col_names(gt)[0]
    gt_match_col = get_col_names(gt)[1]
    
    gt = gt[gt[gt_s1_col].isin(s1_ids)]
    
    pos_pairs = []
    req_s23 = set()
    for _, row in gt.iterrows():
        s1_id = row[gt_s1_col]
        matches = str(row[gt_match_col])
        if pd.isna(matches) or matches == 'nan' or not matches:
            continue
        for cid in matches.split(','):
            cid = cid.strip()
            if cid:
                pos_pairs.append((s1_id, cid))
                req_s23.add(cid)
                
    s2 = pd.read_csv('train/train_source2.tsv', sep='\t')
    s3 = pd.read_csv('train/train_source3.tsv', sep='\t')
    
    s2_id_col = get_col_names(s2)[0]
    s3_id_col = get_col_names(s3)[0]
    
    s2_sub = s2[s2[s2_id_col].isin(req_s23)]
    s3_sub = s3[s3[s3_id_col].isin(req_s23)]
    
    pos_df = pd.DataFrame(pos_pairs, columns=['source1_entity_id', 'candidate_entity_id']).drop_duplicates()
    pos_feats = compute_pairwise_features(pos_df, s1, s2_sub, s3_sub)
    pos_feats['target'] = 1
    
    neg_df = pos_df.copy()
    neg_df['candidate_entity_id'] = np.random.permutation(neg_df['candidate_entity_id'].values)
    neg_feats = compute_pairwise_features(neg_df, s1, s2_sub, s3_sub)
    neg_feats['target'] = 0
    
    train_df = pd.concat([pos_feats, neg_feats], ignore_index=True)
    feat_cols = [c for c in train_df.columns if c.startswith('feat_')]
    
    model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbosity=-1
    )
    model.fit(train_df[feat_cols], train_df['target'])
    print(f"   LightGBM ready! Model trained on {len(train_df):,} samples.")
    return model, feat_cols

# -------------------------------------------------------------------
# 3. TEST INFERENCE & FILE GENERATION
# -------------------------------------------------------------------
def generate_robust_submission():
    start_time = time.time()
    print("==================================================")
    print("    ROBUST LIGHTGBM INFERENCE & SUBMISSION      ")
    print("==================================================\n")
    
    model, feat_cols = train_lightgbm()
    
    print("\n2/3. Pre-indexing Test Datasets with Hash Keys...")
    s1 = pd.read_csv('test/test_source1.tsv', sep='\t')
    s2 = pd.read_csv('test/test_source2.tsv', sep='\t')
    s3 = pd.read_csv('test/test_source3.tsv', sep='\t')
    
    s1_id_col, s1_name_col, _ = get_col_names(s1)
    s2_id_col, s2_name_col, _ = get_col_names(s2)
    s3_id_col, s3_name_col, _ = get_col_names(s3)
    
    s1['key_exact'] = s1[s1_name_col].apply(clean_str)
    s2['key_exact'] = s2[s2_name_col].apply(clean_str)
    s3['key_exact'] = s3[s3_name_col].apply(clean_str)
    
    s1['key_token'] = s1[s1_name_col].apply(make_token_sort_key)
    s2['key_token'] = s2[s2_name_col].apply(make_token_sort_key)
    s3['key_token'] = s3[s3_name_col].apply(make_token_sort_key)
    
    print("   Generating indexed candidate pairs...")
    m2_exact = pd.merge(s1[[s1_id_col, 'key_exact']], s2[[s2_id_col, 'key_exact']], on='key_exact', suffixes=('', '_cand'))
    m2_token = pd.merge(s1[[s1_id_col, 'key_token']], s2[[s2_id_col, 'key_token']], on='key_token', suffixes=('', '_cand'))
    
    m3_exact = pd.merge(s1[[s1_id_col, 'key_exact']], s3[[s3_id_col, 'key_exact']], on='key_exact', suffixes=('', '_cand'))
    m3_token = pd.merge(s1[[s1_id_col, 'key_token']], s3[[s3_id_col, 'key_token']], on='key_token', suffixes=('', '_cand'))
    
    cand_dfs = []
    for m in [m2_exact, m2_token, m3_exact, m3_token]:
        key_col = 'key_exact' if 'key_exact' in m.columns else 'key_token'
        id_cand_col = [c for c in m.columns if '_cand' in c or c.endswith('_y') or c != s1_id_col and c != key_col][0]
        m_filtered = m[m[key_col] != ''][[s1_id_col, id_cand_col]].rename(
            columns={s1_id_col: 'source1_entity_id', id_cand_col: 'candidate_entity_id'}
        )
        cand_dfs.append(m_filtered)
        
    candidates = pd.concat(cand_dfs, ignore_index=True).drop_duplicates()
    print(f"   Candidates generated cleanly: {len(candidates):,} pairs.")
    
    print("\n3/3. Feature Extraction & LightGBM Prediction...")
    if len(candidates) > 0:
        test_feats = compute_pairwise_features(candidates, s1, s2, s3)
        X_test = test_feats[feat_cols]
        candidates['match_prob'] = model.predict_proba(X_test)[:, 1]
        
        # Decision threshold set to tau = 0.70 (strictly > 0.50) to optimize F_0.5 Precision
        DECISION_THRESHOLD = 0.70
        preds = candidates[candidates['match_prob'] >= DECISION_THRESHOLD]
        matches_grouped = preds.groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(x)).reset_index()
        matches_grouped.columns = ['source1_entity_id', 'matched_entity_ids']
    else:
        matches_grouped = pd.DataFrame(columns=['source1_entity_id', 'matched_entity_ids'])
        
    final_results = pd.merge(s1[[s1_id_col]], matches_grouped, left_on=s1_id_col, right_on='source1_entity_id', how='left')
    final_results['source1_entity_id'] = final_results[s1_id_col]
    final_results['matched_entity_ids'] = final_results['matched_entity_ids'].fillna('')
    final_results = final_results[['source1_entity_id', 'matched_entity_ids']]
    
    os.makedirs('output', exist_ok=True)
    cand_out = candidates[['source1_entity_id', 'candidate_entity_id']].drop_duplicates()
    
    cand_out.to_csv('output/candidate_pairs.tsv', sep='\t', index=False)
    final_results.to_csv('output/matching_results.tsv', sep='\t', index=False)
    
    print("\n================ SUBMISSION GENERATED SUCCESSFULLY ================")
    print(f"Total Execution Time: {time.time() - start_time:.2f} seconds")
    print(f"Candidate Pairs Rows: {len(cand_out):,}")
    print(f"Matching Results Rows: {len(final_results):,}")
    print(f"Total Matches Predicted: {(final_results['matched_entity_ids'] != '').sum():,}")
    print(f"Total Singletons (Unmatched): {(final_results['matched_entity_ids'] == '').sum():,}")
    print("Saved -> output/candidate_pairs.tsv")
    print("Saved -> output/matching_results.tsv")
    print("==========================================================")

if __name__ == "__main__":
    generate_robust_submission()