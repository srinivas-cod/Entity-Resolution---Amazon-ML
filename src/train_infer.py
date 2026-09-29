import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, fbeta_score
import time

from blocking import run_multi_pass_blocking
from features import build_pairwise_features

def build_balanced_training_data(gt_path, s1_path, s2_path, s3_path, n_s1=10000):
    print("1. Loading ground truth and S1 subset...")
    s1 = pd.read_csv(s1_path, sep='\t', nrows=n_s1)
    s1_ids = set(s1['entity_id'])
    
    gt = pd.read_csv(gt_path, sep='\t')
    gt = gt[gt['source1_entity_id'].isin(s1_ids)]
    
    # Extract explicit positive pair tuples
    positive_pairs = []
    required_s23_ids = set()
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        matches = str(row['matched_entity_ids'])
        if pd.isna(matches) or matches == 'nan' or not matches:
            continue
        for cand_id in matches.split(','):
            cand_id = cand_id.strip()
            if cand_id:
                positive_pairs.append((s1_id, cand_id))
                required_s23_ids.add(cand_id)
                
    pos_set = set(positive_pairs)
    print(f"Loaded {len(pos_set):,} positive ground-truth pairs.")
    
    print("2. Loading candidate records from S2 and S3...")
    s2_full = pd.read_csv(s2_path, sep='\t')
    s3_full = pd.read_csv(s3_path, sep='\t')
    
    # Include all ground truth targets + random sample for blocking negatives
    s2_sub = pd.concat([
        s2_full[s2_full['entity_id'].isin(required_s23_ids)],
        s2_full.sample(n=min(30000, len(s2_full)), random_state=42)
    ]).drop_duplicates(subset=['entity_id'])
    
    s3_sub = pd.concat([
        s3_full[s3_full['entity_id'].isin(required_s23_ids)],
        s3_full.sample(n=min(30000, len(s3_full)), random_state=42)
    ]).drop_duplicates(subset=['entity_id'])
    
    print("3. Generating candidate pairs via blocking...")
    candidates = run_multi_pass_blocking(s1, s2_sub, s3_sub, top_k=3, min_sim=0.20)
    
    # Ensure all true positives are present in candidates
    pos_df = pd.DataFrame(list(pos_set), columns=['source1_entity_id', 'candidate_entity_id'])
    all_candidates = pd.concat([candidates, pos_df], ignore_index=True).drop_duplicates()
    
    print("4. Extracting similarity features...")
    df_features = build_pairwise_features(all_candidates, s1, s2_sub, s3_sub)
    
    pair_tuples = list(zip(df_features['source1_entity_id'], df_features['candidate_entity_id']))
    df_features['target'] = [1 if pair in pos_set else 0 for pair in pair_tuples]
    
    return df_features

def run_fast_training():
    start = time.time()
    print("==================================================")
    print("    FAST LIGHTGBM TRAINING & THRESHOLD TUNING    ")
    print("==================================================\n")
    
    df_features = build_balanced_training_data(
        gt_path='train/train_ground_truth.tsv',
        s1_path='train/train_source1.tsv',
        s2_path='train/train_source2.tsv',
        s3_path='train/train_source3.tsv',
        n_s1=10000
    )
    
    pos_count = df_features['target'].sum()
    neg_count = (df_features['target'] == 0).sum()
    print(f"\nTraining Set Balance:")
    print(f"  - Positive Pairs (y=1): {pos_count:,}")
    print(f"  - Hard Negatives (y=0): {neg_count:,}")
    
    feature_cols = [col for col in df_features.columns if col.startswith('feat_')]
    X = df_features[feature_cols]
    y = df_features['target']
    
    # Stratified Train/Val Split
    split_idx = int(len(df_features) * 0.75)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print("\nTraining LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=31,
        class_weight='balanced',
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    
    val_probs = model.predict_proba(X_val)[:, 1]
    
    print("\n================ THRESHOLD TUNING (F0.5 METRIC) ================")
    print(f"{'Threshold':<12} | {'Precision':<10} | {'Recall':<10} | {'F0.5 Score':<10}")
    print("-" * 52)
    
    best_thresh, best_fbeta = 0.5, 0.0
    for thresh in np.arange(0.10, 0.96, 0.05):
        y_pred = (val_probs >= thresh).astype(int)
        p = precision_score(y_val, y_pred, zero_division=0)
        r = recall_score(y_val, y_pred, zero_division=0)
        f05 = fbeta_score(y_val, y_pred, beta=0.5, zero_division=0)
        print(f"{thresh:<12.2f} | {p:<10.4f} | {r:<10.4f} | {f05:<10.4f}")
        if f05 > best_fbeta:
            best_fbeta = f05
            best_thresh = thresh
            
    print("-" * 52)
    print(f"Optimal Threshold (tau): {best_thresh:.2f}")
    print(f"Best Validation F0.5 Score: {best_fbeta:.4f}")
    print(f"Total Time: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    run_fast_training()