import pandas as pd
import os

def inspect_all_datasets():
    print("==================================================")
    print("          STEP 1: DATASET PROFILING              ")
    print("==================================================\n")
    
    files = {
        'train_s1': 'train/train_source1.tsv',
        'train_s2': 'train/train_source2.tsv',
        'train_s3': 'train/train_source3.tsv',
        'train_gt': 'train/train_ground_truth.tsv',
        'test_s1':  'test/test_source1.tsv',
        'test_s2':  'test/test_source2.tsv',
        'test_s3':  'test/test_source3.tsv',
    }
    
    for key, filepath in files.items():
        if not os.path.exists(filepath):
            print(f"[MISSING] {filepath} not found!")
            continue
            
        print(f"\n--- FILE: {filepath} ---")
        # Load full dataframe
        df = pd.read_csv(filepath, sep='\t')
        
        print(f"Total Rows: {len(df):,}")
        print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
        print("\nMissing Value Count:")
        print(df.isnull().sum())
        
        print("\nFirst 2 Sample Records:")
        print(df.head(2).to_string())
        print("-" * 60)

if __name__ == "__main__":
    inspect_all_datasets()