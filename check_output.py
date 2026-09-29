import pandas as pd

print("--- CHECKING OUTPUT FILES ---")

# 1. Read Candidate Pairs
df_cand = pd.read_csv('output/candidate_pairs.tsv', sep='\t')
print(f"1. candidate_pairs.tsv Rows: {len(df_cand):,}")

# 2. Read Matching Results
df_match = pd.read_csv('output/matching_results.tsv', sep='\t')
print(f"2. matching_results.tsv Rows: {len(df_match):,}")

# 3. Count Non-Empty Matches
matches_found = (df_match['matched_entity_ids'].fillna('') != '').sum()
singletons = (df_match['matched_entity_ids'].fillna('') == '').sum()

print(f"3. Total Entities Matched: {matches_found:,}")
print(f"4. Total Unmatched Singletons: {singletons:,}")

print("\n--- FIRST 5 ROWS OF MATCHING RESULTS ---")
print(df_match.head(5))