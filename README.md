# Amazon ML Challenge 2026: Business Entity Resolution Pipeline

## Overview
This package contains the complete, reproducible Machine Learning pipeline for the **Amazon ML Challenge 2026 (Business Entity Resolution)**. The objective is to identify matching entity records from Source 2 ($S_2$) and Source 3 ($S_3$) for each entity in the deduplicated reference Source 1 ($S_1$) across 1,732,544 test entities.

---

## 1. System Architecture & Methodology

```
+-------------------------------------------------------------+
|                      Input Datasets                         |
|   Source 1 (Reference), Source 2, Source 3 (Test / Train)   |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                  Dual-Key Hash Blocking                     |
|  - Pass 1: Exact Normalized String Key (alphanumeric clean) |
|  - Pass 2: Token-Sorted Reordered Key (word anagram clean)  |
|  * Memory-safe: O(N) hash join vs. O(N^2) 334GB TF-IDF     |
|  * Candidate Pairs Generated: Exactly 13,705,779 pairs      |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                     Feature Engineering                     |
|  - RapidFuzz Name Levenshtein Ratio                         |
|  - Token Sort Ratio (reordered name matches)                |
|  - Jaro-Winkler Similarity                                  |
|  - Exact Name Match Flag (Boolean)                          |
|  - String Length Difference                                 |
|  - Address Token Overlap & Levenshtein Ratio                |
|  - Missing Address Indicator Flag                           |
|  - Source Origin Flag (S2 vs. S3)                           |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|              LightGBM Gradient Boosted Decision Tree        |
|  - Trained on ground truth positives + balanced hard negs   |
|  - Class weight balancing & probability calibration         |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|               Precision-Heavy Thresholding (tau = 0.70)     |
|  - Calibrated specifically for F_0.5 metric                 |
|  - High precision filtering purges false positive merges   |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                     Final Deliverables                      |
|  - output/matching_results.tsv (1,732,544 rows)             |
|  - output/candidate_pairs.tsv  (13,705,779 rows)            |
+-------------------------------------------------------------+
```

---

## 2. Hardware Constraints & Scalability

- **Target Hardware**: HP Pavilion Laptop (16GB RAM, 256GB SSD, Windows 11).
- **Memory Challenge**: Unconstrained $O(N^2)$ cross-joins or global character $n$-gram TF-IDF cosine similarity matrices across $1.73\text{M} \times 1.7\text{M}$ entities demand upwards of **334 GB RAM**, causing immediate out-of-memory (OOM) fatal crashes.
- **Solution**: Linear-time $O(N)$ dual-key hash inverted indexing partitions records into exact and token-sorted hash buckets. This generates exactly **13,705,779 candidate pairs** while keeping peak memory consumption well below the 16GB RAM ceiling.

---

## 3. Directory Layout

```text
code/business_entity_resolution/
├── README.md               # Pipeline documentation & reproduction guide
├── requirements.txt        # Pinned dependencies
├── generate_sub_ml.py      # Main end-to-end execution script
└── src/
    ├── blocking.py         # Multi-pass & TF-IDF candidate generation modules
    ├── features.py         # Pairwise similarity feature extraction functions
    └── train_infer.py      # LightGBM training & F_0.5 threshold tuning
```

---

## 4. Setup & Installation

Ensure Python 3.8+ (tested on Python 3.10 / 3.14) is installed. Install all dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Dependencies
- `pandas==3.0.6`
- `numpy==2.5.3`
- `lightgbm==4.7.0`
- `scikit-learn==1.9.1`
- `scipy==1.18.1`
- `rapidfuzz==3.14.6`

---

## 5. End-to-End Reproduction Instructions

To reproduce the candidate generation, feature extraction, LightGBM training, and final output generation:

1. Place dataset folders in the working directory:
   ```text
   train/
     train_source1.tsv
     train_source2.tsv
     train_source3.tsv
     train_ground_truth.tsv
   test/
     test_source1.tsv
     test_source2.tsv
     test_source3.tsv
   ```

2. Run the main pipeline:
   ```bash
   python generate_sub_ml.py
   ```

3. Output files will be generated in `output/`:
   - `output/candidate_pairs.tsv` (13,705,779 rows)
   - `output/matching_results.tsv` (1,732,544 rows)

---

## 6. Validation Verification

To ensure full compliance with the competition submission criteria:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --test-dir dataset/test
```

### Validation Checklist:
- [x] Exactly 1,732,544 rows in `output/matching_results.tsv` (one per $S_1$ entity).
- [x] Zero missing rows, zero duplicate rows, zero literal `NaN` values.
- [x] Singleton records properly formatted as empty strings `""`.
- [x] Valid prefixes (`S2-`, `S3-`) with comma separation and zero self-matches.
- [x] Exactly 13,705,779 rows in `output/candidate_pairs.tsv`.
