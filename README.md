# Amazon ML Challenge 2026: Business Entity Resolution

An end-to-end Machine Learning and data engineering pipeline to resolve, link, and deduplicate business entity records across multi-million-row heterogeneous data sources without common identifiers.

---

## Overview

In commercial platforms, business records arrive from multiple independent databases with missing, inconsistent, or noisy fields. The objective of this challenge is to take **1,732,544 ($1.73\text{M}$)** reference entities from **Source 1 ($S_1$)** and identify all corresponding matching records across **Source 2 ($S_2$)** and **Source 3 ($S_3$)** (~10M total records).

### Key Challenges
1. **No Shared Identifiers**: Records share no common foreign keys or national registration numbers.
2. **Real-World Text Noise**: Variations in legal suffixes (*Corp*, *Corporation*, *Pvt Ltd*, *Limited*), token permutations (*Amazon Retail* vs. *Retail Amazon*), typos, abbreviations, and informal address landmarks.
3. **Scale & Memory Constraints**: A brute-force Cartesian comparison ($1.73\text{M} \times 10\text{M}$) requires over 17 trillion operations. Standard all-pairs sparse TF-IDF matrices require over 300GB of RAM. The pipeline is designed to execute efficiently in linear time on standard commodity hardware.
4. **Precision-Biased Metric ($F_{0.5}$)**: The competition evaluates submissions using the macro-averaged $F_{0.5}$ score:
   $$F_{0.5} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$
   False positive merges (linking two different businesses) are penalized twice as heavily as missed matches. Singletons (entities with zero matches) are factored into the average and must be left as empty strings.

---

## Pipeline Architecture

The solution uses a two-stage architecture: **Candidate Blocking** followed by **Pairwise Classification & Thresholding**.

```
[Raw Sources S1, S2, S3]
           │
           ▼
[Dual-Key Hash Inverted Index Blocking]
  ├─ Pass 1: Normalized alphanumeric key
  └─ Pass 2: Alphabetical token-sorted key
           │
           ▼
[Candidate Pairs Pool (~13.7M pairs)]
           │
           ▼
[Pairwise Feature Engineering (8 Similarity Signals)]
           │
           ▼
[LightGBM Binary Classifier]
           │
           ▼
[Precision-Tuned Thresholding (tau = 0.70)]
           │
           ▼
[Output: matching_results.tsv & candidate_pairs.tsv]
```

### 1. Dual-Key Inverted Index Blocking
To avoid memory exhaustion, records are indexed into hash buckets using two complementary keys:
* **Exact Normalized Key (`clean_str`)**: Strips all non-alphanumeric characters, whitespace, and casing (`"Amazon Services, Inc."` $\rightarrow$ `"amazonservicesinc"`).
* **Token-Sorted Key (`make_token_sort_key`)**: Tokenizes the string, sorts tokens alphabetically, and joins them (`"Services Amazon"` $\rightarrow$ `"amazonservices"`).

This reduces the search space by **99.9%** (from 17 trillion down to **13,705,779 candidate pairs**) in $O(N)$ linear time.

### 2. Feature Engineering
For each candidate pair, an 8-dimensional numerical feature vector is computed:
* **Exact Name Match Flag**: Binary flag indicating identical normalized names.
* **RapidFuzz Levenshtein Ratio**: Character edit distance between name strings.
* **Token Sort Ratio**: Word-order-invariant similarity score.
* **Jaro-Winkler Similarity**: Measures common prefix alignment and transposed characters.
* **String Length Delta**: Absolute difference in name lengths.
* **Address Levenshtein Similarity**: Edit distance between address strings.
* **Address Missing Flag**: Binary indicator if either record lacks address data.
* **Source Origin Flag**: Distinguishes Source 2 candidates from Source 3.

### 3. Model Training & Decision Thresholding
* **Model**: LightGBM Binary Classifier (`lgb.LGBMClassifier`) trained on ground-truth matches and balanced hard negative pairs.
* **Threshold Selection**: Because of the 2× precision weighting in $F_{0.5}$, standard $0.50$ thresholds admit too many borderline false positives. The decision threshold was calibrated to **$\tau = 0.70$**, strictly pruning ambiguous candidates to safeguard precision.

---

## Repository Structure

```text
.
├── README.md               # Project documentation & architecture
├── Documentation_template.md # Detailed methodology write-up
├── requirements.txt        # Pinned dependencies
├── generate_sub_ml.py      # Main end-to-end LightGBM inference script
├── generate_sub.py         # Fast country-partitioned exact match baseline
├── baseline_pipeline.py    # Initial exploratory data loader & baseline
├── check_output.py         # Submission verification & row-counting script
├── inspect_data.py         # EDA script for dataset inspection
└── src/
    ├── blocking.py         # Candidate generation & TF-IDF blocking modules
    ├── features.py         # Pairwise similarity feature extraction
    └── train_infer.py      # LightGBM training & threshold tuning
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/srinivas-cod/Entity-Resolution---Amazon-ML.git
   cd Entity-Resolution---Amazon-ML
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### Step 1: Data Placement
Place the dataset folders in the project root:
```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

### Step 2: Run End-to-End Pipeline
To run the full LightGBM pipeline:
```bash
python generate_sub_ml.py
```

Outputs will be saved in `output/`:
* `output/matching_results.tsv`: Final predicted matches for each of the 1,732,544 test entities.
* `output/candidate_pairs.tsv`: Final candidate pair pool (13,705,779 rows).

### Step 3: Verify Output Formatting
To audit the generated outputs against all competition formatting and schema rules:
```bash
python check_output.py
```

---

## Results & Deliverable Verification

* **Total Test Entities Evaluated**: 1,732,544 rows (100% coverage, 0 duplicates, 0 missing).
* **Candidate Pool Generated**: 13,705,779 pairs generated via dual-key hash blocking.
* **Singleton Handling**: 378,867 entities correctly predicted as empty singletons (`""`) with zero literal `NaN` values.
* **Validation F0.5 Score**: ~0.91 on hold-out validation splits.

---

## License

This project is licensed under the MIT License.
