# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityResolvers  
**Team Members:** Data Science & ML Engineering Team  
**Submission Date:** September 2026  

---

## 1. Executive Summary

This report documents our end-to-end Machine Learning pipeline for the Amazon ML Challenge 2026 (Business Entity Resolution). Tasked with matching 1,732,544 Source 1 ($S_1$) test business entities against independent, noisy sources ($S_2$ and $S_3$) on a strict consumer hardware budget (16GB RAM, 256GB SSD), traditional quadratic TF-IDF matrix approaches failed with 334GB out-of-memory errors. We solved this scalability barrier by architecting a **Dual-Key Hash Blocking** system that generated exactly 13,705,779 candidate pairs in linear time. We then trained a **LightGBM Gradient Boosted Decision Tree** utilizing string distance metrics (RapidFuzz, Jaro-Winkler, token sort), address overlap, and structural difference features, calibrated with a precision-heavy decision threshold ($\tau = 0.70$) specifically tuned for the challenge's macro-averaged $F_{0.5}$ metric.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis across the multi-million record datasets, several key characteristics and noise patterns were identified:
1. **Extreme Asymmetric Scale**: Source 1 acts as the deduplicated reference set ($1,732,544$ records in test), while Source 2 and Source 3 represent heterogeneous commercial entity dumps with varied entity representations.
2. **Noise Dimensions**:
   - *Name Variations*: Frequent legal suffix permutations ("Corp", "Corporation", "Pvt Ltd", "Private Limited"), punctuation discrepancies ("&" vs "and"), acronyms, spelling errors, and token-order inversions ("Retail Amazon" vs "Amazon Retail").
   - *Address Inconsistencies*: Truncated strings, missing state or PIN codes, landmark-based descriptions ("Near Metro Station"), and localized transliterations.
   - *Open Set Domain Shift*: While training data only contained `US` and `India`, test data introduced `France`. Our pipeline remained strictly language/country-agnostic in feature extraction.
3. **Hardware Boundary Constraint**: The HP Pavilion workstation operates on 16GB RAM and 256GB SSD. Generating an all-pairs sparse TF-IDF cosine similarity matrix across $1.73\text{M} \times 1.7\text{M}$ entries triggers memory allocations of $334\text{ GB}$, crashing the system instantly.

### 2.2 Solution Strategy
To ensure computational feasibility, high throughput, and maximum precision under $F_{0.5}$, we implemented a decoupled, two-stage architecture:
- **Approach Type**: Dual-Key Hash Inverted Index Blocking + LightGBM Supervised Classifier + Precision-Biased Thresholding.
- **Core Innovation**: Memory-safe dual-pass hash indexing combining alphanumeric normalization with alphabetical token-sorting. This bypassed expensive cross-joins while retaining high candidate recall, reducing the comparison space by $99.5\%$ with zero RAM overflow.

```
[Raw Entity Records (S1, S2, S3)]
               │
               ▼
[Dual-Key Hash Inverted Indexing]
  ├─ Key 1: clean_str (Exact alphanumeric)
  └─ Key 2: make_token_sort_key (Alphabetically sorted tokens)
               │
               ▼
[Candidate Pair Pool: 13,705,779 pairs]
               │
               ▼
[Pairwise Feature Engineering (RapidFuzz, Jaro, Address Overlap)]
               │
               ▼
[LightGBM GBDT Matcher (Trained on GT Positives + Hard Negatives)]
               │
               ▼
[Calibrated Thresholding (tau = 0.70 for F_0.5)]
               │
               ▼
[Final Outputs: matching_results.tsv & candidate_pairs.tsv]
```

---

## 3. Candidate Generation (Blocking)

To reduce the $1.73\text{M} \times 3.5\text{M}$ search space without exhausting system memory, we designed an inverted hash index blocking strategy:

- **Blocking Keys Used**:
  1. *Exact Normalized Key (`clean_str`)*: Lowercases all characters, strips punctuation and whitespace, preserving core alphanumeric tokens (e.g., `"Amazon Services, Inc."` $\rightarrow$ `"amazonservicesinc"`).
  2. *Token-Sorted Key (`make_token_sort_key`)*: Extracts all individual word tokens, sorts them alphabetically, and joins them into a unified string (e.g., `"Services Amazon"` $\rightarrow$ `"amazonservices"`). This directly resolves word-order transpositions and suffix reordering.
- **Candidate Pairs Generated**: Exactly **13,705,779 candidate pairs** were generated across $S_2$ and $S_3$ matches.
- **Ensuring True Matches Were Preserved**: By executing dual-key indexing across both sources, entities with either identical normalized names or word-reordered variations are guaranteed to map into the exact same hash bucket in $O(N)$ lookup time, achieving high candidate recall while avoiding sparse matrix computation.

---

## 4. Matching Model

### Features Used:
For every candidate pair $(e_{S1}, e_{cand})$, we computed an 8-dimensional pairwise similarity feature vector:
1. **Exact Name Match Flag (`feat_name_exact`)**: Binary indicator (1.0 if normalized strings are identical, else 0.0).
2. **Levenshtein Name Similarity (`feat_name_ratio`)**: RapidFuzz fuzzy ratio normalized to $[0.0, 1.0]$.
3. **Token Sort Name Similarity (`feat_name_token_sort`)**: RapidFuzz token sort ratio handling token permutations.
4. **Jaro-Winkler Similarity (`feat_name_jaro`)**: String metric measuring common prefix alignment and transposed characters.
5. **Name Length Absolute Difference (`feat_name_len_diff`)**: Absolute character length difference $|len(n_1) - len(n_2)|$.
6. **Address Fuzzy Ratio (`feat_addr_ratio`)**: Character-level Levenshtein similarity on normalized street and landmark addresses.
7. **Address Missing Indicator (`feat_addr_missing`)**: Boolean flag indicating whether either record lacked address metadata.
8. **Candidate Origin Flag (`feat_is_s2`)**: Source indicator differentiating Source 2 from Source 3 candidates.

### Model Type:
- **Model**: LightGBM Binary Classifier (`lgb.LGBMClassifier`).
- **Configuration**:
  - `n_estimators`: 150 – 250
  - `learning_rate`: 0.03 – 0.05
  - `max_depth`: 5
  - `num_leaves`: 31
  - `objective`: `binary`
  - `class_weight`: `balanced` (compensating for sparse positive matches in candidate space)
  - `random_state`: 42

### Threshold Selection Method:
The competition evaluates models using the macro-averaged $F_{0.5}$ score:
$$F_{0.5} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$
Because Precision is weighted $2\times$ more heavily than Recall, false positives (incorrect merges) penalize the score twice as severely as false negatives (missed matches). A standard $\tau = 0.50$ threshold permits too many borderline candidates to be classified as matches. Through validation sweeps across $\tau \in [0.10, 0.95]$ with step size $0.05$, the optimal operating threshold was determined to be **$\tau = 0.70$**, ensuring high precision and purging low-confidence candidate pairs.

---

## 5. Results & Error Analysis

### Validation Performance:
- **Macro $F_{0.5}$ Score**: **0.9124** on hold-out validation set.
- **Precision**: **0.9418** (reflecting robust false-positive suppression).
- **Recall**: **0.8120**.

### Error Analysis:
- **Common False Positives (Wrong Merges)**:
  - *Franchise & Branch Outlets*: Entities sharing identical brand names (e.g., "Starbucks Coffee", "Subway") located at different regional addresses where address fields were either missing or generic.
  - *Corporate Conglomerates*: Distinct legal subsidiaries sharing a common parent name stem (e.g., "Tata Steel" vs "Tata Motors") when legal suffixes were stripped.
- **Common False Negatives (Missed Matches)**:
  - *Severe Typographical Corruption*: Extreme character misspellings or OCR errors that caused the token-sorted key to miss hash blocking entirely.
  - *Alternate Brand Aliases*: Cases where a business operated under a Doing-Business-As (DBA) trade name completely distinct from its registered legal title without cross-referencing metadata.

---

## 6. Conclusion
By pairing linear-time Dual-Key Hash Blocking with an optimized LightGBM classifier and precision-tuned decision thresholding ($\tau = 0.70$), our pipeline achieves competitive entity resolution performance on 1.73M test entities while strictly conforming to 16GB consumer workstation memory boundaries. The solution is fully reproducible, deterministic, and modular.

---

## Appendix

### A. Code Artefacts
The complete runnable pipeline is packaged in `code/business_entity_resolution/`:
```text
code/business_entity_resolution/
├── README.md               # End-to-end execution guide
├── requirements.txt        # Pinned dependencies
├── generate_sub_ml.py      # Master production execution script
└── src/
    ├── blocking.py         # Multi-pass & TF-IDF candidate generation modules
    ├── features.py         # Pairwise similarity feature extraction functions
    └── train_infer.py      # LightGBM training & F_0.5 threshold tuning
```
To reproduce both `output/matching_results.tsv` and `output/candidate_pairs.tsv`:
```bash
cd code/business_entity_resolution
pip install -r requirements.txt
python generate_sub_ml.py
```

### B. Submission Verification Summary
- **Candidate Pairs Generated**: Exactly `13,705,779` rows (`output/candidate_pairs.tsv`).
- **Matching Results Rows**: Exactly `1,732,544` rows (`output/matching_results.tsv`).
- **Entity Coverage**: 100% of Source 1 test entities represented with 0 duplicate rows.
- **Singleton Handling**: 378,867 singletons correctly formatted as empty strings `""` with zero literal `NaN` values.
