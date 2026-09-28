# Vietnamese Depression Near-Duplicate / Historical Split Leakage Audit v1

This package audits whether highly similar or normalization-equivalent records cross the exact historical train/test boundaries used by the previous supervised experiments.

## Why this audit exists

The LDA diagnostic exposed repeated/templated families that were not exact duplicates. The raw datasets had already passed an exact-duplicate audit, but exact duplicate checks do not detect near-identical templates whose URLs, punctuation, or a few words differ. Because the legacy experiments used row-level random train/test splits, this audit quantifies whether such families cross the held-out boundary.

This audit **does not modify the data and does not retrain models**.

## Inputs expected

Place these files in the same project root already used for reproduction:

```text
Depression_Reproduction/
├── datasets/
│   ├── dataset_1.csv
│   └── dataset_2.csv
├── reproduction_results_v1/
│   ├── preprocessed/
│   │   ├── Dataset_depression_vi1.csv
│   │   └── Dataset_depression_vi2.csv
│   └── runs/.../predictions.csv     # optional; used for sensitivity analysis
├── near_duplicate_leakage_audit.py
└── Vietnamese_Depression_Near_Duplicate_Leakage_Audit_v1.ipynb
```

The script enforces the known SHA-256 hashes of the raw datasets and reconstructs the historical split memberships exactly.

## Historical split protocols audited

1. `tfidf_legacy`
   - reload audited preprocessed CSV;
   - `dropna(subset=['text_clean_tfidf'])`;
   - stratified 80/20 split with `random_state=42`;
   - expected modeled/test sizes: D1 5,921 / 1,185 and D2 7,249 / 1,450.

2. `dl_phobert_outer`
   - use the full non-empty `text_clean_dl` branch;
   - stratified 80/20 outer split with `random_state=42`;
   - expected modeled/test sizes: D1 5,933 / 1,187 and D2 7,345 / 1,469.

The second outer split is the held-out test membership used by BiLSTM, CNN, and PhoBERT; the validation subsets differ inside the training portion and are outside the primary scope of this audit.

## Similarity protocol

Primary fuzzy similarity is cosine similarity over `char_wb` TF-IDF features (`3–5` character n-grams) built from minimally normalized **raw `text_vi`**.

Normalization is deliberately conservative:

- Unicode NFKC;
- lowercase;
- HTML unescape;
- remove URLs, emails, and @mentions;
- punctuation/symbols → spaces;
- collapse whitespace;
- preserve Vietnamese diacritics and digits.

Predeclared thresholds are **0.90, 0.95, and 0.98**. Reporting all three avoids choosing a cutoff after seeing the result.

Fuzzy matching is restricted to normalized texts with at least **30 characters** to reduce false matches among very short generic statements. Normalization-exact duplicates are still counted at every length.

For every test record the audit saves its nearest training neighbor and similarity. All train→test pairs at or above the lowest threshold are also saved. Token-set Jaccard and length ratio are reported as secondary evidence; they are not used to redefine the primary threshold.

## Optional existing-prediction sensitivity

If `reproduction_results_v1/runs/*/*/predictions.csv` files are present, the script recomputes each reproduced model's metrics after excluding test rows flagged at each threshold.

This is only a **post-hoc sensitivity diagnostic**. If meaningful leakage is found, the publication-grade remedy is a group-aware split followed by retraining—not merely deleting difficult/contaminated test rows.

## Run on the server

Inside the existing `depression_repro` environment:

```bash
cd /home/taidinh/Depression_Reproduction

nohup python -u near_duplicate_leakage_audit.py \
  --project-root "$PWD" \
  --n-jobs 4 \
  > logs/near_duplicate_audit_v1.log 2>&1 &

echo $! > logs/near_duplicate_audit_v1.pid
cat logs/near_duplicate_audit_v1.pid
```

Monitor:

```bash
tail -f logs/near_duplicate_audit_v1.log
```

No GPU is required.

## Main outputs

```text
near_duplicate_audit_v1/
├── FINAL_STATUS.json
├── AUDIT_REPORT.md
├── tables/
│   ├── raw_input_audit.csv
│   ├── d1_d2_overlap_audit.csv
│   ├── historical_split_audit.csv
│   ├── near_duplicate_leakage_summary.csv
│   ├── existing_prediction_clean_test_sensitivity.csv
│   ├── prediction_sensitivity_status.csv
│   └── normalized_exact_duplicate_groups_d*.csv
├── pairs/
│   └── cross_split_pairs_*.csv.gz
├── clusters/
│   └── cluster_summary_*.csv
├── internal_review/
│   ├── test_nearest_train_evidence_*.csv.gz
│   ├── cross_split_pairs_WITH_TEXT_INTERNAL_*.csv.gz
│   └── cluster_members_INTERNAL_*.csv.gz
└── provenance/
    ├── audit_configuration.json
    ├── runtime.json
    └── similarity_space_d*.json
```

The `internal_review` files contain raw social-media text and should not be redistributed casually.

## What to send back for interpretation

After completion, package the compact result files:

```bash
tar -czf near_duplicate_audit_review.tar.gz \
  near_duplicate_audit_v1/FINAL_STATUS.json \
  near_duplicate_audit_v1/AUDIT_REPORT.md \
  near_duplicate_audit_v1/tables/near_duplicate_leakage_summary.csv \
  near_duplicate_audit_v1/tables/existing_prediction_clean_test_sensitivity.csv \
  near_duplicate_audit_v1/tables/prediction_sensitivity_status.csv \
  near_duplicate_audit_v1/tables/d1_d2_overlap_audit.csv \
  near_duplicate_audit_v1/tables/historical_split_audit.csv \
  near_duplicate_audit_v1/clusters/cluster_summary_*.csv
```

If we need to inspect individual examples, add the relevant `internal_review/*.csv.gz` files separately.
