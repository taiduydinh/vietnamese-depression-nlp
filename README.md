# Vietnamese Depression-Signal NLP

This repository contains the reproducible code and experiment outputs for a Vietnamese social-media NLP study on **depression-related language signals**. The project combines conventional machine learning, deep learning, PhoBERT, GPT-5.6 Sol, near-duplicate leakage auditing, and unsupervised LDA topic modeling.

> The task concerns signals expressed in text. The dataset labels are research annotations and must not be interpreted as clinical diagnoses.

## Project workflow

```text
Raw Vietnamese datasets
        |
        +--> Audited preprocessing
        |       |
        |       +--> TF-IDF + Logistic Regression
        |       +--> TF-IDF + SVM
        |       +--> Word2Vec + BiLSTM
        |       +--> Word2Vec + CNN
        |       +--> PhoBERT-base / PhoBERT-large
        |       |
        |       +--> Near-duplicate leakage audit
        |
        +--> GPT-5.6 Sol
        |       +--> sentiment classification
        |       +--> T01-T12 thematic annotation
        |
        +--> Shared positive corpus
                +--> near-duplicate/template control
                +--> LDA topic modeling (K = 4..15, five seeds)
```

The supervised models and GPT evaluate depression-signal classification, while LDA is used as an **unsupervised thematic analysis**. LDA-GPT comparisons should therefore be interpreted as thematic correspondence rather than topic-classification accuracy.

## Repository structure

```text
vietnamese-depression-nlp/
|
|-- datasets/
|   `-- README.md                         # expected data format and frozen hashes
|
|-- reproduce_all_previous_models.py     # legacy ML/DL/PhoBERT reproduction
|-- requirements_reproduction_server.txt
|-- reproduction_results_v1/             # reproduced metrics, predictions and audits
|
|-- gpt56_sol/
|   |-- notebooks/                       # GPT-5.6 Sol experiment/recovery notebooks
|   |-- protocol/                        # prompt, schema and T01-T12 topic framework
|   `-- results/                         # final metrics and text-free topic predictions
|
|-- near_duplicate_leakage_audit.py
|-- Vietnamese_Depression_Near_Duplicate_Leakage_Audit_v1.ipynb
|-- requirements_near_duplicate_audit.txt
|-- near_duplicate_audit_v1/             # leakage and sensitivity results
|
|-- lda_topic_modeling_v2.py
|-- Vietnamese_Depression_LDA_Topic_Modeling_v2.ipynb
|-- requirements_lda_v2.txt
|-- LDA_V2_PROTOCOL.md
|-- lda_results_v2/                      # LDA search, selection, topics and figures
|
|-- README_SERVER_REPRODUCTION.md
|-- README_NEAR_DUPLICATE_AUDIT.md
|-- README_LDA_V2.md
|-- RESULTS_STATUS.md
`-- REPRODUCIBILITY_NOTES.md
```

## 1. Set up the environment

Python 3.11 was used for the server-side reproduction workflow. A convenient setup is:

```bash
conda create -n depression_repro python=3.11 -y
conda activate depression_repro

python -m pip install -r requirements_reproduction_server.txt
python -m pip install -r requirements_near_duplicate_audit.txt
python -m pip install -r requirements_lda_v2.txt
```

For the historical preprocessing reproduction, keep the pinned versions in `requirements_reproduction_server.txt`, especially:

```text
underthesea==8.3.0
py_vncorenlp==0.1.4
```

PhoBERT runs require a TensorFlow-compatible GPU for practical execution. The near-duplicate audit and LDA workflow are CPU-compatible.

## 2. Prepare the datasets

Place the two frozen CSV files at:

```text
datasets/dataset_1.csv
datasets/dataset_2.csv
```

Each file must contain:

```text
text_vi      Vietnamese text
sentiment    1 = depression-related signal, 0 = control
```

Expected input profiles are:

| Dataset | Rows | Label 0 | Label 1 | SHA-256 |
|---|---:|---:|---:|---|
| D1 | 5,933 | 2,151 | 3,782 | `b2ad99ebdcf3c34f040a2988fda1c482e9ff9b9cf42795bb64c792b3c7043d9e` |
| D2 | 7,345 | 3,563 | 3,782 | `88161f9388194bbb9cede048223d8f7f6bc6ab68670c373eeffd13bbb45de842` |

The two benchmark variants share the same 3,782 positive texts. See `datasets/README.md` for the data requirements.

## 3. Reproduce the legacy classifiers

Run the preprocessing/audit gate first:

```bash
export TF_USE_LEGACY_KERAS=1

python -u reproduce_all_previous_models.py \
  --project-root "$PWD" \
  --prepare-only \
  --force-preprocess
```

The strict audit should recover the historical usable/test counts before model training begins.

Then run the full reproduction:

```bash
python -u reproduce_all_previous_models.py \
  --project-root "$PWD"
```

The script supports selective execution, for example:

```bash
python -u reproduce_all_previous_models.py \
  --project-root "$PWD" \
  --models lr svm \
  --datasets 1 2
```

Available model names are:

```text
lr
svm
bilstm
cnn
phobert-base
phobert-large
all
```

Main outputs are written under:

```text
reproduction_results_v1/
```

Useful summary files include:

```text
reproduction_results_v1/all_reproduced_results.csv
reproduction_results_v1/comparison_with_legacy.csv
reproduction_results_v1/preprocessing_audit.csv
reproduction_results_v1/runtime_environment.json
```

For the exact historical preprocessing details, see `README_SERVER_REPRODUCTION.md`.

## 4. Run or inspect the GPT-5.6 Sol experiment

The GPT workflow is under:

```text
gpt56_sol/
```

The main notebook is:

```text
gpt56_sol/notebooks/Vietnamese_Depression_GPT56_Sol_Batch_SENTIMENT_TOPICS_v1_4.ipynb
```

It uses the frozen experiment definition in:

```text
gpt56_sol/protocol/prompt_v1_4.txt
gpt56_sol/protocol/output_schema_v1_4.json
gpt56_sol/protocol/topic_codebook_v1_4.csv
gpt56_sol/protocol/TOPIC_FRAMEWORK.md
```

To rerun it, open the notebook, provide your own OpenAI API key when prompted, prepare the batches, submit them, and use the notebook's batch manager to retrieve and parse the completed outputs.

The final result files already produced for this study are in:

```text
gpt56_sol/results/
```

The most useful files are:

```text
gpt56_sol/results/gpt_final_metrics_summary.csv
gpt56_sol/results/d1_predictions_public.csv
gpt56_sol/results/d2_predictions_public.csv
gpt56_sol/results/d1_topic_primary_distribution.csv
gpt56_sol/results/d2_topic_primary_distribution.csv
```

The D2 experiment includes a documented recovery of 139 requests; `d2_final_metrics.json` is the authoritative final D2 result.

## 5. Run the near-duplicate leakage audit

This analysis checks whether highly similar records cross the historical train/test boundaries.

It depends on the audited preprocessing generated in Step 3. Run:

```bash
python -u near_duplicate_leakage_audit.py \
  --project-root "$PWD" \
  --n-jobs 4
```

Main outputs are written to:

```text
near_duplicate_audit_v1/
```

Start with:

```text
near_duplicate_audit_v1/AUDIT_REPORT.md
near_duplicate_audit_v1/tables/near_duplicate_leakage_summary.csv
near_duplicate_audit_v1/tables/existing_prediction_clean_test_sensitivity.csv
near_duplicate_audit_v1/tables/historical_split_audit.csv
```

The audit uses character `char_wb` TF-IDF cosine similarity with 3-5 character n-grams and reports the predeclared thresholds 0.90, 0.95 and 0.98. Full methodological details are in `README_NEAR_DUPLICATE_AUDIT.md`.

## 6. Run the LDA v2 thematic analysis

LDA is applied once to the shared positive corpus because the 3,782 positive texts are identical in D1 and D2.

First run the preparation gate:

```bash
python -u lda_topic_modeling_v2.py \
  --project-root "$PWD" \
  --n-jobs 4 \
  --prepare-only
```

Inspect the corpus-control and dictionary outputs, then launch the full search:

```bash
python -u lda_topic_modeling_v2.py \
  --project-root "$PWD" \
  --n-jobs 4
```

The primary search evaluates:

```text
K = 4..15
seeds = 42, 52, 62, 72, 82
60 total LDA runs
```

Model selection uses the frozen rule defined in `LDA_V2_PROTOCOL.md`: one-standard-error coherence eligibility followed by cross-seed stability, with the final seed chosen as the medoid run.

Main outputs are under:

```text
lda_results_v2/
```

Start with:

```text
lda_results_v2/provenance/lda_selection.json
lda_results_v2/provenance/selected_solution_quality_gate.json
lda_results_v2/tables/lda_k_search_summary.csv
lda_results_v2/tables/lda_topic_terms.csv
lda_results_v2/tables/lda_topic_prevalence.csv
lda_results_v2/figures/
```

The completed primary analysis selected **K=7, seed=82**. See `README_LDA_V2.md` for the full protocol and output map.

## 7. Use the existing results without rerunning the experiments

For readers who only want to inspect or reproduce the paper tables, the principal result locations are:

| Analysis | Main result files |
|---|---|
| Legacy classifiers | `reproduction_results_v1/all_reproduced_results.csv`, `comparison_with_legacy.csv` |
| GPT-5.6 Sol | `gpt56_sol/results/gpt_final_metrics_summary.csv`, `d1_predictions_public.csv`, `d2_predictions_public.csv` |
| Near-duplicate audit | `near_duplicate_audit_v1/tables/near_duplicate_leakage_summary.csv`, `existing_prediction_clean_test_sensitivity.csv` |
| LDA v2 | `lda_results_v2/tables/lda_k_search_summary.csv`, `lda_topic_terms.csv`, `lda_topic_prevalence.csv` |

`RESULTS_STATUS.md` summarizes the completion status of the experimental components, while `REPRODUCIBILITY_NOTES.md` records environment and historical-reproduction caveats.

## Reproducibility notes

The repository includes checksums and provenance files throughout the experiment outputs. For a fresh reproduction, keep the data hashes, preprocessing versions, split settings, model hyperparameters, GPT prompt/schema, near-duplicate thresholds, and LDA selection protocol unchanged unless you are explicitly conducting a separate sensitivity analysis.

The historical PhoBERT-large server rerun was not completed in the final shared-GPU session; the repository therefore distinguishes those archived/original values from the models successfully rerun in the reproduction environment. See `RESULTS_STATUS.md` for the exact status.

## Responsible use

The project analyzes user-generated mental-health-related text. Results should be interpreted as NLP research outputs rather than medical assessments. Do not use the models or topic assignments to diagnose individuals or make clinical decisions.
