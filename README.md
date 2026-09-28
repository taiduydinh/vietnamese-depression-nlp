# Vietnamese Depression-Signal NLP Reproduction and Topic-Modeling Artifacts

This repository snapshot contains the reproducible server-side materials for a Vietnamese social-media depression-signal NLP study: legacy supervised-model reproduction, a historical-split near-duplicate leakage audit, and the final LDA v2 thematic analysis.

> The task detects **depression-related signals in text**. Dataset labels are research annotations and are not clinical diagnoses.

## What is included

- `reproduce_all_previous_models.py` and compact reproduction outputs for TF-IDF+LR, TF-IDF+SVM, Word2Vec+BiLSTM, Word2Vec+CNN, and PhoBERT-base.
- `near_duplicate_leakage_audit.py`, audit notebook, aggregate audit outputs, pair/cluster summaries, and provenance.
- `lda_topic_modeling_v2.py`, notebook, frozen protocol, selected LDA model, all 60 search-run metric JSONs, aggregate model-selection/topic tables, figures, and provenance.
- Input hashes and expected dataset/preprocessing profiles.

## What is intentionally not included

- raw datasets and text-bearing preprocessed files;
- qualitative internal-review files containing social-media text;
- third-party VnCoreNLP/Word2Vec resource binaries;
- intermediate LDA search-model binaries;
- runtime logs/PID/cache files.

See `DATA_AND_PRIVACY.md`, `REPRODUCIBILITY_NOTES.md`, and `RESULTS_STATUS.md`.

## Important experiment status

- Legacy-model reproduction: **10/12 model–dataset runs completed** in the server snapshot; PhoBERT-large reruns were not completed because of GPU OOM in the shared-server session. Archived/original PhoBERT-large values remain in the comparison table.
- Near-duplicate audit: **complete**.
- LDA v2: **60/60 runs complete**, selected `K=7`, seed `82`, diagnostic quality gate passed.
- GPT-5.6 Sol outputs are maintained separately and still need to be merged into the final paper repository.

## Data placement for reproduction

Place the frozen inputs at:

```text
datasets/dataset_1.csv
datasets/dataset_2.csv
```

The expected hashes are documented in `datasets/README.md` and enforced by the scripts.

## Repository integrity

`CHECKSUMS_GITHUB_READY.txt` contains SHA-256 hashes for this curated snapshot. `GITHUB_FILE_MANIFEST.csv` records which files from the private server archive were retained or excluded and why.

## Ethics/privacy

Do not republish identifiable or sensitive user-generated text merely because it appears in a private research archive. The public-ready snapshot keeps aggregate/statistical outputs and removes internal text-bearing review files.
