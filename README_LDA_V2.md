# Vietnamese Depression LDA Topic Modeling v2

This package implements the frozen LDA v2 protocol in `LDA_V2_PROTOCOL.md`.

## Package files

- `lda_topic_modeling_v2.py` — primary CPU-only, resumable analysis script.
- `Vietnamese_Depression_LDA_Topic_Modeling_v2.ipynb` — lightweight notebook interface/inspection helper.
- `LDA_V2_PROTOCOL.md` — frozen scientific protocol.
- `requirements_lda_v2.txt` — package requirements; the existing `depression_repro` environment should already satisfy them.
- `SHA256SUMS_LDA_V2.txt` — checksums for the package files.

## Expected project layout

Place the package files directly under:

```text
/home/taidinh/Depression_Reproduction/
├── datasets/
│   ├── dataset_1.csv
│   └── dataset_2.csv
├── reproduction_results_v1/
│   └── preprocessed/
│       ├── Dataset_depression_vi1.csv
│       └── Dataset_depression_vi2.csv
├── near_duplicate_audit_v1/          # optional provenance cross-check; not required
├── lda_topic_modeling_v2.py
├── Vietnamese_Depression_LDA_Topic_Modeling_v2.ipynb
├── LDA_V2_PROTOCOL.md
└── requirements_lda_v2.txt
```

The script does **not** need LDA v1 files.

## Server workflow

### 1. Enter Docker

From the host:

```bash
docker exec -it kansai_llm bash
```

Inside Docker:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate depression_repro
cd /home/taidinh/Depression_Reproduction
```

### 2. Verify the script

```bash
python -m py_compile lda_topic_modeling_v2.py
```

Optional dependency check:

```bash
python - <<'PY'
import numpy, pandas, scipy, sklearn, gensim, matplotlib
print('numpy:', numpy.__version__)
print('pandas:', pandas.__version__)
print('scipy:', scipy.__version__)
print('sklearn:', sklearn.__version__)
print('gensim:', gensim.__version__)
print('matplotlib:', matplotlib.__version__)
print('DEPENDENCIES: PASS')
PY
```

Do not reinstall packages if this passes.

### 3. Run the prepare-only audit first

This is intentionally a separate gate before the 60-model search:

```bash
python -u lda_topic_modeling_v2.py \
  --project-root "$PWD" \
  --n-jobs 4 \
  --prepare-only
```

Expected high-level messages include:

```text
RAW / SHARED-POSITIVE / PREPROCESSING AUDIT: PASS
CANONICAL CORPUS AUDIT: PASS (3782 rows; 3778 non-empty legacy-clean texts)
FAMILY CONTROL: ...
TOPIC CORPUS READY: ...
PREPARE-ONLY COMPLETE.
```

Inspect these before the full search:

```bash
cat lda_results_v2/provenance/near_duplicate_family_control.json
cat lda_results_v2/provenance/corpus_and_dictionary_stats.json
cat lda_results_v2/provenance/frozen_analysis_configuration.json

python - <<'PY'
import pandas as pd
print('\n=== CORPUS FLOW ===')
print(pd.read_csv('lda_results_v2/tables/lda_v2_corpus_flow.csv').to_string(index=False))
print('\n=== FAMILY SUMMARY (largest first) ===')
f='lda_results_v2/tables/near_duplicate_family_summary.csv'
df=pd.read_csv(f)
print(df.sort_values('family_size', ascending=False).head(20).to_string(index=False) if len(df) else 'No families found')
print('\n=== DICTIONARY REMOVAL COUNTS ===')
t=pd.read_csv('lda_results_v2/tables/dictionary_term_filter_audit.csv')
print(t['dictionary_action'].value_counts().to_string())
PY
```

**Do not start the full LDA search until these preparation outputs look reasonable.**

### 4. Launch the full LDA v2 search

The full run is CPU-only:

```bash
mkdir -p logs

nohup python -u lda_topic_modeling_v2.py \
  --project-root "$PWD" \
  --n-jobs 4 \
  > logs/lda_v2.log 2>&1 &

echo $! > logs/lda_v2.pid
cat logs/lda_v2.pid
```

Monitor:

```bash
tail -f logs/lda_v2.log
```

Count completed LDA search runs:

```bash
find lda_results_v2/models/search -name '*_metrics.json' -type f | wc -l
```

The final count is **60**.

Check process status:

```bash
ps -p $(cat logs/lda_v2.pid) -o pid,etime,%cpu,%mem,cmd
```

The search is resumable. If the process stops, rerun the same command; completed checkpoints are reused only when their stored analysis signature matches the current v2 corpus/configuration.

## Important outputs

### Corpus control and provenance

```text
lda_results_v2/tables/canonical_positive_corpus_3782.csv
lda_results_v2/tables/near_duplicate_family_decisions.csv
lda_results_v2/tables/near_duplicate_family_summary.csv
lda_results_v2/internal_review/near_duplicate_family_members_INTERNAL.csv.gz
lda_results_v2/tables/lda_v2_corpus_flow.csv
lda_results_v2/tables/dictionary_term_filter_audit.csv
lda_results_v2/provenance/frozen_analysis_configuration.json
lda_results_v2/provenance/near_duplicate_family_control.json
lda_results_v2/provenance/corpus_and_dictionary_stats.json
```

### Model selection

```text
lda_results_v2/tables/lda_all_runs.csv
lda_results_v2/tables/lda_stability_pairwise.csv
lda_results_v2/tables/lda_k_search_summary.csv
lda_results_v2/tables/lda_k_selection_table.csv
lda_results_v2/tables/lda_k_medoid_seed_table.csv
lda_results_v2/tables/lda_k_medoid_prevalence_diagnostics.csv
lda_results_v2/tables/lda_selected_k_seed_table.csv
lda_results_v2/provenance/lda_selection.json
lda_results_v2/provenance/selected_solution_quality_gate.json
```

### Interpretation

```text
lda_results_v2/tables/lda_topic_terms.csv
lda_results_v2/tables/lda_topic_prevalence.csv
lda_results_v2/tables/lda_topic_prevalence_family_size_weighted_sensitivity.csv
lda_results_v2/internal_review/lda_representative_documents_INTERNAL.csv
lda_results_v2/internal_review/lda_ambiguous_documents_INTERNAL.csv
lda_results_v2/tables/lda_document_topics.csv
lda_results_v2/tables/lda_gpt_comparison_representatives_only.csv
lda_results_v2/paper_methods_lda_v2.txt
```

Completion marker:

```text
lda_results_v2/RUN_COMPLETE.json
```

## What not to change for the primary paper run

Do not change the default near-duplicate threshold, dictionary thresholds, K range, seeds, passes, iterations, or selection rule after seeing v2 results. If a sensitivity analysis is later needed, run it into a **different results directory** and label it as sensitivity analysis.

Do not force K=12 to match the GPT taxonomy, and do not manually remove terms simply because a resulting topic is inconvenient.
