# Data availability and expected files

The raw Vietnamese social-media datasets are intentionally **not redistributed in this public-ready snapshot**.
They contain user-generated text, and redistribution should be handled only after checking the source licenses and the project's ethics/privacy requirements.

The reproduction scripts expect:

```text
datasets/dataset_1.csv
datasets/dataset_2.csv
```

Both files have columns `text_vi` and `sentiment`, where the project uses `sentiment=1` for depression-related signals and `sentiment=0` for controls. The label is a dataset annotation, not a clinical diagnosis.

Expected frozen input profiles:

| Dataset | Rows | label 0 | label 1 | SHA-256 |
|---|---:|---:|---:|---|
| D1 | 5933 | 2151 | 3782 | `b2ad99ebdcf3c34f040a2988fda1c482e9ff9b9cf42795bb64c792b3c7043d9e` |
| D2 | 7345 | 3563 | 3782 | `88161f9388194bbb9cede048223d8f7f6bc6ab68670c373eeffd13bbb45de842` |

The two benchmark variants share the same 3,782 positive texts. Obtain the source data through the project's documented acquisition/curation workflow, place the CSVs under `datasets/`, and verify the hashes before running experiments.
