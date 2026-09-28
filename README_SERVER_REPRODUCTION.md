# Server reproduction package v1.1

This revision pins the preprocessing versions needed for legacy reproduction.

## Critical correction from v1.0

The archived preprocessing notebook was executed in October 2025 and installed
`underthesea` without a version pin. The stable release available then was 8.3.0.
The current 2026 environment had `underthesea==9.5.0`, which changed tokenization
enough to alter the historical TF-IDF usable-row counts.

Required preprocessing versions:

```text
underthesea==8.3.0
py_vncorenlp==0.1.4
```

Repair an existing environment:

```bash
conda activate depression_repro
export TF_USE_LEGACY_KERAS=1
python -m pip install "underthesea==8.3.0"
```

Verify:

```bash
python - <<'PY'
import underthesea, py_vncorenlp
print("underthesea:", underthesea.__version__)
print("py_vncorenlp:", py_vncorenlp.__version__)
PY
```

Then regenerate preprocessing:

```bash
python -u reproduce_all_previous_models.py \
  --project-root "$PWD" \
  --prepare-only \
  --force-preprocess
```

Expected legacy counts:

```text
TF-IDF D1 usable rows = 5921; test = 1185
TF-IDF D2 usable rows = 7249; test = 1450
DL/PhoBERT D1 test = 1187
DL/PhoBERT D2 test = 1469
```

Do not start model training until the strict preprocessing audit passes.
