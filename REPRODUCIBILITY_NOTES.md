# Reproducibility notes

- Raw dataset hashes, row counts, preprocessing counts, runtime versions, per-run metrics, predictions, and LDA provenance are retained.
- `underthesea==8.3.0` and `py_vncorenlp==0.1.4` are the versions used in the successful 2026 reproduction environment and reproduced the expected historical row counts. This snapshot does **not** independently establish that `underthesea==8.3.0` was the exact package version originally installed in 2025.
- PhoBERT-base in the executable legacy pipeline uses a frozen encoder; PhoBERT-large is configured for full fine-tuning.
- The current server reproduction produced 10/12 model–dataset results; PhoBERT-large reruns failed with GPU OOM under the shared-server memory situation. Archived/original PhoBERT-large metrics are retained separately in the comparison table.
- The near-duplicate sensitivity analysis removes flagged held-out examples post hoc. It should not be described as equivalent to a new group-aware training/test evaluation.
- LDA v2 model selection was frozen before the 60-run search and uses coherence one-SE eligibility followed by cross-seed stability. The quality gate is diagnostic only.
- Public-facing files intentionally omit raw/normalized social-media text and third-party binary resources.
