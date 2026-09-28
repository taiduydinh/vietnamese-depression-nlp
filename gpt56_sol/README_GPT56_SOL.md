# GPT-5.6 Sol experiment artifacts

This directory contains the public, text-safe materials for the GPT-5.6 Sol experiment used in the Vietnamese depression-signal study. The task concerns **depression-related language signals**, not clinical diagnosis.

## Authoritative runs

- **D1:** complete frozen test set, n=1,187. Accuracy=0.937658, class-1 F1=0.948966, macro-F1=0.934439.
- **D2:** the original batch returned 1,330 valid rows and 139 requests failed when API credit was exhausted. Those exact 139 requests were recovered and merged. The authoritative final D2 result is n=1,469, accuracy=0.935330, class-1 F1=0.933982, macro-F1=0.935303.

Do **not** use the original partial D2 metrics for manuscript reporting; use `results/d2_final_metrics.json`.

## What is public here

- clean notebooks with code/markdown preserved and execution outputs removed;
- frozen prompt, topic codebook, output schema, and sanitized run configurations;
- final D1/D2 metrics and aggregate topic tables;
- text-free row-level predictions containing source-row identifiers and SHA-256 hashes of the original texts, enabling controlled merge/audit without redistributing the text;
- a recovery summary for the 139 D2 requests;
- hashes of excluded private batch artifacts.

## What is deliberately excluded

Raw batch inputs/outputs, manifests containing `text_vi`, text-bearing prediction files, pilot batches, batch/upload state files, and the merged raw recovery output are not published. They contain user-generated text or operational API metadata and are unnecessary for reproducing the paper tables.

## Topic taxonomy

The model returned one primary topic and an optional secondary topic from T01-T12, plus T00 as an operational fallback. These are literature-guided LLM annotations, **not ground-truth topic labels**. See `protocol/TOPIC_FRAMEWORK.md` and `protocol/topic_codebook_v1_4.csv`.

## Environment note

Notebook metadata records Python **3.10.15**. Exact historical package versions were not frozen in the original GPT notebooks, so `protocol/requirements_gpt56_sol.txt` lists dependencies without pretending to provide an exact lockfile. API pricing recorded in the run configurations is historical run metadata.
