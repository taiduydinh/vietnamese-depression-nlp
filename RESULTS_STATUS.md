# Results status

This snapshot contains the completed server-side reproduction/audit/LDA work used for the journal extension.

## 1. Legacy-model reproduction

Ten model–dataset runs completed and have per-run metrics/predictions. PhoBERT-large was not successfully rerun in the shared-server session because the training process hit GPU out-of-memory; the comparison table retains the archived/original PhoBERT-large values and leaves reproduced fields blank.

| Model | Dataset | n test | Accuracy | Class-1 F1 | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| PhoBERT-base | D1 | 1187 | 0.9756 | 0.9810 | 0.9734 |
| PhoBERT-base | D2 | 1469 | 0.9809 | 0.9815 | 0.9809 |
| TF-IDF + LR | D1 | 1185 | 0.9662 | 0.9730 | 0.9640 |
| TF-IDF + LR | D2 | 1450 | 0.9717 | 0.9725 | 0.9717 |
| TF-IDF + SVM | D1 | 1185 | 0.9713 | 0.9772 | 0.9692 |
| TF-IDF + SVM | D2 | 1450 | 0.9710 | 0.9720 | 0.9710 |
| Word2Vec + BiLSTM | D1 | 1187 | 0.9697 | 0.9762 | 0.9672 |
| Word2Vec + BiLSTM | D2 | 1469 | 0.9721 | 0.9730 | 0.9721 |
| Word2Vec + CNN | D1 | 1187 | 0.9789 | 0.9836 | 0.9771 |
| Word2Vec + CNN | D2 | 1469 | 0.9796 | 0.9803 | 0.9796 |

See `reproduction_results_v1/comparison_with_legacy.csv` for archived/original values and reproduction deltas.

## 2. Near-duplicate leakage audit

The audit completed successfully for both historical split protocols and thresholds 0.90, 0.95, and 0.98. At the primary 0.90 threshold, held-out rows with a training near-neighbor were approximately 1.7–2.5% depending on dataset/protocol. The post-hoc clean-test sensitivity changed reproduced accuracies only negligibly (all available model–dataset changes under 0.1 percentage points in absolute value). This is a sensitivity diagnostic, not a substitute for retraining with a group-aware split.

## 3. LDA v2

The final LDA v2 search completed 60/60 runs (`K=4..15`, five seeds each) under one frozen analysis signature. The predeclared selection rule chose:

- selected K: **7**
- selected seed: **82**
- best-coherence K: **11**
- coherence one-SE eligible K values: **7, 10, 11, 13**
- maximum dominant-topic share: **0.544**
- zero-dominant topics: **0**
- diagnostic quality gate: **PASS**

Qualitative review of internal representative documents supports five substantive themes plus two event-specific/residual social-media discourse topics. The internal text examples are not redistributed here.

Recommended descriptive labels used in the project analysis:

1. Medication and Psychiatric Treatment Management
2. Anxiety, Panic, and Somatic Symptoms
3. Psychosocial Stressors and Functional Impairment
4. Event-Specific Public Discourse
5. Help-Seeking, Mental-Health Information, and Community Support
6. Event-Specific Social-Media Commentary and Depression Trivialization
7. Severe Depressive Distress, Hopelessness, and Suicidality

## 4. GPT results

The GPT-5.6 Sol batch outputs are **not present in this server snapshot**. They must be added from the separate local GPT experiment archive before this becomes the complete paper repository.
