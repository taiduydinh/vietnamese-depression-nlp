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

See `reproduction_results/comparison_with_legacy.csv` for archived/original values and reproduction deltas.

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

## 4. GPT-5.6 Sol results

The final GPT results are included in public-safe form under `gpt56_sol/`.

| Dataset | n | Accuracy | Class-1 precision | Class-1 recall | Class-1 F1 | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 1187 | 0.9377 | 0.9928 | 0.9089 | 0.9490 | 0.9344 |
| D2 (final merged) | 1469 | 0.9353 | 0.9839 | 0.8889 | 0.9340 | 0.9353 |

The initial D2 batch had 139 quota-related failures and therefore only 1,330 valid outputs. Those exact requests were later recovered; `gpt56_sol/results/d2_final_metrics.json` and `d2_predictions_public.csv` are authoritative. The topic taxonomy is literature-guided LLM annotation, not ground-truth topic labels.

## 5. Text-length confounding and matched-test sensitivity

The saved text-length analysis found that the joint length-only logistic-regression baseline had approximately 63.9% accuracy (50.1% balanced accuracy) in D1 and 92.7% accuracy (92.8% balanced accuracy) in D2. The D2 matched-test subset contains 266 held-out texts, approximately 18.1% of its test set. Match-based results have reduced coverage and should not be presented as independent clinical validation.

Outputs: `length_confounds_results/length_only_baselines.csv`, `matched_test_metrics.csv`, and `RUN_MANIFEST.json`. The available data do not support a PhoBERT-large length-matched evaluation.

## 6. Researcher-reviewed qualitative analysis

The MAXQDA-based component reports 245 initially selected reference-positive texts, 14 uncoded exclusions, 231 retained coded texts, 15 thematic categories, and 523 coded segments. The MAXQDA–LDA–GPT comparisons describe thematic correspondences and differences; they do not establish diagnoses or independent inter-rater agreement.

Publication figures and comparison data reside in `MAXQDA_Triangulation/`. Text-bearing qualitative artifacts require a separate redistribution/privacy review.
