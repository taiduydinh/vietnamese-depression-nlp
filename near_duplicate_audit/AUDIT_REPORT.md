# Near-Duplicate / Historical Split Leakage Audit

Generated UTC: 2026-09-27T14:01:54.620072+00:00

## Scope

This audit reconstructs the exact historical held-out test memberships for the TF-IDF branch and the shared DL/PhoBERT outer split, then searches for normalization-exact and high-similarity raw-text neighbors crossing train→test. It does not alter the datasets or retrain models.

## Primary thresholds

0.90, 0.95, 0.98 cosine similarity.

Fuzzy matching uses char_wb TF-IDF on minimally normalized raw text and is restricted to texts with at least 30 normalized characters. Normalization-exact matches are counted at all lengths.

## Leakage summary

```text
 dataset         protocol  threshold  n_test  cross_split_pairs  test_rows_with_train_near_duplicate  test_leakage_rate  positive_test_rows_leaked  positive_test_leakage_rate  negative_test_rows_leaked  negative_test_leakage_rate  leaked_test_rows_cross_label_nearest
       1     tfidf_legacy       0.90    1185                 87                                   27           0.022785                         19                    0.025132                          8                    0.018648                                     0
       1     tfidf_legacy       0.95    1185                 62                                   23           0.019409                         15                    0.019841                          8                    0.018648                                     0
       1     tfidf_legacy       0.98    1185                 42                                   19           0.016034                         11                    0.014550                          8                    0.018648                                     0
       1 dl_phobert_outer       0.90    1187                104                                   30           0.025274                         23                    0.030383                          7                    0.016279                                     0
       1 dl_phobert_outer       0.95    1187                 76                                   22           0.018534                         18                    0.023778                          4                    0.009302                                     0
       1 dl_phobert_outer       0.98    1187                 56                                   18           0.015164                         14                    0.018494                          4                    0.009302                                     0
       2     tfidf_legacy       0.90    1450                 72                                   25           0.017241                         20                    0.026455                          5                    0.007205                                     0
       2     tfidf_legacy       0.95    1450                 52                                   19           0.013103                         14                    0.018519                          5                    0.007205                                     0
       2     tfidf_legacy       0.98    1450                 41                                   11           0.007586                          6                    0.007937                          5                    0.007205                                     0
       2 dl_phobert_outer       0.90    1469                 84                                   26           0.017699                         24                    0.031746                          2                    0.002805                                     0
       2 dl_phobert_outer       0.95    1469                 65                                   22           0.014976                         20                    0.026455                          2                    0.002805                                     0
       2 dl_phobert_outer       0.98    1469                 52                                   15           0.010211                         13                    0.017196                          2                    0.002805                                     0
```

## Existing-prediction sensitivity

When saved reproduction predictions are available, metrics are recomputed after excluding flagged test rows. This is a post-hoc sensitivity diagnostic, not a replacement for a group-aware train/test split and retraining.

```text
            model  dataset  threshold  n_test_all  n_test_clean  n_test_flagged  all_accuracy  clean_accuracy  clean_minus_all_accuracy  all_macro_f1  clean_macro_f1  clean_minus_all_macro_f1
     PhoBERT-base        1       0.90        1187          1157              30      0.975569        0.974935                 -0.000633      0.973413        0.972831                 -0.000582
     PhoBERT-base        1       0.95        1187          1165              22      0.975569        0.975107                 -0.000461      0.973413        0.973021                 -0.000392
     PhoBERT-base        1       0.98        1187          1169              18      0.975569        0.975192                 -0.000376      0.973413        0.973073                 -0.000340
     PhoBERT-base        2       0.90        1469          1443              26      0.980939        0.980596                 -0.000343      0.980920        0.980590                 -0.000330
     PhoBERT-base        2       0.95        1469          1447              22      0.980939        0.980650                 -0.000290      0.980920        0.980642                 -0.000278
     PhoBERT-base        2       0.98        1469          1454              15      0.980939        0.980743                 -0.000197      0.980920        0.980731                 -0.000189
      TF-IDF + LR        1       0.90        1185          1158              27      0.966245        0.965458                 -0.000787      0.963982        0.963214                 -0.000768
      TF-IDF + LR        1       0.95        1185          1162              23      0.966245        0.965577                 -0.000668      0.963982        0.963290                 -0.000692
      TF-IDF + LR        1       0.98        1185          1166              19      0.966245        0.965695                 -0.000550      0.963982        0.963366                 -0.000617
      TF-IDF + LR        2       0.90        1450          1425              25      0.971724        0.971228                 -0.000496      0.971699        0.971217                 -0.000482
      TF-IDF + LR        2       0.95        1450          1431              19      0.971724        0.971349                 -0.000375      0.971699        0.971333                 -0.000367
      TF-IDF + LR        2       0.98        1450          1439              11      0.971724        0.971508                 -0.000216      0.971699        0.971484                 -0.000216
     TF-IDF + SVM        1       0.90        1185          1158              27      0.971308        0.970639                 -0.000669      0.969244        0.968587                 -0.000657
     TF-IDF + SVM        1       0.95        1185          1162              23      0.971308        0.970740                 -0.000568      0.969244        0.968651                 -0.000593
     TF-IDF + SVM        1       0.98        1185          1166              19      0.971308        0.970840                 -0.000468      0.969244        0.968714                 -0.000530
     TF-IDF + SVM        2       0.90        1450          1425              25      0.971034        0.970526                 -0.000508      0.970997        0.970506                 -0.000491
     TF-IDF + SVM        2       0.95        1450          1431              19      0.971034        0.970650                 -0.000385      0.970997        0.970623                 -0.000374
     TF-IDF + SVM        2       0.98        1450          1439              11      0.971034        0.970813                 -0.000221      0.970997        0.970776                 -0.000221
Word2Vec + BiLSTM        1       0.90        1187          1157              30      0.969671        0.968885                 -0.000786      0.967246        0.966528                 -0.000718
Word2Vec + BiLSTM        1       0.95        1187          1165              22      0.969671        0.969099                 -0.000573      0.967246        0.966760                 -0.000486
Word2Vec + BiLSTM        1       0.98        1187          1169              18      0.969671        0.969204                 -0.000467      0.967246        0.966826                 -0.000420
Word2Vec + BiLSTM        2       0.90        1469          1443              26      0.972090        0.972280                  0.000190      0.972060        0.972270                  0.000210
Word2Vec + BiLSTM        2       0.95        1469          1447              22      0.972090        0.972357                  0.000267      0.972060        0.972344                  0.000284
Word2Vec + BiLSTM        2       0.98        1469          1454              15      0.972090        0.972490                  0.000400      0.972060        0.972471                  0.000411
   Word2Vec + CNN        1       0.90        1187          1157              30      0.978939        0.978392                 -0.000546      0.977128        0.976627                 -0.000501
   Word2Vec + CNN        1       0.95        1187          1165              22      0.978939        0.978541                 -0.000398      0.977128        0.976790                 -0.000338
   Word2Vec + CNN        1       0.98        1187          1169              18      0.978939        0.978614                 -0.000324      0.977128        0.976835                 -0.000293
   Word2Vec + CNN        2       0.90        1469          1443              26      0.979578        0.979903                  0.000325      0.979553        0.979894                  0.000341
   Word2Vec + CNN        2       0.95        1469          1447              22      0.979578        0.979959                  0.000381      0.979553        0.979947                  0.000394
   Word2Vec + CNN        2       0.98        1469          1454              15      0.979578        0.980055                  0.000477      0.979553        0.980039                  0.000486
```

## Interpretation guardrails

- A flagged pair is evidence of high textual similarity, not proof that two records are semantically identical.
- The 0.90/0.95/0.98 thresholds are reported together to avoid choosing a cutoff after seeing results.
- Cross-label near duplicates should be inspected as possible annotation/source conflicts.
- If substantial train→test family overlap is found, the publication-grade remedy is group-aware splitting and retraining, not merely deleting flagged test rows.
- D1 and D2 share the positive core, so they should not be presented as independent replications.
