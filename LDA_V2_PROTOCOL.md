# Vietnamese Depression Project — LDA v2 Frozen Protocol

**Protocol version:** 2.0  
**Status:** Frozen before inspecting any LDA v2 topic solution.

## Scientific objective

LDA v2 is an **unsupervised thematic analysis** of the shared 3,782 depression-related (`sentiment=1`) Vietnamese texts. It is not a depression classifier and its topics are not ground-truth labels. The later comparison with GPT topics is a comparison of **thematic correspondence**, not topic-classification accuracy.

The v1 LDA run was useful diagnostically: it exposed repeated/template-like text families and a highly dominant broad topic. The separate near-duplicate leakage audit then showed that train–test near-duplicate overlap in the historical classification experiments was small and had negligible metric sensitivity. LDA v2 therefore changes the **topic-analysis corpus construction**, not the historical classifier results.

## Frozen v2 decisions

### 1. Input corpus

- Verify the original D1/D2 raw-file SHA-256 hashes.
- Verify audited preprocessing under `reproduction_results_v1/preprocessed/`.
- Verify D1 and D2 contain the same 3,782 positive texts.
- Use one canonical positive corpus in D1 row order.
- Require the previously audited count of **3,778 non-empty positive `text_clean_tfidf` texts** before further topic-specific processing.

### 2. Near-duplicate/template-family control

The primary family threshold is **cosine similarity >= 0.95**.

Rationale: 0.95 is the middle of the three predeclared thresholds (0.90/0.95/0.98) used in the independent leakage audit. For LDA v2 it is used as a conservative removal threshold for very close variants; 0.90 is retained as an audit threshold, not as the primary deletion rule.

Duplicate similarity follows the same normalization and representation as the leakage audit:

- HTML unescape;
- Unicode NFKC normalization;
- lowercase;
- remove URLs, emails, and `@mentions`;
- punctuation/symbols -> spaces;
- collapse whitespace;
- preserve Vietnamese diacritics and digits;
- `TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2, max_features=200000, sublinear_tf=True, norm='l2')`;
- fit the similarity space on all D1 raw rows, matching the audit design;
- fuzzy matching only for normalized texts with at least 30 characters;
- normalization-exact matches are eligible at all lengths.

To avoid single-link chaining, candidate connected components are refined with **complete-linkage clustering at the same 0.95 threshold**. Thus a chain of pairwise-near texts is not automatically collapsed into one large family when some family members are not mutually close.

For each final family, retain one deterministic **medoid representative**:

1. prefer members with non-empty audited `text_clean_tfidf`;
2. maximize mean within-family char-TFIDF cosine similarity;
3. break ties by the lower D1 source row ID.

All excluded members and raw texts are saved for internal review. No manual family deletion is allowed in the primary run.

### 3. Topic-model lexical processing

Start from audited `text_clean_tfidf` so Vietnamese segmentation and the legacy preprocessing remain traceable.

Additional generic topic-model cleanup is intentionally minimal:

- remove one-character tokens;
- remove pure-number tokens;
- remove obvious web residue tokens: `http`, `https`, `www`, `com`.

**No hand-built content stoplist is used.** In particular, the script does not manually remove terms related to depression, suicide, treatment, medication, school, relationships, brands, advertisements, or any theme observed in v1.

Dictionary filtering is frozen at:

- `no_below=5` documents;
- `no_above=0.40` document share.

The 0.40 upper-document-frequency filter is a topic-specific corpus-generic-term control. Every term removed by the lower/upper DF rules is saved in `dictionary_term_filter_audit.csv`.

### 4. LDA model search

Search:

- `K = 4..15`;
- seeds `[42, 52, 62, 72, 82]`;
- 60 total models;
- `gensim.models.LdaModel`;
- `passes=20`;
- `iterations=400`;
- `chunksize=512`;
- `alpha='auto'`;
- `eta='auto'`;
- `minimum_probability=0.0`;
- single-process C_v coherence calculation for deterministic/reproducible evaluation.

Metrics:

- C_v coherence;
- topic diversity among top-10 terms;
- log-perplexity (supplemental, not the primary selector);
- cross-seed stability via Hungarian-matched cosine similarity of full topic-word distributions.

### 5. K and seed selection

Use the same predeclared v1 selection rule:

1. identify the K with highest **mean C_v coherence** across the five seeds;
2. define the eligibility threshold as `best mean coherence - SE(best K)`;
3. retain K values meeting that threshold;
4. among eligible K values, choose the **highest mean cross-seed stability**;
5. tie-break by higher mean topic diversity, then smaller K;
6. within the selected K, choose the **medoid seed** (highest mean similarity to the other seeds), tie-breaking by coherence and then lower seed.

K is **not forced to 12** to match the GPT taxonomy.

### 6. Post-selection diagnostic quality gate

The quality gate is diagnostic and **does not alter K selection**.

Flag the selected solution for additional review when either:

- one dominant topic accounts for **>70%** of retained documents; or
- one or more selected topics are dominant for **zero** documents.

Passing the gate does not prove semantic validity. Topic labels are assigned only after reviewing both top terms and representative documents.

### 7. Primary versus sensitivity prevalence

Primary prevalence treats each retained representative/singleton equally, preventing large repeated families from dominating the topic model.

A separate **family-size-weighted prevalence sensitivity** is saved to show what the mean topic distribution would look like if the removed near-duplicate family multiplicities were restored. It is not the primary LDA prevalence estimate.

### 8. Privacy and reporting

Representative and family-member social-media texts are saved under `internal_review/`. They are for qualitative inspection only. Do not quote identifiable or sensitive user text in a publication without appropriate ethical/privacy review.

The final paper should clearly distinguish:

- historical supervised classification results;
- the near-duplicate leakage sensitivity audit;
- LDA v2 unsupervised thematic discovery;
- GPT topic assignments;
- later human validation.
