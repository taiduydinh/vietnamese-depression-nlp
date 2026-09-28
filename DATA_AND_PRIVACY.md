# Data, licensing, and privacy

The private research archive contains Vietnamese social-media text and derived text-bearing files. Those materials are not copied into this public-ready snapshot. Before publishing any raw dataset, translated text, or representative example, verify the source license/redistribution terms and the project's ethical/privacy approvals.

Files omitted for this reason include:

- `datasets/*.csv`
- `reproduction_results_v1/preprocessed/*.csv`
- `near_duplicate_audit_v1/internal_review/*`
- near-duplicate split-membership tables containing `text_vi`
- `lda_results_v2/internal_review/*`
- LDA tables containing `text_vi`, normalized text, or full document-topic rows with text

The repository retains input hashes, row counts, labels, aggregate metrics, source-row identifiers where useful for auditing, and a text-free `lda_gpt_comparison_representatives_only.csv` for later GPT–LDA alignment.
