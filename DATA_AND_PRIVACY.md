# Data, licensing, and privacy

The two raw benchmark datasets and several preprocessing artifacts are not distributed in the public repository. However, some publicly tracked qualitative materials have included MAXQDA projects, a QDPX exchange, and verbatim quotation files. Such artifacts can contain sensitive user-generated mental-health text and must **not** be presumed anonymous merely because the main raw CSV datasets are absent.

Before distributing any dataset, MAXQDA/QDPX project, workbook, translated passage, quotation, or derived text-bearing file, verify legal redistribution rights, institutional research-ethics requirements, and re-identification risk. If an artifact should not be publicly available, removing it from the latest working tree alone does not erase it from Git history or external copies.

Raw/private material expected to be excluded from public distribution includes:

- `datasets/*.csv`
- `reproduction_results/preprocessed/*.csv`
- `near_duplicate_audit/internal_review/*`
- near-duplicate split-membership tables containing `text_vi`
- `lda_results/internal_review/*`
- LDA tables containing `text_vi`, normalized text, or full document-topic rows with text

The repository retains input hashes, row counts, labels, aggregate metrics, source-row identifiers where useful for auditing, and a text-free `lda_gpt_comparison_representatives_only.csv` for later GPT–LDA alignment.

## GPT batch artifacts

The private GPT archive contains the original text in batch inputs, manifests, raw outputs, and prediction CSVs. Those files are intentionally excluded. Public GPT prediction files contain only dataset/source-row identifiers, labels/topics, and SHA-256 hashes of the original text. The hashes support reproducible matching without publishing the text itself.

## Qualitative artifacts requiring a separate release decision

Review these tracked locations before the next public push (and review existing Git history):

- `Vietnamese_Depression_Label1_231_Texts_15_Codes.mqda`
- `MAXQDA_Triangulation/data/maxqda_label1_231.qdpx`
- `MAXQDA_Triangulation/data/source_comparison_workbook.xlsx`
- `MAXQDA_Triangulation/outputs/representative_quotations_REVIEW_BEFORE_PUBLICATION.csv`
- The publication quotation figure and any other files containing verbatim passages

Aggregated thematic counts and topic-level summaries are generally less revealing than exact original posts, but release suitability must still be assessed. Hashes and row identifiers are pseudonymous technical identifiers, not a guarantee of anonymization.
