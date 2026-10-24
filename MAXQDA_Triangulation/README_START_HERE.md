# MAXQDA × LDA × GPT — Reproducible publication figures

## Start

1. Extract the entire ZIP, keeping the `data/` and `outputs/` folders next to the notebook.
2. Open `MAXQDA_LDA_GPT_Publication_Figures.ipynb` in JupyterLab, VS Code, or classic Jupyter.
3. Use a Python 3.10+ environment with `numpy`, `matplotlib`, `ipython`, `ipykernel`, and `notebook` (install from `requirements.txt` if needed).
4. Run **Kernel → Restart & Run All**. Outputs will be written to the `outputs/` folder.

The notebook was executed and verified successfully before delivery. It requires **no internet access, no API key, and no pandas/openpyxl**. The `data/` folder contains portable CSVs extracted from the attached comparison workbook (using the spreadsheet import capability) and the self-contained MAXQDA QDPX text source, plus a reference copy of the original .xlsx.

## Files you can use in the manuscript

| File | Purpose |
|---|---|
| `outputs/figure_maxqda_lda_heatmap.pdf` | Primary 15 × 7 document-level topic/code heatmap |
| `outputs/figure_three_way_joint_display.pdf` | Seven-row LDA–MAXQDA–GPT descriptive joint display |
| `outputs/figure_bilingual_representative_quotations.pdf` | Exact Vietnamese coded passages with draft English translations |
| Corresponding `.svg`, `.png` | Vector and 300-dpi alternatives |
| `outputs/three_way_joint_display.csv` | Editable figure/table source |
| `outputs/representative_quotations_REVIEW_BEFORE_PUBLICATION.csv` | Verbatim coding spans, QDA IDs, offsets, and translations to verify |
| `outputs/lda_within_topic_percentages.csv` | Full 15 × 7 heatmap source |

## Study-specific constraints and interpretation

- Initial purposive Label 1 sample: **245**; excluded uncoded texts: **14**; retained coded sample: **231**.
- Human-developed 15-code framework; **523** coded passages were initially AI-assisted, then researcher-validated as reported by the research team. Earlier metadata in the supplied QDPX may still mention preliminary AI coding.
- Seven **dominant LDA topics** partition the 231 retained texts; MAXQDA theme assignments can be **many-to-many**.
- GPT subset: **210 source-specific annotation records** from **182 distinct texts**; **49 texts** have no GPT annotation. Do not equate record counts with person counts or independent documents.
- The heatmap's outlined cells indicate an **analyst-selected conceptual link**, not a statistical significance test or accuracy measure.
- Descriptive thematic overlaps are **not** clinical validation or independent human–model inter-rater agreement.
- Important sensitivity: dropping 14 uncoded texts changes some topic denominators. See the original-sample versus retained-sample comparison in the notebook.
- User-generated excerpts may contain sensitive health experiences and identifying context. Read and verify **all English translations** and anonymization/ethics suitability before publication.

## Customizing the charts

- Edit `anchor_pairs` to change conceptual LDA–MAXQDA mappings.
- Edit `primary_anchor` and `interpretation` to change the joint-display selection and interpretation (editorial decisions).
- Edit `QUOTATION_SELECTIONS` to choose different coded passages by QDA document ID and MAXQDA code name, and to improve the English translation.
- Edit plotting cells to change fonts, dimensions, colors, and export resolutions.

## Source files

- `data/source_comparison_workbook.xlsx`: uploaded, previously calculated 231-text triangulation workbook, not changed here.
- `data/maxqda_label1_231.qdpx`: final provided 231-text, 15-code MAXQDA exchange project, used to verify all quote excerpts against their original code-to-text spans.

This package produces figures and tables, **not** a revised manuscript.
