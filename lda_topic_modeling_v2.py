#!/usr/bin/env python3
"""
Vietnamese depression-signal project: LDA topic modeling v2.

Purpose
-------
Discover latent themes in the shared 3,782 positive/depression-related texts after a
conservative, reproducible near-duplicate/template-family control and topic-model-specific
lexical filtering. This is an unsupervised thematic analysis; LDA topics are not ground-truth
classes and later GPT-LDA analysis must be described as thematic correspondence, not accuracy.

Primary v2 protocol (defaults are frozen for the paper run)
-----------------------------------------------------------
1. Verify the two raw datasets by known SHA-256 hashes and verify row-aligned audited
   preprocessing from reproduction_results_v1/preprocessed/.
2. Verify that D1 and D2 share the same 3,782 positive texts.
3. Build one canonical positive corpus using D1 row order.
4. Recompute the SAME conservative duplicate-normalization and char_wb TF-IDF similarity
   representation used in the near-duplicate audit (fit on all D1 rows).
5. Collapse only very-high-similarity positive families at cosine >= 0.95. Fuzzy matching is
   limited to normalized texts with >=30 characters. Normalization-exact matches are eligible
   at all lengths. Candidate graph components are refined with complete-linkage clustering at
   the same 0.95 threshold to prevent single-link chaining. One deterministic medoid is retained
   per family, preferring a member with non-empty audited text_clean_tfidf.
6. Start lexical topic modeling from the audited legacy text_clean_tfidf representation; apply
   only generic topic-model cleanup (remove pure-number, one-character, and obvious web-residue
   tokens). NO hand-built removal of depression, treatment, school, relationship, brand, or other
   content terms is performed.
7. Dictionary filtering: no_below=5 and no_above=0.40. Every removed term and excluded document
   is logged.
8. Search K=4..15 with seeds 42,52,62,72,82 using gensim LdaModel, passes=20,
   iterations=400, alpha='auto', eta='auto'.
9. Evaluate C_v coherence, topic diversity, log-perplexity, and cross-seed stability using
   Hungarian-matched cosine similarity of full topic-word distributions.
10. Select K using the same predeclared one-standard-error coherence rule as v1, then highest
    mean stability; choose the final seed as the medoid seed.
11. Apply a POST-SELECTION diagnostic quality gate (not a K-selection criterion): warn if a
    single topic dominates >70% of retained documents or if any selected topic is dominant for
    zero documents. The gate prevents silently declaring a degenerate solution final.
12. Save all corpus decisions, family evidence, model-search metrics, selected model, topic terms,
    prevalence, representative/ambiguous documents, figures, provenance, and checksums.

The script is CPU-only. It is resumable: completed LDA runs are reused only when their recorded
analysis signature exactly matches the current corpus and frozen settings.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gensim
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel


SCRIPT_VERSION = "2.0"

EXPECTED_SHA256 = {
    1: "b2ad99ebdcf3c34f040a2988fda1c482e9ff9b9cf42795bb64c792b3c7043d9e",
    2: "88161f9388194bbb9cede048223d8f7f6bc6ab68670c373eeffd13bbb45de842",
}
EXPECTED_RAW_ROWS = {1: 5933, 2: 7345}
EXPECTED_LABEL_COUNTS = {
    1: {0: 2151, 1: 3782},
    2: {0: 3563, 1: 3782},
}
EXPECTED_POSITIVE_ROWS = 3782
EXPECTED_POSITIVE_NONEMPTY_LEGACY = 3778  # verified on the audited server preprocessing
EXPECTED_DL_TEST_POSITIVES = {1: 757, 2: 756}

TEXT_COL = "text_vi"
LABEL_COL = "sentiment"
CLEAN_COL = "text_clean_tfidf"

# Frozen primary v2 protocol.
DEFAULT_DEDUP_THRESHOLD = 0.95
DEFAULT_MIN_CHARS = 30
DEFAULT_CHAR_NGRAM_RANGE = (3, 5)
DEFAULT_MAX_FEATURES = 200_000
DEFAULT_NO_BELOW = 5
DEFAULT_NO_ABOVE = 0.40
DEFAULT_K_VALUES = list(range(4, 16))
DEFAULT_SEEDS = [42, 52, 62, 72, 82]
DEFAULT_PASSES = 20
DEFAULT_ITERATIONS = 400
DEFAULT_CHUNKSIZE = 512
DEFAULT_TOP_TERMS = 15
DEFAULT_DIVERSITY_TOPN = 10
DEFAULT_REP_DOCS = 10
DEFAULT_AMBIGUOUS_DOCS = 100
DEFAULT_DOMINANCE_WARNING_THRESHOLD = 0.70

URL_RE = re.compile(
    r"(?:https?://\S+|www\.\S+|\bhttps?\b\s*(?:[:/._-]\s*|\s+)[a-z0-9][a-z0-9\s._/-]{2,})",
    flags=re.IGNORECASE,
)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b")
MENTION_RE = re.compile(r"(?<!\w)@[\w_.-]+")
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
WHITESPACE_RE = re.compile(r"\s+")
PURE_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:[.,]\d+)*)$")
WEB_RESIDUE = {"http", "https", "www", "com"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json_hash(obj: object) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def normalize_for_duplicate_control(text: str) -> str:
    """Exact normalization copied from the near-duplicate audit protocol."""
    s = html.unescape(str(text))
    s = unicodedata.normalize("NFKC", s)
    s = ZERO_WIDTH_RE.sub("", s)
    s = s.lower()
    s = URL_RE.sub(" ", s)
    s = EMAIL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            out.append(" ")
        else:
            out.append(ch)
    return WHITESPACE_RE.sub(" ", "".join(out)).strip()


def topic_token_cleanup(text: str) -> Tuple[List[str], Dict[str, int]]:
    """Generic lexical cleanup only; no hand-built content/topic stoplist."""
    kept: List[str] = []
    stats = Counter()
    for raw_tok in str(text).split():
        tok = unicodedata.normalize("NFKC", raw_tok).strip().lower()
        if not tok:
            stats["empty"] += 1
            continue
        if len(tok) < 2:
            stats["one_character"] += 1
            continue
        if PURE_NUMBER_RE.fullmatch(tok):
            stats["pure_number"] += 1
            continue
        if tok in WEB_RESIDUE:
            stats["web_residue"] += 1
            continue
        kept.append(tok)
        stats["kept"] += 1
    return kept, dict(stats)


class DSU:
    def __init__(self, items: Iterable[int]):
        self.parent = {int(x): int(x) for x in items}
        self.rank = {int(x): 0 for x in items}

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(int(a)), self.find(int(b))
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--results-dir", default="lda_results_v2")
    p.add_argument("--n-jobs", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--prepare-only", action="store_true", help="Stop after corpus/family/dictionary audit.")
    p.add_argument("--force-search", action="store_true", help="Retrain LDA search models even if matching checkpoints exist.")
    p.add_argument("--dedup-threshold", type=float, default=DEFAULT_DEDUP_THRESHOLD)
    p.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    p.add_argument("--no-below", type=int, default=DEFAULT_NO_BELOW)
    p.add_argument("--no-above", type=float, default=DEFAULT_NO_ABOVE)
    return p.parse_args()


class LDAV2:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = args.project_root.resolve()
        self.results = self.root / args.results_dir
        self.tables = self.results / "tables"
        self.figures = self.results / "figures"
        self.models = self.results / "models"
        self.search_models = self.models / "search"
        self.provenance = self.results / "provenance"
        self.internal = self.results / "internal_review"
        for d in [self.results, self.tables, self.figures, self.models, self.search_models, self.provenance, self.internal]:
            d.mkdir(parents=True, exist_ok=True)

        if not (0.0 < args.dedup_threshold <= 1.0):
            raise ValueError("--dedup-threshold must be in (0,1].")
        if not (0.0 < args.no_above <= 1.0):
            raise ValueError("--no-above must be in (0,1].")
        if args.no_below < 1:
            raise ValueError("--no-below must be >=1.")

        self.raw: Dict[int, pd.DataFrame] = {}
        self.pre: Dict[int, pd.DataFrame] = {}
        self.canonical: pd.DataFrame | None = None
        self.family_decisions: pd.DataFrame | None = None
        self.retained: pd.DataFrame | None = None
        self.model_docs: pd.DataFrame | None = None
        self.model_tokens: List[List[str]] = []
        self.model_bow: List[List[Tuple[int, int]]] = []
        self.dictionary: Dictionary | None = None
        self.analysis_signature: str | None = None
        self.corpus_sha: str | None = None
        self.audit_reference: Dict[str, object] = {}

        self.K_VALUES = DEFAULT_K_VALUES
        self.SEEDS = DEFAULT_SEEDS
        self.PASSES = DEFAULT_PASSES
        self.ITERATIONS = DEFAULT_ITERATIONS
        self.CHUNKSIZE = DEFAULT_CHUNKSIZE

    # ---------- input and provenance ----------
    def write_runtime(self) -> None:
        runtime = {
            "created_at_utc": utc_now(),
            "script_version": SCRIPT_VERSION,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "gensim": gensim.__version__,
            "matplotlib": matplotlib.__version__,
            "cpu_count": os.cpu_count(),
            "n_jobs_near_duplicate": self.args.n_jobs,
        }
        (self.provenance / "runtime.json").write_text(json.dumps(runtime, indent=2, ensure_ascii=False), encoding="utf-8")

    def audit_inputs(self) -> None:
        log("Auditing raw datasets and audited preprocessing...")
        rows = []
        for d in [1, 2]:
            raw_path = self.root / "datasets" / f"dataset_{d}.csv"
            pre_path = self.root / "reproduction_results_v1" / "preprocessed" / f"Dataset_depression_vi{d}.csv"
            if not raw_path.exists():
                raise FileNotFoundError(raw_path)
            if not pre_path.exists():
                raise FileNotFoundError(f"Missing audited preprocessing file: {pre_path}")

            digest = sha256_file(raw_path)
            if digest != EXPECTED_SHA256[d]:
                raise RuntimeError(f"D{d} raw SHA-256 mismatch: {digest} != {EXPECTED_SHA256[d]}")
            raw = pd.read_csv(raw_path)
            if list(raw.columns) != [TEXT_COL, LABEL_COL]:
                raise RuntimeError(f"D{d} raw schema changed: {raw.columns.tolist()}")
            if len(raw) != EXPECTED_RAW_ROWS[d]:
                raise RuntimeError(f"D{d} raw rows {len(raw)} != {EXPECTED_RAW_ROWS[d]}")
            if raw[[TEXT_COL, LABEL_COL]].isna().any().any():
                raise RuntimeError(f"D{d} raw data contain missing text/label values")
            raw[LABEL_COL] = raw[LABEL_COL].astype(int)
            labels = raw[LABEL_COL].value_counts().sort_index().to_dict()
            if labels != EXPECTED_LABEL_COUNTS[d]:
                raise RuntimeError(f"D{d} label counts changed: {labels} != {EXPECTED_LABEL_COUNTS[d]}")
            raw = raw.copy()
            raw["source_row_id"] = raw.index.astype(int)

            pre = pd.read_csv(pre_path)
            required = {TEXT_COL, LABEL_COL, CLEAN_COL}
            if not required.issubset(pre.columns):
                raise RuntimeError(f"D{d} preprocessed file missing columns: {sorted(required - set(pre.columns))}")
            if len(pre) != len(raw):
                raise RuntimeError(f"D{d} preprocessed rows {len(pre)} != raw rows {len(raw)}")
            if not pre[TEXT_COL].fillna("").astype(str).equals(raw[TEXT_COL].fillna("").astype(str)):
                raise RuntimeError(f"D{d} preprocessed text_vi is not row-aligned with raw data")
            if not pre[LABEL_COL].astype(int).equals(raw[LABEL_COL].astype(int)):
                raise RuntimeError(f"D{d} labels changed during preprocessing")
            pre = pre.copy()
            pre["source_row_id"] = pre.index.astype(int)
            pre[LABEL_COL] = pre[LABEL_COL].astype(int)

            self.raw[d], self.pre[d] = raw, pre
            rows.append({
                "dataset": d,
                "rows": len(raw),
                "label_0": labels[0],
                "label_1": labels[1],
                "raw_sha256": digest,
                "raw_exact_text_duplicate_rows": int(raw.duplicated(subset=[TEXT_COL], keep=False).sum()),
            })

        pd.DataFrame(rows).to_csv(self.tables / "raw_dataset_audit.csv", index=False)

        pos1 = self.raw[1].loc[self.raw[1][LABEL_COL].eq(1), TEXT_COL].astype(str).tolist()
        pos2 = self.raw[2].loc[self.raw[2][LABEL_COL].eq(1), TEXT_COL].astype(str).tolist()
        if len(pos1) != EXPECTED_POSITIVE_ROWS or len(pos2) != EXPECTED_POSITIVE_ROWS:
            raise RuntimeError(f"Positive count mismatch: D1={len(pos1)} D2={len(pos2)}")
        if Counter(pos1) != Counter(pos2):
            raise RuntimeError("D1 and D2 positive corpora are not identical multisets")
        if len(set(pos1)) != len(pos1):
            raise RuntimeError("Unexpected exact duplicate positive raw texts; mapping assumptions would be ambiguous")

        # Verify cleaned representations for the shared positive core.
        p1 = self.pre[1].loc[self.pre[1][LABEL_COL].eq(1), [TEXT_COL, CLEAN_COL]]
        p2 = self.pre[2].loc[self.pre[2][LABEL_COL].eq(1), [TEXT_COL, CLEAN_COL]]
        m1 = dict(zip(p1[TEXT_COL].astype(str), p1[CLEAN_COL].fillna("").astype(str)))
        m2 = dict(zip(p2[TEXT_COL].astype(str), p2[CLEAN_COL].fillna("").astype(str)))
        if m1.keys() != m2.keys():
            raise RuntimeError("Positive text keys differ after preprocessing")
        mismatches = [t for t in m1 if m1[t] != m2[t]]
        if mismatches:
            raise RuntimeError(f"{len(mismatches)} shared positive texts have different cleaned representations across D1/D2")

        log("RAW / SHARED-POSITIVE / PREPROCESSING AUDIT: PASS")

        audit_cfg = self.root / "near_duplicate_audit_v1" / "provenance" / "audit_configuration.json"
        if audit_cfg.exists():
            try:
                cfg = json.loads(audit_cfg.read_text(encoding="utf-8"))
                self.audit_reference = {
                    "path": str(audit_cfg),
                    "sha256": sha256_file(audit_cfg),
                    "thresholds": cfg.get("thresholds"),
                    "min_chars": cfg.get("min_chars"),
                    "char_analyzer": cfg.get("char_analyzer"),
                    "char_ngram_range": cfg.get("char_ngram_range"),
                    "max_features": cfg.get("max_features"),
                }
                log("Found near_duplicate_audit_v1 configuration; recorded it as provenance reference.")
            except Exception as exc:
                log(f"WARNING: could not parse near-duplicate audit configuration: {exc}")

    def build_canonical(self) -> None:
        log("Building canonical shared positive corpus...")
        raw1, pre1 = self.raw[1], self.pre[1]
        pos_mask = raw1[LABEL_COL].eq(1)
        d2_text_to_idx = {
            str(text): int(idx)
            for idx, text in self.raw[2].loc[self.raw[2][LABEL_COL].eq(1), TEXT_COL].items()
        }
        canonical = pd.DataFrame({
            "dataset1_source_row_id": raw1.loc[pos_mask, "source_row_id"].astype(int).to_numpy(),
            TEXT_COL: raw1.loc[pos_mask, TEXT_COL].astype(str).to_numpy(),
            CLEAN_COL: pre1.loc[pos_mask, CLEAN_COL].fillna("").astype(str).to_numpy(),
        })
        canonical["dataset2_source_row_id"] = canonical[TEXT_COL].map(d2_text_to_idx).astype(int)
        canonical["text_sha256"] = canonical[TEXT_COL].map(sha256_text)
        canonical["legacy_clean_nonempty"] = canonical[CLEAN_COL].str.strip().ne("")
        canonical["duplicate_normalized_text"] = canonical[TEXT_COL].map(normalize_for_duplicate_control)
        canonical["duplicate_normalized_chars"] = canonical["duplicate_normalized_text"].str.len().astype(int)
        canonical["duplicate_normalized_sha256"] = canonical["duplicate_normalized_text"].map(sha256_text)

        # Frozen raw 80/20 test memberships used by DL/PhoBERT/GPT for later merge convenience.
        from sklearn.model_selection import train_test_split
        for d in [1, 2]:
            idx = np.arange(len(self.raw[d]))
            _, test_idx = train_test_split(
                idx,
                test_size=0.2,
                random_state=42,
                stratify=self.raw[d][LABEL_COL].to_numpy(),
            )
            test_set = set(map(int, test_idx))
            col = "dataset1_source_row_id" if d == 1 else "dataset2_source_row_id"
            canonical[f"in_d{d}_frozen_test"] = canonical[col].isin(test_set)
            got = int(canonical[f"in_d{d}_frozen_test"].sum())
            if got != EXPECTED_DL_TEST_POSITIVES[d]:
                raise RuntimeError(f"D{d} frozen-test positive count {got} != {EXPECTED_DL_TEST_POSITIVES[d]}")

        nonempty = int(canonical["legacy_clean_nonempty"].sum())
        if len(canonical) != EXPECTED_POSITIVE_ROWS:
            raise RuntimeError(f"Canonical positive rows {len(canonical)} != {EXPECTED_POSITIVE_ROWS}")
        if nonempty != EXPECTED_POSITIVE_NONEMPTY_LEGACY:
            raise RuntimeError(
                f"Expected {EXPECTED_POSITIVE_NONEMPTY_LEGACY} non-empty positive text_clean_tfidf rows; got {nonempty}. "
                "Do not continue until preprocessing drift is explained."
            )
        canonical.to_csv(self.tables / "canonical_positive_corpus_3782.csv", index=False)
        canonical.loc[~canonical["legacy_clean_nonempty"]].to_csv(
            self.tables / "excluded_empty_after_legacy_preprocessing.csv", index=False
        )
        self.canonical = canonical
        log(f"CANONICAL CORPUS AUDIT: PASS ({len(canonical)} rows; {nonempty} non-empty legacy-clean texts)")

    # ---------- near-duplicate family control ----------
    def build_family_control(self) -> None:
        assert self.canonical is not None
        log(
            f"Building conservative near-duplicate/template families at cosine >= {self.args.dedup_threshold:.2f} "
            f"with complete-link refinement..."
        )
        d1 = self.raw[1]
        all_norm = d1[TEXT_COL].astype(str).map(normalize_for_duplicate_control).tolist()
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=DEFAULT_CHAR_NGRAM_RANGE,
            min_df=2,
            max_features=DEFAULT_MAX_FEATURES,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        X_all = vectorizer.fit_transform(all_norm)
        (self.provenance / "duplicate_similarity_space.json").write_text(
            json.dumps({
                "fit_corpus": "all D1 raw rows, matching near-duplicate audit design",
                "n_documents": int(X_all.shape[0]),
                "n_features": int(X_all.shape[1]),
                "nnz": int(X_all.nnz),
                "analyzer": "char_wb",
                "ngram_range": list(DEFAULT_CHAR_NGRAM_RANGE),
                "min_df": 2,
                "max_features": DEFAULT_MAX_FEATURES,
                "sublinear_tf": True,
                "norm": "l2",
                "dedup_threshold": self.args.dedup_threshold,
                "min_chars_for_fuzzy": self.args.min_chars,
            }, indent=2), encoding="utf-8"
        )

        pos_ids = self.canonical["dataset1_source_row_id"].astype(int).to_numpy()
        norm_by_id = {int(i): all_norm[int(i)] for i in pos_ids}
        dsu = DSU(pos_ids)
        candidate_edges: Dict[Tuple[int, int], Dict[str, object]] = {}

        # Normalization-exact pairs at all lengths; use star edges to avoid quadratic output.
        exact_groups: Dict[str, List[int]] = defaultdict(list)
        for rid in pos_ids:
            s = norm_by_id[int(rid)]
            if s:
                exact_groups[s].append(int(rid))
        for members in exact_groups.values():
            if len(members) >= 2:
                anchor = members[0]
                for other in members[1:]:
                    a, b = sorted((anchor, other))
                    dsu.union(a, b)
                    candidate_edges[(a, b)] = {"similarity": 1.0, "kind": "normalized_exact"}

        eligible = np.array([rid for rid in pos_ids if len(norm_by_id[int(rid)]) >= self.args.min_chars], dtype=int)
        if len(eligible) >= 2:
            nn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=self.args.n_jobs)
            nn.fit(X_all[eligible])
            radius = max(0.0, 1.0 - self.args.dedup_threshold + 1e-8)
            distances, indices = nn.radius_neighbors(X_all[eligible], radius=radius, return_distance=True, sort_results=True)
            for q, rid in enumerate(eligible):
                for dist, local_idx in zip(distances[q], indices[q]):
                    other = int(eligible[int(local_idx)])
                    if other == int(rid):
                        continue
                    sim = float(np.clip(1.0 - float(dist), -1.0, 1.0))
                    if sim + 1e-8 < self.args.dedup_threshold:
                        continue
                    a, b = sorted((int(rid), other))
                    prev = candidate_edges.get((a, b))
                    if prev is None or sim > float(prev["similarity"]):
                        candidate_edges[(a, b)] = {"similarity": sim, "kind": "fuzzy_char_tfidf"}
                    dsu.union(a, b)

        # Candidate connected components, then complete-link refinement to prevent transitive chaining.
        comps: Dict[int, List[int]] = defaultdict(list)
        for rid in pos_ids:
            comps[dsu.find(int(rid))].append(int(rid))
        candidate_components = [sorted(m) for m in comps.values() if len(m) >= 2]

        final_families: List[List[int]] = []
        for members in candidate_components:
            if len(members) == 2:
                final_families.append(members)
                continue
            sim = cosine_similarity(X_all[members], dense_output=True)
            # Force normalization-exact relations to similarity 1, including very short/zero-feature cases.
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if norm_by_id[members[i]] and norm_by_id[members[i]] == norm_by_id[members[j]]:
                        sim[i, j] = sim[j, i] = 1.0
            np.fill_diagonal(sim, 1.0)
            dist = np.clip(1.0 - sim, 0.0, 2.0)
            model = AgglomerativeClustering(
                n_clusters=None,
                metric="precomputed",
                linkage="complete",
                distance_threshold=max(0.0, 1.0 - self.args.dedup_threshold + 1e-10),
                compute_full_tree=True,
            )
            labels = model.fit_predict(dist)
            for lab in sorted(set(map(int, labels))):
                sub = [members[i] for i, x in enumerate(labels) if int(x) == lab]
                if len(sub) >= 2:
                    final_families.append(sorted(sub))

        final_families = sorted(final_families, key=lambda xs: min(xs))
        canonical_by_d1 = self.canonical.set_index("dataset1_source_row_id", drop=False)
        family_rows = []
        member_rows = []
        rep_by_row: Dict[int, int] = {int(r): int(r) for r in pos_ids}
        family_id_by_row: Dict[int, str] = {int(r): "" for r in pos_ids}

        for n, members in enumerate(final_families, start=1):
            fid = f"F{n:04d}"
            sim = cosine_similarity(X_all[members], dense_output=True)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if norm_by_id[members[i]] and norm_by_id[members[i]] == norm_by_id[members[j]]:
                        sim[i, j] = sim[j, i] = 1.0
            np.fill_diagonal(sim, 1.0)

            # Prefer a family representative that survives legacy lexical preprocessing.
            nonempty_idx = [i for i, rid in enumerate(members) if bool(canonical_by_d1.at[rid, "legacy_clean_nonempty"])]
            candidates = nonempty_idx if nonempty_idx else list(range(len(members)))
            mean_sim = sim.mean(axis=1)
            best_i = sorted(candidates, key=lambda i: (-float(mean_sim[i]), int(members[i])))[0]
            rep = int(members[best_i])

            offdiag = sim[np.triu_indices(len(members), k=1)]
            if len(offdiag) and float(offdiag.min()) + 1e-6 < self.args.dedup_threshold:
                raise RuntimeError(
                    f"Complete-link family audit failed for {fid}: minimum pair similarity "
                    f"{float(offdiag.min()):.6f} < threshold {self.args.dedup_threshold:.6f}"
                )
            family_rows.append({
                "family_id": fid,
                "family_size": len(members),
                "representative_dataset1_source_row_id": rep,
                "min_pair_similarity": float(offdiag.min()) if len(offdiag) else 1.0,
                "mean_pair_similarity": float(offdiag.mean()) if len(offdiag) else 1.0,
                "max_pair_similarity": float(offdiag.max()) if len(offdiag) else 1.0,
                "all_members_legacy_clean_nonempty": bool(all(bool(canonical_by_d1.at[r, "legacy_clean_nonempty"]) for r in members)),
                "representative_legacy_clean_nonempty": bool(canonical_by_d1.at[rep, "legacy_clean_nonempty"]),
                "member_row_ids": ";".join(map(str, members)),
            })
            for rid in members:
                rep_by_row[rid] = rep
                family_id_by_row[rid] = fid
                member_rows.append({
                    "family_id": fid,
                    "family_size": len(members),
                    "is_representative": int(rid == rep),
                    "representative_dataset1_source_row_id": rep,
                    "dataset1_source_row_id": rid,
                    "dataset2_source_row_id": int(canonical_by_d1.at[rid, "dataset2_source_row_id"]),
                    "legacy_clean_nonempty": bool(canonical_by_d1.at[rid, "legacy_clean_nonempty"]),
                    "text_sha256": canonical_by_d1.at[rid, "text_sha256"],
                    TEXT_COL: canonical_by_d1.at[rid, TEXT_COL],
                    CLEAN_COL: canonical_by_d1.at[rid, CLEAN_COL],
                })

        family_summary = pd.DataFrame(family_rows)
        family_members = pd.DataFrame(member_rows)
        family_summary.to_csv(self.tables / "near_duplicate_family_summary.csv", index=False)
        family_members.to_csv(self.internal / "near_duplicate_family_members_INTERNAL.csv.gz", index=False, compression="gzip")

        decisions = self.canonical.copy()
        decisions["near_duplicate_family_id"] = decisions["dataset1_source_row_id"].map(family_id_by_row)
        decisions["representative_dataset1_source_row_id"] = decisions["dataset1_source_row_id"].map(rep_by_row).astype(int)
        decisions["is_family_representative"] = decisions["dataset1_source_row_id"].eq(decisions["representative_dataset1_source_row_id"])
        decisions["excluded_as_near_duplicate_nonrepresentative"] = ~decisions["is_family_representative"]
        decisions["retained_after_family_control"] = decisions["is_family_representative"]
        decisions["family_size"] = decisions["representative_dataset1_source_row_id"].map(
            decisions.groupby("representative_dataset1_source_row_id").size().to_dict()
        ).astype(int)
        decisions.to_csv(self.tables / "near_duplicate_family_decisions.csv", index=False)

        candidate_edge_df = pd.DataFrame([
            {"row_a": a, "row_b": b, **meta} for (a, b), meta in sorted(candidate_edges.items())
        ])
        candidate_edge_df.to_csv(self.internal / "near_duplicate_candidate_edges_INTERNAL.csv.gz", index=False, compression="gzip")

        retained = decisions.loc[decisions["retained_after_family_control"]].copy().reset_index(drop=True)
        self.family_decisions = decisions
        self.retained = retained

        n_family_docs = int(decisions["near_duplicate_family_id"].ne("").sum())
        n_families = len(family_summary)
        n_removed = int(decisions["excluded_as_near_duplicate_nonrepresentative"].sum())
        max_family = int(family_summary["family_size"].max()) if len(family_summary) else 1
        control_summary = {
            "created_at_utc": utc_now(),
            "threshold": self.args.dedup_threshold,
            "min_chars_for_fuzzy": self.args.min_chars,
            "candidate_graph_edges": len(candidate_edges),
            "candidate_components_size_ge_2": len(candidate_components),
            "final_complete_link_families": n_families,
            "documents_in_families": n_family_docs,
            "nonrepresentatives_removed": n_removed,
            "documents_retained_after_family_control": len(retained),
            "largest_family_size": max_family,
            "representative_rule": "prefer legacy-clean-nonempty member; then highest within-family mean char-TFIDF cosine similarity; tie lower D1 row id",
            "complete_link_refinement": True,
            "audit_reference": self.audit_reference,
        }
        (self.provenance / "near_duplicate_family_control.json").write_text(
            json.dumps(control_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log(
            f"FAMILY CONTROL: {n_families} families; {n_removed} nonrepresentative rows removed; "
            f"{len(retained)} representatives/singletons retained."
        )

    # ---------- topic lexical corpus ----------
    def build_topic_corpus(self) -> None:
        assert self.retained is not None
        log("Applying generic topic-model lexical cleanup and dictionary filtering...")
        retained = self.retained.copy()
        legacy_empty = retained.loc[~retained["legacy_clean_nonempty"]].copy()
        lexical = retained.loc[retained["legacy_clean_nonempty"]].copy().reset_index(drop=True)
        legacy_empty.to_csv(self.tables / "excluded_representatives_empty_after_legacy_preprocessing.csv", index=False)

        token_lists: List[List[str]] = []
        cleanup_stats = Counter()
        for text in lexical[CLEAN_COL].astype(str):
            toks, stats = topic_token_cleanup(text)
            token_lists.append(toks)
            cleanup_stats.update(stats)
        lexical["topic_tokens_before_dictionary"] = [" ".join(t) for t in token_lists]
        pre_dict_nonempty = np.array([len(t) > 0 for t in token_lists], dtype=bool)
        empty_generic = lexical.loc[~pre_dict_nonempty].copy()
        empty_generic.to_csv(self.tables / "excluded_empty_after_generic_topic_cleanup.csv", index=False)
        lexical2 = lexical.loc[pre_dict_nonempty].copy().reset_index(drop=True)
        tokens2 = [t for t, keep in zip(token_lists, pre_dict_nonempty) if keep]

        dictionary = Dictionary(tokens2)
        vocab_before = len(dictionary)
        # Record document frequencies BEFORE filter_extremes.
        token_df_rows = []
        n_docs = len(tokens2)
        for token_id, token in dictionary.iteritems():
            df = int(dictionary.dfs.get(token_id, 0))
            share = df / n_docs if n_docs else np.nan
            if df < self.args.no_below:
                action = "remove_low_df"
            elif share > self.args.no_above:
                action = "remove_high_df"
            else:
                action = "keep"
            token_df_rows.append({"term": token, "document_frequency": df, "document_share": share, "dictionary_action": action})
        token_df = pd.DataFrame(token_df_rows).sort_values(["document_frequency", "term"], ascending=[False, True])
        token_df.to_csv(self.tables / "dictionary_term_filter_audit.csv", index=False)

        dictionary.filter_extremes(no_below=self.args.no_below, no_above=self.args.no_above, keep_n=None)
        dictionary.compactify()
        vocab_after = len(dictionary)
        if vocab_after == 0:
            raise RuntimeError("Dictionary is empty after filtering")

        bows = [dictionary.doc2bow(t) for t in tokens2]
        bow_nonempty = np.array([len(b) > 0 for b in bows], dtype=bool)
        empty_dict = lexical2.loc[~bow_nonempty].copy()
        empty_dict.to_csv(self.tables / "excluded_empty_after_dictionary_filtering.csv", index=False)
        model_docs = lexical2.loc[bow_nonempty].copy().reset_index(drop=True)
        model_tokens = [t for t, keep in zip(tokens2, bow_nonempty) if keep]
        model_bow = [b for b, keep in zip(bows, bow_nonempty) if keep]
        if len(model_docs) == 0:
            raise RuntimeError("No documents remain for LDA")

        dictionary.save(str(self.models / "lda_dictionary.gensim"))
        self.dictionary = dictionary
        self.model_docs = model_docs
        self.model_tokens = model_tokens
        self.model_bow = model_bow

        flow = pd.DataFrame([
            {"stage": "shared_positive_raw", "documents": len(self.canonical)},
            {"stage": "after_near_duplicate_family_control", "documents": len(retained)},
            {"stage": "legacy_clean_nonempty_representatives", "documents": len(lexical)},
            {"stage": "generic_topic_cleanup_nonempty", "documents": len(lexical2)},
            {"stage": "final_lda_documents_after_dictionary", "documents": len(model_docs)},
        ])
        flow.to_csv(self.tables / "lda_v2_corpus_flow.csv", index=False)

        clean_payload = "\n".join(
            f"{int(r.dataset1_source_row_id)}\t{r.text_sha256}\t{' '.join(model_tokens[i])}"
            for i, (_, r) in enumerate(model_docs.iterrows())
        )
        self.corpus_sha = sha256_text(clean_payload)

        dictionary_hash = stable_json_hash(sorted(dictionary.token2id.items(), key=lambda x: x[1]))
        corpus_stats = {
            "shared_positive_raw": int(len(self.canonical)),
            "retained_after_family_control": int(len(retained)),
            "excluded_family_nonrepresentatives": int(len(self.canonical) - len(retained)),
            "representatives_empty_after_legacy_preprocessing": int(len(legacy_empty)),
            "empty_after_generic_topic_cleanup": int(len(empty_generic)),
            "empty_after_dictionary_filtering": int(len(empty_dict)),
            "documents_used_for_lda": int(len(model_docs)),
            "vocab_before_filtering": int(vocab_before),
            "vocab_after_filtering": int(vocab_after),
            "no_below": int(self.args.no_below),
            "no_above": float(self.args.no_above),
            "generic_token_cleanup_counts": dict(cleanup_stats),
            "lda_corpus_sha256": self.corpus_sha,
            "dictionary_sha256": dictionary_hash,
        }
        (self.provenance / "corpus_and_dictionary_stats.json").write_text(
            json.dumps(corpus_stats, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        settings = {
            "script_version": SCRIPT_VERSION,
            "raw_sha256": EXPECTED_SHA256,
            "corpus_sha256": self.corpus_sha,
            "dictionary_sha256": dictionary_hash,
            "dedup_threshold": self.args.dedup_threshold,
            "min_chars": self.args.min_chars,
            "duplicate_normalization": "same as near_duplicate_audit_v1",
            "duplicate_similarity": {"analyzer": "char_wb", "ngram_range": list(DEFAULT_CHAR_NGRAM_RANGE), "min_df": 2, "max_features": DEFAULT_MAX_FEATURES},
            "complete_link_refinement": True,
            "generic_token_cleanup": ["remove one-character tokens", "remove pure-number tokens", "remove http/https/www/com residues"],
            "no_manual_content_stoplist": True,
            "no_below": self.args.no_below,
            "no_above": self.args.no_above,
            "k_values": self.K_VALUES,
            "seeds": self.SEEDS,
            "passes": self.PASSES,
            "iterations": self.ITERATIONS,
            "chunksize": self.CHUNKSIZE,
            "alpha": "auto",
            "eta": "auto",
            "coherence": "c_v",
            "stability": "Hungarian-matched cosine similarity of full topic-word distributions",
            "k_selection": "best mean coherence; one-SE eligible set; highest mean stability; tie diversity then smaller K",
            "seed_selection": "medoid seed by mean similarity to other seeds; tie coherence then smaller seed",
            "post_selection_quality_gate": {"max_dominant_topic_share": DEFAULT_DOMINANCE_WARNING_THRESHOLD, "zero_dominant_topics_allowed": 0},
        }
        self.analysis_signature = stable_json_hash(settings)
        settings["analysis_signature_sha256"] = self.analysis_signature
        settings["audit_reference"] = self.audit_reference
        (self.provenance / "frozen_analysis_configuration.json").write_text(
            json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log(
            f"TOPIC CORPUS READY: {len(model_docs)} documents, vocabulary={vocab_after}, "
            f"analysis_signature={self.analysis_signature[:12]}..."
        )

    # ---------- LDA search ----------
    def model_path(self, k: int, seed: int) -> Path:
        return self.search_models / f"lda_K{k:02d}_seed{seed}.model"

    def metric_path(self, k: int, seed: int) -> Path:
        return self.search_models / f"lda_K{k:02d}_seed{seed}_metrics.json"

    @staticmethod
    def topic_diversity(model: LdaModel, topn: int = DEFAULT_DIVERSITY_TOPN) -> float:
        words: List[str] = []
        for topic_id in range(model.num_topics):
            words.extend([w for w, _ in model.show_topic(topic_id, topn=topn)])
        return len(set(words)) / len(words) if words else float("nan")

    def train_or_load(self, k: int, seed: int) -> Tuple[LdaModel, Dict[str, object], bool]:
        assert self.dictionary is not None and self.analysis_signature is not None
        mp, jp = self.model_path(k, seed), self.metric_path(k, seed)
        if mp.exists() and jp.exists() and not self.args.force_search:
            meta = json.loads(jp.read_text(encoding="utf-8"))
            if meta.get("analysis_signature") != self.analysis_signature:
                raise RuntimeError(
                    f"Stale LDA checkpoint detected for K={k}, seed={seed}. Existing signature differs. "
                    "Use a fresh results directory or --force-search after removing stale checkpoints."
                )
            return LdaModel.load(str(mp)), meta, True

        started = time.time()
        model = LdaModel(
            corpus=self.model_bow,
            id2word=self.dictionary,
            num_topics=k,
            random_state=seed,
            chunksize=self.CHUNKSIZE,
            passes=self.PASSES,
            iterations=self.ITERATIONS,
            alpha="auto",
            eta="auto",
            eval_every=None,
            minimum_probability=0.0,
            per_word_topics=False,
        )
        coherence = float(CoherenceModel(
            model=model,
            texts=self.model_tokens,
            dictionary=self.dictionary,
            coherence="c_v",
            processes=1,
        ).get_coherence())
        metrics = {
            "created_at_utc": utc_now(),
            "analysis_signature": self.analysis_signature,
            "k": int(k),
            "seed": int(seed),
            "coherence_cv": coherence,
            "topic_diversity": float(self.topic_diversity(model)),
            "log_perplexity": float(model.log_perplexity(self.model_bow)),
            "elapsed_seconds": float(time.time() - started),
        }
        model.save(str(mp))
        jp.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return model, metrics, False

    @staticmethod
    def matched_topic_similarity(a: LdaModel, b: LdaModel) -> float:
        A, B = np.asarray(a.get_topics(), dtype=float), np.asarray(b.get_topics(), dtype=float)
        sim = cosine_similarity(A, B)
        rows, cols = linear_sum_assignment(-sim)
        return float(sim[rows, cols].mean())

    def search(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[Tuple[int, int], LdaModel]]:
        total = len(self.K_VALUES) * len(self.SEEDS)
        counter = 0
        run_rows: List[Dict[str, object]] = []
        models_by_key: Dict[Tuple[int, int], LdaModel] = {}
        log(f"Starting/resuming LDA search: {total} runs ({len(self.K_VALUES)} K values x {len(self.SEEDS)} seeds)...")
        for k in self.K_VALUES:
            for seed in self.SEEDS:
                counter += 1
                model, metrics, reused = self.train_or_load(k, seed)
                models_by_key[(k, seed)] = model
                run_rows.append(metrics)
                log(
                    f"[{counter:02d}/{total}] K={k:02d} seed={seed} "
                    f"C_v={float(metrics['coherence_cv']):.6f} diversity={float(metrics['topic_diversity']):.4f} "
                    f"{'REUSED' if reused else 'TRAINED'}"
                )
        all_runs = pd.DataFrame(run_rows).sort_values(["k", "seed"]).reset_index(drop=True)
        all_runs.to_csv(self.tables / "lda_all_runs.csv", index=False)

        pair_rows = []
        run_stability_rows = []
        for k in self.K_VALUES:
            per_seed = {seed: [] for seed in self.SEEDS}
            for sa, sb in combinations(self.SEEDS, 2):
                score = self.matched_topic_similarity(models_by_key[(k, sa)], models_by_key[(k, sb)])
                per_seed[sa].append(score)
                per_seed[sb].append(score)
                pair_rows.append({"k": k, "seed_a": sa, "seed_b": sb, "matched_cosine_similarity": score})
            for seed in self.SEEDS:
                run_stability_rows.append({
                    "k": k,
                    "seed": seed,
                    "mean_similarity_to_other_seeds": float(np.mean(per_seed[seed])),
                })
        stability_pairs = pd.DataFrame(pair_rows)
        run_stability = pd.DataFrame(run_stability_rows)
        stability_pairs.to_csv(self.tables / "lda_stability_pairwise.csv", index=False)
        run_stability.to_csv(self.tables / "lda_run_stability.csv", index=False)

        summary = all_runs.groupby("k", as_index=False).agg(
            coherence_mean=("coherence_cv", "mean"),
            coherence_sd=("coherence_cv", "std"),
            diversity_mean=("topic_diversity", "mean"),
            diversity_sd=("topic_diversity", "std"),
            log_perplexity_mean=("log_perplexity", "mean"),
        )
        summary["coherence_se"] = summary["coherence_sd"] / math.sqrt(len(self.SEEDS))
        stab_summary = stability_pairs.groupby("k", as_index=False).agg(
            stability_mean=("matched_cosine_similarity", "mean"),
            stability_sd=("matched_cosine_similarity", "std"),
            stability_min=("matched_cosine_similarity", "min"),
        )
        summary = summary.merge(stab_summary, on="k", how="left")
        summary.to_csv(self.tables / "lda_k_search_summary.csv", index=False)
        return all_runs, stability_pairs, run_stability, models_by_key

    def select(self, all_runs: pd.DataFrame, run_stability: pd.DataFrame, models_by_key: Dict[Tuple[int, int], LdaModel]):
        summary = pd.read_csv(self.tables / "lda_k_search_summary.csv")
        best = summary.loc[summary["coherence_mean"].idxmax()]
        threshold = float(best["coherence_mean"] - best["coherence_se"])
        selection = summary.copy()
        selection["coherence_eligible"] = selection["coherence_mean"].ge(threshold)
        eligible = selection.loc[selection["coherence_eligible"]].copy()
        selected = eligible.sort_values(["stability_mean", "diversity_mean", "k"], ascending=[False, False, True]).iloc[0]
        selected_k = int(selected["k"])

        # Seed medoids for every K; also create per-K prevalence diagnostics.
        medoid_rows = []
        diagnostic_rows = []
        for k in self.K_VALUES:
            seed_table = all_runs.loc[all_runs["k"].eq(k)].merge(
                run_stability.loc[run_stability["k"].eq(k)], on=["k", "seed"], how="left"
            )
            row = seed_table.sort_values(
                ["mean_similarity_to_other_seeds", "coherence_cv", "seed"], ascending=[False, False, True]
            ).iloc[0]
            medoid_seed = int(row["seed"])
            medoid_rows.append({"k": k, "medoid_seed": medoid_seed, **row.to_dict()})
            model = models_by_key[(k, medoid_seed)]
            probs = np.vstack([
                [p for _, p in model.get_document_topics(bow, minimum_probability=0.0)] for bow in self.model_bow
            ])
            dominant = probs.argmax(axis=1)
            counts = np.bincount(dominant, minlength=k)
            mean_prev = probs.mean(axis=0)
            entropy_prev = -float(np.sum(mean_prev * np.log(mean_prev + 1e-12)))
            diagnostic_rows.append({
                "k": k,
                "medoid_seed": medoid_seed,
                "max_dominant_topic_share": float(counts.max() / len(dominant)),
                "zero_dominant_topics": int((counts == 0).sum()),
                "min_nonzero_dominant_count": int(counts[counts > 0].min()) if np.any(counts > 0) else 0,
                "effective_topics_from_mean_prevalence": float(np.exp(entropy_prev)),
                "max_mean_topic_probability": float(mean_prev.max()),
            })
        medoids = pd.DataFrame(medoid_rows)
        diagnostics = pd.DataFrame(diagnostic_rows)
        medoids.to_csv(self.tables / "lda_k_medoid_seed_table.csv", index=False)
        diagnostics.to_csv(self.tables / "lda_k_medoid_prevalence_diagnostics.csv", index=False)

        selected_seed = int(medoids.loc[medoids["k"].eq(selected_k), "medoid_seed"].iloc[0])
        selection.to_csv(self.tables / "lda_k_selection_table.csv", index=False)
        medoids.loc[medoids["k"].eq(selected_k)].to_csv(self.tables / "lda_selected_k_seed_table.csv", index=False)
        payload = {
            "selection_rule": "One-standard-error coherence eligibility; among eligible K choose highest mean cross-seed stability, then diversity, then smaller K. Within selected K choose medoid seed.",
            "best_coherence_k": int(best["k"]),
            "best_coherence_mean": float(best["coherence_mean"]),
            "best_coherence_se": float(best["coherence_se"]),
            "coherence_threshold": threshold,
            "eligible_k": [int(x) for x in eligible["k"].tolist()],
            "selected_k": selected_k,
            "selected_seed": selected_seed,
        }
        (self.provenance / "lda_selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return selected_k, selected_seed, selection, diagnostics, payload

    # ---------- final outputs ----------
    def save_figures(self, summary: pd.DataFrame, threshold: float, prevalence: pd.DataFrame, selected_k: int) -> None:
        def line_plot(ycol: str, ylabel: str, filename: str, errcol: str | None = None, hline: float | None = None):
            fig, ax = plt.subplots(figsize=(7.2, 4.6))
            if errcol:
                ax.errorbar(summary["k"], summary[ycol], yerr=summary[errcol], marker="o", capsize=3)
            else:
                ax.plot(summary["k"], summary[ycol], marker="o")
            if hline is not None:
                ax.axhline(hline, linestyle="--")
            ax.set_xlabel("Number of topics (K)")
            ax.set_ylabel(ylabel)
            ax.set_xticks(self.K_VALUES)
            fig.tight_layout()
            fig.savefig(self.figures / filename, dpi=300, bbox_inches="tight")
            plt.close(fig)
        line_plot("coherence_mean", "Mean C_v coherence", "fig_01_coherence_by_k.png", "coherence_sd", threshold)
        line_plot("stability_mean", "Mean matched topic stability", "fig_02_stability_by_k.png", "stability_sd")
        line_plot("diversity_mean", f"Topic diversity (top {DEFAULT_DIVERSITY_TOPN})", "fig_03_topic_diversity_by_k.png", "diversity_sd")
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.bar(prevalence["topic_id"].astype(str), prevalence["mean_topic_probability"])
        ax.set_xlabel("LDA topic")
        ax.set_ylabel("Mean topic probability")
        ax.set_title(f"LDA v2 selected solution: K={selected_k}")
        fig.tight_layout()
        fig.savefig(self.figures / "fig_04_selected_topic_prevalence.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def finalize(self, all_runs, run_stability, models_by_key, selected_k, selected_seed, selection, diagnostics, selection_payload):
        assert self.model_docs is not None and self.dictionary is not None
        final_model = models_by_key[(selected_k, selected_seed)]
        final_model.save(str(self.models / "lda_selected_model.gensim"))

        topic_rows = []
        for tid in range(selected_k):
            for rank, (term, weight) in enumerate(final_model.show_topic(tid, topn=DEFAULT_TOP_TERMS), start=1):
                topic_rows.append({"topic_id": tid + 1, "rank": rank, "term": term, "term_probability": float(weight)})
        topic_terms = pd.DataFrame(topic_rows)
        topic_terms.to_csv(self.tables / "lda_topic_terms.csv", index=False)

        probs = np.vstack([
            [p for _, p in final_model.get_document_topics(bow, minimum_probability=0.0)] for bow in self.model_bow
        ])
        topic_cols = [f"lda_topic_{i+1:02d}_prob" for i in range(selected_k)]
        base_cols = [
            "dataset1_source_row_id", "dataset2_source_row_id", "text_sha256", TEXT_COL, CLEAN_COL,
            "near_duplicate_family_id", "family_size", "in_d1_frozen_test", "in_d2_frozen_test",
        ]
        doc_topics = self.model_docs[base_cols].copy()
        for i, col in enumerate(topic_cols):
            doc_topics[col] = probs[:, i]
        doc_topics["lda_dominant_topic"] = probs.argmax(axis=1) + 1
        doc_topics["lda_dominant_probability"] = probs.max(axis=1)
        sorted_probs = np.sort(probs, axis=1)
        doc_topics["lda_top1_top2_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2] if selected_k > 1 else 1.0
        ent = -(probs * np.log(probs + 1e-12)).sum(axis=1)
        doc_topics["lda_normalized_entropy"] = ent / np.log(selected_k)
        doc_topics.to_csv(self.tables / "lda_document_topics.csv", index=False)

        prevalence_rows = []
        for i, col in enumerate(topic_cols, start=1):
            prevalence_rows.append({
                "topic_id": i,
                "mean_topic_probability": float(doc_topics[col].mean()),
                "dominant_document_count": int(doc_topics["lda_dominant_topic"].eq(i).sum()),
                "dominant_document_share": float(doc_topics["lda_dominant_topic"].eq(i).mean()),
            })
        prevalence = pd.DataFrame(prevalence_rows)
        prevalence.to_csv(self.tables / "lda_topic_prevalence.csv", index=False)

        # Secondary sensitivity: reconstruct original-corpus weighting by family size for retained representatives.
        weights = doc_topics["family_size"].astype(float).to_numpy()
        weighted_mean = np.average(probs, axis=0, weights=weights)
        weighted_prev = prevalence[["topic_id"]].copy()
        weighted_prev["family_size_weighted_mean_topic_probability"] = weighted_mean
        weighted_prev.to_csv(self.tables / "lda_topic_prevalence_family_size_weighted_sensitivity.csv", index=False)

        rep_parts = []
        for tid in range(1, selected_k + 1):
            col = f"lda_topic_{tid:02d}_prob"
            top = doc_topics.nlargest(DEFAULT_REP_DOCS, col).copy()
            top.insert(0, "representative_for_topic", tid)
            top.insert(1, "representative_rank", np.arange(1, len(top) + 1))
            rep_parts.append(top)
        representatives = pd.concat(rep_parts, ignore_index=True)
        representatives.to_csv(self.internal / "lda_representative_documents_INTERNAL.csv", index=False)
        ambiguous = doc_topics.sort_values(["lda_normalized_entropy", "lda_top1_top2_margin"], ascending=[False, True]).head(DEFAULT_AMBIGUOUS_DOCS)
        ambiguous.to_csv(self.internal / "lda_ambiguous_documents_INTERNAL.csv", index=False)

        merge_cols = [
            "dataset1_source_row_id", "dataset2_source_row_id", "text_sha256", "near_duplicate_family_id", "family_size",
            "in_d1_frozen_test", "in_d2_frozen_test", "lda_dominant_topic", "lda_dominant_probability",
            "lda_top1_top2_margin", "lda_normalized_entropy",
        ] + topic_cols
        doc_topics[merge_cols].to_csv(self.tables / "lda_gpt_comparison_representatives_only.csv", index=False)

        summary = pd.read_csv(self.tables / "lda_k_search_summary.csv")
        best = summary.loc[summary["coherence_mean"].idxmax()]
        coh_threshold = float(best["coherence_mean"] - best["coherence_se"])
        self.save_figures(summary, coh_threshold, prevalence, selected_k)

        selected_diag = diagnostics.loc[diagnostics["k"].eq(selected_k)].iloc[0]
        max_dom = float(selected_diag["max_dominant_topic_share"])
        zero_dom = int(selected_diag["zero_dominant_topics"])
        gate_pass = (max_dom <= DEFAULT_DOMINANCE_WARNING_THRESHOLD) and (zero_dom == 0)
        gate = {
            "quality_gate_is_diagnostic_not_selection_criterion": True,
            "selected_k": selected_k,
            "selected_seed": selected_seed,
            "max_dominant_topic_share": max_dom,
            "warning_threshold_max_dominant_topic_share": DEFAULT_DOMINANCE_WARNING_THRESHOLD,
            "zero_dominant_topics": zero_dom,
            "quality_gate_pass": gate_pass,
            "interpretation": "PASS means the selected solution does not trigger the predeclared dominance/unused-topic warnings; it does not prove semantic validity. Topic labels still require representative-document review.",
        }
        (self.provenance / "selected_solution_quality_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

        selected_metrics = all_runs.loc[(all_runs["k"].eq(selected_k)) & (all_runs["seed"].eq(selected_seed))].iloc[0].to_dict()
        provenance = {
            "created_at_utc": utc_now(),
            "script_version": SCRIPT_VERSION,
            "analysis_signature": self.analysis_signature,
            "raw_dataset_sha256": EXPECTED_SHA256,
            "shared_positive_raw_count": int(len(self.canonical)),
            "documents_used_for_lda": int(len(self.model_docs)),
            "lda_corpus_sha256": self.corpus_sha,
            "selection": selection_payload,
            "selected_run_metrics": {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in selected_metrics.items()},
            "quality_gate": gate,
            "audit_reference": self.audit_reference,
        }
        (self.provenance / "lda_v2_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

        family_control = json.loads((self.provenance / "near_duplicate_family_control.json").read_text(encoding="utf-8"))
        corpus_stats = json.loads((self.provenance / "corpus_and_dictionary_stats.json").read_text(encoding="utf-8"))
        selected_stability = float(summary.loc[summary["k"].eq(selected_k), "stability_mean"].iloc[0])
        methods = f"""LDA topic-modeling v2 protocol (generated automatically)\n\nCorpus and quality control:\n- Starting corpus: {len(self.canonical)} shared depression-related texts (D1/D2 positive core).\n- Near-duplicate/template-family threshold: cosine >= {self.args.dedup_threshold:.2f} using char_wb TF-IDF (3-5 grams) on conservatively normalized raw text; fuzzy matching only for >= {self.args.min_chars} normalized characters.\n- Candidate families were refined by complete-linkage at the same threshold to prevent single-link chaining.\n- Near-duplicate families: {family_control['final_complete_link_families']}.\n- Nonrepresentative family members removed: {family_control['nonrepresentatives_removed']}.\n- Representatives/singletons after family control: {family_control['documents_retained_after_family_control']}.\n- Documents used for LDA after lexical and dictionary filters: {len(self.model_docs)}.\n- LDA corpus SHA-256: {self.corpus_sha}.\n\nLexical representation:\n- Starting representation: audited legacy text_clean_tfidf.\n- Additional generic topic cleanup: one-character tokens, pure-number tokens, and http/https/www/com residues removed.\n- No manual content/topic stoplist was used.\n- Dictionary no_below={self.args.no_below}; no_above={self.args.no_above}.\n- Final vocabulary size={corpus_stats['vocab_after_filtering']}.\n\nModel search:\n- K={min(self.K_VALUES)}..{max(self.K_VALUES)}.\n- Random seeds={self.SEEDS}.\n- passes={self.PASSES}; iterations={self.ITERATIONS}; chunksize={self.CHUNKSIZE}.\n- alpha='auto'; eta='auto'.\n- Primary model-quality metrics: C_v coherence, topic diversity, and cross-seed stability.\n- Stability: Hungarian-matched cosine similarity of full topic-word distributions.\n\nSelection:\n- Highest mean C_v identifies the reference K; K values within one SE of that mean are eligible.\n- Among eligible K, select highest mean cross-seed stability; tie-break by diversity then smaller K.\n- Final seed is the medoid run by similarity to the other seeds.\n- Selected K={selected_k}; selected seed={selected_seed}.\n- Selected-run C_v={float(selected_metrics['coherence_cv']):.6f}.\n- Selected-run topic diversity={float(selected_metrics['topic_diversity']):.6f}.\n- Mean stability for selected K={selected_stability:.6f}.\n\nPost-selection diagnostic quality gate:\n- Maximum dominant-topic share={max_dom:.6f}; warning threshold={DEFAULT_DOMINANCE_WARNING_THRESHOLD:.2f}.\n- Zero-dominant topics={zero_dom}.\n- Quality gate pass={gate_pass}.\n- This gate is diagnostic and did not alter K selection. Semantic topic validity still requires review of top terms and representative documents.\n\nInterpretation:\n- Topic names must be assigned only after reviewing both top terms and representative documents.\n- Do not force the LDA solution to K=12 or rename topics to match GPT T01-T12.\n- Later GPT-LDA analysis is thematic correspondence, not classification accuracy.\n- Representative social-media texts are internal review artifacts and should not be quoted publicly without appropriate ethics/privacy review.\n"""
        (self.results / "paper_methods_lda_v2.txt").write_text(methods, encoding="utf-8")

        self.write_checksums()
        complete = {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "analysis_signature": self.analysis_signature,
            "selected_k": selected_k,
            "selected_seed": selected_seed,
            "quality_gate_pass": gate_pass,
        }
        (self.results / "RUN_COMPLETE.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
        if not gate_pass:
            log(
                "WARNING: selected solution triggered the predeclared dominance/unused-topic diagnostic gate. "
                "Do not label the topic solution final until top terms and representative documents are reviewed."
            )
        log(f"LDA V2 COMPLETE: selected K={selected_k}, seed={selected_seed}, quality_gate_pass={gate_pass}")
        log(f"Results: {self.results}")

    def write_checksums(self) -> None:
        paths = []
        for p in sorted(self.results.rglob("*")):
            if p.is_file() and p.name != "SHA256SUMS_LDA_V2_OUTPUTS.txt":
                paths.append(p)
        lines = [f"{sha256_file(p)}  {p.relative_to(self.results).as_posix()}" for p in paths]
        (self.results / "SHA256SUMS_LDA_V2_OUTPUTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run(self) -> None:
        self.write_runtime()
        self.audit_inputs()
        self.build_canonical()
        self.build_family_control()
        self.build_topic_corpus()
        if self.args.prepare_only:
            self.write_checksums()
            (self.results / "PREPARE_ONLY_COMPLETE.json").write_text(
                json.dumps({"status": "prepare_only_complete", "completed_at_utc": utc_now(), "analysis_signature": self.analysis_signature}, indent=2),
                encoding="utf-8",
            )
            log("PREPARE-ONLY COMPLETE. Inspect corpus/family/dictionary audits before launching the 60-run LDA search.")
            log(f"Results: {self.results}")
            return
        all_runs, stability_pairs, run_stability, models_by_key = self.search()
        selected_k, selected_seed, selection, diagnostics, payload = self.select(all_runs, run_stability, models_by_key)
        self.finalize(all_runs, run_stability, models_by_key, selected_k, selected_seed, selection, diagnostics, payload)


def main() -> None:
    args = parse_args()
    LDAV2(args).run()


if __name__ == "__main__":
    main()
