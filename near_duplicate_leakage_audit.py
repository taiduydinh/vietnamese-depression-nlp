#!/usr/bin/env python3
"""
Near-duplicate / train-test leakage audit for the Vietnamese depression-signal project.

Primary question
----------------
Do highly similar or normalization-equivalent texts cross the exact historical train/test
boundaries used by the legacy supervised experiments?

Design principles
-----------------
* Verify the two raw datasets by their known SHA-256 hashes.
* Reconstruct the exact historical TF-IDF and DL/PhoBERT outer splits from the audited
  preprocessed CSVs.
* Measure similarity on minimally normalized ORIGINAL `text_vi`, not on model features,
  to avoid manufacturing leakage through the preprocessing pipeline itself.
* Report three predeclared fuzzy similarity thresholds: 0.90, 0.95, 0.98.
* Count normalization-exact matches at all lengths. Fuzzy matching is restricted to texts
  with at least MIN_CHARS normalized characters to reduce short-text false positives.
* Save row-level evidence, pair-level evidence, and an optional post-hoc sensitivity analysis
  of already-saved model predictions. The sensitivity analysis is diagnostic; it is not a
  substitute for group-aware retraining if leakage is material.

The script is CPU-only and does not require TensorFlow or a GPU.
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
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


SCRIPT_VERSION = "1.0"

EXPECTED_SHA256 = {
    1: "b2ad99ebdcf3c34f040a2988fda1c482e9ff9b9cf42795bb64c792b3c7043d9e",
    2: "88161f9388194bbb9cede048223d8f7f6bc6ab68670c373eeffd13bbb45de842",
}
EXPECTED_RAW_ROWS = {1: 5933, 2: 7345}
EXPECTED_RAW_LABELS = {
    1: {0: 2151, 1: 3782},
    2: {0: 3563, 1: 3782},
}
EXPECTED_TFIDF_ROWS = {1: 5921, 2: 7249}
EXPECTED_TFIDF_TEST = {
    1: {"rows": 1185, "labels": {0: 429, 1: 756}},
    2: {"rows": 1450, "labels": {0: 694, 1: 756}},
}
EXPECTED_DL_ROWS = {1: 5933, 2: 7345}
EXPECTED_DL_TEST = {
    1: {"rows": 1187, "labels": {0: 430, 1: 757}},
    2: {"rows": 1469, "labels": {0: 713, 1: 756}},
}

TEXT_COL = "text_vi"
LABEL_COL = "sentiment"
TFIDF_COL = "text_clean_tfidf"
DL_COL = "text_clean_dl"


URL_RE = re.compile(
    r"(?:https?://\S+|www\.\S+|\bhttps?\b\s*(?:[:/._-]\s*|\s+)[a-z0-9][a-z0-9\s._/-]{2,})",
    flags=re.IGNORECASE,
)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b")
MENTION_RE = re.compile(r"(?<!\w)@[\w_.-]+")
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
WHITESPACE_RE = re.compile(r"\s+")


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


def normalize_for_duplicate_audit(text: str) -> str:
    """Conservative, deterministic normalization for duplicate detection.

    Keeps Vietnamese diacritics and digits. Removes URLs/emails/@mentions because differing
    tracking links or handles should not make otherwise templated text look unique. Punctuation
    is converted to spaces rather than deleting characters, preventing token concatenation.
    """
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
    s = "".join(out)
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s


def token_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def length_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if max(la, lb) == 0:
        return 1.0
    return min(la, lb) / max(la, lb)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return {
            "n": 0,
            "accuracy": np.nan,
            "precision_class1": np.nan,
            "recall_class1": np.nan,
            "f1_class1": np.nan,
            "macro_f1": np.nan,
            "tn": np.nan,
            "fp": np.nan,
            "fn": np.nan,
            "tp": np.nan,
        }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_class1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_class1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_class1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


@dataclass
class SplitSpec:
    dataset: int
    protocol: str
    train_ids: np.ndarray
    test_ids: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray


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
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--results-dir", type=str, default="near_duplicate_audit_v1")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.90, 0.95, 0.98])
    parser.add_argument("--min-chars", type=int, default=30)
    parser.add_argument("--char-ngram-min", type=int, default=3)
    parser.add_argument("--char-ngram-max", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=200000)
    parser.add_argument("--n-jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--cluster-neighbors", type=int, default=25)
    parser.add_argument("--skip-clusters", action="store_true")
    parser.add_argument("--skip-prediction-sensitivity", action="store_true")
    return parser.parse_args()


class AuditRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = args.project_root.resolve()
        self.results = self.root / args.results_dir
        self.tables = self.results / "tables"
        self.pairs_dir = self.results / "pairs"
        self.clusters_dir = self.results / "clusters"
        self.provenance = self.results / "provenance"
        self.internal = self.results / "internal_review"
        for p in [self.results, self.tables, self.pairs_dir, self.clusters_dir, self.provenance, self.internal]:
            p.mkdir(parents=True, exist_ok=True)

        self.thresholds = sorted(set(float(x) for x in args.thresholds))
        if not self.thresholds or min(self.thresholds) < 0 or max(self.thresholds) > 1:
            raise ValueError("Thresholds must lie in [0,1].")
        self.min_threshold = min(self.thresholds)

        self.raw: Dict[int, pd.DataFrame] = {}
        self.pre: Dict[int, pd.DataFrame] = {}
        self.splits: Dict[Tuple[int, str], SplitSpec] = {}
        self.norm_texts: Dict[int, List[str]] = {}
        self.vectorizers: Dict[int, TfidfVectorizer] = {}
        self.X: Dict[int, object] = {}
        self.nearest_tables: Dict[Tuple[int, str], pd.DataFrame] = {}
        self.pair_tables: Dict[Tuple[int, str], pd.DataFrame] = {}

    def write_config(self) -> None:
        cfg = {
            "created_at_utc": utc_now(),
            "script_version": SCRIPT_VERSION,
            "project_root": str(self.root),
            "thresholds": self.thresholds,
            "min_chars": self.args.min_chars,
            "char_analyzer": "char_wb",
            "char_ngram_range": [self.args.char_ngram_min, self.args.char_ngram_max],
            "max_features": self.args.max_features,
            "tfidf_min_df": 2,
            "n_jobs": self.args.n_jobs,
            "cluster_neighbors": self.args.cluster_neighbors,
            "primary_similarity": "cosine similarity of L2-normalized char_wb TF-IDF on minimally normalized raw text_vi",
            "exact_normalization": "NFKC + lowercase + HTML unescape + URL/email/@mention removal + punctuation/symbol-to-space + whitespace collapse; Vietnamese diacritics and digits preserved",
            "fuzzy_short_text_policy": f"fuzzy matching restricted to normalized texts >= {self.args.min_chars} characters; normalization-exact matches counted at all lengths",
            "historical_split_reference": "train_test_split(test_size=0.2, random_state=42, stratify=sentiment); TF-IDF after dropna(text_clean_tfidf), DL/PhoBERT on text_clean_dl",
        }
        (self.provenance / "audit_configuration.json").write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        runtime = {
            "created_at_utc": utc_now(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "cpu_count": os.cpu_count(),
        }
        (self.provenance / "runtime.json").write_text(
            json.dumps(runtime, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def audit_inputs(self) -> None:
        log("Auditing raw datasets and audited preprocessing files...")
        rows = []
        for d in [1, 2]:
            raw_path = self.root / "datasets" / f"dataset_{d}.csv"
            pre_path = self.root / "reproduction_results_v1" / "preprocessed" / f"Dataset_depression_vi{d}.csv"
            if not raw_path.exists():
                raise FileNotFoundError(raw_path)
            if not pre_path.exists():
                raise FileNotFoundError(
                    f"Missing audited preprocessing file: {pre_path}. Run the reproduction preprocessing audit first."
                )

            digest = sha256_file(raw_path)
            if digest != EXPECTED_SHA256[d]:
                raise RuntimeError(f"D{d} raw SHA-256 mismatch: {digest} != {EXPECTED_SHA256[d]}")
            raw = pd.read_csv(raw_path)
            if list(raw.columns) != [TEXT_COL, LABEL_COL]:
                raise RuntimeError(f"D{d} raw schema changed: {raw.columns.tolist()}")
            if len(raw) != EXPECTED_RAW_ROWS[d]:
                raise RuntimeError(f"D{d} raw rows {len(raw)} != {EXPECTED_RAW_ROWS[d]}")
            labels = raw[LABEL_COL].astype(int).value_counts().sort_index().to_dict()
            if labels != EXPECTED_RAW_LABELS[d]:
                raise RuntimeError(f"D{d} raw label counts {labels} != {EXPECTED_RAW_LABELS[d]}")
            if raw[[TEXT_COL, LABEL_COL]].isna().any().any():
                raise RuntimeError(f"D{d} has missing raw text/label values")
            raw = raw.copy()
            raw[LABEL_COL] = raw[LABEL_COL].astype(int)
            raw["source_row_id"] = raw.index.astype(int)

            pre = pd.read_csv(pre_path)
            required = {TEXT_COL, LABEL_COL, TFIDF_COL, DL_COL}
            missing = required - set(pre.columns)
            if missing:
                raise RuntimeError(f"D{d} preprocessed file missing {sorted(missing)}")
            if len(pre) != len(raw):
                raise RuntimeError(f"D{d} preprocessed rows {len(pre)} != raw rows {len(raw)}")
            if not pre[TEXT_COL].fillna("").astype(str).equals(raw[TEXT_COL].fillna("").astype(str)):
                raise RuntimeError(f"D{d} preprocessed text_vi is not row-aligned with raw data")
            if not pre[LABEL_COL].astype(int).equals(raw[LABEL_COL].astype(int)):
                raise RuntimeError(f"D{d} labels changed during preprocessing")
            pre = pre.copy()
            pre["source_row_id"] = pre.index.astype(int)
            pre[LABEL_COL] = pre[LABEL_COL].astype(int)

            self.raw[d] = raw
            self.pre[d] = pre

            rows.append({
                "dataset": d,
                "raw_rows": len(raw),
                "label_0": labels[0],
                "label_1": labels[1],
                "raw_sha256": digest,
                "raw_exact_text_duplicate_rows": int(raw.duplicated(subset=[TEXT_COL], keep=False).sum()),
                "raw_exact_fullrow_duplicate_rows": int(raw.duplicated(subset=[TEXT_COL, LABEL_COL], keep=False).sum()),
            })

        pd.DataFrame(rows).to_csv(self.tables / "raw_input_audit.csv", index=False)
        log("RAW / PREPROCESSING ALIGNMENT AUDIT: PASS")

    def dataset_overlap_audit(self) -> None:
        log("Auditing D1/D2 overlap...")
        d1, d2 = self.raw[1], self.raw[2]
        rows = []
        for label in [None, 0, 1]:
            a = d1 if label is None else d1[d1[LABEL_COL] == label]
            b = d2 if label is None else d2[d2[LABEL_COL] == label]
            exact_a, exact_b = set(a[TEXT_COL].astype(str)), set(b[TEXT_COL].astype(str))
            norm_a = set(a[TEXT_COL].astype(str).map(normalize_for_duplicate_audit))
            norm_b = set(b[TEXT_COL].astype(str).map(normalize_for_duplicate_audit))
            rows.append({
                "label_scope": "all" if label is None else str(label),
                "d1_n": len(a),
                "d2_n": len(b),
                "exact_text_intersection": len(exact_a & exact_b),
                "normalized_text_intersection": len(norm_a & norm_b),
            })
        pd.DataFrame(rows).to_csv(self.tables / "d1_d2_overlap_audit.csv", index=False)

    def reconstruct_splits(self) -> None:
        log("Reconstructing exact historical held-out test memberships...")
        split_rows = []
        for d in [1, 2]:
            # TF-IDF legacy branch: exactly match reload -> dropna(text_clean_tfidf) -> stratified 80/20.
            tfidf_df = self.pre[d].dropna(subset=[TFIDF_COL]).copy()
            if len(tfidf_df) != EXPECTED_TFIDF_ROWS[d]:
                raise RuntimeError(f"D{d} TF-IDF rows {len(tfidf_df)} != historical {EXPECTED_TFIDF_ROWS[d]}")
            train_df, test_df = train_test_split(
                tfidf_df,
                test_size=0.2,
                random_state=42,
                stratify=tfidf_df[LABEL_COL],
            )
            got = test_df[LABEL_COL].value_counts().sort_index().to_dict()
            exp = EXPECTED_TFIDF_TEST[d]
            if len(test_df) != exp["rows"] or got != exp["labels"]:
                raise RuntimeError(f"D{d} TF-IDF split mismatch: n={len(test_df)}, labels={got}, expected={exp}")
            self.splits[(d, "tfidf_legacy")] = SplitSpec(
                d, "tfidf_legacy",
                train_df["source_row_id"].astype(int).to_numpy(),
                test_df["source_row_id"].astype(int).to_numpy(),
                train_df[LABEL_COL].astype(int).to_numpy(),
                test_df[LABEL_COL].astype(int).to_numpy(),
            )

            # DL/PhoBERT outer branch: exact legacy 80/20 held-out test split.
            dl_df = self.pre[d].dropna(subset=[DL_COL]).copy()
            if len(dl_df) != EXPECTED_DL_ROWS[d]:
                raise RuntimeError(f"D{d} DL rows {len(dl_df)} != historical {EXPECTED_DL_ROWS[d]}")
            train_df, test_df = train_test_split(
                dl_df,
                test_size=0.2,
                random_state=42,
                stratify=dl_df[LABEL_COL],
            )
            got = test_df[LABEL_COL].value_counts().sort_index().to_dict()
            exp = EXPECTED_DL_TEST[d]
            if len(test_df) != exp["rows"] or got != exp["labels"]:
                raise RuntimeError(f"D{d} DL split mismatch: n={len(test_df)}, labels={got}, expected={exp}")
            self.splits[(d, "dl_phobert_outer")] = SplitSpec(
                d, "dl_phobert_outer",
                train_df["source_row_id"].astype(int).to_numpy(),
                test_df["source_row_id"].astype(int).to_numpy(),
                train_df[LABEL_COL].astype(int).to_numpy(),
                test_df[LABEL_COL].astype(int).to_numpy(),
            )

        # Membership evidence table.
        for (d, protocol), spec in self.splits.items():
            membership = pd.DataFrame({
                "source_row_id": np.concatenate([spec.train_ids, spec.test_ids]),
                "split": ["train"] * len(spec.train_ids) + ["test"] * len(spec.test_ids),
            })
            membership = membership.merge(
                self.raw[d][["source_row_id", LABEL_COL, TEXT_COL]], on="source_row_id", how="left", validate="one_to_one"
            )
            membership.to_csv(self.tables / f"split_membership_d{d}_{protocol}.csv.gz", index=False, compression="gzip")
            split_rows.append({
                "dataset": d,
                "protocol": protocol,
                "n_train": len(spec.train_ids),
                "n_test": len(spec.test_ids),
                "test_label_0": int((spec.test_labels == 0).sum()),
                "test_label_1": int((spec.test_labels == 1).sum()),
            })
        pd.DataFrame(split_rows).to_csv(self.tables / "historical_split_audit.csv", index=False)
        log("HISTORICAL SPLIT AUDIT: PASS")

    def prepare_similarity_space(self, d: int) -> None:
        log(f"D{d}: normalizing raw texts and fitting char TF-IDF duplicate-detection space...")
        texts = self.raw[d][TEXT_COL].astype(str).tolist()
        norm = [normalize_for_duplicate_audit(t) for t in texts]
        self.norm_texts[d] = norm
        norm_meta = self.raw[d][["source_row_id", LABEL_COL]].copy()
        norm_meta["normalized_text"] = norm
        norm_meta["normalized_chars"] = [len(x) for x in norm]
        norm_meta["normalized_sha256"] = [sha256_text(x) for x in norm]
        norm_meta.to_csv(self.internal / f"normalized_text_metadata_d{d}.csv.gz", index=False, compression="gzip")

        # Normalization-exact duplicate groups (all lengths).
        groups = norm_meta[norm_meta["normalized_text"].ne("")].groupby("normalized_sha256", sort=False)
        exact_rows = []
        for h, g in groups:
            if len(g) >= 2:
                labels = g[LABEL_COL].value_counts().to_dict()
                exact_rows.append({
                    "dataset": d,
                    "normalized_sha256": h,
                    "group_size": len(g),
                    "label_0": int(labels.get(0, 0)),
                    "label_1": int(labels.get(1, 0)),
                    "label_pure": int(g[LABEL_COL].nunique() == 1),
                    "source_row_ids": ";".join(map(str, sorted(g["source_row_id"].astype(int).tolist()))),
                })
        pd.DataFrame(exact_rows).to_csv(self.tables / f"normalized_exact_duplicate_groups_d{d}.csv", index=False)

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(self.args.char_ngram_min, self.args.char_ngram_max),
            min_df=2,
            max_features=self.args.max_features,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        X = vectorizer.fit_transform(norm)
        self.vectorizers[d] = vectorizer
        self.X[d] = X
        vocab_info = {
            "dataset": d,
            "n_documents": X.shape[0],
            "n_features": X.shape[1],
            "nnz": int(X.nnz),
            "min_chars": self.args.min_chars,
        }
        (self.provenance / f"similarity_space_d{d}.json").write_text(
            json.dumps(vocab_info, indent=2), encoding="utf-8"
        )
        log(f"D{d}: TF-IDF similarity matrix {X.shape[0]} x {X.shape[1]} (nnz={X.nnz:,})")

    def audit_one_split(self, spec: SplitSpec) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, object]]]:
        d, protocol = spec.dataset, spec.protocol
        log(f"D{d} {protocol}: querying test->train near duplicates (threshold >= {self.min_threshold:.2f})...")
        raw = self.raw[d].set_index("source_row_id", drop=False)
        norm = self.norm_texts[d]
        X = self.X[d]

        train_ids = np.asarray(spec.train_ids, dtype=int)
        test_ids = np.asarray(spec.test_ids, dtype=int)
        train_eligible = np.array([i for i in train_ids if len(norm[i]) >= self.args.min_chars], dtype=int)
        test_eligible = np.array([i for i in test_ids if len(norm[i]) >= self.args.min_chars], dtype=int)

        # Exact-normalized train lookup, including short texts.
        exact_map: Dict[str, List[int]] = defaultdict(list)
        for rid in train_ids:
            s = norm[int(rid)]
            if s:
                exact_map[s].append(int(rid))

        nearest = pd.DataFrame({"source_row_id": test_ids})
        nearest["max_train_similarity"] = np.nan
        nearest["nearest_train_row_id"] = pd.Series([pd.NA] * len(nearest), dtype="Int64")
        nearest["nearest_kind"] = "none"

        # Map test row -> position in nearest table for fast updates.
        pos = {int(r): i for i, r in enumerate(test_ids)}

        # Fuzzy nearest neighbor and all cross pairs >= min threshold, only for eligible lengths.
        pair_map: Dict[Tuple[int, int], Dict[str, object]] = {}
        if len(train_eligible) and len(test_eligible):
            nn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=self.args.n_jobs)
            nn.fit(X[train_eligible])

            dist1, ind1 = nn.kneighbors(X[test_eligible], n_neighbors=1, return_distance=True)
            for q, rid in enumerate(test_eligible):
                trid = int(train_eligible[int(ind1[q, 0])])
                sim = float(np.clip(1.0 - dist1[q, 0], -1.0, 1.0))
                j = pos[int(rid)]
                nearest.at[j, "max_train_similarity"] = sim
                nearest.at[j, "nearest_train_row_id"] = trid
                nearest.at[j, "nearest_kind"] = "fuzzy_nearest"

            radius = max(0.0, 1.0 - self.min_threshold + 1e-12)
            dists, inds = nn.radius_neighbors(X[test_eligible], radius=radius, return_distance=True, sort_results=True)
            for q, rid in enumerate(test_eligible):
                test_id = int(rid)
                for dist, local_idx in zip(dists[q], inds[q]):
                    train_id = int(train_eligible[int(local_idx)])
                    sim = float(np.clip(1.0 - dist, -1.0, 1.0))
                    if sim + 1e-8 < self.min_threshold:
                        continue
                    key = (test_id, train_id)
                    pair_map[key] = {
                        "dataset": d,
                        "protocol": protocol,
                        "test_row_id": test_id,
                        "train_row_id": train_id,
                        "similarity": sim,
                        "match_kind": "fuzzy",
                    }

        # Exact-normalized matches override nearest similarity to exactly 1.0 and add any short-text pairs.
        for test_id in test_ids:
            s = norm[int(test_id)]
            if not s or s not in exact_map:
                continue
            train_matches = exact_map[s]
            j = pos[int(test_id)]
            nearest.at[j, "max_train_similarity"] = 1.0
            nearest.at[j, "nearest_train_row_id"] = int(train_matches[0])
            nearest.at[j, "nearest_kind"] = "normalized_exact"
            for train_id in train_matches:
                key = (int(test_id), int(train_id))
                pair_map[key] = {
                    "dataset": d,
                    "protocol": protocol,
                    "test_row_id": int(test_id),
                    "train_row_id": int(train_id),
                    "similarity": 1.0,
                    "match_kind": "normalized_exact",
                }

        pairs = pd.DataFrame(list(pair_map.values()))
        if len(pairs):
            # Attach evidence. Raw texts are intentionally stored only in internal-review outputs.
            pairs["test_label"] = pairs["test_row_id"].map(raw[LABEL_COL].to_dict()).astype(int)
            pairs["train_label"] = pairs["train_row_id"].map(raw[LABEL_COL].to_dict()).astype(int)
            pairs["same_label"] = pairs["test_label"].eq(pairs["train_label"])
            pairs["test_norm_chars"] = pairs["test_row_id"].map({i: len(norm[i]) for i in range(len(norm))})
            pairs["train_norm_chars"] = pairs["train_row_id"].map({i: len(norm[i]) for i in range(len(norm))})
            pairs["length_ratio"] = [length_ratio(norm[t], norm[r]) for t, r in zip(pairs["test_row_id"], pairs["train_row_id"])]
            pairs["token_jaccard"] = [token_jaccard(norm[t], norm[r]) for t, r in zip(pairs["test_row_id"], pairs["train_row_id"])]
            pairs = pairs.sort_values(["similarity", "test_row_id", "train_row_id"], ascending=[False, True, True]).reset_index(drop=True)
        else:
            pairs = pd.DataFrame(columns=[
                "dataset", "protocol", "test_row_id", "train_row_id", "similarity", "match_kind",
                "test_label", "train_label", "same_label", "test_norm_chars", "train_norm_chars",
                "length_ratio", "token_jaccard"
            ])

        # Attach nearest-neighbor labels/text evidence.
        nearest = nearest.merge(raw[[LABEL_COL, TEXT_COL]], left_on="source_row_id", right_index=True, how="left")
        nearest = nearest.rename(columns={LABEL_COL: "test_label", TEXT_COL: "test_text"})
        train_label_map = raw[LABEL_COL].to_dict()
        train_text_map = raw[TEXT_COL].to_dict()
        nearest["nearest_train_label"] = nearest["nearest_train_row_id"].map(train_label_map).astype("Int64")
        nearest["nearest_train_text"] = nearest["nearest_train_row_id"].map(train_text_map)
        nearest["same_label_as_nearest"] = (
            nearest["nearest_train_label"].notna() &
            nearest["test_label"].eq(nearest["nearest_train_label"].astype("float"))
        )
        nearest["normalized_chars"] = nearest["source_row_id"].map({i: len(norm[i]) for i in range(len(norm))})
        nearest["fuzzy_eligible"] = nearest["normalized_chars"].ge(self.args.min_chars)

        summary_rows = []
        for thr in self.thresholds:
            leaked = nearest["max_train_similarity"].fillna(-np.inf).ge(thr - 1e-8)
            pos_mask = nearest["test_label"].eq(1)
            neg_mask = nearest["test_label"].eq(0)
            same_label = leaked & nearest["same_label_as_nearest"].fillna(False)
            cross_label = leaked & ~nearest["same_label_as_nearest"].fillna(False)
            pthr = pairs[pairs["similarity"].ge(thr - 1e-8)] if len(pairs) else pairs
            summary_rows.append({
                "dataset": d,
                "protocol": protocol,
                "threshold": thr,
                "n_train": len(train_ids),
                "n_test": len(test_ids),
                "n_test_fuzzy_eligible": int(nearest["fuzzy_eligible"].sum()),
                "n_test_short_fuzzy_excluded": int((~nearest["fuzzy_eligible"]).sum()),
                "cross_split_pairs": int(len(pthr)),
                "test_rows_with_train_near_duplicate": int(leaked.sum()),
                "test_leakage_rate": float(leaked.mean()),
                "positive_test_rows": int(pos_mask.sum()),
                "positive_test_rows_leaked": int((leaked & pos_mask).sum()),
                "positive_test_leakage_rate": float((leaked & pos_mask).sum() / pos_mask.sum()) if pos_mask.sum() else np.nan,
                "negative_test_rows": int(neg_mask.sum()),
                "negative_test_rows_leaked": int((leaked & neg_mask).sum()),
                "negative_test_leakage_rate": float((leaked & neg_mask).sum() / neg_mask.sum()) if neg_mask.sum() else np.nan,
                "leaked_test_rows_same_label_nearest": int(same_label.sum()),
                "leaked_test_rows_cross_label_nearest": int(cross_label.sum()),
                "cross_label_pair_count": int((~pthr["same_label"]).sum()) if len(pthr) else 0,
                "normalized_exact_pair_count": int((pthr["match_kind"] == "normalized_exact").sum()) if len(pthr) else 0,
            })

        nearest.to_csv(
            self.internal / f"test_nearest_train_evidence_d{d}_{protocol}.csv.gz",
            index=False,
            compression="gzip",
        )
        pairs.to_csv(
            self.pairs_dir / f"cross_split_pairs_d{d}_{protocol}_gte_{self.min_threshold:.2f}.csv.gz",
            index=False,
            compression="gzip",
        )
        pairs_internal = pairs.copy()
        if len(pairs_internal):
            pairs_internal["test_text"] = pairs_internal["test_row_id"].map(raw[TEXT_COL].to_dict())
            pairs_internal["train_text"] = pairs_internal["train_row_id"].map(raw[TEXT_COL].to_dict())
        pairs_internal.to_csv(
            self.internal / f"cross_split_pairs_WITH_TEXT_INTERNAL_d{d}_{protocol}_gte_{self.min_threshold:.2f}.csv.gz",
            index=False,
            compression="gzip",
        )
        return nearest, pairs, summary_rows

    def similarity_audit(self) -> pd.DataFrame:
        summary_rows: List[Dict[str, object]] = []
        for d in [1, 2]:
            self.prepare_similarity_space(d)
            for protocol in ["tfidf_legacy", "dl_phobert_outer"]:
                nearest, pairs, rows = self.audit_one_split(self.splits[(d, protocol)])
                self.nearest_tables[(d, protocol)] = nearest
                self.pair_tables[(d, protocol)] = pairs
                summary_rows.extend(rows)
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(self.tables / "near_duplicate_leakage_summary.csv", index=False)
        return summary

    def build_exploratory_clusters(self) -> None:
        if self.args.skip_clusters:
            log("Skipping exploratory whole-dataset near-duplicate clustering (--skip-clusters).")
            return
        log("Building exploratory whole-dataset near-duplicate family clusters...")
        for d in [1, 2]:
            raw = self.raw[d]
            norm = self.norm_texts[d]
            X = self.X[d]
            eligible = np.array([i for i, s in enumerate(norm) if len(s) >= self.args.min_chars], dtype=int)
            edges: Dict[Tuple[int, int], float] = {}

            # Exact-normalized edges at all lengths, using star edges to keep large groups compact.
            exact_groups: Dict[str, List[int]] = defaultdict(list)
            for i, s in enumerate(norm):
                if s:
                    exact_groups[s].append(i)
            for members in exact_groups.values():
                if len(members) >= 2:
                    anchor = int(members[0])
                    for m in members[1:]:
                        a, b = sorted((anchor, int(m)))
                        edges[(a, b)] = 1.0

            # kNN graph for fuzzy family discovery. Primary train-test leakage statistics above
            # use radius search and do NOT depend on this approximate clustering step.
            if len(eligible) >= 2:
                k = min(self.args.cluster_neighbors + 1, len(eligible))
                nn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=self.args.n_jobs)
                nn.fit(X[eligible])
                dists, inds = nn.kneighbors(X[eligible], n_neighbors=k, return_distance=True)
                for q, rid in enumerate(eligible):
                    for dist, local_idx in zip(dists[q], inds[q]):
                        other = int(eligible[int(local_idx)])
                        if other == int(rid):
                            continue
                        sim = float(np.clip(1.0 - dist, -1.0, 1.0))
                        if sim + 1e-8 < self.min_threshold:
                            continue
                        a, b = sorted((int(rid), other))
                        if sim > edges.get((a, b), -1):
                            edges[(a, b)] = sim

            edge_items = [(a, b, s) for (a, b), s in edges.items()]
            log(f"D{d}: exploratory cluster graph has {len(edge_items):,} edges >= {self.min_threshold:.2f}")

            for thr in self.thresholds:
                dsu = DSU(range(len(raw)))
                for a, b, sim in edge_items:
                    if sim + 1e-8 >= thr:
                        dsu.union(a, b)
                comps: Dict[int, List[int]] = defaultdict(list)
                for rid in range(len(raw)):
                    comps[dsu.find(rid)].append(rid)
                comps = {root: m for root, m in comps.items() if len(m) >= 2}

                # Stable cluster IDs ordered by minimum source row id.
                ordered = sorted(comps.values(), key=lambda xs: min(xs))
                member_rows = []
                summary_rows = []
                for cid, members in enumerate(ordered, start=1):
                    labels = raw.loc[members, LABEL_COL].astype(int)
                    counts = labels.value_counts().to_dict()
                    for rid in members:
                        member_rows.append({
                            "dataset": d,
                            "threshold": thr,
                            "cluster_id": cid,
                            "cluster_size": len(members),
                            "source_row_id": rid,
                            "label": int(raw.at[rid, LABEL_COL]),
                            "text_vi": raw.at[rid, TEXT_COL],
                        })
                    summary_rows.append({
                        "dataset": d,
                        "threshold": thr,
                        "cluster_id": cid,
                        "cluster_size": len(members),
                        "label_0": int(counts.get(0, 0)),
                        "label_1": int(counts.get(1, 0)),
                        "label_pure": int(labels.nunique() == 1),
                        "min_source_row_id": min(members),
                    })
                pd.DataFrame(summary_rows).to_csv(
                    self.clusters_dir / f"cluster_summary_d{d}_t{int(round(thr*100)):02d}.csv", index=False
                )
                pd.DataFrame(member_rows).to_csv(
                    self.internal / f"cluster_members_INTERNAL_d{d}_t{int(round(thr*100)):02d}.csv.gz",
                    index=False, compression="gzip"
                )

    def prediction_sensitivity(self) -> pd.DataFrame:
        if self.args.skip_prediction_sensitivity:
            log("Skipping prediction sensitivity analysis (--skip-prediction-sensitivity).")
            return pd.DataFrame()

        run_root = self.root / "reproduction_results_v1" / "runs"
        pred_files = sorted(run_root.glob("*/*/predictions.csv")) if run_root.exists() else []
        if not pred_files:
            log("No reproduction predictions.csv files found; skipping post-hoc metric sensitivity analysis.")
            return pd.DataFrame()

        log(f"Found {len(pred_files)} saved prediction files; calculating post-hoc clean-test sensitivity...")
        rows = []
        status_rows = []
        for pred_path in pred_files:
            metrics_path = pred_path.with_name("metrics.json")
            if not metrics_path.exists():
                continue
            meta = json.loads(metrics_path.read_text(encoding="utf-8"))
            model = str(meta.get("model", pred_path.parent.parent.name))
            dataset = int(meta.get("dataset", pred_path.parent.name.lstrip("d")))
            protocol = "tfidf_legacy" if model in {"TF-IDF + LR", "TF-IDF + SVM"} else "dl_phobert_outer"
            nearest = self.nearest_tables[(dataset, protocol)][
                ["source_row_id", "test_label", "max_train_similarity"]
            ].copy()
            pred = pd.read_csv(pred_path)
            required = {"source_row_id", "gold_label", "predicted_label"}
            if not required.issubset(pred.columns):
                status_rows.append({"model": model, "dataset": dataset, "status": "missing_prediction_columns", "path": str(pred_path)})
                continue
            pred["source_row_id"] = pred["source_row_id"].astype(int)
            pred["gold_label"] = pred["gold_label"].astype(int)
            pred["predicted_label"] = pred["predicted_label"].astype(int)

            expected_ids = set(nearest["source_row_id"].astype(int))
            got_ids = set(pred["source_row_id"].astype(int))
            if got_ids != expected_ids:
                status_rows.append({
                    "model": model, "dataset": dataset, "status": "membership_mismatch",
                    "expected_n": len(expected_ids), "prediction_n": len(got_ids), "path": str(pred_path)
                })
                continue
            merged = pred.merge(nearest, on="source_row_id", how="left", validate="one_to_one")
            if not merged["gold_label"].eq(merged["test_label"].astype(int)).all():
                status_rows.append({"model": model, "dataset": dataset, "status": "gold_label_mismatch", "path": str(pred_path)})
                continue

            all_m = metrics_dict(merged["gold_label"].to_numpy(), merged["predicted_label"].to_numpy())
            for thr in self.thresholds:
                leak = merged["max_train_similarity"].fillna(-np.inf).ge(thr - 1e-8)
                clean = merged.loc[~leak]
                leaked = merged.loc[leak]
                clean_m = metrics_dict(clean["gold_label"].to_numpy(), clean["predicted_label"].to_numpy())
                leak_m = metrics_dict(leaked["gold_label"].to_numpy(), leaked["predicted_label"].to_numpy())
                row = {
                    "model": model,
                    "dataset": dataset,
                    "protocol": protocol,
                    "threshold": thr,
                    "n_test_all": all_m["n"],
                    "n_test_clean": clean_m["n"],
                    "n_test_flagged": leak_m["n"],
                }
                for prefix, m in [("all", all_m), ("clean", clean_m), ("flagged", leak_m)]:
                    for key in ["accuracy", "precision_class1", "recall_class1", "f1_class1", "macro_f1"]:
                        row[f"{prefix}_{key}"] = m[key]
                row["clean_minus_all_accuracy"] = (
                    clean_m["accuracy"] - all_m["accuracy"] if not pd.isna(clean_m["accuracy"]) else np.nan
                )
                row["clean_minus_all_macro_f1"] = (
                    clean_m["macro_f1"] - all_m["macro_f1"] if not pd.isna(clean_m["macro_f1"]) else np.nan
                )
                rows.append(row)
            status_rows.append({"model": model, "dataset": dataset, "status": "ok", "path": str(pred_path)})

        pd.DataFrame(status_rows).to_csv(self.tables / "prediction_sensitivity_status.csv", index=False)
        out = pd.DataFrame(rows)
        out.to_csv(self.tables / "existing_prediction_clean_test_sensitivity.csv", index=False)
        return out

    def write_report(self, leakage: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
        lines = []
        lines.append("# Near-Duplicate / Historical Split Leakage Audit\n")
        lines.append(f"Generated UTC: {utc_now()}\n")
        lines.append("## Scope\n")
        lines.append(
            "This audit reconstructs the exact historical held-out test memberships for the TF-IDF branch "
            "and the shared DL/PhoBERT outer split, then searches for normalization-exact and high-similarity "
            "raw-text neighbors crossing train→test. It does not alter the datasets or retrain models.\n"
        )
        lines.append("## Primary thresholds\n")
        lines.append(", ".join(f"{t:.2f}" for t in self.thresholds) + " cosine similarity.\n")
        lines.append(
            f"Fuzzy matching uses char_wb TF-IDF on minimally normalized raw text and is restricted to texts "
            f"with at least {self.args.min_chars} normalized characters. Normalization-exact matches are counted at all lengths.\n"
        )
        lines.append("## Leakage summary\n")
        if len(leakage):
            show_cols = [
                "dataset", "protocol", "threshold", "n_test", "cross_split_pairs",
                "test_rows_with_train_near_duplicate", "test_leakage_rate",
                "positive_test_rows_leaked", "positive_test_leakage_rate",
                "negative_test_rows_leaked", "negative_test_leakage_rate",
                "leaked_test_rows_cross_label_nearest",
            ]
            lines.append("```text\n" + leakage[show_cols].to_string(index=False) + "\n```\n")
        lines.append("## Existing-prediction sensitivity\n")
        lines.append(
            "When saved reproduction predictions are available, metrics are recomputed after excluding flagged test rows. "
            "This is a post-hoc sensitivity diagnostic, not a replacement for a group-aware train/test split and retraining.\n"
        )
        if len(sensitivity):
            show_cols = [
                "model", "dataset", "threshold", "n_test_all", "n_test_clean", "n_test_flagged",
                "all_accuracy", "clean_accuracy", "clean_minus_all_accuracy",
                "all_macro_f1", "clean_macro_f1", "clean_minus_all_macro_f1",
            ]
            lines.append("```text\n" + sensitivity[show_cols].to_string(index=False) + "\n```\n")
        else:
            lines.append("No compatible saved prediction files were analyzed.\n")
        lines.append("## Interpretation guardrails\n")
        lines.append(
            "- A flagged pair is evidence of high textual similarity, not proof that two records are semantically identical.\n"
            "- The 0.90/0.95/0.98 thresholds are reported together to avoid choosing a cutoff after seeing results.\n"
            "- Cross-label near duplicates should be inspected as possible annotation/source conflicts.\n"
            "- If substantial train→test family overlap is found, the publication-grade remedy is group-aware splitting and retraining, not merely deleting flagged test rows.\n"
            "- D1 and D2 share the positive core, so they should not be presented as independent replications.\n"
        )
        (self.results / "AUDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    def run(self) -> None:
        t0 = time.time()
        self.write_config()
        self.audit_inputs()
        self.dataset_overlap_audit()
        self.reconstruct_splits()
        leakage = self.similarity_audit()
        self.build_exploratory_clusters()
        sensitivity = self.prediction_sensitivity()
        self.write_report(leakage, sensitivity)

        final = {
            "created_at_utc": utc_now(),
            "status": "completed",
            "script_version": SCRIPT_VERSION,
            "raw_input_audit": "PASS",
            "historical_split_audit": "PASS",
            "datasets": [1, 2],
            "protocols": ["tfidf_legacy", "dl_phobert_outer"],
            "thresholds": self.thresholds,
            "prediction_files_analyzed": int(sensitivity[["model", "dataset"]].drop_duplicates().shape[0]) if len(sensitivity) else 0,
            "elapsed_seconds": time.time() - t0,
        }
        (self.results / "FINAL_STATUS.json").write_text(
            json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log("AUDIT COMPLETE")
        log(f"Results: {self.results}")
        if len(leakage):
            print("\n=== Primary leakage summary ===")
            print(leakage.to_string(index=False))
        if len(sensitivity):
            print("\n=== Existing-prediction sensitivity ===")
            cols = [
                "model", "dataset", "threshold", "n_test_all", "n_test_clean", "n_test_flagged",
                "all_accuracy", "clean_accuracy", "all_macro_f1", "clean_macro_f1",
            ]
            print(sensitivity[cols].to_string(index=False))


def main() -> int:
    args = parse_args()
    runner = AuditRunner(args)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
