#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reproduce all previous Vietnamese depression-detection experiments.

Models:
  1) TF-IDF + Logistic Regression
  2) TF-IDF + SVM
  3) Word2Vec + BiLSTM
  4) Word2Vec + CNN
  5) PhoBERT-base (frozen encoder in the archived executable notebook)
  6) PhoBERT-large (full fine-tuning)

Each model is run on Dataset 1 and Dataset 2.

Version 1.2. This script is based on the audited co-author notebooks. It preserves the
legacy preprocessing and model-family-specific split protocols, while adding
controlled seeds, checkpointed outputs, and resumability.

Typical server command:
  python -u reproduce_all_previous_models.py --project-root "/path/to/Depression_Thay Cach"

Resume is automatic: completed model×dataset runs are skipped by default.
"""

from __future__ import annotations

import os
# Important for Hugging Face TF models with modern Keras installations.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import gc
import gzip
import hashlib
import json
import math
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


# ---------------------------------------------------------------------
# Historical reference results extracted from the archived notebooks.
# ---------------------------------------------------------------------

LEGACY_REFERENCE_ROWS = [
    ["TF-IDF + LR", 1, 1185, 0.97,   0.99, 0.96, 0.97,
     "accuracy printed to 2 d.p.; GridSearch CV macro-F1=0.9644"],
    ["TF-IDF + LR", 2, 1450, 0.97,   0.99, 0.96, 0.97,
     "accuracy printed to 2 d.p.; GridSearch CV macro-F1=0.9670"],
    ["TF-IDF + SVM", 1, 1185, 0.97,  0.99, 0.96, 0.98,
     "accuracy printed to 2 d.p.; GridSearch CV macro-F1=0.9653"],
    ["TF-IDF + SVM", 2, 1450, 0.97,  0.98, 0.96, 0.97,
     "accuracy printed to 2 d.p.; GridSearch CV macro-F1=0.9674"],
    ["Word2Vec + BiLSTM", 1, 1187, 0.9655, 0.98, 0.97, 0.97,
     "explicit test accuracy 96.55%"],
    ["Word2Vec + BiLSTM", 2, 1469, 0.9769, 0.97, 0.98, 0.98,
     "explicit test accuracy 97.69%"],
    ["Word2Vec + CNN", 1, 1187, 0.9773, 0.98, 0.98, 0.98,
     "explicit test accuracy 97.73%"],
    ["Word2Vec + CNN", 2, 1469, 0.9789, 0.98, 0.98, 0.98,
     "explicit test accuracy 97.89%"],
    ["PhoBERT-base", 1, 1187, 0.9697, 0.97, 0.99, 0.98,
     "encoder frozen in executable notebook"],
    ["PhoBERT-base", 2, 1469, 0.9816, 0.98, 0.99, 0.98,
     "encoder frozen in executable notebook"],
    ["PhoBERT-large", 1, 1187, 0.9890, 0.99, 0.99, 0.99,
     "full fine-tuning; explicit test accuracy 98.90%"],
    ["PhoBERT-large", 2, 1469, 0.9912, 0.99, 0.99, 0.99,
     "full fine-tuning; explicit test accuracy 99.12%"],
]

LEGACY_REFERENCE = pd.DataFrame(
    LEGACY_REFERENCE_ROWS,
    columns=[
        "model",
        "dataset",
        "legacy_test_n",
        "legacy_accuracy",
        "legacy_precision_class1",
        "legacy_recall_class1",
        "legacy_f1_class1",
        "legacy_note",
    ],
)


# ---------------------------------------------------------------------
# Audited dataset facts.
# ---------------------------------------------------------------------

EXPECTED_RAW_SHA256 = {
    1: "b2ad99ebdcf3c34f040a2988fda1c482e9ff9b9cf42795bb64c792b3c7043d9e",
    2: "88161f9388194bbb9cede048223d8f7f6bc6ab68670c373eeffd13bbb45de842",
}

EXPECTED_RAW_ROWS = {1: 5933, 2: 7345}

EXPECTED_RAW_LABELS = {
    1: {0: 2151, 1: 3782},
    2: {0: 3563, 1: 3782},
}

# Historical counts after legacy TF-IDF preprocessing/reload.
EXPECTED_TFIDF_MODEL_ROWS = {1: 5921, 2: 7249}

EXPECTED_TFIDF_TEST = {
    1: {"rows": 1185, "labels": {0: 429, 1: 756}},
    2: {"rows": 1450, "labels": {0: 694, 1: 756}},
}

EXPECTED_DL_TEST = {
    1: {"rows": 1187, "labels": {0: 430, 1: 757}},
    2: {"rows": 1469, "labels": {0: 713, 1: 756}},
}


# ---------------------------------------------------------------------
# Legacy preprocessing dictionary from the archived notebook.
# ---------------------------------------------------------------------

ABBREVIATIONS = {
    # ===== Phủ định / khẳng định =====
    "ko": "không", "k": "không", "kh": "không", "hk": "không",
    "dc": "được", "đc": "được",
    "ok": "đồng ý", "oke": "đồng ý", "okie": "đồng ý",

    # ===== Xưng hô =====
    "vk": "vợ", "ck": "chồng",
    "ny": "người yêu", "ngy": "người yêu",
    "bff": "bạn thân", "bb": "bạn bè",
    "m": "mình", "mik": "mình", "mịk": "mình",
    "t": "tôi", "toi": "tôi", "tớ": "tôi",
    "bn": "bạn", "bno": "bạn", "bạn": "bạn",
    "ae": "anh em", "cj": "chị", "a": "anh", "e": "em",
    "crush": "người thầm thích", "nyc": "người yêu cũ",
    "bae": "người yêu", "bồ": "người yêu",

    # ===== Thời gian =====
    "hnay": "hôm nay", "hqua": "hôm qua", "mai": "ngày mai",
    "cn": "chủ nhật", "t2": "thứ hai", "t3": "thứ ba",
    "t4": "thứ tư", "t5": "thứ năm", "t6": "thứ sáu", "t7": "thứ bảy",
    "hn": "hôm nay", "mai n": "ngày mai", "trc": "trước", "trước đó": "trước",
    "sau n": "sau này", "nay": "hôm nay", "nãy": "lúc nãy",

    # ===== Cảm xúc / trạng thái =====
    "vs": "với", "vsz": "với", "w": "với",
    "wa": "quá", "wá": "quá", "qá": "quá", "qa": "quá",
    "zui": "vui", "hpx": "hạnh phúc", "hp": "hạnh phúc", "hpny": "happy new year",
    "bùn": "buồn", "sad": "buồn", "sr": "xin lỗi",
    "thk": "cảm ơn", "tks": "cảm ơn", "thanks": "cảm ơn", "ty": "cảm ơn", "thx": "cảm ơn",
    "bt": "bình thường", "thjk": "thích", "thik": "thích",
    "huhu": "(khóc)", "hic": "(thở dài)", "haizz": "(thở dài)", "haiz": "(thở dài)",
    "kaka": "(cười)", "hehe": "(cười)", "hihi": "(cười)", "kkk": "(cười)",
    "kk": "(cười)", "zz": "(chán)", "zzz": "(buồn ngủ)",
    "g9": "chúc ngủ ngon", "gn": "chúc ngủ ngon", "gm": "chào buổi sáng",
    "max": "rất", "siêu": "rất", "vđ": "vãi", "vđk": "vãi",
    "xịn": "tốt", "xịn sò": "tốt", "dth": "dễ thương", "dthw": "dễ thương",

    # ===== Khác =====
    "mn": "mọi người", "ms": "mới", "r": "rồi", "òi": "rồi", "oy": "rồi", "rùi": "rồi",
    "j": "gì", "ji": "gì", "z": "gì",
    "cg": "cũng", "chx": "chưa", "nch": "nói chung",
    "qtqđ": "quá trời quá đất", "vl": "vãi", "vcl": "vãi", "vlcl": "vãi", "vkl": "vãi",
    "lm": "làm", "lj": "lý do", "nge": "nghe", "ngek": "nghe",
    "nx": "nhận xét", "kq": "kết quả", "nv": "như vậy", "ntna": "như thế nào",
    "hum": "hôm", "humk": "hôm",
    "ib": "inbox", "inb": "inbox", "inbox": "inbox", "rep": "trả lời", "rv": "review",
    "má ơi": "trời ơi", "má": "mẹ",
    "lol": "cười lớn", "lmao": "cười lăn lộn",
    "omfg": "trời ơi", "wtf": "cái gì vậy",
    "dm": "chết tiệt", "đm": "chết tiệt",
    "cc": "", "cl": "",
    "ad": "admin", "acc": "tài khoản", "add": "thêm",
    "fb": "facebook", "tt": "tiktok", "yt": "youtube",
    "sg": "sài gòn", "dn": "đà nẵng", "q1": "quận 1", "q3": "quận 3", "q7": "quận 7",
    "pls": "làm ơn", "plz": "làm ơn", "idk": "không biết",
    "imo": "theo ý tôi", "imho": "theo ý tôi", "tbh": "thật lòng mà nói",
    "brb": "quay lại ngay", "btw": "nhân tiện", "aka": "còn gọi là",
    "asap": "càng sớm càng tốt", "fyi": "để bạn biết", "np": "không sao",
    "u": "bạn", "ur": "của bạn",
    "fs": "free ship", "freeship": "free ship",
    "sale off": "giảm giá", "km": "khuyến mãi", "kmđb": "khuyến mãi đặc biệt",
    "ib deal": "inbox thương lượng", "deal": "thương lượng",
    "tl": "tài liệu", "tg": "tác giả", "sv": "sinh viên",
    "cv": "hồ sơ", "jd": "mô tả công việc", "hr": "nhân sự",
    "bk": "bách khoa", "cntt": "công nghệ thông tin",
    "clgt": "cái quái gì thế", "sml": "xỉu luôn",
    "toang": "hỏng bét", "toang rồi": "hỏng bét",
}


# ---------------------------------------------------------------------
# Fixed model hyperparameters and resources.
# ---------------------------------------------------------------------

GLOBAL_SEED = 42
EXPECTED_ABBREVIATIONS_COUNT = 179
EXPECTED_ABBREVIATIONS_SHA256 = "239d69600b8845621ad503454f510c921655de6ff23c75f5e25bf77f70e7db8d"

WORD2VEC_URL = (
    "https://thiaisotajppub.s3-ap-northeast-1.amazonaws.com/"
    "publicfiles/wiki.vi.model.bin.gz"
)
STOPWORDS_URL = (
    "https://raw.githubusercontent.com/stopwords/"
    "vietnamese-stopwords/master/vietnamese-stopwords.txt"
)

VOCAB_SIZE = 20000
WORD2VEC_MAX_LEN = 150

LEGACY_BILSTM_HPS = {
    1: {
        "lstm_units": 32,
        "dropout_rate": 0.4,
        "dense_units": 96,
        "learning_rate": 1e-2,
    },
    2: {
        "lstm_units": 32,
        "dropout_rate": 0.2,
        "dense_units": 96,
        "learning_rate": 1e-4,
    },
}

LEGACY_CNN_HPS = {
    1: {
        "filters": 256,
        "kernel_size": 3,
        "dense_units": 32,
        "dropout_rate": 0.3,
        "learning_rate": 1e-3,
    },
    2: {
        "filters": 256,
        "kernel_size": 3,
        "dense_units": 32,
        "dropout_rate": 0.3,
        "learning_rate": 1e-3,
    },
}

MODEL_ALIASES = {
    "lr": "lr",
    "logistic": "lr",
    "logistic-regression": "lr",
    "svm": "svm",
    "bilstm": "bilstm",
    "bi-lstm": "bilstm",
    "cnn": "cnn",
    "phobert-base": "phobert-base",
    "base": "phobert-base",
    "phobert-large": "phobert-large",
    "large": "phobert-large",
}

ALL_MODELS = ["lr", "svm", "bilstm", "cnn", "phobert-base", "phobert-large"]


# ---------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def safe_slug(model_name: str) -> str:
    return (
        model_name.lower()
        .replace("+", "plus")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def normalize_models(values: Sequence[str]) -> list[str]:
    out = []
    for raw in values:
        for piece in raw.split(","):
            key = piece.strip().lower()
            if not key:
                continue
            if key == "all":
                return list(ALL_MODELS)
            if key not in MODEL_ALIASES:
                raise ValueError(
                    f"Unknown model {piece!r}. Allowed: {', '.join(ALL_MODELS)}"
                )
            resolved = MODEL_ALIASES[key]
            if resolved not in out:
                out.append(resolved)
    return out


# ---------------------------------------------------------------------
# Reproduction runner.
# ---------------------------------------------------------------------

class ReproductionRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.project_root = Path(args.project_root).expanduser().resolve()
        self.data_dir = self.project_root / "datasets"
        self.results_root = self.project_root / args.results_dir
        self.preprocessed_dir = self.results_root / "preprocessed"
        self.resources_dir = self.results_root / "resources"
        self.logs_dir = self.results_root / "logs"

        for p in [
            self.results_root,
            self.preprocessed_dir,
            self.resources_dir,
            self.logs_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)

        self.raw_datasets = {
            1: self.data_dir / "dataset_1.csv",
            2: self.data_dir / "dataset_2.csv",
        }

        self.stopwords_path = self.resources_dir / "vietnamese-stopwords.txt"
        self.vncorenlp_dir = self.resources_dir / "vncorenlp"
        self.word2vec_gz = self.resources_dir / "wiki.vi.model.bin.gz"
        self.word2vec_bin = self.resources_dir / "wiki.vi.model.bin"

        self.preprocessed = {
            1: self.preprocessed_dir / "Dataset_depression_vi1.csv",
            2: self.preprocessed_dir / "Dataset_depression_vi2.csv",
        }

        self.models = normalize_models(args.models)
        self.datasets = list(args.datasets)

        set_global_seed(GLOBAL_SEED)

    # ------------------------------
    # Paths / status
    # ------------------------------

    def run_dir(self, model_name: str, dataset_id: int) -> Path:
        p = (
            self.results_root
            / "runs"
            / safe_slug(model_name)
            / f"d{dataset_id}"
        )
        p.mkdir(parents=True, exist_ok=True)
        return p

    def metrics_path(self, model_name: str, dataset_id: int) -> Path:
        return self.run_dir(model_name, dataset_id) / "metrics.json"

    def completed(self, model_name: str, dataset_id: int) -> bool:
        return self.metrics_path(model_name, dataset_id).exists()

    def should_skip(self, model_name: str, dataset_id: int) -> bool:
        return (
            self.args.skip_completed
            and self.completed(model_name, dataset_id)
        )

    # ------------------------------
    # Runtime / GPU
    # ------------------------------

    def runtime_audit(self) -> None:
        runtime: Dict[str, Any] = {
            "created_at_utc": utc_now(),
            "python": sys.version,
            "platform": platform.platform(),
            "global_seed_added_for_reproduction": GLOBAL_SEED,
            "neural_hparam_mode": self.args.neural_hparam_mode,
            "selected_models": self.models,
            "selected_datasets": self.datasets,
        }

        package_imports = {
            "numpy": "numpy",
            "pandas": "pandas",
            "sklearn": "sklearn",
            "tensorflow": "tensorflow",
            "transformers": "transformers",
            "keras_tuner": "keras_tuner",
            "gensim": "gensim",
            "underthesea": "underthesea",
            "py_vncorenlp": "py_vncorenlp",
        }

        for key, module_name in package_imports.items():
            try:
                mod = __import__(module_name)
                runtime[key] = getattr(mod, "__version__", "unknown")
            except Exception as exc:
                runtime[key] = f"unavailable: {type(exc).__name__}"

        gpu_devices = []
        try:
            import tensorflow as tf
            gpu_devices = [
                d.name for d in tf.config.list_physical_devices("GPU")
            ]
        except Exception:
            pass

        runtime["tensorflow_gpu_devices"] = gpu_devices

        # Also capture nvidia-smi when available.
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
            runtime["nvidia_smi"] = [
                line.strip() for line in r.stdout.splitlines() if line.strip()
            ]
        except Exception:
            runtime["nvidia_smi"] = []

        (self.results_root / "runtime_environment.json").write_text(
            json.dumps(runtime, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n=== Runtime audit ===")
        print("Project root:", self.project_root)
        print("Results root:", self.results_root)
        print("Python:", sys.version.split()[0])
        print("TensorFlow GPU devices:", gpu_devices)
        if runtime["nvidia_smi"]:
            print("NVIDIA GPU:", " | ".join(runtime["nvidia_smi"]))

        phobert_selected = (
            "phobert-base" in self.models
            or "phobert-large" in self.models
        )
        if phobert_selected and not gpu_devices and not self.args.allow_cpu_phobert:
            raise RuntimeError(
                "PhoBERT is selected but TensorFlow sees no GPU. "
                "Use a GPU environment, or pass --allow-cpu-phobert "
                "only if you intentionally accept a very slow CPU run."
            )

    # ------------------------------
    # Dataset audit
    # ------------------------------

    def audit_raw_datasets(self) -> None:
        rows = []

        for dataset_id, path in self.raw_datasets.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing raw dataset: {path}")

            digest = sha256_file(path)
            df = pd.read_csv(path)

            if digest != EXPECTED_RAW_SHA256[dataset_id]:
                raise RuntimeError(
                    f"Dataset {dataset_id} SHA-256 mismatch.\n"
                    f"Expected: {EXPECTED_RAW_SHA256[dataset_id]}\n"
                    f"Actual:   {digest}"
                )

            if list(df.columns) != ["text_vi", "sentiment"]:
                raise RuntimeError(
                    f"Dataset {dataset_id} columns changed: "
                    f"{df.columns.tolist()}"
                )

            labels = (
                df["sentiment"]
                .value_counts()
                .sort_index()
                .to_dict()
            )

            assert len(df) == EXPECTED_RAW_ROWS[dataset_id]
            assert labels == EXPECTED_RAW_LABELS[dataset_id]
            assert not df[["text_vi", "sentiment"]].isna().any().any()

            rows.append({
                "dataset": dataset_id,
                "rows": len(df),
                "label_0": labels.get(0, 0),
                "label_1": labels.get(1, 0),
                "sha256": digest,
            })

        pd.DataFrame(rows).to_csv(
            self.results_root / "raw_dataset_audit.csv",
            index=False,
        )
        print("\nRAW DATASET AUDIT: PASS")

    # ------------------------------
    # Legacy preprocessing
    # ------------------------------

    def ensure_stopwords(self) -> set[str]:
        if not self.stopwords_path.exists():
            print("Downloading legacy stopword source...")
            r = requests.get(STOPWORDS_URL, timeout=60)
            r.raise_for_status()
            self.stopwords_path.write_bytes(r.content)

        text = self.stopwords_path.read_text(encoding="utf-8")
        stopwords = set(line.strip() for line in text.splitlines())

        print("Stopwords:", len(stopwords))
        print("Stopword SHA256:", sha256_file(self.stopwords_path))
        return stopwords

    def ensure_vncorenlp(self):
        try:
            subprocess.run(
                ["java", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Java was not found. VnCoreNLP is required."
            ) from exc

        import py_vncorenlp

        self.vncorenlp_dir.mkdir(parents=True, exist_ok=True)
        if len(list(self.vncorenlp_dir.iterdir())) == 0:
            print("Downloading VnCoreNLP resources...")
            py_vncorenlp.download_model(
                save_dir=str(self.vncorenlp_dir)
            )

        return py_vncorenlp.VnCoreNLP(
            annotators=["wseg"],
            save_dir=str(self.vncorenlp_dir),
        )

    @staticmethod
    def base_text_cleaner(text: str) -> str:
        if not isinstance(text, str) or not text:
            return ""

        text = unicodedata.normalize("NFC", text).lower().strip()

        for k, v in ABBREVIATIONS.items():
            pattern = r"\b{}\b".format(re.escape(k))
            text = re.sub(pattern, v, text)

        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"[@#]\w+", " ", text)
        text = re.sub(r"\bhttps\b", " ", text)
        text = re.sub(r"\bhttp\b", " ", text)
        text = re.sub(r"\bper\b", "", text)

        # Legacy executed behavior: remove emojis.
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess_for_tfidf(
        self,
        text: str,
        segmenter,
        stopwords: set[str],
    ) -> str:
        from underthesea import word_tokenize

        cleaned_text = self.base_text_cleaner(text)
        segmented_sentences_list = segmenter.word_segment(cleaned_text)
        full_segmented_text = (
            segmented_sentences_list[0]
            if segmented_sentences_list
            else ""
        )
        word_tokens = word_tokenize(full_segmented_text)

        filtered_words = []
        for word in word_tokens:
            if (
                word.replace("_", "").isalpha()
                and word not in stopwords
                and len(word) > 1
            ):
                filtered_words.append(word)

        return " ".join(filtered_words)

    def preprocess_for_dl(self, text: str, segmenter) -> str:
        cleaned_text = self.base_text_cleaner(text)
        segmented_words = segmenter.word_segment(cleaned_text)
        return " ".join(segmented_words)

    def generate_preprocessed_datasets(self) -> None:
        if (
            not self.args.force_preprocess
            and all(p.exists() for p in self.preprocessed.values())
        ):
            print("\nReusing existing preprocessed datasets.")
            return

        stopwords = self.ensure_stopwords()
        segmenter = self.ensure_vncorenlp()

        provenance = {
            "created_at_utc": utc_now(),
            "stopwords_url": STOPWORDS_URL,
            "stopwords_sha256": sha256_file(self.stopwords_path),
            "legacy_emoji_behavior":
                "emoji dictionary was defined but cleaner removed emoji",
            "raw_sha256": EXPECTED_RAW_SHA256,
        }

        for dataset_id in [1, 2]:
            print(f"\nPreprocessing Dataset {dataset_id}...")
            df = pd.read_csv(self.raw_datasets[dataset_id]).copy()
            df["source_row_id"] = np.arange(len(df), dtype=int)

            df["text_clean_tfidf"] = df["text_vi"].apply(
                lambda x: self.preprocess_for_tfidf(
                    x, segmenter, stopwords
                )
            )
            df["text_clean_dl"] = df["text_vi"].apply(
                lambda x: self.preprocess_for_dl(x, segmenter)
            )

            df.to_csv(
                self.preprocessed[dataset_id],
                index=False,
                encoding="utf-8-sig",
            )
            print("Saved:", self.preprocessed[dataset_id])

        (self.results_root / "preprocessing_provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        del segmenter
        gc.collect()

    def load_model_df(
        self,
        dataset_id: int,
        text_col: str,
    ) -> pd.DataFrame:
        df = pd.read_csv(self.preprocessed[dataset_id])
        df = df.dropna(subset=[text_col]).copy()
        df[text_col] = df[text_col].astype(str)
        df["sentiment"] = df["sentiment"].astype(int)
        return df

    def audit_preprocessing(self) -> None:
        rows = []

        for d in [1, 2]:
            tfidf_df = self.load_model_df(d, "text_clean_tfidf")
            dl_df = self.load_model_df(d, "text_clean_dl")

            _, tfidf_test = train_test_split(
                tfidf_df,
                test_size=0.2,
                random_state=42,
                stratify=tfidf_df["sentiment"],
            )
            _, dl_test = train_test_split(
                dl_df,
                test_size=0.2,
                random_state=42,
                stratify=dl_df["sentiment"],
            )

            tfidf_labels = (
                tfidf_test["sentiment"]
                .value_counts()
                .sort_index()
                .to_dict()
            )
            dl_labels = (
                dl_test["sentiment"]
                .value_counts()
                .sort_index()
                .to_dict()
            )

            rows.append({
                "dataset": d,
                "tfidf_rows": len(tfidf_df),
                "expected_tfidf_rows":
                    EXPECTED_TFIDF_MODEL_ROWS[d],
                "tfidf_test_n": len(tfidf_test),
                "tfidf_test_labels": str(tfidf_labels),
                "dl_rows": len(dl_df),
                "expected_dl_rows": EXPECTED_RAW_ROWS[d],
                "dl_test_n": len(dl_test),
                "dl_test_labels": str(dl_labels),
            })

            problems = []
            if len(tfidf_df) != EXPECTED_TFIDF_MODEL_ROWS[d]:
                problems.append(
                    f"TF-IDF rows {len(tfidf_df)} != "
                    f"{EXPECTED_TFIDF_MODEL_ROWS[d]}"
                )
            if len(tfidf_test) != EXPECTED_TFIDF_TEST[d]["rows"]:
                problems.append("TF-IDF test size differs")
            if tfidf_labels != EXPECTED_TFIDF_TEST[d]["labels"]:
                problems.append("TF-IDF test labels differ")
            if len(dl_df) != EXPECTED_RAW_ROWS[d]:
                problems.append(
                    f"DL rows {len(dl_df)} != {EXPECTED_RAW_ROWS[d]}"
                )
            if len(dl_test) != EXPECTED_DL_TEST[d]["rows"]:
                problems.append("DL test size differs")
            if dl_labels != EXPECTED_DL_TEST[d]["labels"]:
                problems.append("DL test labels differ")

            if problems and self.args.strict_preprocess_audit:
                raise RuntimeError(
                    f"Dataset {d} preprocessing audit failed: "
                    + "; ".join(problems)
                )

        pd.DataFrame(rows).to_csv(
            self.results_root / "preprocessing_audit.csv",
            index=False,
        )
        print("\nPREPROCESSING / SPLIT AUDIT: PASS")

    # ------------------------------
    # Evaluation
    # ------------------------------

    def evaluate_and_save(
        self,
        model_name: str,
        dataset_id: int,
        test_df: pd.DataFrame,
        y_pred,
        config: Dict[str, Any],
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        y_true = test_df["sentiment"].astype(int).to_numpy()
        y_pred = np.asarray(y_pred).reshape(-1).astype(int)

        if len(y_true) != len(y_pred):
            raise RuntimeError(
                f"{model_name} D{dataset_id}: predictions "
                f"{len(y_pred)} != labels {len(y_true)}"
            )

        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        metrics = {
            "created_at_utc": utc_now(),
            "model": model_name,
            "dataset": dataset_id,
            "n_test": int(len(y_true)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_class1": float(
                precision_score(
                    y_true, y_pred, pos_label=1, zero_division=0
                )
            ),
            "recall_class1": float(
                recall_score(
                    y_true, y_pred, pos_label=1, zero_division=0
                )
            ),
            "f1_class1": float(
                f1_score(
                    y_true, y_pred, pos_label=1, zero_division=0
                )
            ),
            "macro_f1": float(
                f1_score(
                    y_true, y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "confusion_matrix_labels_0_1": cm.tolist(),
            "classification_report": report,
            "config": config,
        }

        if extra_metrics:
            metrics.update(extra_metrics)

        rdir = self.run_dir(model_name, dataset_id)

        pred_df = pd.DataFrame({
            "source_row_id":
                test_df["source_row_id"].astype(int).to_numpy(),
            "gold_label": y_true,
            "predicted_label": y_pred,
        })
        pred_df.to_csv(
            rdir / "predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        (rdir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(
            f"\n{model_name} | D{dataset_id} | n={len(y_true)} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"P1={metrics['precision_class1']:.4f} | "
            f"R1={metrics['recall_class1']:.4f} | "
            f"F1={metrics['f1_class1']:.4f} | "
            f"macro-F1={metrics['macro_f1']:.4f}"
        )
        print(
            "Confusion matrix [[TN,FP],[FN,TP]]:",
            cm.tolist(),
        )
        return metrics

    # ------------------------------
    # TF-IDF + LR
    # ------------------------------

    def run_tfidf_lr(self, dataset_id: int) -> None:
        model_name = "TF-IDF + LR"

        if self.should_skip(model_name, dataset_id):
            print(
                f"SKIP: {model_name} D{dataset_id} already complete."
            )
            return

        df = self.load_model_df(
            dataset_id, "text_clean_tfidf"
        )
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["sentiment"],
        )

        X_train = train_df["text_clean_tfidf"].astype(str)
        y_train = train_df["sentiment"].astype(int)
        X_test = test_df["text_clean_tfidf"].astype(str)

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ])

        param_grid = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__max_features": [1000, 5000, 10000],
            "clf__C": [0.1, 0.5, 1, 2, 10],
        }

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=self.args.n_jobs,
            verbose=1,
        )

        t0 = time.time()
        grid.fit(X_train, y_train)
        y_pred = grid.best_estimator_.predict(X_test)
        elapsed = time.time() - t0

        rdir = self.run_dir(model_name, dataset_id)

        pd.DataFrame(grid.cv_results_).to_csv(
            rdir / "gridsearch_results.csv",
            index=False,
        )

        if self.args.save_models:
            joblib.dump(
                grid.best_estimator_,
                rdir / "model.joblib",
            )

        self.evaluate_and_save(
            model_name,
            dataset_id,
            test_df,
            y_pred,
            config={
                "text_column": "text_clean_tfidf",
                "test_size": 0.2,
                "random_state": 42,
                "grid_best_params": grid.best_params_,
                "grid_best_macro_f1":
                    float(grid.best_score_),
                "param_grid": {
                    k: [str(x) for x in v]
                    for k, v in param_grid.items()
                },
            },
            extra_metrics={"elapsed_seconds": elapsed},
        )

    # ------------------------------
    # TF-IDF + SVM
    # ------------------------------

    def run_tfidf_svm(self, dataset_id: int) -> None:
        model_name = "TF-IDF + SVM"

        if self.should_skip(model_name, dataset_id):
            print(
                f"SKIP: {model_name} D{dataset_id} already complete."
            )
            return

        df = self.load_model_df(
            dataset_id, "text_clean_tfidf"
        )
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["sentiment"],
        )

        X_train = train_df["text_clean_tfidf"].astype(str)
        y_train = train_df["sentiment"].astype(int)
        X_test = test_df["text_clean_tfidf"].astype(str)

        pipe = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2)),
            ),
            (
                "clf",
                SVC(
                    class_weight="balanced",
                    random_state=42,
                    probability=True,
                ),
            ),
        ])

        param_grid = {
            "tfidf__max_features": [10000],
            "clf__C": [1, 10],
            "clf__kernel": ["linear", "rbf"],
        }

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=self.args.n_jobs,
            verbose=1,
        )

        t0 = time.time()
        grid.fit(X_train, y_train)
        y_pred = grid.best_estimator_.predict(X_test)
        elapsed = time.time() - t0

        rdir = self.run_dir(model_name, dataset_id)

        pd.DataFrame(grid.cv_results_).to_csv(
            rdir / "gridsearch_results.csv",
            index=False,
        )

        if self.args.save_models:
            joblib.dump(
                grid.best_estimator_,
                rdir / "model.joblib",
            )

        self.evaluate_and_save(
            model_name,
            dataset_id,
            test_df,
            y_pred,
            config={
                "text_column": "text_clean_tfidf",
                "test_size": 0.2,
                "random_state": 42,
                "fixed_ngram_range": [1, 2],
                "grid_best_params": grid.best_params_,
                "grid_best_macro_f1":
                    float(grid.best_score_),
                "param_grid": param_grid,
            },
            extra_metrics={"elapsed_seconds": elapsed},
        )

    # ------------------------------
    # Word2Vec
    # ------------------------------

    def ensure_word2vec(self) -> Path:
        if not self.word2vec_bin.exists():
            if not self.word2vec_gz.exists():
                print("Downloading legacy Vietnamese Word2Vec...")
                with requests.get(
                    WORD2VEC_URL,
                    stream=True,
                    timeout=120,
                ) as r:
                    r.raise_for_status()
                    with self.word2vec_gz.open("wb") as f:
                        for chunk in r.iter_content(
                            chunk_size=1024 * 1024
                        ):
                            if chunk:
                                f.write(chunk)

            print("Decompressing Word2Vec...")
            with gzip.open(
                self.word2vec_gz, "rb"
            ) as src, self.word2vec_bin.open("wb") as dst:
                shutil.copyfileobj(src, dst)

        print("Word2Vec SHA256:", sha256_file(self.word2vec_bin))
        return self.word2vec_bin

    def load_word2vec(self):
        import gensim

        path = self.ensure_word2vec()
        print("Loading Word2Vec into memory...")
        return gensim.models.KeyedVectors.load_word2vec_format(
            str(path),
            binary=True,
        )

    def prepare_word2vec_split(
        self,
        dataset_id: int,
        wv_model,
    ) -> Dict[str, Any]:
        from tensorflow.keras.preprocessing.sequence import (
            pad_sequences,
        )
        from tensorflow.keras.preprocessing.text import Tokenizer

        df = self.load_model_df(
            dataset_id, "text_clean_dl"
        )

        trainval_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["sentiment"],
        )

        train_df, val_df = train_test_split(
            trainval_df,
            test_size=0.125,
            random_state=42,
            stratify=trainval_df["sentiment"],
        )

        expected = EXPECTED_DL_TEST[dataset_id]
        assert len(test_df) == expected["rows"]
        assert (
            test_df["sentiment"]
            .value_counts()
            .sort_index()
            .to_dict()
            == expected["labels"]
        )

        tokenizer = Tokenizer(
            num_words=VOCAB_SIZE,
            oov_token="<unk>",
        )
        tokenizer.fit_on_texts(
            train_df["text_clean_dl"].astype(str)
        )
        word_index = tokenizer.word_index

        def encode(series):
            return pad_sequences(
                tokenizer.texts_to_sequences(
                    series.astype(str)
                ),
                maxlen=WORD2VEC_MAX_LEN,
                padding="post",
                truncating="post",
            )

        X_train = encode(train_df["text_clean_dl"])
        X_val = encode(val_df["text_clean_dl"])
        X_test = encode(test_df["text_clean_dl"])

        embedding_dim = int(wv_model.vector_size)
        embedding_matrix = np.zeros(
            (len(word_index) + 1, embedding_dim),
            dtype=np.float32,
        )

        hits = 0
        misses = 0

        for word, i in word_index.items():
            if word in wv_model:
                embedding_matrix[i] = wv_model[word]
                hits += 1
            else:
                misses += 1

        return {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train":
                train_df["sentiment"].astype(int).to_numpy(),
            "y_val":
                val_df["sentiment"].astype(int).to_numpy(),
            "embedding_matrix": embedding_matrix,
            "embedding_dim": embedding_dim,
            "word_index_size": len(word_index),
            "hits": hits,
            "misses": misses,
        }

    # ------------------------------
    # BiLSTM
    # ------------------------------

    def run_bilstm(
        self,
        dataset_id: int,
        wv_model,
    ) -> None:
        model_name = "Word2Vec + BiLSTM"

        if self.should_skip(model_name, dataset_id):
            print(
                f"SKIP: {model_name} D{dataset_id} already complete."
            )
            return

        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import (
            Bidirectional,
            Dense,
            Dropout,
            Embedding,
            LSTM,
        )
        from tensorflow.keras.models import Sequential

        set_global_seed(GLOBAL_SEED)
        data = self.prepare_word2vec_split(
            dataset_id, wv_model
        )

        def build_model(hps):
            model = Sequential([
                Embedding(
                    input_dim=data["word_index_size"] + 1,
                    output_dim=data["embedding_dim"],
                    weights=[data["embedding_matrix"]],
                    trainable=False,
                ),
                Bidirectional(
                    LSTM(units=int(hps["lstm_units"]))
                ),
                Dropout(
                    rate=float(hps["dropout_rate"])
                ),
                Dense(
                    units=int(hps["dense_units"]),
                    activation="relu",
                ),
                Dense(1, activation="sigmoid"),
            ])

            model.compile(
                loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=float(
                        hps["learning_rate"]
                    )
                ),
                metrics=["accuracy"],
            )
            return model

        if self.args.neural_hparam_mode == "legacy_recorded":
            hps = dict(
                LEGACY_BILSTM_HPS[dataset_id]
            )
            tuning_note = (
                "recorded best hyperparameters from "
                "archived notebook output"
            )
        else:
            import keras_tuner as kt

            def tuner_builder(hp):
                return build_model({
                    "lstm_units":
                        hp.Int(
                            "lstm_units",
                            32,
                            128,
                            step=32,
                        ),
                    "dropout_rate":
                        hp.Float(
                            "dropout_rate",
                            0.2,
                            0.5,
                            step=0.1,
                        ),
                    "dense_units":
                        hp.Int(
                            "dense_units",
                            32,
                            128,
                            step=32,
                        ),
                    "learning_rate":
                        hp.Choice(
                            "learning_rate",
                            [1e-2, 1e-3, 1e-4],
                        ),
                })

            tuner_dir = (
                self.run_dir(model_name, dataset_id)
                / "tuner"
            )

            tuner = kt.RandomSearch(
                tuner_builder,
                objective="val_accuracy",
                max_trials=10,
                executions_per_trial=1,
                directory=str(tuner_dir),
                project_name="bilstm_hyper_tuning",
                overwrite=True,
                seed=GLOBAL_SEED,
            )

            tuner.search(
                data["X_train"],
                data["y_train"],
                epochs=15,
                validation_data=(
                    data["X_val"],
                    data["y_val"],
                ),
                callbacks=[
                    EarlyStopping(
                        monitor="val_loss",
                        patience=3,
                    )
                ],
                verbose=1,
            )

            best = tuner.get_best_hyperparameters(1)[0]

            hps = {
                "lstm_units":
                    best.get("lstm_units"),
                "dropout_rate":
                    best.get("dropout_rate"),
                "dense_units":
                    best.get("dense_units"),
                "learning_rate":
                    best.get("learning_rate"),
            }
            tuning_note = (
                "controlled retune with tuner seed=42; "
                "legacy tuner had no seed"
            )

        set_global_seed(GLOBAL_SEED)
        model = build_model(hps)

        t0 = time.time()
        history = model.fit(
            data["X_train"],
            data["y_train"],
            epochs=15,
            batch_size=32,
            validation_data=(
                data["X_val"],
                data["y_val"],
            ),
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    restore_best_weights=True,
                    verbose=1,
                )
            ],
            verbose=1,
        )
        elapsed = time.time() - t0

        y_pred = (
            model.predict(
                data["X_test"],
                verbose=0,
            ).reshape(-1)
            > 0.5
        ).astype(int)

        rdir = self.run_dir(model_name, dataset_id)

        pd.DataFrame(
            history.history
        ).to_csv(
            rdir / "training_history.csv",
            index=False,
        )

        if self.args.save_models:
            model.save(rdir / "model.keras")

        self.evaluate_and_save(
            model_name,
            dataset_id,
            data["test_df"],
            y_pred,
            config={
                "text_column": "text_clean_dl",
                "split": "70/10/20 effective",
                "test_size": 0.2,
                "trainval_to_val": 0.125,
                "split_random_state": 42,
                "global_training_seed_added":
                    GLOBAL_SEED,
                "vocab_size": VOCAB_SIZE,
                "max_len": WORD2VEC_MAX_LEN,
                "embedding_trainable": False,
                "embedding_dim":
                    data["embedding_dim"],
                "word2vec_hits": data["hits"],
                "word2vec_misses":
                    data["misses"],
                "hyperparameters": hps,
                "hyperparameter_mode":
                    self.args.neural_hparam_mode,
                "tuning_note": tuning_note,
                "epochs_max": 15,
                "batch_size": 32,
                "early_stopping_patience": 3,
            },
            extra_metrics={
                "elapsed_seconds": elapsed
            },
        )

        del model, data
        tf.keras.backend.clear_session()
        gc.collect()

    # ------------------------------
    # CNN
    # ------------------------------

    def run_cnn(
        self,
        dataset_id: int,
        wv_model,
    ) -> None:
        model_name = "Word2Vec + CNN"

        if self.should_skip(model_name, dataset_id):
            print(
                f"SKIP: {model_name} D{dataset_id} already complete."
            )
            return

        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import (
            Conv1D,
            Dense,
            Dropout,
            Embedding,
            GlobalMaxPooling1D,
        )
        from tensorflow.keras.models import Sequential

        set_global_seed(GLOBAL_SEED)
        data = self.prepare_word2vec_split(
            dataset_id, wv_model
        )

        def build_model(hps):
            model = Sequential([
                Embedding(
                    input_dim=data["word_index_size"] + 1,
                    output_dim=data["embedding_dim"],
                    weights=[data["embedding_matrix"]],
                    trainable=False,
                ),
                Conv1D(
                    filters=int(hps["filters"]),
                    kernel_size=int(hps["kernel_size"]),
                    activation="relu",
                ),
                GlobalMaxPooling1D(),
                Dense(
                    units=int(hps["dense_units"]),
                    activation="relu",
                ),
                Dropout(
                    rate=float(hps["dropout_rate"])
                ),
                Dense(1, activation="sigmoid"),
            ])

            model.compile(
                loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=float(
                        hps["learning_rate"]
                    )
                ),
                metrics=["accuracy"],
            )
            return model

        if self.args.neural_hparam_mode == "legacy_recorded":
            hps = dict(
                LEGACY_CNN_HPS[dataset_id]
            )
            tuning_note = (
                "recorded best hyperparameters from "
                "archived notebook output"
            )
        else:
            import keras_tuner as kt

            def tuner_builder(hp):
                return build_model({
                    "filters":
                        hp.Int(
                            "filters",
                            64,
                            256,
                            step=64,
                        ),
                    "kernel_size":
                        hp.Choice(
                            "kernel_size",
                            [3, 5, 7],
                        ),
                    "dense_units":
                        hp.Int(
                            "dense_units",
                            32,
                            128,
                            step=32,
                        ),
                    "dropout_rate":
                        hp.Float(
                            "dropout_rate",
                            0.2,
                            0.5,
                            step=0.1,
                        ),
                    "learning_rate":
                        hp.Choice(
                            "learning_rate",
                            [1e-2, 1e-3, 1e-4],
                        ),
                })

            tuner_dir = (
                self.run_dir(model_name, dataset_id)
                / "tuner"
            )

            tuner = kt.RandomSearch(
                tuner_builder,
                objective="val_accuracy",
                max_trials=10,
                executions_per_trial=1,
                directory=str(tuner_dir),
                project_name="cnn_hyper_tuning",
                overwrite=True,
                seed=GLOBAL_SEED,
            )

            tuner.search(
                data["X_train"],
                data["y_train"],
                epochs=15,
                validation_data=(
                    data["X_val"],
                    data["y_val"],
                ),
                callbacks=[
                    EarlyStopping(
                        monitor="val_loss",
                        patience=3,
                    )
                ],
                verbose=1,
            )

            best = tuner.get_best_hyperparameters(1)[0]

            hps = {
                "filters": best.get("filters"),
                "kernel_size":
                    best.get("kernel_size"),
                "dense_units":
                    best.get("dense_units"),
                "dropout_rate":
                    best.get("dropout_rate"),
                "learning_rate":
                    best.get("learning_rate"),
            }
            tuning_note = (
                "controlled retune with tuner seed=42; "
                "legacy tuner had no seed"
            )

        set_global_seed(GLOBAL_SEED)
        model = build_model(hps)

        t0 = time.time()
        history = model.fit(
            data["X_train"],
            data["y_train"],
            epochs=15,
            batch_size=32,
            validation_data=(
                data["X_val"],
                data["y_val"],
            ),
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    restore_best_weights=True,
                    verbose=1,
                )
            ],
            verbose=1,
        )
        elapsed = time.time() - t0

        y_pred = (
            model.predict(
                data["X_test"],
                verbose=0,
            ).reshape(-1)
            > 0.5
        ).astype(int)

        rdir = self.run_dir(model_name, dataset_id)

        pd.DataFrame(
            history.history
        ).to_csv(
            rdir / "training_history.csv",
            index=False,
        )

        if self.args.save_models:
            model.save(rdir / "model.keras")

        self.evaluate_and_save(
            model_name,
            dataset_id,
            data["test_df"],
            y_pred,
            config={
                "text_column": "text_clean_dl",
                "split": "70/10/20 effective",
                "test_size": 0.2,
                "trainval_to_val": 0.125,
                "split_random_state": 42,
                "global_training_seed_added":
                    GLOBAL_SEED,
                "vocab_size": VOCAB_SIZE,
                "max_len": WORD2VEC_MAX_LEN,
                "embedding_trainable": False,
                "embedding_dim":
                    data["embedding_dim"],
                "word2vec_hits": data["hits"],
                "word2vec_misses":
                    data["misses"],
                "hyperparameters": hps,
                "hyperparameter_mode":
                    self.args.neural_hparam_mode,
                "tuning_note": tuning_note,
                "epochs_max": 15,
                "batch_size": 32,
                "early_stopping_patience": 3,
            },
            extra_metrics={
                "elapsed_seconds": elapsed
            },
        )

        del model, data
        tf.keras.backend.clear_session()
        gc.collect()

    # ------------------------------
    # PhoBERT
    # ------------------------------

    def run_phobert(
        self,
        dataset_id: int,
        variant: str,
    ) -> None:
        import tensorflow as tf
        from transformers import (
            AutoTokenizer,
            TFAutoModel,
        )

        if variant == "base":
            model_name = "PhoBERT-base"
            hf_name = "vinai/phobert-base"
            encoder_trainable = False
            epochs = 25
            batch_size = 16
        elif variant == "large":
            model_name = "PhoBERT-large"
            hf_name = "vinai/phobert-large"
            encoder_trainable = True
            epochs = 10
            batch_size = 32
        else:
            raise ValueError(
                "variant must be 'base' or 'large'"
            )

        if self.should_skip(model_name, dataset_id):
            print(
                f"SKIP: {model_name} D{dataset_id} already complete."
            )
            return

        tf.keras.backend.clear_session()
        gc.collect()
        set_global_seed(GLOBAL_SEED)

        df = self.load_model_df(
            dataset_id, "text_clean_dl"
        )

        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["sentiment"],
        )

        expected = EXPECTED_DL_TEST[dataset_id]
        assert len(test_df) == expected["rows"]
        assert (
            test_df["sentiment"]
            .value_counts()
            .sort_index()
            .to_dict()
            == expected["labels"]
        )

        tokenizer = AutoTokenizer.from_pretrained(
            hf_name
        )

        max_len = 256

        train_tok = tokenizer(
            train_df["text_clean_dl"]
            .astype(str)
            .tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="tf",
        )

        test_tok = tokenizer(
            test_df["text_clean_dl"]
            .astype(str)
            .tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="tf",
        )

        encoder = TFAutoModel.from_pretrained(hf_name)
        encoder.trainable = encoder_trainable

        input_ids = tf.keras.layers.Input(
            shape=(max_len,),
            dtype=tf.int32,
            name="input_ids",
        )
        attention_mask = tf.keras.layers.Input(
            shape=(max_len,),
            dtype=tf.int32,
            name="attention_mask",
        )

        outputs = encoder(
            [input_ids, attention_mask]
        )
        pooled = outputs["pooler_output"]

        x = tf.keras.layers.Dropout(0.3)(pooled)
        x = tf.keras.layers.Dense(
            128,
            activation="relu",
        )(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        prediction = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
        )(x)

        model = tf.keras.Model(
            inputs=[input_ids, attention_mask],
            outputs=prediction,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=3e-5
            ),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        train_inputs = {
            "input_ids": train_tok["input_ids"],
            "attention_mask":
                train_tok["attention_mask"],
        }
        test_inputs = {
            "input_ids": test_tok["input_ids"],
            "attention_mask":
                test_tok["attention_mask"],
        }

        t0 = time.time()

        history = model.fit(
            train_inputs,
            train_df["sentiment"]
            .astype(int)
            .to_numpy(),
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=2,
                    restore_best_weights=True,
                    verbose=1,
                )
            ],
            verbose=1,
        )

        elapsed = time.time() - t0

        probs = model.predict(
            test_inputs,
            verbose=0,
        ).reshape(-1)
        y_pred = (probs > 0.5).astype(int)

        rdir = self.run_dir(
            model_name,
            dataset_id,
        )

        pd.DataFrame(
            history.history
        ).to_csv(
            rdir / "training_history.csv",
            index=False,
        )

        if self.args.save_models:
            model.save_weights(
                rdir / "best_weights.weights.h5"
            )

        self.evaluate_and_save(
            model_name,
            dataset_id,
            test_df,
            y_pred,
            config={
                "text_column": "text_clean_dl",
                "hf_model": hf_name,
                "encoder_trainable":
                    encoder_trainable,
                "max_len": max_len,
                "test_size": 0.2,
                "split_random_state": 42,
                "validation_split_of_train": 0.2,
                "effective_split": "64/16/20",
                "global_training_seed_added":
                    GLOBAL_SEED,
                "head":
                    "pooler_output -> dropout(0.3) -> "
                    "dense(128,relu) -> dropout(0.2) -> sigmoid",
                "optimizer": "Adam",
                "learning_rate": 3e-5,
                "epochs_max": epochs,
                "batch_size": batch_size,
                "early_stopping_patience": 2,
            },
            extra_metrics={
                "elapsed_seconds": elapsed
            },
        )

        del model
        del encoder
        del tokenizer
        del train_tok
        del test_tok

        tf.keras.backend.clear_session()
        gc.collect()

    # ------------------------------
    # Result consolidation
    # ------------------------------

    def collect_completed_results(self) -> pd.DataFrame:
        rows = []

        runs_root = self.results_root / "runs"
        if not runs_root.exists():
            return pd.DataFrame()

        for p in runs_root.glob("*/*/metrics.json"):
            try:
                m = json.loads(
                    p.read_text(encoding="utf-8")
                )
                rows.append({
                    "model": m["model"],
                    "dataset":
                        int(m["dataset"]),
                    "n_test":
                        int(m["n_test"]),
                    "accuracy":
                        float(m["accuracy"]),
                    "precision_class1":
                        float(m["precision_class1"]),
                    "recall_class1":
                        float(m["recall_class1"]),
                    "f1_class1":
                        float(m["f1_class1"]),
                    "macro_f1":
                        float(m["macro_f1"]),
                    "metrics_file": str(p),
                })
            except Exception as exc:
                print(
                    "Could not read result:",
                    p,
                    exc,
                )

        return pd.DataFrame(rows)

    def write_summary(self) -> None:
        reproduced = self.collect_completed_results()

        if reproduced.empty:
            print("\nNo completed model results yet.")
            return

        reproduced = (
            reproduced
            .sort_values(["model", "dataset"])
            .reset_index(drop=True)
        )

        comparison = LEGACY_REFERENCE.merge(
            reproduced,
            on=["model", "dataset"],
            how="left",
        )

        comparison["accuracy_delta_pp"] = (
            comparison["accuracy"]
            - comparison["legacy_accuracy"]
        ) * 100.0

        comparison["f1_class1_delta_pp"] = (
            comparison["f1_class1"]
            - comparison["legacy_f1_class1"]
        ) * 100.0

        reproduced.to_csv(
            self.results_root
            / "all_reproduced_results.csv",
            index=False,
        )
        comparison.to_csv(
            self.results_root
            / "comparison_with_legacy.csv",
            index=False,
        )

        print("\n=== Completed reproduction results ===")
        show_cols = [
            "model",
            "dataset",
            "n_test",
            "accuracy",
            "precision_class1",
            "recall_class1",
            "f1_class1",
            "macro_f1",
        ]
        print(
            reproduced[show_cols].to_string(
                index=False
            )
        )

        print(
            "\nSaved:",
            self.results_root
            / "comparison_with_legacy.csv",
        )

    # ------------------------------
    # Main manager
    # ------------------------------

    def run(self) -> None:
        print(
            "\n=================================================="
        )
        print(
            "Vietnamese Depression Previous-Experiment Reproduction"
        )
        print(
            "=================================================="
        )
        print("Selected models:", self.models)
        print("Selected datasets:", self.datasets)

        canonical_abbr = json.dumps(
            ABBREVIATIONS,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        actual_abbr_sha = hashlib.sha256(
            canonical_abbr.encode("utf-8")
        ).hexdigest()

        if len(ABBREVIATIONS) != EXPECTED_ABBREVIATIONS_COUNT:
            raise RuntimeError(
                f"Legacy ABBREVIATIONS count mismatch: "
                f"{len(ABBREVIATIONS)} != {EXPECTED_ABBREVIATIONS_COUNT}"
            )
        if actual_abbr_sha != EXPECTED_ABBREVIATIONS_SHA256:
            raise RuntimeError(
                "Legacy ABBREVIATIONS SHA-256 mismatch."
            )

        print(
            "Legacy abbreviations audit: PASS | "
            f"{len(ABBREVIATIONS)} entries | "
            f"SHA256={actual_abbr_sha}"
        )

        print(
            "Skip completed:",
            self.args.skip_completed,
        )
        print(
            "Neural hyperparameter mode:",
            self.args.neural_hparam_mode,
        )

        self.runtime_audit()
        self.audit_raw_datasets()
        self.generate_preprocessed_datasets()
        self.audit_preprocessing()

        if self.args.prepare_only:
            print(
                "\nPREPARE-ONLY mode complete. "
                "No model training was started."
            )
            self.write_summary()
            return

        # Classical models.
        for d in self.datasets:
            if "lr" in self.models:
                self.run_tfidf_lr(d)
                self.write_summary()

            if "svm" in self.models:
                self.run_tfidf_svm(d)
                self.write_summary()

        # Word2Vec models share one loaded embedding model.
        if (
            "bilstm" in self.models
            or "cnn" in self.models
        ):
            wv_model = self.load_word2vec()

            try:
                for d in self.datasets:
                    if "bilstm" in self.models:
                        self.run_bilstm(d, wv_model)
                        self.write_summary()

                    if "cnn" in self.models:
                        self.run_cnn(d, wv_model)
                        self.write_summary()
            finally:
                del wv_model
                gc.collect()

        # PhoBERT base.
        if "phobert-base" in self.models:
            for d in self.datasets:
                self.run_phobert(d, "base")
                self.write_summary()

        # PhoBERT large.
        if "phobert-large" in self.models:
            for d in self.datasets:
                self.run_phobert(d, "large")
                self.write_summary()

        print(
            "\nSELECTED REPRODUCTION RUNS FINISHED."
        )
        self.write_summary()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Reproduce all archived Vietnamese depression "
            "classification experiments."
        )
    )

    p.add_argument(
        "--project-root",
        default=".",
        help=(
            "Project root containing datasets/dataset_1.csv "
            "and datasets/dataset_2.csv."
        ),
    )

    p.add_argument(
        "--results-dir",
        default="reproduction_results_v1",
        help="Output directory under the project root.",
    )

    p.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=(
            "Models: all, lr, svm, bilstm, cnn, "
            "phobert-base, phobert-large. "
            "Space- or comma-separated."
        ),
    )

    p.add_argument(
        "--datasets",
        nargs="+",
        type=int,
        choices=[1, 2],
        default=[1, 2],
        help="Dataset IDs to run.",
    )

    p.add_argument(
        "--neural-hparam-mode",
        choices=["legacy_recorded", "retune"],
        default="legacy_recorded",
        help=(
            "Use recorded best BiLSTM/CNN hyperparameters "
            "or rerun KerasTuner."
        ),
    )

    p.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Regenerate preprocessing even if files exist.",
    )

    p.add_argument(
        "--no-strict-preprocess-audit",
        dest="strict_preprocess_audit",
        action="store_false",
        help=(
            "Do not stop if preprocessing row counts differ "
            "from the archived historical counts."
        ),
    )
    p.set_defaults(strict_preprocess_audit=True)

    p.add_argument(
        "--rerun-completed",
        dest="skip_completed",
        action="store_false",
        help="Train even when metrics.json already exists.",
    )
    p.set_defaults(skip_completed=True)

    p.add_argument(
        "--save-models",
        action="store_true",
        help=(
            "Save trained model weights/artifacts. "
            "Metrics and predictions are always saved."
        ),
    )

    p.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Audit raw data, regenerate preprocessing, "
            "audit splits, then stop before training."
        ),
    )

    p.add_argument(
        "--allow-cpu-phobert",
        action="store_true",
        help=(
            "Allow PhoBERT without a TensorFlow GPU. "
            "Not recommended."
        ),
    )

    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help=(
            "Parallel jobs for scikit-learn GridSearchCV. "
            "Default -1 uses all CPU cores."
        ),
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    runner = ReproductionRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
