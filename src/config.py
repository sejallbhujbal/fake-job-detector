"""
Central configuration: paths, random seeds and tunable constants.

Every other module imports from here, so changing a path or a seed in one
place changes it everywhere (no "magic numbers" scattered around the code).
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CUSTOM_INDIA_DIR = DATA_DIR / "custom_india"
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT_DIR / "models"))
RESULTS_DIR = ROOT_DIR / "results"

EMSCAD_CSV = RAW_DIR / "fake_job_postings.csv"
CUSTOM_INDIA_CSV = CUSTOM_INDIA_DIR / "custom_india.csv"
KAGGLE_DATASET = "shivamb/real-or-fake-fake-jobposting-prediction"

# Saved artefacts
BEST_CLASSICAL_MODEL = MODELS_DIR / "best_classical.joblib"
MODEL_META_JSON = MODELS_DIR / "model_meta.json"
BERT_DIR = MODELS_DIR / "distilbert"
RESULTS_CSV = RESULTS_DIR / "model_comparison.csv"
README_PATH = ROOT_DIR / "README.md"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# Split sizes (fractions of the full dataset)
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# --------------------------------------------------------------------------- #
# Decision settings
# --------------------------------------------------------------------------- #
# F-beta with beta=2 weights recall twice as much as precision. We use it to
# choose the decision threshold because missing a scam (false negative) is
# far more costly to a student than a false alarm.
THRESHOLD_BETA = 2.0

# How strongly the rule-based red-flag score can push the risk up.
# final = 1 - (1 - p_model) * (1 - RULE_WEIGHT * rule_score)   (noisy-OR)
RULE_WEIGHT = 0.7

# Hard override: if the red-flag score alone reaches this value (e.g. "asks
# for a fee" + "Gmail address"), the post is labelled Fake whatever the model
# says. Protects against the model having never seen Indian scam patterns.
RULE_OVERRIDE_SCORE = 0.70

# Inputs shorter than this many words get a "low confidence" warning.
MIN_WORDS_FOR_CONFIDENCE = 15

# Risk-score bands for the UI gauge (score is 0-100, 50 == decision threshold)
RISK_BANDS = {"low": 30, "high": 50}


def set_seed(seed: int = SEED) -> None:
    """Fix all random seeds we control so experiments are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is optional (only needed for DistilBERT)
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dirs() -> None:
    """Create output folders if they do not exist yet."""
    for d in (RAW_DIR, PROCESSED_DIR, CUSTOM_INDIA_DIR, MODELS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
