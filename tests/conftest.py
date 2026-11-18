"""Shared fixtures: a tiny trained model so tests run in seconds without EMSCAD."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CUSTOM_INDIA_CSV  # noqa: E402
from src.features import build_model_frame  # noqa: E402
from src.train_classical import build_logreg  # noqa: E402

SCAM = ("URGENT!!! Earn ₹50,000 per week from home. Pay registration fee ₹999 and "
        "WhatsApp HR on 9000000001. No interview, direct joining. Limited seats, hurry!")
LEGIT = ("Software Engineering Intern at Brightpath Software Pvt. Ltd., Pune. You will work with our "
         "backend team on Python microservices, write unit tests and join code reviews. Selection: "
         "online coding test and two technical interviews. Stipend ₹25,000 per month for 8 weeks. "
         "Apply through our careers page.")


@pytest.fixture(scope="session")
def tiny_models_dir(tmp_path_factory) -> Path:
    """Train a Logistic Regression on the custom India CSV and save it in the
    same layout as `python -m src.train_classical` produces."""
    d = tmp_path_factory.mktemp("models")
    df = pd.read_csv(CUSTOM_INDIA_CSV, dtype=str, keep_default_na=False)
    df["label"] = df["label"].astype(int)
    frame = build_model_frame(df)
    model = build_logreg()
    model.set_params(prep__tfidf__min_df=1)
    model.fit(frame, frame["label"])
    joblib.dump(model, d / "best_classical.joblib")
    joblib.dump(model, d / "tfidf_logreg.joblib")
    meta = {"best": "tfidf_logreg", "trained_at": "test",
            "models": {"tfidf_logreg": {"file": "tfidf_logreg.joblib", "threshold": 0.5,
                                        "hybrid_threshold": 0.5}}}
    (d / "model_meta.json").write_text(json.dumps(meta))
    return d
