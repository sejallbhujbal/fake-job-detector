"""
Train and compare the classical models.

Models
------
1. tfidf_logreg          TF-IDF (1-2 grams) + Logistic Regression, class_weight='balanced'
2. tfidf_logreg_smote    same, but SMOTE oversampling instead of class weights
3. tfidf_meta_xgb        TF-IDF + metadata features + XGBoost, scale_pos_weight
4. tfidf_meta_xgb_smote  same, but SMOTE instead of scale_pos_weight

For every model we:
  * fit on the TRAIN split,
  * tune the decision threshold on the VALIDATION split (max F2 -> recall first),
  * report metrics on the untouched TEST split and on the custom Indian set,
  * in two modes: `model_only` and `hybrid` (model + rule-based red flags).
The best model (highest validation F2 in hybrid mode, PR-AUC as tie-break) is
saved as models/best_classical.joblib and used by the app.

Run:  python -m src.train_classical [--quick] [--no-smote] [--update-readme]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from .config import (
    BEST_CLASSICAL_MODEL, MODEL_META_JSON, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR,
    RULE_WEIGHT, SEED, ensure_dirs, set_seed,
)
from .data_loader import load_custom_india, load_emscad, stratified_split
from .evaluate import (
    compute_metrics, find_best_threshold, majority_baseline, plot_confusion_matrix,
    plot_pr_curves, save_results, update_readme,
)
from .features import METADATA_COLUMNS, build_model_frame, rule_scores_df
from .predict import apply_rule_override, hybrid_probability

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Optional dependencies
# --------------------------------------------------------------------------- #
try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - not installed / broken install
    XGBClassifier = None
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:  # pragma: no cover - not installed / incompatible with sklearn
    SMOTE = ImbPipeline = None


# --------------------------------------------------------------------------- #
# Data preparation (with a small cache, preprocessing 17k posts takes ~1 min)
# --------------------------------------------------------------------------- #
def prepare_frame(df: pd.DataFrame, cache_name: str | None = None) -> pd.DataFrame:
    """Add clean_text + metadata features. Optionally cache to data/processed/."""
    if cache_name:
        cache = PROCESSED_DIR / f"{cache_name}.pkl"
        if cache.exists():
            cached = pd.read_pickle(cache)
            if len(cached) == len(df) and (cached["text"].values == df["text"].values).all():
                return cached
    out = build_model_frame(df)
    if cache_name:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out.to_pickle(PROCESSED_DIR / f"{cache_name}.pkl")
    return out


# --------------------------------------------------------------------------- #
# Model factories
# --------------------------------------------------------------------------- #
def _tfidf(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),      # unigrams + bigrams ("registration fee")
        min_df=2,                # ignore one-off typos
        max_df=0.95,             # ignore words in almost every post
        max_features=max_features,
        sublinear_tf=True,       # log(tf): long posts don't dominate
        dtype=np.float32,
    )


def build_logreg(smote: bool = False):
    prep = ColumnTransformer([("tfidf", _tfidf(50_000), "clean_text")])
    clf = LogisticRegression(
        C=4.0, solver="liblinear", max_iter=3000, random_state=SEED,
        class_weight=None if smote else "balanced",
    )
    if smote:
        return ImbPipeline([("prep", prep), ("smote", SMOTE(random_state=SEED)), ("clf", clf)])
    return Pipeline([("prep", prep), ("clf", clf)])


def build_xgb(scale_pos_weight: float, smote: bool = False):
    prep = ColumnTransformer(
        [("tfidf", _tfidf(30_000), "clean_text"), ("meta", MaxAbsScaler(), METADATA_COLUMNS)],
        sparse_threshold=1.0,
    )
    clf = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8,
        colsample_bytree=0.5, min_child_weight=1, reg_lambda=1.0,
        tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=SEED,
        scale_pos_weight=1.0 if smote else scale_pos_weight,
    )
    if smote:
        return ImbPipeline([("prep", prep), ("smote", SMOTE(random_state=SEED)), ("clf", clf)])
    return Pipeline([("prep", prep), ("clf", clf)])


def model_zoo(y_train, use_smote: bool = True) -> dict:
    """Return {name: unfitted pipeline} for all models available in this env."""
    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    zoo = {"tfidf_logreg": build_logreg()}
    if use_smote and SMOTE is not None:
        zoo["tfidf_logreg_smote"] = build_logreg(smote=True)
    if XGBClassifier is not None:
        zoo["tfidf_meta_xgb"] = build_xgb(spw)
        if use_smote and SMOTE is not None:
            zoo["tfidf_meta_xgb_smote"] = build_xgb(spw, smote=True)
    else:
        logger.warning("xgboost not installed -> XGBoost models skipped (pip install xgboost)")
    if use_smote and SMOTE is None:
        logger.warning("imbalanced-learn not installed -> SMOTE variants skipped")
    return zoo


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train_and_compare(quick: bool = False, use_smote: bool = True, readme: bool = False) -> pd.DataFrame:
    set_seed()
    ensure_dirs()

    logger.info("Loading EMSCAD ...")
    df = load_emscad()
    if quick:  # fast dev loop: 30 % stratified sample
        df = df.groupby("label", group_keys=False).sample(frac=0.3, random_state=SEED)
    train, val, test = stratified_split(df)
    logger.info("train=%d val=%d test=%d | fake ratio=%.2f%%", len(train), len(val), len(test),
                100 * df["label"].mean())

    tag = "quick" if quick else "full"
    t0 = time.time()
    train_f = prepare_frame(train, f"emscad_train_{tag}")
    val_f = prepare_frame(val, f"emscad_val_{tag}")
    test_f = prepare_frame(test, f"emscad_test_{tag}")
    india_f = prepare_frame(load_custom_india())
    logger.info("Preprocessing done in %.1fs", time.time() - t0)

    rule = {
        "val": rule_scores_df(val_f).values,
        "emscad_test": rule_scores_df(test_f).values,
        "custom_india": rule_scores_df(india_f).values,
    }
    eval_sets = {"emscad_test": test_f, "custom_india": india_f}

    rows, meta_models, curves = [], {}, {k: {} for k in eval_sets}
    for name, model in model_zoo(train_f["label"].values, use_smote).items():
        logger.info("Training %s ...", name)
        t0 = time.time()
        model.fit(train_f, train_f["label"].values)
        fit_s = time.time() - t0

        # ---- threshold tuning on validation (never on test!) ----
        p_val = model.predict_proba(val_f)[:, 1]
        thr = find_best_threshold(val_f["label"], p_val)
        h_val = hybrid_probability(p_val, rule["val"])
        h_thr = find_best_threshold(val_f["label"], h_val)
        h_val = apply_rule_override(h_val, rule["val"], h_thr)
        val_hybrid = compute_metrics(val_f["label"], h_val, h_thr)

        # ---- evaluation ----
        for ds, frame in eval_sets.items():
            p = model.predict_proba(frame)[:, 1]
            curves[ds][name] = (frame["label"].values, p)
            m = compute_metrics(frame["label"], p, thr)
            rows.append({"model": name, "dataset": ds, "mode": "model_only", **m})
            h = apply_rule_override(hybrid_probability(p, rule[ds]), rule[ds], h_thr)
            mh = compute_metrics(frame["label"], h, h_thr)
            rows.append({"model": name, "dataset": ds, "mode": "hybrid", **mh})
            plot_confusion_matrix(mh, f"{name} (hybrid)\n{ds}", RESULTS_DIR / f"cm_{name}_{ds}.png")

        joblib.dump(model, MODELS_DIR / f"{name}.joblib", compress=3)
        meta_models[name] = {
            "file": f"{name}.joblib",
            "threshold": thr,
            "hybrid_threshold": h_thr,
            "val_f2_hybrid": val_hybrid["f2"],
            "val_pr_auc_hybrid": val_hybrid["pr_auc"],
            "fit_seconds": round(fit_s, 1),
        }
        logger.info("  %s: val F2(hybrid)=%.3f PR-AUC=%.3f thr=%.3f (%.0fs)",
                    name, val_hybrid["f2"], val_hybrid["pr_auc"], h_thr, fit_s)

    # Dummy baseline row -> demonstrates the accuracy paradox in the table
    for ds, frame in eval_sets.items():
        rows.append({"model": "always_real_baseline", "dataset": ds, "mode": "model_only",
                     **majority_baseline(frame["label"])})
        plot_pr_curves(curves[ds], f"Precision-Recall - {ds}", RESULTS_DIR / f"pr_curve_{ds}.png")

    # ---- choose + save the best model (selection on VALIDATION only) ----
    best = max(meta_models, key=lambda n: (meta_models[n]["val_f2_hybrid"], meta_models[n]["val_pr_auc_hybrid"]))
    shutil.copy(MODELS_DIR / meta_models[best]["file"], BEST_CLASSICAL_MODEL)
    meta = {
        "best": best,
        "models": meta_models,
        "rule_weight": RULE_WEIGHT,
        "seed": SEED,
        "quick": quick,
        "n_train": len(train_f), "n_val": len(val_f), "n_test": len(test_f),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    MODEL_META_JSON.write_text(json.dumps(meta, indent=2))
    logger.info("Best classical model: %s -> %s", best, BEST_CLASSICAL_MODEL)

    table = save_results(rows, merge=True)
    print("\n" + table.to_string(index=False))
    if readme:
        update_readme(table)
    return table


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Train classical fake-job models")
    ap.add_argument("--quick", action="store_true", help="use a 30%% sample for a fast run")
    ap.add_argument("--no-smote", action="store_true", help="skip SMOTE variants")
    ap.add_argument("--update-readme", action="store_true", help="write results table into README.md")
    args = ap.parse_args()
    train_and_compare(quick=args.quick, use_smote=not args.no_smote, readme=args.update_readme)


if __name__ == "__main__":
    main()
