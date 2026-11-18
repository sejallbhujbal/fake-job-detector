"""
Evaluation utilities: metrics, threshold tuning, plots and the results table.

Why not accuracy?
    EMSCAD is ~95 % real posts. A model that ALWAYS says "real" scores ~95 %
    accuracy while catching zero scams. We therefore report precision,
    recall, F1/F2 and PR-AUC for the *fake* class, and tune the decision
    threshold for recall (F2) because a missed scam costs a student money,
    while a false alarm only costs a second look.

CLI:
    python -m src.evaluate                 # re-evaluate saved models, rebuild plots + table
    python -m src.evaluate --update-readme # also write the table into README.md
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    fbeta_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)

from .config import README_PATH, RESULTS_CSV, RESULTS_DIR, THRESHOLD_BETA

logger = logging.getLogger(__name__)

README_START = "<!-- RESULTS_START -->"
README_END = "<!-- RESULTS_END -->"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def find_best_threshold(y_true, y_prob, beta: float = THRESHOLD_BETA) -> float:
    """Pick the probability threshold that maximises F-beta on validation data.

    beta=2 -> recall counts twice as much as precision.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5  # degenerate split: nothing to tune
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision/recall have one more element than thresholds
    p, r = precision[:-1], recall[:-1]
    denom = beta**2 * p + r
    fbeta = np.where(denom > 0, (1 + beta**2) * p * r / np.maximum(denom, 1e-12), 0)
    best = int(np.argmax(fbeta))
    return float(np.clip(thresholds[best], 0.01, 0.99))


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """All metrics for the *fake* (positive) class at the given threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_prob) if both_classes else float("nan"),
        "roc_auc": roc_auc_score(y_true, y_prob) if both_classes else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y_true)),
    }


def majority_baseline(y_true) -> dict:
    """Metrics of the 'always predict real' dummy model (shows why accuracy lies)."""
    return compute_metrics(y_true, np.zeros(len(y_true)), threshold=0.5)


# --------------------------------------------------------------------------- #
# Plots (matplotlib imported lazily so the API never needs it)
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(metrics: dict, title: str, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["Real", "Fake"])
    ax.set_yticks([0, 1], ["Real", "Fake"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pr_curves(curves: dict, title: str, path) -> None:
    """curves = {model_name: (y_true, y_prob)}"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for name, (y, p) in curves.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{name} (AP={average_precision_score(y, p):.3f})", lw=1.8)
    base = float(np.mean(next(iter(curves.values()))[0])) if curves else 0
    ax.axhline(base, ls="--", c="grey", lw=1, label=f"random ({base:.3f})")
    ax.set_xlabel("Recall (fake)")
    ax.set_ylabel("Precision (fake)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Results table
# --------------------------------------------------------------------------- #
TABLE_COLUMNS = ["model", "dataset", "mode", "threshold", "precision", "recall", "f1", "f2",
                 "pr_auc", "roc_auc", "accuracy", "tp", "fp", "fn", "tn", "n"]


def save_results(rows: list[dict], merge: bool = True) -> pd.DataFrame:
    """Write/merge rows into results/model_comparison.csv (+ json)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if merge and RESULTS_CSV.exists():
        old = pd.read_csv(RESULTS_CSV)
        key = ["model", "dataset", "mode"]
        old = old[~old.set_index(key).index.isin(new.set_index(key).index)]
        new = pd.concat([old, new], ignore_index=True)
    new = new[[c for c in TABLE_COLUMNS if c in new.columns]]
    new = new.sort_values(["dataset", "mode", "model"]).reset_index(drop=True)
    new.to_csv(RESULTS_CSV, index=False)
    new.to_json(RESULTS_CSV.with_suffix(".json"), orient="records", indent=2)
    return new


def results_markdown(df: pd.DataFrame) -> str:
    """Pretty markdown tables (one per dataset) for the README."""
    parts = []
    names = {"emscad_test": "Kaggle EMSCAD test set", "custom_india": "Custom Indian dataset"}
    for ds in ["emscad_test", "custom_india"]:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue
        parts.append(f"**{names.get(ds, ds)}** (fake = positive class)\n")
        parts.append("| Model | Mode | Precision | Recall | F1 | PR-AUC | Accuracy | TP / FN / FP |")
        parts.append("|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            parts.append(
                f"| {r['model']} | {r['mode']} | {r['precision']:.3f} | {r['recall']:.3f} | "
                f"{r['f1']:.3f} | {r['pr_auc']:.3f} | {r['accuracy']:.3f} | "
                f"{int(r['tp'])} / {int(r['fn'])} / {int(r['fp'])} |"
            )
        parts.append("")
    return "\n".join(parts)


def update_readme(df: pd.DataFrame) -> None:
    if not README_PATH.exists():
        logger.warning("README.md not found; skipping README update.")
        return
    text = README_PATH.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        logger.warning("README markers not found; skipping README update.")
        return
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    block = f"{README_START}\n_Auto-generated by `python -m src.evaluate --update-readme`._\n\n{results_markdown(df)}\n{README_END}"
    README_PATH.write_text(head + block + tail, encoding="utf-8")
    print(f"README updated: {README_PATH}")


# --------------------------------------------------------------------------- #
# Re-evaluate saved models (CLI)
# --------------------------------------------------------------------------- #
def evaluate_saved_models(update: bool = False) -> pd.DataFrame:
    """Load every saved classical model + DistilBERT (if present), evaluate on
    the EMSCAD test split and the custom Indian set, write plots + table."""
    import joblib

    from .config import BERT_DIR, MODELS_DIR, MODEL_META_JSON, set_seed
    from .data_loader import load_custom_india, load_emscad, stratified_split
    from .features import rule_scores_df
    from .predict import apply_rule_override, hybrid_probability
    from .train_classical import prepare_frame

    set_seed()
    meta = json.loads(MODEL_META_JSON.read_text()) if MODEL_META_JSON.exists() else {}
    _, _, test = stratified_split(load_emscad())
    india = load_custom_india()
    frames = {"emscad_test": prepare_frame(test), "custom_india": prepare_frame(india)}
    rule = {k: rule_scores_df(v).values for k, v in frames.items()}

    rows, curves = [], {k: {} for k in frames}
    for name, info in meta.get("models", {}).items():
        path = MODELS_DIR / info["file"]
        if not path.exists():
            continue
        model = joblib.load(path)
        for ds, frame in frames.items():
            p = model.predict_proba(frame)[:, 1]
            curves[ds][name] = (frame["label"].values, p)
            m = compute_metrics(frame["label"], p, info["threshold"])
            rows.append({"model": name, "dataset": ds, "mode": "model_only", **m})
            h = apply_rule_override(hybrid_probability(p, rule[ds]), rule[ds], info["hybrid_threshold"])
            m = compute_metrics(frame["label"], h, info["hybrid_threshold"])
            rows.append({"model": name, "dataset": ds, "mode": "hybrid", **m})
            plot_confusion_matrix(m, f"{name} (hybrid) - {ds}", RESULTS_DIR / f"cm_{name}_{ds}.png")

    if (BERT_DIR / "config.json").exists():
        try:
            from .train_bert import BertScorer

            scorer = BertScorer(BERT_DIR)
            bmeta = json.loads((BERT_DIR / "bert_meta.json").read_text())
            for ds, frame in frames.items():
                p = scorer.predict_proba(frame["text"].tolist())
                curves[ds]["distilbert"] = (frame["label"].values, p)
                m = compute_metrics(frame["label"], p, bmeta["threshold"])
                rows.append({"model": "distilbert", "dataset": ds, "mode": "model_only", **m})
                h = apply_rule_override(hybrid_probability(p, rule[ds]), rule[ds], bmeta["hybrid_threshold"])
                m = compute_metrics(frame["label"], h, bmeta["hybrid_threshold"])
                rows.append({"model": "distilbert", "dataset": ds, "mode": "hybrid", **m})
        except ImportError:
            logger.warning("transformers/torch not installed - skipping DistilBERT evaluation.")

    for ds in frames:
        rows.append({"model": "always_real_baseline", "dataset": ds, "mode": "model_only",
                     **majority_baseline(frames[ds]["label"])})
        if curves[ds]:
            plot_pr_curves(curves[ds], f"Precision-Recall - {ds}", RESULTS_DIR / f"pr_curve_{ds}.png")

    df = save_results(rows, merge=True)
    print(df.to_string(index=False))
    if update:
        update_readme(df)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Evaluate saved models")
    ap.add_argument("--update-readme", action="store_true")
    ap.add_argument("--readme-only", action="store_true", help="only rewrite README from existing CSV")
    args = ap.parse_args()
    if args.readme_only:
        update_readme(pd.read_csv(RESULTS_CSV))
        return
    evaluate_saved_models(update=args.update_readme)


if __name__ == "__main__":
    main()
