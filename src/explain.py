"""
Explainability: which words / features pushed a post towards "Fake"?

Three methods, from fastest to most general:

1. Linear models (Logistic Regression): exact contribution of each term
       contribution_i = tfidf_value_i * coefficient_i
2. XGBoost: TreeSHAP values computed natively by XGBoost
       booster.predict(..., pred_contribs=True)   (no `shap` package needed)
3. LIME (any model, incl. DistilBERT): perturbs the text by removing words and
   fits a local linear model to see which words change the prediction most.

Plus `highlight_segments`, which maps important terms and rule-based red-flag
phrases back onto the ORIGINAL text so the UI can colour them.
"""
from __future__ import annotations

import html
import logging
import re
from typing import Callable

import numpy as np

from .preprocess import clean_text

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers to look inside a fitted pipeline
# --------------------------------------------------------------------------- #
def _split_pipeline(model):
    """Return (preprocessor, classifier) from an sklearn/imblearn Pipeline."""
    steps = model.named_steps
    return steps["prep"], steps["clf"]


def _pretty_name(raw: str) -> str:
    """'tfidf__registration fee' -> 'registration fee'; 'meta__asks_payment' -> '[asks payment]'"""
    if raw.startswith("tfidf__"):
        return raw[len("tfidf__"):]
    if raw.startswith("meta__"):
        return "[" + raw[len("meta__"):].replace("_", " ") + "]"
    return raw


def _top_terms(names, values, top_k: int) -> tuple[list[dict], list[dict]]:
    """Split contributions into suspicious (+) and trustworthy (-) lists."""
    order = np.argsort(values)
    pos = [{"term": _pretty_name(names[i]), "weight": round(float(values[i]), 4)}
           for i in order[::-1][:top_k] if values[i] > 0]
    neg = [{"term": _pretty_name(names[i]), "weight": round(float(values[i]), 4)}
           for i in order[:top_k] if values[i] < 0]
    return pos, neg


def is_linear(model) -> bool:
    return hasattr(_split_pipeline(model)[1], "coef_")


def is_xgboost(model) -> bool:
    return hasattr(_split_pipeline(model)[1], "get_booster")


# --------------------------------------------------------------------------- #
# 1. Linear models
# --------------------------------------------------------------------------- #
def explain_linear(model, frame, top_k: int = 10):
    prep, clf = _split_pipeline(model)
    X = prep.transform(frame)  # 1 x n_features sparse
    names = prep.get_feature_names_out()
    X = X.tocsr()
    idx = X.indices
    contrib = X.data * clf.coef_[0][idx]
    return _top_terms(names[idx], contrib, top_k)


# --------------------------------------------------------------------------- #
# 2. XGBoost (native TreeSHAP)
# --------------------------------------------------------------------------- #
def explain_xgboost(model, frame, top_k: int = 10):
    import xgboost as xgb

    prep, clf = _split_pipeline(model)
    X = prep.transform(frame)
    names = prep.get_feature_names_out()
    contribs = clf.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]  # drop bias
    nz = np.flatnonzero(contribs)
    return _top_terms(names[nz], contribs[nz], top_k)


def explain_classical(model, frame, top_k: int = 10):
    """Dispatch to the fast exact explainer for this model type."""
    if is_linear(model):
        return explain_linear(model, frame, top_k), "linear_contributions"
    if is_xgboost(model):
        return explain_xgboost(model, frame, top_k), "xgboost_treeshap"
    raise TypeError("Unsupported model type for fast explanation")


# --------------------------------------------------------------------------- #
# 3. LIME (model-agnostic)
# --------------------------------------------------------------------------- #
def explain_with_lime(text: str, predict_proba: Callable[[list[str]], np.ndarray],
                      top_k: int = 10, num_samples: int = 400, seed: int = 42):
    """predict_proba(list_of_texts) -> array (n, 2). Returns (pos, neg) term lists.

    Raises ImportError if `lime` is not installed.
    """
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=["real", "fake"], random_state=seed, bow=True)
    exp = explainer.explain_instance(text, predict_proba, num_features=top_k * 2,
                                     num_samples=num_samples, labels=(1,))
    weights = exp.as_list(label=1)
    pos = [{"term": w.lower(), "weight": round(float(v), 4)} for w, v in weights if v > 0][:top_k]
    neg = [{"term": w.lower(), "weight": round(float(v), 4)} for w, v in weights if v < 0][:top_k]
    return pos, neg


# --------------------------------------------------------------------------- #
# Highlighting on the original text
# --------------------------------------------------------------------------- #
TOKEN_RE = re.compile(r"\S+")


def highlight_segments(text: str, terms: list[dict], red_flags: list[dict] | None = None) -> list[dict]:
    """Return the original text split into segments:
        [{"text": "...", "score": 0-1, "kind": "model"|"rule"|None}, ...]
    `score` is the highlight intensity (normalised to the strongest term).
    """
    if not text:
        return []
    weights = np.zeros(len(text), dtype=float)
    kinds = np.array([""] * len(text), dtype=object)

    # ---- model terms (unigrams and bigrams on cleaned tokens) ----
    term_w = {}
    for t in terms:
        if t["term"].startswith("["):  # metadata feature, not a word
            continue
        term_w[t["term"]] = max(term_w.get(t["term"], 0.0), float(t["weight"]))
    max_w = max(term_w.values(), default=0.0)
    if max_w > 0:
        tokens = [(m.start(), m.end(), clean_text(m.group(0))) for m in TOKEN_RE.finditer(text)]
        # keep tokens that survive cleaning (stop-words become '')
        kept = [(s, e, c) for s, e, c in tokens if c]
        for i, (s, e, c) in enumerate(kept):
            for word in c.split():
                w = term_w.get(word, 0.0)
                if w > 0:
                    _paint(weights, kinds, s, e, w / max_w, "model")
            if i + 1 < len(kept):
                s2, e2, c2 = kept[i + 1]
                bigram = f"{c.split()[-1]} {c2.split()[0]}"
                w = term_w.get(bigram, 0.0)
                if w > 0:
                    _paint(weights, kinds, s, e2, w / max_w, "model")

    # ---- rule evidence phrases (exact text matches) ----
    for flag in red_flags or []:
        if not flag.get("triggered"):
            continue
        for phrase in flag.get("evidence", []):
            if not phrase or not any(ch.isalpha() for ch in phrase):
                continue
            for m in re.finditer(re.escape(phrase), text, flags=re.I):
                _paint(weights, kinds, m.start(), m.end(), max(0.6, flag["weight"]), "rule")

    # ---- merge characters into segments ----
    segments, start = [], 0
    for i in range(1, len(text) + 1):
        if i == len(text) or weights[i] != weights[start] or kinds[i] != kinds[start]:
            segments.append({"text": text[start:i], "score": round(float(weights[start]), 3),
                             "kind": kinds[start] or None})
            start = i
    return segments


def _paint(weights, kinds, s, e, w, kind):
    region = weights[s:e]
    mask = w > region
    region[mask] = w
    kinds[s:e][mask] = kind


def segments_to_html(segments: list[dict]) -> str:
    """Render segments as HTML with red (model) / orange (rule) highlights."""
    out = []
    for seg in segments:
        t = html.escape(seg["text"]).replace("\n", "<br>")
        if seg["score"] > 0:
            alpha = 0.25 + 0.6 * seg["score"]
            color = f"rgba(220,38,38,{alpha:.2f})" if seg["kind"] == "model" else f"rgba(245,158,11,{alpha:.2f})"
            title = "model: suspicious term" if seg["kind"] == "model" else "rule: red-flag phrase"
            out.append(f'<mark title="{title}" style="background:{color};padding:0 2px;border-radius:3px">{t}</mark>')
        else:
            out.append(t)
    return "".join(out)
