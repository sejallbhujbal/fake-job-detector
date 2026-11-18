"""
Inference: one post in -> label, risk score, highlighted words, red flags out.

Decision logic ("hybrid" scoring)
---------------------------------
    p_model = probability of FAKE from the ML model
    r       = rule-based red-flag score in [0, 1]       (features.rule_score)
    p_final = 1 - (1 - p_model) * (1 - RULE_WEIGHT * r)  (noisy-OR)

Rules can only *raise* the risk, which lets domain knowledge about Indian scam
patterns (fees, WhatsApp-only contact...) override a model trained on US data.
If the rule score alone is >= RULE_OVERRIDE_SCORE the post is always labelled
Fake (hard override for unmistakable scams).
`p_final` is compared with a threshold tuned on the validation set and then
mapped to a 0-100 risk score where 50 == the decision threshold.

Model priority: DistilBERT (if USE_BERT=1 and installed) -> best classical
model -> rule-only fallback (when no trained model exists, or the text is in
Hindi/Marathi, which the models were not trained on).

CLI:  python -m src.predict "paste a job post here" [--company X --email Y --salary Z]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MIN_WORDS_FOR_CONFIDENCE, MODELS_DIR, RISK_BANDS, RULE_OVERRIDE_SCORE, RULE_WEIGHT
from .features import build_model_frame, red_flag_checklist, rule_score
from .preprocess import detect_language, light_clean, validate_input, word_count

logger = logging.getLogger(__name__)


def hybrid_probability(p_model, rule, weight: float = RULE_WEIGHT):
    """Noisy-OR combination of model probability and rule score (vectorised)."""
    p = np.asarray(p_model, dtype=float)
    r = np.asarray(rule, dtype=float)
    return 1.0 - (1.0 - p) * (1.0 - weight * r)


def apply_rule_override(p_final, rule, threshold: float, override: float = RULE_OVERRIDE_SCORE):
    """If the rule score alone is >= `override`, lift the probability so the
    post is classified as fake: max(p_final, threshold + a bit proportional to
    how far the rule score exceeds the override level). Vectorised."""
    p = np.asarray(p_final, dtype=float)
    r = np.asarray(rule, dtype=float)
    boost = threshold + (1 - threshold) * np.clip((r - override) / (1 - override), 0, 1) * 0.9
    return np.where(r >= override, np.maximum(p, boost), p)


def to_risk_score(prob: float, threshold: float) -> float:
    """Piece-wise linear map so that threshold -> 50, 0 -> 0, 1 -> 100.

    Class-weighted models are not calibrated (probabilities are shifted up),
    so showing the raw probability would confuse users. With this mapping
    'score >= 50' always means 'classified as fake'.
    """
    threshold = float(np.clip(threshold, 1e-3, 1 - 1e-3))
    if prob < threshold:
        score = 50.0 * prob / threshold
    else:
        score = 50.0 + 50.0 * (prob - threshold) / (1.0 - threshold)
    return round(float(np.clip(score, 0, 100)), 1)


def risk_level(score: float) -> str:
    if score >= RISK_BANDS["high"]:
        return "high"
    if score >= RISK_BANDS["low"]:
        return "medium"
    return "low"


class Predictor:
    """Loads models once and serves predictions."""

    def __init__(self, models_dir: Path | str = MODELS_DIR, use_bert: bool | None = None):
        self.models_dir = Path(models_dir)
        self.model = None
        self.model_name = "rules_only"
        self.threshold = 0.5
        self.hybrid_threshold = 0.5
        self.bert = None
        self.meta: dict = {}

        if use_bert is None:
            use_bert = os.getenv("USE_BERT", "0") == "1"
        if use_bert:
            self._try_load_bert()
        self._try_load_classical()
        if self.bert is None and self.model is None:
            logger.warning("No trained model found in %s - running in RULES-ONLY mode. "
                           "Train with: python -m src.train_classical", self.models_dir)

    # ----------------------------------------------------------------- loading
    def _try_load_classical(self):
        meta_path = self.models_dir / "model_meta.json"
        model_path = self.models_dir / "best_classical.joblib"
        if not (meta_path.exists() and model_path.exists()):
            return
        import joblib

        try:
            self.meta = json.loads(meta_path.read_text())
            self.model = joblib.load(model_path)
        except Exception as exc:  # corrupted file / version mismatch
            logger.error("Could not load classical model: %s", exc)
            self.model = None
            return
        if self.bert is None:  # BERT, if loaded, keeps priority
            best = self.meta["best"]
            info = self.meta["models"][best]
            self.model_name = best
            self.threshold = info["threshold"]
            self.hybrid_threshold = info["hybrid_threshold"]

    def _try_load_bert(self):
        bert_dir = self.models_dir / "distilbert"
        if not (bert_dir / "config.json").exists():
            logger.warning("USE_BERT=1 but %s not found; using classical model.", bert_dir)
            return
        try:
            from .train_bert import BertScorer

            self.bert = BertScorer(bert_dir)
            bmeta = json.loads((bert_dir / "bert_meta.json").read_text())
            self.model_name = "distilbert"
            self.threshold = bmeta["threshold"]
            self.hybrid_threshold = bmeta["hybrid_threshold"]
        except ImportError:
            logger.warning("transformers/torch not installed; DistilBERT disabled.")
            self.bert = None

    @property
    def mode(self) -> str:
        return "distilbert" if self.bert else "classical" if self.model is not None else "rules_only"

    # ------------------------------------------------------------- probability
    def _proba_texts(self, texts: list[str], company="", email="", salary="") -> np.ndarray:
        """P(fake) for a list of texts (used by LIME). Returns shape (n, 2)."""
        if self.bert is not None:
            p = self.bert.predict_proba([light_clean(t) for t in texts])
        else:
            frame = build_model_frame(pd.DataFrame(
                {"text": texts, "company": company, "email": email, "salary": salary}))
            p = self.model.predict_proba(frame)[:, 1]
        return np.column_stack([1 - p, p])

    # ----------------------------------------------------------------- predict
    def predict(self, text: str, company: str = "", email: str = "", salary: str = "",
                explain: str = "auto", top_k: int = 10) -> dict:
        """Full prediction for one post.

        explain: 'auto' (fast exact method), 'lime' (slower, model-agnostic) or 'none'.
        Raises ValueError on empty text.
        """
        text, warnings = validate_input(text)
        company, email, salary = (str(x or "").strip() for x in (company, email, salary))
        language = detect_language(text)

        checklist = red_flag_checklist(text, company, email, salary)
        r_score = rule_score(checklist)

        use_model = self.mode != "rules_only" and language not in {"hi_mr"}
        frame = build_model_frame(pd.DataFrame([{"text": text, "company": company,
                                                 "email": email, "salary": salary}]))
        if use_model:
            p_model = float(self._proba_texts([text], company, email, salary)[0, 1])
            threshold = self.hybrid_threshold
            p_final = float(apply_rule_override(hybrid_probability(p_model, r_score), r_score, threshold))
            if r_score >= RULE_OVERRIDE_SCORE and p_model < self.threshold:
                warnings.append("Several strong red flags found: labelled Fake by the rule override "
                                "even though the ML model alone was not confident.")
            model_name = self.model_name
        else:
            # Rule-only: the rule score itself is the probability.
            p_model = None
            p_final = r_score
            threshold = 0.5
            model_name = "rules_only"
            if self.mode != "rules_only":
                warnings.append("ML model skipped for non-English text; using red-flag rules only.")
            else:
                warnings.append("No trained model loaded; using red-flag rules only.")

        # ---------------- explanation ----------------
        suspicious, trustworthy, method = [], [], "none"
        if use_model and explain != "none":
            try:
                if explain == "lime" or self.bert is not None:
                    suspicious, trustworthy = explain_with_lime_safe(self, text, company, email, salary, top_k)
                    if suspicious or trustworthy:
                        method = "lime"
                if method == "none" and self.model is not None:
                    from .explain import explain_classical

                    (suspicious, trustworthy), method = explain_classical(self.model, frame, top_k)
                    if self.bert is not None:  # classical model used as a surrogate explainer
                        method += "_surrogate"
            except Exception as exc:  # never fail a prediction because of XAI
                logger.warning("Explanation failed: %s", exc)
                warnings.append("Word-level explanation unavailable for this input.")

        from .explain import highlight_segments

        segments = highlight_segments(text, suspicious, checklist)
        score = to_risk_score(p_final, threshold)
        is_fake = p_final >= threshold
        low_conf = word_count(text) < MIN_WORDS_FOR_CONFIDENCE

        return {
            "label": "Fake" if is_fake else "Real",
            "is_fake": bool(is_fake),
            "risk_score": score,
            "risk_level": risk_level(score),
            "probability": round(p_final, 4),
            "model_probability": None if p_model is None else round(p_model, 4),
            "rule_score": r_score,
            "threshold": round(threshold, 4),
            "model_name": model_name,
            "explanation_method": method,
            "highlighted_words": suspicious,
            "trust_words": trustworthy,
            "highlighted_segments": segments,
            "red_flags": checklist,
            "triggered_flags": sum(f["triggered"] for f in checklist),
            "language": language,
            "low_confidence": low_conf,
            "warnings": warnings,
        }

    def info(self) -> dict:
        return {"mode": self.mode, "model_name": self.model_name,
                "threshold": self.threshold, "hybrid_threshold": self.hybrid_threshold,
                "trained_at": self.meta.get("trained_at"), "rule_weight": RULE_WEIGHT,
                "rule_override": RULE_OVERRIDE_SCORE}


def explain_with_lime_safe(predictor: Predictor, text, company, email, salary, top_k):
    """LIME if installed; otherwise raise ImportError (caller falls back)."""
    from .explain import explain_with_lime

    try:
        return explain_with_lime(
            text, lambda ts: predictor._proba_texts(ts, company, email, salary), top_k=top_k,
            num_samples=200 if predictor.bert is not None else 400,
        )
    except ImportError:
        if predictor.model is None:
            raise
        return [], []  # fall back to classical fast explainer (method stays 'none')


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Process-wide singleton (models are loaded only once)."""
    return Predictor()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Predict whether a job post is fake")
    ap.add_argument("text")
    ap.add_argument("--company", default="")
    ap.add_argument("--email", default="")
    ap.add_argument("--salary", default="")
    ap.add_argument("--explain", default="auto", choices=["auto", "lime", "none"])
    args = ap.parse_args()
    out = Predictor().predict(args.text, args.company, args.email, args.salary, explain=args.explain)
    out.pop("highlighted_segments")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
