import pytest

from src.predict import Predictor, hybrid_probability, risk_level, to_risk_score
from tests.conftest import LEGIT, SCAM


def test_hybrid_probability_only_increases_risk():
    assert hybrid_probability(0.2, 0.0) == pytest.approx(0.2)
    assert hybrid_probability(0.2, 1.0) > 0.2
    assert hybrid_probability(1.0, 0.5) == pytest.approx(1.0)


def test_risk_score_mapping():
    assert to_risk_score(0.3, 0.3) == 50
    assert to_risk_score(0.0, 0.3) == 0
    assert to_risk_score(1.0, 0.3) == 100
    assert risk_level(10) == "low" and risk_level(40) == "medium" and risk_level(70) == "high"


def test_rules_only_mode_without_models(tmp_path):
    p = Predictor(models_dir=tmp_path)
    assert p.mode == "rules_only"
    out = p.predict(SCAM, email="x@gmail.com")
    assert out["label"] == "Fake" and out["model_probability"] is None
    assert any("rules only" in w for w in out["warnings"])


def test_classical_prediction_schema(tiny_models_dir):
    p = Predictor(models_dir=tiny_models_dir)
    assert p.mode == "classical"
    out = p.predict(SCAM, email="hr@gmail.com")
    for key in ["label", "risk_score", "risk_level", "highlighted_words", "red_flags",
                "highlighted_segments", "warnings"]:
        assert key in out
    assert 0 <= out["risk_score"] <= 100
    assert out["label"] == "Fake"
    assert out["explanation_method"] == "linear_contributions"
    assert "".join(s["text"] for s in out["highlighted_segments"]) == SCAM


def test_legit_post_scores_lower(tiny_models_dir):
    p = Predictor(models_dir=tiny_models_dir)
    legit = p.predict(LEGIT, company="Brightpath Software Pvt. Ltd.", email="campus@brightpath.in")
    scam = p.predict(SCAM)
    assert legit["risk_score"] < scam["risk_score"]
    assert legit["label"] == "Real"


def test_empty_and_short_input(tiny_models_dir):
    p = Predictor(models_dir=tiny_models_dir)
    with pytest.raises(ValueError):
        p.predict("   ")
    out = p.predict("hello")
    assert out["low_confidence"] is True


def test_devanagari_uses_rules_only(tiny_models_dir):
    p = Predictor(models_dir=tiny_models_dir)
    out = p.predict("घर बैठे कमाएं ₹50000 प्रति सप्ताह। रजिस्ट्रेशन फीस ₹999। WhatsApp करें 9000000001")
    assert out["model_name"] == "rules_only"
    assert out["language"] == "hi_mr"
