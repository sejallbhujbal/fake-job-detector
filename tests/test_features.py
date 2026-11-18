import pytest

from src.features import (
    METADATA_COLUMNS, build_model_frame, extract_features, parse_salary, red_flag_checklist,
    rule_score,
)
import pandas as pd

from tests.conftest import LEGIT, SCAM


def flags(text, **kw):
    return {f["id"]: f for f in red_flag_checklist(text, **kw)}


def test_feature_vector_shape_and_types():
    f = extract_features(SCAM)
    assert list(f) == METADATA_COLUMNS
    assert all(isinstance(v, (int, float)) for v in f.values())


def test_empty_input_features():
    f = extract_features("")
    assert f["short_description"] == 1 and f["asks_payment"] == 0


def test_scam_patterns_detected():
    f = flags(SCAM)
    for fid in ["asks_payment", "offplatform_contact", "no_interview", "urgency", "salary_too_high"]:
        assert f[fid]["triggered"], fid
    assert any("registration fee" in e.lower() for e in f["asks_payment"]["evidence"])


def test_legit_post_has_few_flags():
    f = flags(LEGIT, company="Brightpath Software Pvt. Ltd.", email="campus@brightpath.in")
    assert not f["asks_payment"]["triggered"]
    assert not f["free_email_domain"]["triggered"]
    assert not f["offplatform_contact"]["triggered"]
    assert not f["missing_company"]["triggered"]


def test_free_email_domain_from_field_and_text():
    assert flags("contact us", email="HR.Jobs@Gmail.com")["free_email_domain"]["triggered"]
    assert flags("mail your cv to recruit2024@yahoo.co.in")["free_email_domain"]["triggered"]
    assert not flags("mail cv to hr@infosys.com")["free_email_domain"]["triggered"]


@pytest.mark.parametrize("salary,monthly,currency", [
    ("₹50,000 per week", 50_000 * 52 / 12, "INR"),
    ("5 LPA", 500_000 / 12, "INR"),
    ("₹25,000/month", 25_000, "INR"),
    ("40000-50000", 50_000 / 12, "UNKNOWN"),
    ("20k-40k monthly", 40_000, "UNKNOWN"),
    ("1.5 lakh per month", 150_000, "INR"),
])
def test_parse_salary(salary, monthly, currency):
    p = parse_salary(salary)
    assert p["amount_monthly"] == pytest.approx(monthly, rel=1e-3)
    assert p["currency"] == currency


@pytest.mark.parametrize("bad", ["", "Oct-15", "15-Nov", "competitive", None])
def test_parse_salary_garbage(bad):
    assert parse_salary(bad)["amount_monthly"] is None


def test_salary_too_high_for_intern():
    assert flags("Data analyst internship for freshers", salary="₹90,000 per month")["salary_too_high"]["triggered"]
    assert not flags("Senior architect, 12 years experience", salary="₹90,000 per month")["salary_too_high"]["triggered"]


def test_rule_score_bounds_and_monotonic():
    empty = red_flag_checklist(LEGIT, company="X Pvt Ltd")
    scam = red_flag_checklist(SCAM, email="a@gmail.com")
    assert 0 <= rule_score(empty) < rule_score(scam) <= 1
    assert rule_score(scam) > 0.8


def test_build_model_frame_columns():
    df = build_model_frame(pd.DataFrame({"text": [SCAM, None]}))
    assert "clean_text" in df and set(METADATA_COLUMNS) <= set(df.columns)
    assert df.loc[1, "clean_text"] == ""


def test_negated_payment_is_not_a_flag():
    assert not flags("Fellowship of ₹15,000 per month. No application fee.")["asks_payment"]["triggered"]
    assert not flags("We never ask for your OTP or bank details.")["personal_info_request"]["triggered"]
    assert flags("Pay the application fee of ₹500 to proceed.")["asks_payment"]["triggered"]
