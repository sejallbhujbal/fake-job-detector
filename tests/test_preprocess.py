import pytest

from src.preprocess import (
    clean_text, detect_language, light_clean, replace_contacts, strip_html, validate_input,
)


def test_strip_html_removes_tags_and_entities():
    assert strip_html("<p>Hello&nbsp;<b>world</b> &amp; co</p>").split() == ["Hello", "world", "&", "co"]


def test_urls_emails_phones_become_tokens():
    out = replace_contacts("Visit https://scam.xyz/apply or mail hr.jobs@gmail.com or call 9876543210")
    assert "urltoken" in out and "emailtoken" in out and "phonetoken" in out
    assert "gmail" not in out and "9876543210" not in out


def test_emscad_anonymised_tokens():
    assert clean_text("contact #EMAIL_0b13a2cfd4718ce252c09b2353d692a73bd32552# now").split()[0] == "contact"
    assert "emailtoken" in clean_text("#EMAIL_abc123#")
    assert "urltoken" in clean_text("#URL_99f#")


def test_clean_text_lowercases_and_drops_stopwords_and_digits():
    out = clean_text("The Company WILL pay 5000 Rupees!!!")
    assert out == out.lower()
    assert "the" not in out.split() and "5000" not in out


def test_clean_text_keeps_negations():
    assert "no" in clean_text("No interview required").split()


def test_lemmatization_plural():
    assert "fee" in clean_text("registration fees").split()


@pytest.mark.parametrize("bad", [None, float("nan"), "", "   ", 123])
def test_clean_text_handles_weird_input(bad):
    assert isinstance(clean_text(bad), str)
    assert isinstance(light_clean(bad), str)


def test_validate_empty_raises():
    with pytest.raises(ValueError):
        validate_input("   ")
    with pytest.raises(ValueError):
        validate_input(None)


def test_validate_short_text_warns():
    text, warnings = validate_input("Earn money fast")
    assert text == "Earn money fast"
    assert any("short" in w.lower() for w in warnings)


def test_detect_language():
    assert detect_language("") == "unknown"
    assert detect_language("घर बैठे कमाएं ₹50000 प्रति सप्ताह, रजिस्ट्रेशन फीस ₹999") == "hi_mr"
    assert detect_language("We are hiring a software engineering intern in Pune") == "en"


def test_validate_devanagari_warns():
    _, warnings = validate_input("घर बैठे कमाएं ₹50000 प्रति सप्ताह रजिस्ट्रेशन फीस ₹999 व्हाट्सएप करें")
    assert any("Devanagari" in w for w in warnings)
