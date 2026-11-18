"""
Text preprocessing for job / internship posts.

Pipeline (classical models):
    raw text -> HTML unescape + tag removal -> replace URLs / emails / phones
    with placeholder tokens -> lowercase -> keep letters only -> remove
    stop-words -> lemmatize

DistilBERT uses `light_clean` instead (no stop-word removal / lemmatization),
because transformers learn from the full sentence context.

All functions are safe to call with None / NaN / non-string input.
"""
from __future__ import annotations

import html
import logging
import re
import unicodedata
from functools import lru_cache

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Regular expressions (compiled once)
# --------------------------------------------------------------------------- #
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|in|net|org|co|io|xyz|info)/\S*)", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Indian mobile numbers (+91 optional, starts with 6-9) and generic long digit runs
PHONE_RE = re.compile(r"(\+?91[\s-]?)?\b[6-9]\d{4}[\s-]?\d{5}\b|\b\d{3}[\s-]\d{3}[\s-]\d{4}\b")
# EMSCAD anonymised tokens look like  #URL_0a1b...#  #EMAIL_...#  #PHONE_...#
EMSCAD_TOKEN_RE = re.compile(r"#(URL|EMAIL|PHONE)_[^#\s]*#", re.I)
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")
# Devanagari block covers Hindi and Marathi
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")

# Keep negations and a few words that matter for scams even though they are
# in the generic stop-word list (e.g. "no interview", "free").
KEEP_WORDS = {"no", "not", "nor", "without", "free", "only", "per", "call", "now"}
STOP_WORDS = frozenset(w for w in ENGLISH_STOP_WORDS if w not in KEEP_WORDS)

# Placeholder tokens (kept as words so the model can learn from them)
URL_TOKEN, EMAIL_TOKEN, PHONE_TOKEN = " urltoken ", " emailtoken ", " phonetoken "


# --------------------------------------------------------------------------- #
# Lemmatizer: use NLTK WordNet if available, otherwise a tiny rule fallback
# --------------------------------------------------------------------------- #
class _FallbackLemmatizer:
    """Very small suffix-stripping lemmatizer used when NLTK data is missing.

    It is deliberately conservative: it only handles the commonest English
    plural / verb endings so it never destroys words.
    """

    def lemmatize(self, word: str) -> str:
        if len(word) <= 3:
            return word
        if word.endswith("ies"):
            return word[:-3] + "y"
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("s") and not word.endswith(("ss", "us", "is")):
            return word[:-1]
        return word


@lru_cache(maxsize=1)
def get_lemmatizer():
    """Return a WordNet lemmatizer if NLTK + wordnet data are installed.

    Newer NLTK versions keep WordNet as a .zip file, so instead of checking
    for a folder we simply try to use it; if the data is missing we try a
    quiet download once, and otherwise fall back to the simple lemmatizer.
    """
    try:
        from nltk.stem import WordNetLemmatizer

        lem = WordNetLemmatizer()
        try:
            lem.lemmatize("tests")  # loads WordNet (works with zipped data too)
        except LookupError:
            import nltk

            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            lem = WordNetLemmatizer()
            lem.lemmatize("tests")
        return lem
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.warning("NLTK WordNet unavailable (%s); using fallback lemmatizer.", exc)
        return _FallbackLemmatizer()


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #
def to_str(text) -> str:
    """Convert None / NaN / numbers to a clean string."""
    if text is None:
        return ""
    if isinstance(text, float) and text != text:  # NaN check without numpy
        return ""
    return str(text)


def strip_html(text: str) -> str:
    """Unescape HTML entities (&amp; -> &) and drop tags (<br>, <p>, ...)."""
    text = html.unescape(text)
    return HTML_TAG_RE.sub(" ", text)


def replace_contacts(text: str) -> str:
    """Replace URLs, e-mails and phone numbers by placeholder tokens."""
    text = EMSCAD_TOKEN_RE.sub(lambda m: f" {m.group(1).lower()}token ", text)
    text = EMAIL_RE.sub(EMAIL_TOKEN, text)
    text = URL_RE.sub(URL_TOKEN, text)
    text = PHONE_RE.sub(PHONE_TOKEN, text)
    return text


def light_clean(text) -> str:
    """Minimal cleaning for transformer models: keeps casing-free sentences."""
    text = to_str(text)
    text = unicodedata.normalize("NFKC", text)
    text = strip_html(text)
    text = replace_contacts(text)
    return MULTI_SPACE_RE.sub(" ", text).strip()


def clean_text(text, lemmatize: bool = True, remove_stopwords: bool = True) -> str:
    """Full cleaning used by TF-IDF models.

    >>> clean_text("<p>Pay the Registration FEES now!!! www.scam.xyz</p>")
    'pay registration fee now urltoken'
    """
    text = light_clean(text).lower()
    # The rupee sign carries signal in Indian scams ("earn ₹5000 daily").
    text = text.replace("₹", " rupee ").replace("rs.", " rupee ")
    text = NON_ALPHA_RE.sub(" ", text)
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [t for t in tokens if len(t) > 1 or t in {"no"}]
    if lemmatize:
        lem = get_lemmatizer()
        tokens = [lem.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def clean_series(texts, lemmatize: bool = True):
    """Vectorised helper for pandas Series / lists."""
    return [clean_text(t, lemmatize=lemmatize) for t in texts]


def word_count(text) -> int:
    return len(to_str(text).split())


def detect_language(text) -> str:
    """Return 'en', 'hi_mr' (Devanagari script), 'other' or 'unknown'.

    Uses `langdetect` when installed; otherwise falls back to a simple
    script-based heuristic that is good enough to warn the user.
    """
    text = to_str(text).strip()
    if not text:
        return "unknown"
    deva = len(DEVANAGARI_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    letters = deva + latin
    if letters == 0:
        return "unknown"
    if deva / letters > 0.3:
        return "hi_mr"
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        lang = detect(text)
        return "en" if lang == "en" else "other"
    except Exception:
        # langdetect missing or failed: Latin script -> assume English.
        # (Romanised Hindi "Hinglish" will be treated as English.)
        return "en" if latin / letters > 0.7 else "other"


def validate_input(text) -> tuple[str, list[str]]:
    """Check user input and return (stripped_text, warnings).

    Raises ValueError for empty input. Short or non-English input is allowed
    but produces warnings so the UI can tell the user to be careful.
    """
    text = to_str(text).strip()
    warnings: list[str] = []
    if not text:
        raise ValueError("Input text is empty. Paste the full job/internship post.")
    n_words = word_count(text)
    from .config import MIN_WORDS_FOR_CONFIDENCE

    if n_words < MIN_WORDS_FOR_CONFIDENCE:
        warnings.append(
            f"Very short text ({n_words} words). Prediction is low-confidence; "
            "paste the complete post for a reliable result."
        )
    lang = detect_language(text)
    if lang == "hi_mr":
        warnings.append(
            "Hindi/Marathi (Devanagari) text detected. The model is trained on "
            "English posts, so only rule-based checks are reliable for this input."
        )
    elif lang == "other":
        warnings.append("Text does not look like English; prediction may be unreliable.")
    return text, warnings