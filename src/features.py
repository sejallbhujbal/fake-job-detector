"""
Hand-crafted metadata features + a rule-based red-flag checklist.

Why both?
* The ML models learn patterns from data, but EMSCAD (US/UK posts, 2012-14)
  has almost no examples of Indian scam tactics such as "registration fee",
  WhatsApp-only contact or "earn ₹50k/week from home".
* The rules encode that domain knowledge explicitly, are fully explainable,
  and are combined with the model probability in `predict.py`.

Inputs are the same four fields the user can give in the app:
    text, company, email, salary
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .preprocess import EMAIL_RE, to_str, word_count

# --------------------------------------------------------------------------- #
# Vocabulary / patterns
# --------------------------------------------------------------------------- #
FREE_EMAIL_DOMAINS = {
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.in", "ymail.com",
    "rediffmail.com", "hotmail.com", "outlook.com", "live.com", "aol.com",
    "protonmail.com", "proton.me", "icloud.com", "mail.com", "gmx.com",
    "zoho.com", "yandex.com",
}

def _rx(*patterns: str) -> re.Pattern:
    """Compile several alternatives into one case-insensitive regex."""
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.I)


MONEY = r"(?:rs\.?|₹|inr|rupees?|\$)\s?\d[\d,]*"

PAYMENT_RE = _rx(
    r"registration\s*(?:fee|fees|charges?|amount)",
    r"(?:security|refundable)\s*(?:deposit|amount|fee)",
    r"refundable",
    r"(?:processing|training|joining|application|verification|onboarding|"
    r"documentation|enrol?lment|certificate|id\s*card|kit|uniform)\s*(?:fee|fees|charges?)",
    rf"(?:pay|deposit|transfer|send)\s+(?:a\s+|an\s+|the\s+|only\s+|just\s+|minimum\s+)?(?:amount\s+of\s+)?{MONEY}",
    r"(?:buy|purchase)\s+(?:our|the|a)\s+(?:paid\s+)?(?:training|course|kit|package|starter)",
    rf"(?:fee|fees|charges?)\s+(?:of\s+)?{MONEY}",
    r"pay\s+(?:via|through|on)\s+(?:upi|gpay|google\s*pay|phonepe|paytm)",
)
OFFPLATFORM_RE = _rx(r"whats\s?app", r"\bwa\.me\b", r"telegram", r"\bt\.me\b", r"\bsignal app\b")
NO_INTERVIEW_RE = _rx(
    r"no\s+interview", r"without\s+(?:any\s+)?interview", r"direct\s+(?:joining|selection)",
    r"instant\s+(?:selection|joining|offer)", r"selected\s+without",
    r"100\s?%\s+(?:job|placement|selection)", r"guaranteed\s+(?:job|placement|selection|income)",
    r"(?:job|placement)\s+guarantee",
)
EASY_MONEY_RE = _rx(
    rf"earn\s+(?:up\s*to\s+|upto\s+|minimum\s+)?{MONEY}",
    r"earn\s+\d[\d,]*\s*(?:k|rs|rupees|/-)",
    r"earn\s+\d[\d,]*\s*(?:daily|weekly|per\s+(?:day|week))",
    r"daily\s+(?:payout|payment|income|earning)",
    r"easy\s+(?:money|income|earning|work)",
    r"(?:per|a|every)\s+day\s+earning",
    r"work\s+from\s+home\s+(?:and|&)\s+earn",
    r"(?:like|liking|subscribe|subscribing)\s+(?:youtube\s+)?videos",
    r"(?:rate|rating|review)\s+(?:products|hotels|restaurants)\s+(?:and|&)\s+earn",
    r"simple\s+(?:copy\s*paste|typing)\s+work",
    r"(?:just|only)\s+\d+\s*(?:hours?|hrs?|mins?|minutes)\s+(?:a|per)\s+day",
    r"part[\s-]?time\s+(?:job\s+)?(?:and\s+)?earn",
)
URGENCY_STRONG_RE = _rx(
    r"\bhurry\b", r"limited\s+(?:seats|slots|vacancies|openings)", r"last\s+chance",
    r"act\s+(?:fast|now)", r"today\s+only", r"within\s+24\s*(?:hours|hrs)",
    r"only\s+(?:few|\d+)\s+(?:seats|slots)\s+left", r"offer\s+expires",
)
URGENCY_MILD_RE = _rx(r"\burgent(?:ly)?\b", r"\bimmediate(?:ly)?\b", r"\basap\b", r"apply\s+now", r"join\s+now")
PERSONAL_INFO_RE = _rx(
    r"aadha?ar", r"pan\s*card", r"bank\s+(?:account\s+)?details", r"\botp\b",
    r"upi\s+pin", r"atm\s+pin", r"cvv", r"passport\s+copy",
)
COMPANY_IN_TEXT_RE = _rx(r"pvt\.?\s*ltd", r"private\s+limited", r"\bllp\b", r"\bltd\b", r"\binc\b\.?", r"\bcorp(?:oration)?\b")
INTERN_CONTEXT_RE = _rx(r"intern(?:ship)?", r"fresher", r"no\s+experience", r"students?")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐✅]")

# Salary parsing helpers
SALARY_IN_TEXT_RE = re.compile(
    r"(?:(?:rs\.?|₹|inr|\$|usd)\s?\d[\d,]*(?:\.\d+)?\s*(?:k|lakhs?|lacs?|lpa)?"
    r"|\d[\d,.]*\s*(?:lpa|lakhs?\s+per\s+annum|lakhs?|lacs?))"
    r"(?:\s*(?:-|to)\s*(?:rs\.?|₹|inr|\$)?\s?\d[\d,]*(?:\.\d+)?\s*(?:k|lakhs?|lacs?|lpa)?)?"
    r"(?:\s*(?:/|per|a|every)\s*(?:hour|hr|day|week|month|mo|year|annum))?",
    re.I,
)
NUMBER_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(k\b|lakhs?|lacs?|lpa|l\b|cr\b|crores?)?", re.I)
PERIOD_PATTERNS = [
    ("hour", re.compile(r"\b(?:per|/|an|a)\s*(?:hour|hr)\b|hourly", re.I), 160.0),
    ("day", re.compile(r"\b(?:per|/|a|every)\s*day\b|daily", re.I), 26.0),
    ("week", re.compile(r"\b(?:per|/|a|every)\s*week\b|weekly", re.I), 52 / 12),
    ("month", re.compile(r"\b(?:per|/|a)\s*(?:month|mo)\b|monthly|p\.?m\.?\b|stipend", re.I), 1.0),
    ("year", re.compile(r"\b(?:per|/|a)\s*(?:year|annum)\b|annual|yearly|\blpa\b|\bp\.?a\.?\b|\bctc\b", re.I), 1 / 12),
]
INR_RE = re.compile(r"₹|\brs\.?|\binr\b|rupee|lakh|\blac|\blpa\b|crore", re.I)
OTHER_CUR_RE = re.compile(r"\$|\busd\b|£|€|\bgbp\b|\beur\b", re.I)
# Excel-corrupted EMSCAD ranges such as "Oct-15" or "15-Nov"
EXCEL_DATE_RE = re.compile(
    r"^\s*(?:\d{1,2}-)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?:-\d{1,2})?\s*$", re.I
)


# --------------------------------------------------------------------------- #
# Salary parsing
# --------------------------------------------------------------------------- #
def parse_salary(salary: str) -> dict:
    """Parse a free-text salary into an approximate *monthly* amount.

    Returns dict(amount_monthly: float|None, currency: 'INR'|'OTHER'|'UNKNOWN',
                 period: str|None).

    Heuristics (documented so they can be defended in an interview):
      * k = thousand, lakh/lac/L/LPA = 100,000, crore = 10,000,000
      * Period taken from words like "per week", "daily", "LPA".
      * No period: INR >= 3,00,000 is assumed yearly, otherwise monthly;
        other currencies >= 10,000 assumed yearly (EMSCAD style "40000-50000").
      * EMSCAD has corrupted ranges such as "Oct-15" (Excel dates): ignored.
    """
    s = to_str(salary).strip()
    out = {"amount_monthly": None, "currency": "UNKNOWN", "period": None}
    if not s or EXCEL_DATE_RE.match(s):
        return out

    amounts = []
    for num, unit in NUMBER_RE.findall(s):
        try:
            value = float(num.replace(",", ""))
        except ValueError:
            continue
        unit = (unit or "").lower()
        if unit == "k":
            value *= 1_000
        elif unit.startswith(("lakh", "lac", "lpa")) or unit == "l":
            value *= 100_000
        elif unit.startswith(("cr", "crore")):
            value *= 10_000_000
        amounts.append(value)
    amounts = [a for a in amounts if a >= 100]  # drop "5 days", "2 hours" etc.
    if not amounts:
        return out

    currency = "INR" if INR_RE.search(s) else "OTHER" if OTHER_CUR_RE.search(s) else "UNKNOWN"
    amount = max(amounts)  # scams advertise the top of the range

    period, factor = None, None
    for name, rx, f in PERIOD_PATTERNS:
        if rx.search(s):
            period, factor = name, f
            break
    if factor is None:
        if currency == "INR":
            period, factor = ("year", 1 / 12) if amount >= 300_000 else ("month", 1.0)
        else:
            period, factor = ("year", 1 / 12) if amount >= 10_000 else ("month", 1.0)

    out.update(amount_monthly=amount * factor, currency=currency, period=period)
    return out


def find_salary_in_text(text: str) -> str:
    """Return the first salary-looking phrase in the post (or '')."""
    m = SALARY_IN_TEXT_RE.search(to_str(text))
    return m.group(0) if m else ""


def is_salary_too_high(parsed: dict, text: str = "") -> bool:
    """Flag pay that is unrealistic for the role (thresholds are monthly)."""
    amt = parsed.get("amount_monthly")
    if amt is None:
        return False
    if parsed["currency"] == "INR":
        if amt > 200_000:
            return True
        # ₹75k+/month for an internship / fresher role is a classic bait
        return amt > 75_000 and bool(INTERN_CONTEXT_RE.search(to_str(text)))
    return amt > 25_000  # USD/GBP etc.: > ~300k per year


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def _emails(text: str, email: str) -> list[str]:
    found = EMAIL_RE.findall(to_str(text))
    if to_str(email).strip():
        found.insert(0, to_str(email).strip())
    return [e.lower() for e in found]


def _free_email(text: str, email: str) -> tuple[bool, list[str]]:
    hits = [e for e in _emails(text, email) if e.split("@")[-1] in FREE_EMAIL_DOMAINS]
    return bool(hits), hits


NEGATION_RE = re.compile(r"\b(?:no|never|not|without|zero|free\s+of|don'?t|do\s+not)\b[^.!?\n]{0,25}$", re.I)


def _matches(rx: re.Pattern, text: str, limit: int = 3, skip_negated: bool = False) -> list[str]:
    """Return up to `limit` matched phrases. With skip_negated, ignore matches
    preceded by a negation in the same sentence ("no application fee",
    "we never ask for your OTP")."""
    hits = []
    for m in rx.finditer(text):
        if skip_negated and NEGATION_RE.search(text[max(0, m.start() - 40):m.start()]):
            continue
        hits.append(m.group(0).strip())
    return hits[:limit]


def extract_features(text: str, company: str = "", email: str = "", salary: str = "") -> dict:
    """Compute the numeric metadata feature vector for ONE post.

    All features are computed only from the 4 user-supplied fields, so the
    exact same code runs at training time and at prediction time (no
    train/serve skew).
    """
    text = to_str(text)
    company = to_str(company).strip()
    salary_str = to_str(salary).strip() or find_salary_in_text(text)
    letters = [c for c in text if c.isalpha()]
    caps_ratio = (sum(c.isupper() for c in letters) / len(letters)) if len(letters) >= 20 else 0.0
    parsed = parse_salary(salary_str)
    n_words = word_count(text)

    free_email, _ = _free_email(text, email)
    urgency_count = len(URGENCY_STRONG_RE.findall(text)) * 2 + len(URGENCY_MILD_RE.findall(text))

    return {
        "free_email_domain": int(free_email),
        "salary_missing": int(parsed["amount_monthly"] is None),
        "salary_too_high": int(is_salary_too_high(parsed, text)),
        "missing_company": int(not company and not COMPANY_IN_TEXT_RE.search(text)),
        "asks_payment": int(bool(_matches(PAYMENT_RE, text, skip_negated=True))),
        "offplatform_contact": int(bool(OFFPLATFORM_RE.search(text))),
        "no_interview": int(bool(NO_INTERVIEW_RE.search(text))),
        "easy_money": int(bool(EASY_MONEY_RE.search(text))),
        "personal_info_request": int(bool(_matches(PERSONAL_INFO_RE, text, skip_negated=True))),
        "urgency_count": urgency_count,
        "caps_ratio": round(caps_ratio, 4),
        "exclamation_count": text.count("!"),
        "emoji_count": len(EMOJI_RE.findall(text)),
        "log_char_len": math.log1p(len(text)),
        "log_word_count": math.log1p(n_words),
        "short_description": int(n_words < 50),
    }


METADATA_COLUMNS = list(extract_features("").keys())


def extract_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply `extract_features` row-wise to a DataFrame with columns
    text, company, email, salary (missing columns are treated as empty)."""
    cols = {c: df[c] if c in df else pd.Series([""] * len(df), index=df.index)
            for c in ("text", "company", "email", "salary")}
    rows = [
        extract_features(t, c, e, s)
        for t, c, e, s in zip(cols["text"], cols["company"], cols["email"], cols["salary"])
    ]
    return pd.DataFrame(rows, index=df.index)[METADATA_COLUMNS]


# --------------------------------------------------------------------------- #
# Rule-based red-flag checklist
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RedFlag:
    id: str
    name: str
    description: str
    weight: float  # how suspicious this flag is on its own (0-1)
    check: Callable[[str, str, str, str], tuple[bool, list[str]]]


def _rx_check(rx: re.Pattern, skip_negated: bool = False):
    def check(text, company, email, salary):
        hits = _matches(rx, text, skip_negated=skip_negated)
        return bool(hits), hits
    return check


def _check_free_email(text, company, email, salary):
    return _free_email(text, email)


def _check_salary(text, company, email, salary):
    s = to_str(salary).strip() or find_salary_in_text(text)
    parsed = parse_salary(s)
    too_high = is_salary_too_high(parsed, text)
    return too_high, ([s] if too_high and s else [])


def _check_missing_company(text, company, email, salary):
    missing = not to_str(company).strip() and not COMPANY_IN_TEXT_RE.search(to_str(text))
    return missing, []


def _check_urgency(text, company, email, salary):
    strong = _matches(URGENCY_STRONG_RE, text)
    mild = _matches(URGENCY_MILD_RE, text)
    return bool(strong) or len(mild) >= 2, (strong + mild)[:3]


def _check_caps_emoji(text, company, email, salary):
    f = extract_features(text)
    hit = f["caps_ratio"] > 0.3 or f["emoji_count"] >= 3 or f["exclamation_count"] >= 3
    ev = []
    if f["emoji_count"] >= 3:
        ev.append(f"{f['emoji_count']} emojis")
    if f["exclamation_count"] >= 3:
        ev.append(f"{f['exclamation_count']} exclamation marks")
    if f["caps_ratio"] > 0.3:
        ev.append(f"{int(f['caps_ratio'] * 100)}% capital letters")
    return hit, ev


def _check_short(text, company, email, salary):
    n = word_count(text)
    return n < 50, ([f"only {n} words"] if n < 50 else [])


RED_FLAGS: list[RedFlag] = [
    RedFlag("asks_payment", "Asks for money / registration fee",
            "Genuine employers never charge candidates to apply, train or join.", 0.60,
            _rx_check(PAYMENT_RE, skip_negated=True)),
    RedFlag("personal_info_request", "Requests sensitive personal/bank info",
            "Aadhaar, PAN, bank details or OTP requested before an offer.", 0.40,
            _rx_check(PERSONAL_INFO_RE, skip_negated=True)),
    RedFlag("easy_money", "Promises easy / quick money",
            "e.g. 'earn ₹5,000 daily from home', 'like videos and earn'.", 0.40,
            _rx_check(EASY_MONEY_RE)),
    RedFlag("offplatform_contact", "Contact only via WhatsApp / Telegram",
            "Recruiters avoid official channels to stay untraceable.", 0.35,
            _rx_check(OFFPLATFORM_RE)),
    RedFlag("no_interview", "No interview / guaranteed selection",
            "Real hiring involves screening; 'direct joining' is a lure.", 0.35,
            _rx_check(NO_INTERVIEW_RE)),
    RedFlag("free_email_domain", "Uses free e-mail (Gmail/Yahoo)",
            "Companies normally use their own domain (hr@company.com).", 0.30,
            _check_free_email),
    RedFlag("salary_too_high", "Unrealistically high pay",
            "Pay far above market for the role/experience level.", 0.30,
            _check_salary),
    RedFlag("urgency", "Pressure / urgency tactics",
            "'Hurry', 'limited seats', 'today only' push you to act without checking.", 0.20,
            _check_urgency),
    RedFlag("missing_company", "Company not identified",
            "No company name or registered entity (Pvt Ltd / LLP) mentioned.", 0.15,
            _check_missing_company),
    RedFlag("caps_emojis", "Excessive caps / emojis / '!!!'",
            "Professional posts rarely shout or use many emojis.", 0.15,
            _check_caps_emoji),
    RedFlag("short_description", "Very short / vague description",
            "Few details about role, team, or responsibilities.", 0.10,
            _check_short),
]


def red_flag_checklist(text: str, company: str = "", email: str = "", salary: str = "") -> list[dict]:
    """Run every rule and return a list of dicts suitable for JSON/UI."""
    text = to_str(text)
    results = []
    for flag in RED_FLAGS:
        triggered, evidence = flag.check(text, company, email, salary)
        results.append({
            "id": flag.id,
            "name": flag.name,
            "description": flag.description,
            "weight": flag.weight,
            "triggered": bool(triggered),
            "evidence": evidence,
        })
    return results


def rule_score(checklist: list[dict]) -> float:
    """Combine triggered flags with a noisy-OR: 1 - prod(1 - w_i).

    Each additional flag raises the score, but it can never exceed 1, and a
    single weak flag (e.g. short text, 0.10) cannot dominate.
    """
    p_clean = 1.0
    for item in checklist:
        if item["triggered"]:
            p_clean *= 1.0 - item["weight"]
    return round(1.0 - p_clean, 4)


def rule_scores_df(df: pd.DataFrame) -> pd.Series:
    """Vectorised rule score for a DataFrame (used to tune the hybrid threshold)."""
    get = lambda c: df[c] if c in df else pd.Series([""] * len(df), index=df.index)  # noqa: E731
    return pd.Series(
        [rule_score(red_flag_checklist(t, c, e, s))
         for t, c, e, s in zip(get("text"), get("company"), get("email"), get("salary"))],
        index=df.index,
    )


# --------------------------------------------------------------------------- #
# Model input frame (shared by training and prediction)
# --------------------------------------------------------------------------- #
def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return df + `clean_text` + all METADATA_COLUMNS.

    Every classical model is a scikit-learn Pipeline that takes THIS frame as
    input, so training and inference always use identical preprocessing.
    """
    from .preprocess import clean_text

    out = df.copy()
    for c in ("text", "company", "email", "salary"):
        if c not in out:
            out[c] = ""
        out[c] = out[c].fillna("").astype(str)
    out["clean_text"] = [clean_text(t) for t in out["text"]]
    meta = extract_features_df(out)
    for c in METADATA_COLUMNS:
        out[c] = meta[c].values
    return out
