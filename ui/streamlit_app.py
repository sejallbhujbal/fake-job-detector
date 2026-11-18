"""
Streamlit front-end.

Run:   streamlit run ui/streamlit_app.py
By default the model runs in-process (simplest for laptops and Hugging Face
Spaces). Set API_URL=http://localhost:8000 to call the FastAPI backend instead.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CUSTOM_INDIA_CSV, RESULTS_CSV, RESULTS_DIR, RISK_BANDS  # noqa: E402
from src.explain import segments_to_html  # noqa: E402

API_URL = os.getenv("API_URL", "").rstrip("/")

st.set_page_config(page_title="Job Scam Detector", page_icon="🛡️", layout="wide")


# --------------------------------------------------------------------------- #
# Backend access
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model ...")
def load_predictor():
    from src.predict import Predictor

    return Predictor()


def run_prediction(text, company, email, salary, explain):
    """Call the API if API_URL is set, otherwise predict in-process."""
    if API_URL:
        import requests

        r = requests.post(f"{API_URL}/predict", timeout=60, json={
            "text": text, "company": company, "email": email, "salary": salary, "explain": explain})
        if r.status_code == 422:
            raise ValueError(r.json().get("detail", "Invalid input"))
        r.raise_for_status()
        return r.json()
    return load_predictor().predict(text, company, email, salary, explain=explain)


def backend_info() -> dict:
    if API_URL:
        import requests

        try:
            return requests.get(f"{API_URL}/health", timeout=5).json()
        except Exception as exc:
            return {"mode": f"API unreachable ({exc})", "model_name": "-"}
    return load_predictor().info()


# --------------------------------------------------------------------------- #
# UI components
# --------------------------------------------------------------------------- #
COLORS = {"low": "#16a34a", "medium": "#eab308", "high": "#dc2626"}


def gauge(score: float, level: str):
    try:
        import plotly.graph_objects as go
    except ImportError:  # plain HTML fallback
        st.markdown(
            f"<div style='background:#e5e7eb;border-radius:8px'><div style='width:{score}%;"
            f"background:{COLORS[level]};padding:6px;border-radius:8px;color:white'>"
            f"<b>{score:.0f}/100</b></div></div>", unsafe_allow_html=True)
        return
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 100", "font": {"size": 34}},
        title={"text": "Scam risk score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#111827", "thickness": 0.25},
            "steps": [
                {"range": [0, RISK_BANDS["low"]], "color": "#bbf7d0"},
                {"range": [RISK_BANDS["low"], RISK_BANDS["high"]], "color": "#fef08a"},
                {"range": [RISK_BANDS["high"], 100], "color": "#fecaca"},
            ],
            "threshold": {"line": {"color": COLORS[level], "width": 4}, "value": score},
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def load_examples() -> dict:
    examples = {"— type or paste your own —": None}
    if CUSTOM_INDIA_CSV.exists():
        df = pd.read_csv(CUSTOM_INDIA_CSV, dtype=str, keep_default_na=False)
        for label, name in [("1", "Scam example"), ("0", "Legit example")]:
            for i, row in enumerate(df[df["label"] == label].head(3).itertuples(), 1):
                examples[f"{name} {i}: {row.text[:50]}…"] = row
    return examples


# --------------------------------------------------------------------------- #
# Page 1: check a post
# --------------------------------------------------------------------------- #
def page_check():
    st.title("🛡️ Fake Job & Internship Scam Detector")
    st.caption("Paste a job / internship post. The model predicts whether it is a scam, gives a risk "
               "score and shows which phrases look suspicious.")

    examples = load_examples()
    choice = st.selectbox("Load an example (synthetic)", list(examples))
    ex = examples[choice]

    with st.form("post_form"):
        text = st.text_area("Job / internship post *", value=ex.text if ex else "", height=220,
                            placeholder="Paste the full post here…")
        c1, c2, c3 = st.columns(3)
        company = c1.text_input("Company name (optional)", value=ex.company if ex else "")
        email = c2.text_input("Contact e-mail (optional)", value=ex.email if ex else "")
        salary = c3.text_input("Salary / stipend (optional)", value=ex.salary if ex else "")
        explain = st.radio("Explanation method", ["auto", "lime"], horizontal=True,
                           help="auto = fast exact method for the classical model; "
                                "lime = slower, model-agnostic")
        submitted = st.form_submit_button("Check post", type="primary")

    if not submitted:
        return
    if not text.strip():
        st.error("Please paste the post text first.")
        return

    with st.spinner("Analysing…"):
        try:
            res = run_prediction(text, company, email, salary, explain)
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

    for w in res["warnings"]:
        st.warning(w)

    left, right = st.columns([1, 1.2])
    with left:
        gauge(res["risk_score"], res["risk_level"])
        verdict = {"high": "🚨 Likely FAKE", "medium": "⚠️ Be careful", "low": "✅ Looks legitimate"}
        st.markdown(f"<h3 style='text-align:center;color:{COLORS[res['risk_level']]}'>"
                    f"{verdict[res['risk_level']]} — predicted <b>{res['label']}</b></h3>",
                    unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        mp = res["model_probability"]
        m1.metric("Model P(fake)", "—" if mp is None else f"{mp:.2f}")
        m2.metric("Rule score", f"{res['rule_score']:.2f}")
        m3.metric("Red flags", f"{res['triggered_flags']}/{len(res['red_flags'])}")
        st.caption(f"Model: `{res['model_name']}` · explanation: `{res['explanation_method']}` · "
                   f"language: `{res['language']}`")

    with right:
        st.subheader("Red-flag checklist")
        for f in res["red_flags"]:
            icon = "❌" if f["triggered"] else "✅"
            ev = f" — _“{'”, “'.join(f['evidence'])}”_" if f["triggered"] and f["evidence"] else ""
            st.markdown(f"{icon} **{f['name']}**{ev}", help=f["description"])
        st.caption("❌ = red flag found · ✅ = check passed")

    st.subheader("Highlighted post")
    st.caption("🟥 red = words the model found suspicious · 🟧 orange = rule-based red-flag phrase")
    st.markdown(
        f"<div style='border:1px solid #e5e7eb;border-radius:8px;padding:12px;line-height:1.7'>"
        f"{segments_to_html(res['highlighted_segments'])}</div>", unsafe_allow_html=True)

    if res["highlighted_words"] or res["trust_words"]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top suspicious terms**")
            st.dataframe(pd.DataFrame(res["highlighted_words"]), hide_index=True, use_container_width=True)
        with c2:
            st.markdown("**Terms that looked legitimate**")
            st.dataframe(pd.DataFrame(res["trust_words"]), hide_index=True, use_container_width=True)

    st.info("This tool gives a risk estimate, not a guarantee. Never pay any fee to get a job, and "
            "verify the company on its official website / MCA portal before sharing documents.")


# --------------------------------------------------------------------------- #
# Page 2: model comparison
# --------------------------------------------------------------------------- #
def page_models():
    st.title("📊 Model comparison")
    if not RESULTS_CSV.exists():
        st.warning("No results yet. Train the models first:\n\n`python -m src.train_classical`")
        return
    df = pd.read_csv(RESULTS_CSV)
    c1, c2 = st.columns(2)
    ds_names = {"emscad_test": "Kaggle EMSCAD test set", "custom_india": "Custom Indian dataset"}
    dataset = c1.selectbox("Dataset", df["dataset"].unique(), format_func=lambda d: ds_names.get(d, d))
    mode = c2.selectbox("Mode", ["hybrid", "model_only"],
                        help="hybrid = model + rule-based red flags (what the app uses)")
    sub = df[(df["dataset"] == dataset) & ((df["mode"] == mode) | (df["model"] == "always_real_baseline"))]

    cols = ["model", "precision", "recall", "f1", "pr_auc", "accuracy", "tp", "fn", "fp", "tn", "threshold"]
    st.dataframe(sub[cols].style.format({c: "{:.3f}" for c in ["precision", "recall", "f1", "pr_auc",
                                                                "accuracy", "threshold"]}),
                 hide_index=True, use_container_width=True)

    try:
        import plotly.express as px

        long = sub.melt(id_vars="model", value_vars=["precision", "recall", "f1", "pr_auc", "accuracy"],
                        var_name="metric", value_name="value")
        fig = px.bar(long, x="metric", y="value", color="model", barmode="group", range_y=[0, 1],
                     title=f"Metrics for the FAKE class — {ds_names.get(dataset, dataset)} ({mode})")
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.bar_chart(sub.set_index("model")[["precision", "recall", "f1", "pr_auc"]])

    st.markdown(
        "**Why not accuracy?** Only ~5 % of EMSCAD posts are fake. The `always_real_baseline` row never "
        "predicts *fake* yet reaches ~95 % accuracy with **0 recall**. We optimise **recall** on the fake "
        "class (threshold tuned for F2 on the validation set) and compare models with **PR-AUC**, "
        "because missing a scam is far costlier than a false alarm.")

    pr = RESULTS_DIR / f"pr_curve_{dataset}.png"
    if pr.exists():
        st.image(str(pr), caption="Precision-recall curves (model only)")
    cms = sorted(RESULTS_DIR.glob(f"cm_*_{dataset}.png"))
    if cms:
        st.subheader("Confusion matrices (hybrid)")
        grid = st.columns(min(4, len(cms)))
        for i, p in enumerate(cms):
            grid[i % len(grid)].image(str(p), use_container_width=True)


# --------------------------------------------------------------------------- #
# Page 3: about
# --------------------------------------------------------------------------- #
def page_about():
    st.title("ℹ️ About")
    info = backend_info()
    st.json(info)
    st.markdown("""
**How it works**
1. Text is cleaned (HTML, URLs, e-mails, phone numbers → tokens), lower-cased and lemmatised.
2. 16 metadata features are computed (free e-mail domain, fees, WhatsApp/Telegram contact, urgency, caps…).
3. A trained model (TF-IDF + Logistic Regression / XGBoost, or DistilBERT) outputs P(fake).
4. A rule-based red-flag score is combined with it (noisy-OR), so Indian scam patterns not present in the
   Kaggle data still raise the risk.
5. The risk score is scaled so that **50 = decision threshold** (tuned for recall on validation data).

**Limitations:** English only, trained mostly on 2012-14 US/UK posts, the custom Indian set starts with
synthetic examples, and scammers adapt — always verify independently.
""")


PAGES = {"🔍 Check a post": page_check, "📊 Model comparison": page_models, "ℹ️ About": page_about}
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", list(PAGES), label_visibility="collapsed")
    info = backend_info()
    st.caption(f"Backend: **{'API ' + API_URL if API_URL else 'in-process'}**  \n"
               f"Mode: `{info.get('mode')}`  \nModel: `{info.get('model_name')}`")
PAGES[page]()
