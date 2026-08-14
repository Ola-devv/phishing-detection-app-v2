# ============================================================
# app.py — Phishing Detection System (URL + Email)
# URL model: Random Forest (v3 features)
# Email model: XGBoost (v2 features + TF-IDF)
#
# IMPORTANT: feature_extraction.py must sit in the same folder —
# it is the single source of truth for how features are computed,
# shared between training and this app, so predictions here match
# training exactly.
# ============================================================
import streamlit as st
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from feature_extraction import (
    url_features_to_vector,
    email_features_to_vector,
    clean_text_for_tfidf,
    get_risk_level,
    get_url_risk_factors,
    get_email_risk_factors,
)

MODEL_DIR = "models"  # folder containing all .joblib files, relative to this app.py

# ── Load all models once, cached across requests ────────────────
@st.cache_resource
def load_models():
    url_model = joblib.load(f"{MODEL_DIR}/url_model_random_forest.joblib")

    email_model = joblib.load(f"{MODEL_DIR}/email_model_v2_xgboost.joblib")
    email_tfidf = joblib.load(f"{MODEL_DIR}/email_tfidf_vectorizer_v2.joblib")
    email_engineered_cols = joblib.load(f"{MODEL_DIR}/email_engineered_cols_v2.joblib")

    return url_model, email_model, email_tfidf, email_engineered_cols

try:
    url_model, email_model, email_tfidf, email_engineered_cols = load_models()
    models_loaded = True
    load_error = None
except Exception as e:
    models_loaded = False
    load_error = str(e)

# ── Prediction functions ─────────────────────────────────────────
def predict_url(url):
    vec = url_features_to_vector(url)
    prob = url_model.predict_proba(vec)[0][1]
    return float(prob)

def predict_email(subject, body, sender=None):
    text_clean = clean_text_for_tfidf((subject or '') + ' ' + (body or ''))
    tfidf_vec = email_tfidf.transform([text_clean])
    eng_vec = email_features_to_vector(subject, body, sender)
    # column order in eng_vec must match email_engineered_cols exactly —
    # email_features_to_vector already enforces this via EMAIL_FEATURE_ORDER
    combined = hstack([tfidf_vec, csr_matrix(eng_vec.values)]).tocsr()
    prob = email_model.predict_proba(combined)[0][1]
    return float(prob)

# ── UI ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Phishing Detection System", page_icon="🛡️")
st.title("🛡️ Phishing Detection System")
st.caption("URL detection: Random Forest · Email detection: XGBoost")

if not models_loaded:
    st.error(
        "Models failed to load. Check that the 'models' folder is present "
        "alongside app.py and contains all required .joblib files.\n\n"
        f"Error: {load_error}"
    )
    st.stop()

tab_url, tab_email = st.tabs(["🔗 Check a URL", "📧 Check an Email"])

with tab_url:
    url_input = st.text_input("Enter a URL to check", placeholder="https://example.com/login")
    if st.button("Analyze URL", key="url_btn"):
        if not url_input.strip():
            st.warning("Please enter a URL.")
        else:
            prob = predict_url(url_input)
            pct = prob * 100
            risk_label, risk_color = get_risk_level(prob)

            st.markdown(f"### Risk Score: {pct:.1f}/100 — :{risk_color}[{risk_label}]")
            st.progress(float(min(max(prob, 0.0), 1.0)))

            if prob >= 0.5:
                st.error(f"⚠️ Likely PHISHING")
            else:
                st.success(f"✅ Likely LEGITIMATE")

            reasons = get_url_risk_factors(url_input)
            if reasons:
                st.markdown("**Contributing factors:**")
                for r in reasons:
                    st.markdown(f"- {r}")
            else:
                st.markdown("_No notable risk indicators detected._")

with tab_email:
    subject_input = st.text_input("Email subject (optional)", key="email_subject")
    sender_input = st.text_input("Sender email address (optional)", key="email_sender",
                                   placeholder="alerts@examplebank.com")
    body_input = st.text_area("Email body", height=200, key="email_body",
                                placeholder="Paste the email content here...")
    if st.button("Analyze Email", key="email_btn"):
        if not body_input.strip():
            st.warning("Please enter the email body.")
        else:
            prob = predict_email(subject_input, body_input, sender_input)
            pct = prob * 100
            risk_label, risk_color = get_risk_level(prob)

            st.markdown(f"### Risk Score: {pct:.1f}/100 — :{risk_color}[{risk_label}]")
            st.progress(float(min(max(prob, 0.0), 1.0)))

            if prob >= 0.5:
                st.error(f"⚠️ Likely PHISHING")
            else:
                st.success(f"✅ Likely LEGITIMATE")

            reasons = get_email_risk_factors(subject_input, body_input, sender_input)
            if reasons:
                st.markdown("**Contributing factors:**")
                for r in reasons:
                    st.markdown(f"- {r}")
            else:
                st.markdown("_No notable risk indicators detected._")

st.divider()
st.caption(
    "This tool provides a probability estimate, not a certainty. "
    "Always verify suspicious messages directly with your bank through "
    "official channels before taking any action."
)
