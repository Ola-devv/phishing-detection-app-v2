# Nigerian financial institution phishing detector

The Streamlit prototype uses XGBoost for URLs and Logistic Regression for email. The email classifier was fitted on 22,889 public emails, 94 genuine banking inbox emails and 2,426 fraud-labelled messages from Nigerian_Fraud.csv. No generated emails were used. Raw email messages and private split records are not included in the repository.

| Task | Model | Threshold | Public test accuracy | Phishing F1 |
|---|---|---:|---:|---:|
| Email | Logistic Regression | 0.668639 | 99.76% | 96.77% |
| URL | XGBoost | 0.686150 | 96.90% | 97.35% |

The email model did not flag any of the 12 held-out genuine banking emails and flagged 802 of 810 separately held-out fraud-labelled emails. Similar fraud templates were grouped before splitting. The fraud flag rate is not recall on verified Nigerian banking phishing. The original public partitions were inspected during development. A low score does not verify that an input is safe.

Run locally with `streamlit run app.py`. The deployment aggregate email results are in `results/email_final.json`; files under `results/collected-data-v1/` and `reproduction/repair_v2/` document earlier runs.
