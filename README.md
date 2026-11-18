# Nigerian financial institution phishing detector

The `collected-data-v1` research prototype uses XGBoost for URLs and Logistic Regression for email. The revised training and validation used collected public datasets and authentic inbox records, with no generated examples. The repository does not include the raw inbox, master datasets or private split records.

| Task | Model | Threshold | Historical test accuracy | Phishing F1 |
|---|---|---:|---:|---:|
| Email | Logistic Regression | 0.525960 | 99.79% | 97.29% |
| URL | XGBoost | 0.686150 | 96.90% | 97.35% |

The figures describe raw binary outputs on the previously inspected historical test splits. They do not establish performance on representative Nigerian banking traffic. The email model cleared 11 of 12 reserved legitimate bank emails. When recognised warning, transaction or sensitive-request wording conflicts with the email score, the interface can show **Needs review**. This display safeguard does not change model scores or benchmark metrics. A lower score does not establish safety.

```bash
python -m pip install -r requirements.txt
python -m unittest test_release.py test_explanations.py
streamlit run app.py
```

`REPAIR_RESULTS.md`, `Repair_Candidate_Colab.ipynb`, and the other files under `reproduction/repair_v2/` describe an earlier experiment. Their results and constructed diagnostics are historical and are not inputs to the current models.
