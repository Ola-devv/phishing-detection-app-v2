# Nigerian financial institution phishing detector
The existing URL/email application now uses the repaired, validated model pair. The repository owner approved publishing the selected bundles and evaluation results on 23 October 2026. Raw emails, datasets and split references remain private.

```bash
python -m pip install -r requirements.txt
python -m unittest -q test_release.py
streamlit run app.py
```

The app uses XGBoost for URLs (35 features, threshold 0.6046873927116394) and Logistic Regression for email (8,000 TF-IDF features plus 10 engineered inputs, threshold 0.629481294875887). See REPAIR_RESULTS.md for all algorithms, known failures and changed validation. This is a research demonstration; scores are uncalibrated and predictions may be wrong.

For retraining, put the original master_email_dataset_v2.csv and master_url_dataset.csv under data/. Extract Private_Reproduction_References.zip into the project root to populate data/reproduction_refs. Keep all these files private. Then:
```bash
python reproduction/repair_v2/train_candidates.py email
python reproduction/repair_v2/train_candidates.py url
python reproduction/repair_v2/fresh_checks.py
```
Candidate training never automatically replaces the app models. The original raw-bank and external-phishing mailbox evaluations are not reproduced by these runs. Earlier source and model versions remain in Git history. The thesis review copy needs corresponding updates.
