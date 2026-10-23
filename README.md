# Phishing Detection System
Research prototype for Nigerian financial institutions, adapted from the original URL and email Streamlit application.

## Run
Use Python 3.12, install `requirements.txt`, then run:
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Models used by the app
- URL: Logistic Regression, frozen validation threshold 0.878675231262895.
- Email: XGBoost, frozen validation threshold 0.6766026020050049, with TF-IDF and 20 engineered features.
- `inference.py` checks the saved feature schema and feature-code hashes. Keep the model bundles, feature modules and mailbox reader together.
- These choices follow the recorded validation rules. They are not claims that the selected algorithm wins on every inspected test cohort.

The displayed score is uncalibrated, not a verified probability of phishing. No alert does not establish safety. Manual-review notices do not alter the saved binary prediction. Links are not visited, sender identities are not authenticated, and bank ownership is not verified.

## Verification
```bash
python -m unittest -q test_release.py
```
Six integration checks cover schema and thresholds, saved authored-probe reproduction, MIME hidden links, malformed URLs and domain boundaries, and separation of review notices from model predictions. Authored cases are constructed diagnostics, not independent banking benchmark evidence.

## Deployment
Streamlit Community Cloud uses this repository, branch `main`, entrypoint `app.py`. Commit changes to the connected branch to update the app. The new `*_bundle.joblib` files are ordinary small Git blobs; older model files remain as historical assets and are not loaded by the current app.

No raw private emails or training mailboxes are included. The earlier application remains available in Git history, including commit `5337eae8d929b9fa2948f9392b0ce640e58eec01`. The full evaluation review and Colab notebook are supplied separately.

## Limitations
The genuine Nigerian banking diagnostic is small and negative-only. Historical scores do not establish Nigerian banking accuracy. Source/class confounding, approximate family grouping and repeated development inspection limit generalization claims. Remove confidential information before submitting demo inputs and verify suspicious requests through a known official bank channel.
