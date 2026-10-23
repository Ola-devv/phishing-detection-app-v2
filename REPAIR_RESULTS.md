# Repaired model candidate — review before public deployment
Date: 23 October 2026

## Status
Rebuilt and locally verified. The repository owner explicitly approved public publication of this model pair and evaluation results on 23 October 2026. Raw datasets, emails and private split references are excluded. Hosting verification is performed separately after the release commit.

## Changes
The existing project design is retained. The selected candidate pair is XGBoost for URLs and Logistic Regression for email, following the updated validation procedure. No bank-domain allowlist or rule forces the classifier output.

- Email: TF-IDF word/bigram features capped at 8,000, plus ten engineered context inputs. Formatting/length proxies are omitted. Credential-request parsing now covers card numbers, CVV, tokens and more request forms, while negation and education remain learned inputs.
- Email training: 22,889 original historical training records plus 1,344 explicitly authored messages from 48 training template families. Validation includes 560 authored variants from 20 separate template families. The authored material is artificial, not authenticated Nigerian bank mail. No exact inspected diagnostic email was inserted into training.
- URL: the original 35-feature schema and historical domain split are retained. Added 10,000 derived training URLs combine training benign hosts with paths sampled only from benign training URLs. A separate 1,500 derived validation paths use validation hosts. These are lexical robustness exercises, not verified safe pages.
- The app always shows the model classification; review notices are separate and do not change predictions.

## Main inspected failures
| Case | Previous deployed result | Repaired candidate result |
|---|---|---|
| Official GTBank internet-banking information page | Phishing, 96.7/100 | Likely legitimate, 20.4/100 |
| Authored harmless credit alert | Phishing, 99.9/100 | Likely legitimate, 23.6/100 |
| Authored card-number/CVV/PIN request | Legitimate, 51.3/100 | Potential phishing, 99.8/100 |

Model scores are uncalibrated outputs, not accuracy or verified probability. These specific improvements do not prove population performance.

## Diagnostic results
| Check | Previously deployed pair | Repaired pair |
|---|---:|---:|
| Original 12 official banking pages cleared | 11/12 | 12/12 |
| Additional official pages cleared | Not checked in that run | 8/8 |
| Constructed suspicious URLs flagged | 16/16 | 16/16 |
| Original authored benign pasted emails cleared | 4/12 | 9/12 |
| Original authored phishing pasted emails flagged | 11/12 | 12/12 |
| Four authored HTML emails correct | 2/4 | 4/4 |
| Additional authored benign emails cleared | Not checked in that run | 11/12 |
| Additional authored phishing emails flagged | Not checked in that run | 12/12 |

Eight invalid URLs were rejected in the preceding run; URL parsing is unchanged. The email review mechanism still handles short input, and the app blocks an empty pasted body. The 24 additional authored email cases were excluded from fitting and threshold selection but were already inspected before the rebuild. They are not fresh independent final testing. Official page URLs came from the banks’ websites. Other positive/negative examples are labelled by construction and are not a representative benchmark.

## Validation comparison
Email rule: maximum historical validation phishing F1 subject to historical FPR ≤2%, historical recall ≥85%, authored-family validation FPR ≤10% and recall ≥90%; recall then lower FPR break ties. URL rule: maximum validation recall subject to historical FPR ≤2% and derived-path FPR ≤5%; choose the algorithm by recall, then F1 and lower FPR. Historical test and inspected regression cases do not set thresholds.

| Task | Algorithm | Eligible | Threshold | Historical validation recall | F1 | FPR | Constructed validation recall / FPR |
|---|---|---|---:|---:|---:|---:|---|
| email | LR | True | 0.629481 | 96.20% | 95.90% | 0.18% | 90.00% / 0.00% |
| email | RF | True | 0.483164 | 91.46% | 92.48% | 0.26% | 91.07% / 0.36% |
| email | XGB | False | 0.500000 | 95.89% | 89.38% | 0.77% | 75.00% / 0.00% |
| url | LR | True | 0.823392 | 87.28% | 92.85% | 1.45% | N/A / 5.00% |
| url | RF | True | 0.588542 | 93.15% | 95.95% | 2.00% | N/A / 2.40% |
| url | XGB | True | 0.604687 | 93.43% | 96.10% | 2.00% | N/A / 1.53% |

## Historical development test results
These are previously inspected historical partitions, not fresh final testing. The email set has 8,248 inherited negatives and 329 Nazario-attributed positives; the URL set has 20,043 negatives and 28,954 positives.

| Task | Algorithm | TN | FP | FN | TP | Accuracy | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| email | LR | 8235 | 13 | 13 | 316 | 99.70% | 96.05% | 96.05% |
| email | RF | 8229 | 19 | 44 | 285 | 99.27% | 86.63% | 90.05% |
| email | XGB | 8197 | 51 | 18 | 311 | 99.20% | 94.53% | 90.01% |
| url | LR | 19750 | 293 | 5741 | 23213 | 87.68% | 80.17% | 88.50% |
| url | RF | 19676 | 367 | 2300 | 26654 | 94.56% | 92.06% | 95.24% |
| url | XGB | 19629 | 414 | 1763 | 27191 | 95.56% | 93.91% | 96.15% |

## Remaining errors and limitations
The selected email model still falsely flags the original password-change notice, maintenance notice and security-education example. It also flags a fee notification in the additional authored cases. Thus this is an improved research demonstration, not a solved or production-ready detector. A high score may still be wrong.

The full original 115-message bank mailbox and 436-message external phishing mailbox could not be recovered for this retraining. In particular, the preceding 94 raw-bank training representatives and nine raw-bank validation messages are not reused here. Their old results must not be transferred to these candidates. An earlier cleaned-text subset check does not replace a raw-message evaluation. The current honest banking evidence is limited to the disclosed official URL checks and authored email diagnostics.

All retained historical positives remain Nazario-attributed and the negative labels are inherited. Source/class confounding, class imbalance and repeated development inspection still limit the high historical scores. The candidate is not uniformly better than every earlier model: the old LR historical test F1 was 97.26%, while this new LR is 96.05%; the repaired candidate improves specific local-language diagnostics but that does not establish universal superiority.

The historical email family split reconstructs 39,429 retained rows exactly; no historical validation/test rows are used for fitting. URL domain reservations are likewise preserved. Authored train/validation families are distinct and exact cleaned train overlap with held-out historical records and inspected pasted examples is checked. This does not guarantee exhaustive semantic independence of authored wording.

## Verification and thesis impact
Seven integration checks pass, including feature schemas, saved thresholds, exported prediction parity, MIME hidden-link handling, malformed URLs/domain boundaries and the three previous live failures. Local Streamlit smoke checks pass for the GTBank page, credit notice, card-details request and simultaneous display of a review notice and model label.

The current thesis review copy describes the preceding pair. Before submission, its model selections, thresholds, email feature schema, vocabulary size, augmentation, validation procedure and metrics must be updated. Do not describe the previous raw-bank or external-mailbox metrics as results of the new candidates. The additional tests are diagnostics, not new thesis accuracy figures.

## Reproduction and privacy
The private model ZIP contains learned weights and the email vocabulary, but no raw email records. The private references ZIP contains split/row-selection metadata and no raw email bodies. Keep the reference ZIP private and extract it so data/reproduction_refs contains the NPZ files. Original master CSVs are still required under data/ for retraining.

Reproduction source is on the repair branch. Candidate scripts write to reproduction/repair_v2/models and never automatically replace production bundles. The learned model files have not been published because explicit approval for their public destination is pending.

## Exact remaining diagnostic mistakes
| Case | Score /100 | Expected scenario | Model result |
|---|---:|---|---|
| Password-change notice | 86.39 | Benign | Phishing |
| Maintenance notice | 75.65 | Benign | Phishing |
| Support education | 98.31 | Benign | Phishing |
| Fee notification | 78.21 | Benign | Phishing |
