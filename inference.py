"""Frozen feature schemas, preprocessing, reader and validation thresholds."""
from pathlib import Path
import hashlib
import joblib
import numpy as np
from scipy.sparse import csr_matrix,hstack
import features_url as u
import features_email as e
from mailbox_reader import readable_message
ROOT=Path(__file__).resolve().parent
def load_bundle(kind):
 if kind not in ('url','email'):raise ValueError('Unknown input type')
 b=joblib.load(ROOT/'models'/(kind+'_bundle.joblib'))
 module='features_url.py' if kind=='url' else 'features_email.py'
 order=u.URL_ORDER if kind=='url' else e.EMAIL_ORDER
 if hashlib.sha256((ROOT/module).read_bytes()).hexdigest()!=b['feature_sha256']:raise RuntimeError('Feature code differs from training')
 if b['transforms']['feature_order']!=order:raise RuntimeError('Feature order differs from training')
 if kind=='email' and hashlib.sha256((ROOT/'mailbox_reader.py').read_bytes()).hexdigest()!=b['reader_sha256']:raise RuntimeError('Email extraction differs from training')
 return b
def score(b,url=None,subject='',body=''):
 if b['kind']=='url':X=u.url_frame([url])
 else:X=hstack([b['transforms']['tfidf'].transform([e.clean_text((subject or '')+' '+(body or ''))]),csr_matrix(e.email_frame([subject or ''],[body or '']).values)]).tocsr()
 if b['scaler'] is not None:X=b['scaler'].transform(X)
 p=float(b['model'].predict_proba(X)[0,list(b['model'].classes_).index(1)])
 if not np.isfinite(p):raise RuntimeError('Invalid model score')
 return dict(score=p,threshold=float(b['threshold']),predicted_label=int(p>=b['threshold']))
def score_message(b,raw):
 msg=readable_message(raw)
 return score(b,subject=msg['subject'],body=msg['model_body']),msg
def review_reason(result,subject,body):
 v=e.extract_email(subject or '',body or '')
 if len(e.clean_text(body).split())<8:return 'Too little readable text for a dependable assessment.'
 if not result['predicted_label'] and v['direct_secret_request_count']:return 'A sensitive-information request conflicts with the lower model score.'
 warning=any(p.search(((subject or '')+' '+(body or '')).lower()) for p in e.WARN)
 if result['predicted_label'] and (warning or v['educational_secret_request_count']) and not v['direct_secret_request_count']:return 'Protective or educational wording conflicts with the model alert.'
 return None

