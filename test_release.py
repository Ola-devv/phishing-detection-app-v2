import unittest,sys,json,copy
from pathlib import Path
import numpy as np,pandas as pd
from inference import load_bundle,score,score_message,review_reason
import features_url as u
ROOT=Path(__file__).resolve().parent
class FixedEstimator:
 classes_=np.array([0,1])
 def predict_proba(self,X):return np.tile([.4,.6],(X.shape[0],1))
class ReleaseChecks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.url=load_bundle('url');cls.email=load_bundle('email')
 def test_schemas_and_saved_thresholds(self):
  self.assertEqual(len(self.url['transforms']['feature_order']),35)
  self.assertEqual(len(self.email['transforms']['feature_order']),10)
  self.assertAlmostEqual(self.url['threshold'],.6046873927116394)
  self.assertAlmostEqual(self.email['threshold'],.556350991464082)
 def test_saved_email_diagnostics(self):
  rows=pd.read_csv(ROOT/'results/email_authored_probes_predictions.csv')
  raw=json.loads((ROOT/'results/authored_inputs.json').read_text())
  p=[score(self.email,subject=s,body=b)['score'] for _,_,s,b in raw]
  np.testing.assert_allclose(p,rows.score,atol=1e-7)
  self.assertEqual((np.array(p)>=self.email['threshold']).tolist(),rows.flagged.tolist())
 def test_threshold_is_not_half(self):
  b=dict(self.email);b['model']=FixedEstimator()
  r=score(b,body='A sufficiently long ordinary message for a threshold check.')
  self.assertEqual(r['score'],.6);self.assertEqual(r['predicted_label'],int(.6>=b['threshold']))
 def test_mime_hidden_link_and_parity(self):
  raw=b'Subject: Account notice\nMIME-Version: 1.0\nContent-Type: text/html; charset=utf-8\n\n<p>Review your account update</p><a href="https://example.com/verify">Continue</a>'
  r,m=score_message(self.email,raw)
  self.assertIn('https://example.com/verify',m['model_body'])
  self.assertEqual(r,score(self.email,subject=m['subject'],body=m['model_body']))
 def test_url_validation_and_domain_boundary(self):
  for x in ['ftp://example.com','https://example.com:99999','not a url']:
   with self.assertRaises(ValueError):score(self.url,url=x)
  self.assertFalse(u.bounded_domain_match('evilfcmb.com','fcmb.com'))
  self.assertTrue(u.bounded_domain_match('alerts.fcmb.com','fcmb.com'))
 def test_previous_live_failures(self):
  self.assertEqual(score(self.url,url='https://www.gtbank.com/personal-banking/ways-to-bank/internet-banking')['predicted_label'],0)
  self.assertEqual(score(self.email,subject='Credit transaction',body='Your account ending 0000 has been credited with NGN 25000. This notification is for your records only. Please check your mobile banking app for details.')['predicted_label'],0)
  self.assertEqual(score(self.email,subject='Card renewal',body='To renew your debit card, reply with the full card number, expiry date, CVV and your card PIN. Our agent will process the renewal today.')['predicted_label'],1)
 def test_review_does_not_change_binary_prediction(self):
  r={'score':.1,'threshold':.6766,'predicted_label':0};old=copy.deepcopy(r)
  self.assertIsNotNone(review_reason(r,'','Enter your password and OTP immediately to restore your account.'))
  self.assertEqual(r,old)
if __name__=='__main__':unittest.main()

