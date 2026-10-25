import copy
import json
import unittest
from pathlib import Path
from inference import load_bundle, score, score_message
from result_explanations import explain

ROOT=Path(__file__).resolve().parent

class ExplanationChecks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.url=load_bundle('url');cls.email=load_bundle('email')
 def check_explanation(self,bundle,**kwargs):
  result=score(bundle,**kwargs);before=copy.deepcopy(result)
  info=explain(bundle,result,**kwargs)
  self.assertTrue(info['available'],kwargs)
  self.assertEqual(result,before)
  self.assertEqual(result,score(bundle,**kwargs))
  return info
 def test_saved_url_and_email_cases(self):
  for row in json.loads((ROOT/'results/url_XGB_regression.json').read_text()):
   self.check_explanation(self.url,url=row['url'])
  for _,_,subject,body in json.loads((ROOT/'results/authored_inputs.json').read_text()):
   self.check_explanation(self.email,subject=subject,body=body)
  for row in json.loads((ROOT/'results/email_fresh_authored.json').read_text()):
   self.check_explanation(self.email,subject=row['subject'],body=row['body'])
 def test_specific_destination_and_no_bank_verification(self):
  info=self.check_explanation(self.url,url='https://accessbankplc.com@unrelated.invalid/verify')
  self.assertIn('Registered domain: unrelated.invalid',info['observations'][0])
  self.assertIn('actual destination is unrelated.invalid',info['observations'][1])
 def test_request_and_negation_are_distinct(self):
  malicious=self.check_explanation(self.email,body='Send your PIN and CVV to our agent now.')
  warning=self.check_explanation(self.email,body='Do not send your PIN or CVV to anyone. Keep your account secure.')
  self.assertIn('Direct sensitive-information requests: 1.',malicious['observations'])
  self.assertIn('Instructions not to share sensitive information: 1.',warning['observations'])
  self.assertNotEqual(malicious['higher'],warning['higher'])
 def test_no_secret_values_in_observations(self):
  info=self.check_explanation(self.email,body='Your OTP is 987654. Never share it with anyone.')
  self.assertNotIn('987654',' '.join(info['observations']))
 def test_failed_attribution_keeps_result_and_input_details(self):
  kwargs={'url':'https://www.accessbankplc.com/'}
  result=score(self.url,**kwargs);result['score']=.5;before=copy.deepcopy(result)
  info=explain(self.url,result,**kwargs)
  self.assertFalse(info['available']);self.assertEqual(result,before)
  self.assertEqual(info['higher'],[]);self.assertTrue(info['observations'])
 def test_eml_uses_same_readable_input(self):
  raw=b'Subject: Support\nMIME-Version: 1.0\nContent-Type: text/html; charset=utf-8\n\n<p>Send your PIN to our agent.</p><a href="https://example.invalid/verify">Continue</a>'
  result,msg=score_message(self.email,raw)
  info=explain(self.email,result,subject=msg['subject'],body=msg['model_body'])
  self.assertTrue(info['available']);self.assertIn('Links detected: 1.', ' '.join(info['observations']))

if __name__=='__main__':unittest.main()
