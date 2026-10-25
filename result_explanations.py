"""Input evidence and verified local contributions; never changes predictions."""
import re
import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit
import features_url as u
import features_email as e

LABELS = {
 'url_length':'URL length', 'has_https':'HTTPS indicator',
 'hostname_length':'Hostname length', 'registered_domain_length':'Registered-domain length',
 'subdomain_depth':'Subdomain depth', 'host_hyphens':'Hyphens in hostname',
 'host_digit_ratio':'Digit proportion in hostname', 'has_ip':'IP-address hostname',
 'has_userinfo':'User information before hostname', 'has_port':'Explicit port',
 'path_length':'Path length', 'path_depth':'Path depth', 'query_length':'Query length',
 'query_params':'Query parameter count', 'percent_count':'Percent-sign count',
 'domain_entropy':'Domain character entropy', 'path_entropy':'Path character entropy',
 'longest_digit_run':'Longest digit sequence in hostname', 'brand_in_host':'Bank-name matches in hostname',
 'brand_in_path':'Bank-name matches in path/query', 'brand_similarity':'Domain similarity to bank names',
 'has_free_hosting':'Shared-hosting indicator', 'punycode_host':'Internationalised hostname encoding',
 'suspicious_token_count':'Sensitive/account URL keyword count', 'repeated_slashes':'Repeated path slashes',
 'hostname_entropy':'Hostname character entropy', 'first_label_entropy':'First hostname label entropy',
 'first_label_vowel_ratio':'Vowel proportion in first hostname label', 'host_dot_count':'Dots in hostname',
 'total_dot_count':'Dots in URL', 'total_hyphen_count':'Hyphens in URL',
 'special_char_count':'Special-character count', 'url_length_category':'URL length category',
 'suspicious_tld':'Flagged domain-suffix indicator',
 'first_label_brand_similarity':'First hostname label similarity to bank names',
 'url_count':'Links in message', 'direct_secret_request_count':'Direct sensitive-information requests',
 'negated_secret_request_count':'Instructions not to share sensitive information',
 'educational_secret_request_count':'Sensitive-information requests discussed as warnings',
 'security_warning_count':'Security-warning patterns', 'otp_notice_count':'OTP notification patterns',
 'transaction_notice_count':'Transaction notification patterns',
 'request_and_warning':'Direct request alongside a security warning',
 'reply_with_secret_count':'Requests to send or reply with sensitive information',
 'has_suspension':'Suspension or restriction wording',
}

def _matrix(bundle, url, subject, body):
 if bundle['kind']=='url':
  frame=u.url_frame([url]); X=frame; names=list(u.URL_ORDER)
 else:
  frame=e.email_frame([subject],[body])
  vectorizer=bundle['transforms']['tfidf']
  words=vectorizer.transform([e.clean_text(subject+' '+body)])
  X=hstack([words,csr_matrix(frame.values)]).tocsr()
  names=['word: '+w for w in vectorizer.get_feature_names_out()]+list(e.EMAIL_ORDER)
 if bundle['scaler'] is not None:X=bundle['scaler'].transform(X)
 return X,names,frame.iloc[0].to_dict()

def _contributions(bundle, X):
 model=bundle['model']
 if hasattr(model,'steps'):
  original_width=X.shape[1]
  for _,transform in model.steps[:-1]:X=transform.transform(X)
  if X.shape[1]!=original_width:raise ValueError('Pipeline changed feature coordinates')
  model=model.steps[-1][1]
 if hasattr(model,'coef_'):
  if list(model.classes_)!=[0,1] or model.coef_.shape[0]!=1:
   raise ValueError('Unsupported linear class order')
  values=np.asarray(X.toarray() if hasattr(X,'toarray') else X).reshape(-1)
  contributions=values*np.asarray(model.coef_[0])
  baseline=float(model.intercept_[0])
 elif hasattr(model,'get_booster'):
  import xgboost as xgb
  # TreeSHAP values are additive in the binary model's raw margin.
  row=model.get_booster().predict(xgb.DMatrix(X),pred_contribs=True)[0]
  contributions=np.asarray(row[:-1]);baseline=float(row[-1])
 else:raise ValueError('Local explanation unsupported for this model')
 return baseline,np.asarray(contributions,dtype=float)

def _url_evidence(url, v):
 _,p,host=u.parse_url(url);domain=u.PSL(host).top_domain_under_public_suffix
 out=[f'Destination hostname: {host}. '+(f'Registered domain: {domain}.' if domain else 'The offline suffix list could not resolve a registered domain.')]
 if v['has_userinfo']:out.append(f'The text before @ is user information. The actual destination is {host}.')
 if v['has_ip']:out.append('The hostname is an IP address rather than a named domain.')
 if v['punycode_host']:out.append('At least one hostname label uses xn-- internationalised encoding. This alone does not prove impersonation.')
 tokens=sorted(set(re.findall(r'[a-z]+',(host+' '+p.path+' '+p.query).lower())) & set(u.SUSPICIOUS))
 if tokens:out.append('URL keywords detected: '+', '.join(tokens)+f". Matched-token count: {int(v['suspicious_token_count'])}.")
 if v['brand_in_host'] or v['brand_in_path']:
  out.append(f"Bank-name pattern matches: {int(v['brand_in_host'])} in hostname and {int(v['brand_in_path'])} in path/query. A name match does not verify bank ownership.")
 out.append(f"URL length: {int(v['url_length'])} characters. Subdomain levels: {int(v['subdomain_depth'])}. Path segments: {int(v['path_depth'])}.")
 return out

def _email_evidence(subject,body,v):
 text=subject+'\n'+body
 out=[]
 for key in ['direct_secret_request_count','negated_secret_request_count',
             'educational_secret_request_count','security_warning_count','otp_notice_count',
             'transaction_notice_count','request_and_warning','reply_with_secret_count','has_suspension']:
  if v[key]:out.append(f'{LABELS[key]}: {int(v[key])}.')
 # Show matched vocabulary only, avoiding account numbers and secret values.
 secrets=sorted(set(m.group().lower() for m in e.BROAD_SECRET.finditer(text.lower())))
 if secrets:out.append('Sensitive-information terms present: '+', '.join(secrets)+'.')
 out.append(f"Links detected: {int(v['url_count'])}. Direct sensitive-information requests detected: {int(v['direct_secret_request_count'])}.")
 if not any(v[k] for k in e.EMAIL_ORDER if k!='url_count'):
  out.append('No engineered request or notification pattern matched. The model also uses the wording learned through TF-IDF.')
 return out

def explain(bundle,result,url=None,subject='',body=''):
 """Fail closed on attribution mismatch, retaining factual observations."""
 subject=subject or '';body=body or ''
 if bundle['kind']=='url':v=u.extract_url(url);observations=_url_evidence(url,v)
 else:v=e.extract_email(subject,body);observations=_email_evidence(subject,body,v)
 answer={'observations':observations,'higher':[],'lower':[],'details':[],
         'available':False,'threshold':float(result['threshold']),
         'summary':f"The score {result['score']*100:.1f}/100 is {'at or above' if result['predicted_label'] else 'below'} the model's decision threshold of {result['threshold']*100:.2f}/100."}
 try:
  X,names,raw=_matrix(bundle,url,subject,body)
  baseline,values=_contributions(bundle,X)
  if len(values)!=len(names) or not np.isfinite(values).all():raise ValueError('Invalid contributions')
  reconstructed=float(expit(baseline+values.sum()))
  if not np.isclose(reconstructed,result['score'],atol=2e-5,rtol=2e-5):
   raise ValueError('Explanation does not reconstruct prediction')
  def item(i):
   name=names[i];label=LABELS.get(name,name)
   observed=raw.get(name)
   suffix=f' (observed value: {observed:g})' if observed is not None else ''
   return label+suffix
  higher=sorted((i for i,x in enumerate(values) if x>1e-8),key=lambda i:-values[i])[:3]
  lower=sorted((i for i,x in enumerate(values) if x< -1e-8),key=lambda i:values[i])[:3]
  answer.update(available=True,higher=[item(i) for i in higher],lower=[item(i) for i in lower],
   details=[{'Feature':LABELS.get(names[i],names[i]),'Contribution (log-odds)':float(values[i])} for i in higher+lower],
   baseline=baseline)
 except Exception:
  # A failed explanation must never hide or replace the classifier result.
  answer['unavailable_reason']='Model feature contributions could not be verified for this input. The detected details below describe the input, rather than the cause of the prediction.'
 return answer
