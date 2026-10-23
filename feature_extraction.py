"""Compatibility adapter for the original application interface."""
import features_url as u
import features_email as e
URL_FEATURE_ORDER=u.URL_ORDER
EMAIL_FEATURE_ORDER=e.EMAIL_ORDER
def url_features_to_vector(url):return u.url_frame([url])
def email_features_to_vector(subject,body,sender=None):return e.email_frame([subject or ''],[body or ''])
def clean_text_for_tfidf(text):return e.clean_text(text)
def get_url_risk_factors(url):
 v=u.extract_url(url);out=[]
 if v['has_userinfo']:out.append('User information appears before the destination hostname.')
 if v['has_ip']:out.append('Destination hostname is an IP address.')
 if v['punycode_host']:out.append('Hostname uses internationalized domain encoding.')
 if v['brand_in_host'] or v['brand_in_path']:out.append('A financial brand name appears; ownership is unverified.')
 return out
def get_email_risk_factors(subject,body,sender=None):
 v=e.extract_email(subject or '',body or '');risks=[];warnings=[]
 if v['credential_request_count']:risks.append('Credential-request or verification phrases appear.')
 if v['urgency_density']:risks.append('Urgency wording appears.')
 if v['security_warning_count']:warnings.append('Security-warning wording appears; attackers may also copy it.')
 return risks,warnings
