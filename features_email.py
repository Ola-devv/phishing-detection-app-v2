"""Shared, offline feature extraction used by training and inference."""
import re
import math
import ipaddress
from urllib.parse import urlsplit, parse_qsl
from difflib import SequenceMatcher
from collections import Counter
import numpy as np
import pandas as pd
import tldextract

PSL = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)
BRANDS = ['gtbank','accessbank','zenithbank','opay','palmpay','firstbank','uba','fcmb',
          'moniepoint','kuda','flutterwave','paystack','sterling','fidelity','stanbic','ecobank']
HOSTING = ['weebly.com','wixsite.com','netlify.app','blogspot.com','wordpress.com','github.io',
           'glitch.me','firebaseapp.com','web.app','myclickfunnels.com','carrd.co','yolasite.com']
SUSPICIOUS = ['login','signin','verify','secure','account','update','banking','confirm',
              'bvn','transfer','suspended','blocked','urgent','otp','pin','password']
EMAIL_ORDER = ['text_length_log','word_count_log','url_count','credential_request_count',
 'security_warning_count','credential_request_net_score','urgency_density','nigerian_keyword_density',
 'exclamation_ratio','caps_ratio','digit_count','avg_word_length','has_suspension']
URL_ORDER = ['url_length','has_https','hostname_length','registered_domain_length','subdomain_depth',
 'host_hyphens','host_digit_ratio','has_ip','has_userinfo','has_port','path_length','path_depth',
 'query_length','query_params','percent_count','domain_entropy','path_entropy','longest_digit_run',
 'brand_in_host','brand_in_path','brand_similarity','has_free_hosting','punycode_host',
 'suspicious_token_count','repeated_slashes','hostname_entropy','first_label_entropy',
 'first_label_vowel_ratio','host_dot_count','total_dot_count','total_hyphen_count',
 'special_char_count','url_length_category','suspicious_tld','first_label_brand_similarity']

def bounded_domain_match(host, domain):
    host, domain = host.lower().rstrip('.'), domain.lower().rstrip('.')
    return host == domain or host.endswith('.' + domain)

def parse_url(raw):
    raw = str(raw).strip()
    if not raw or re.search(r'\s',raw):
        raise ValueError('Enter one URL without spaces.')
    if '://' not in raw:
        raw = 'https://' + raw if not raw.startswith('//') else 'https:' + raw
    p = urlsplit(raw)
    if p.scheme.lower() not in ('http','https') or not p.hostname:
        raise ValueError('Only HTTP or HTTPS URLs with a hostname are supported.')
    host = p.hostname.lower().rstrip('.').encode('idna').decode('ascii')
    _ = p.port  # Reject invalid port syntax/range.
    if not host or '..' in host:
        raise ValueError('The hostname is invalid.')
    return raw, p, host

def url_group(raw):
    _, p, host = parse_url(raw)
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        e = PSL(host)
        return e.top_domain_under_public_suffix or host

def url_identity(raw):
    _, p, host = parse_url(raw)
    port = p.port
    authority = host + (':' + str(port) if port and port != (443 if p.scheme=='https' else 80) else '')
    # Keep path/query case and ordering; discard fragment, which is not sent to server.
    return p.scheme.lower() + '://' + authority + (p.path or '/') + ('?' + p.query if p.query else '') + ('|userinfo' if p.username is not None else '')

def entropy(s):
    counts = Counter(s)
    return -sum((v/len(s))*math.log2(v/len(s)) for v in counts.values()) if s else 0.

def brand_hits(s):
    # Short uba needs boundaries to avoid matching aruba.
    return sum(bool(re.search(r'(?<![a-z])uba(?![a-z])',s)) if b=='uba' else b in re.sub(r'[-\s]','',s) for b in BRANDS)

def extract_url(raw):
    raw, p, host = parse_url(raw)
    e = PSL(host); core=e.domain or host; reg=e.top_domain_under_public_suffix or host
    try: ipaddress.ip_address(host); is_ip=1
    except ValueError: is_ip=0
    tokens=re.findall(r'[a-z]+',(host+' '+p.path+' '+p.query).lower())
    f = dict(url_length=len(raw),has_https=int(p.scheme.lower()=='https'),hostname_length=len(host),
      registered_domain_length=len(reg),subdomain_depth=0 if is_ip else len(e.subdomain.split('.')) if e.subdomain else 0,
      host_hyphens=host.count('-'),host_digit_ratio=sum(c.isdigit() for c in host)/max(1,len(host)),
      has_ip=is_ip,has_userinfo=int(p.username is not None),has_port=int(p.port is not None),
      path_length=len(p.path),path_depth=len([v for v in p.path.split('/') if v]),query_length=len(p.query),
      query_params=len(parse_qsl(p.query,keep_blank_values=True)),percent_count=raw.count('%'),
      domain_entropy=entropy(core),path_entropy=entropy(p.path),
      longest_digit_run=max((len(v) for v in re.findall(r'\d+',host)),default=0),
      brand_in_host=brand_hits(host),brand_in_path=brand_hits((p.path+' '+p.query).lower()),
      brand_similarity=max(SequenceMatcher(None,core,b).ratio() for b in BRANDS),
      has_free_hosting=int(any(bounded_domain_match(host,d) for d in HOSTING)),
      punycode_host=int(any(s.startswith('xn--') for s in host.split('.'))),
      suspicious_token_count=sum(t in SUSPICIOUS for t in tokens),repeated_slashes=int('//' in p.path),
      hostname_entropy=entropy(host),first_label_entropy=entropy(host.split('.')[0]),
      first_label_vowel_ratio=sum(c in 'aeiou' for c in host.split('.')[0])/(len(host.split('.')[0])+1),
      host_dot_count=host.count('.'),total_dot_count=raw.count('.'),total_hyphen_count=raw.count('-'),
      special_char_count=len(re.findall(r'[%=&#~?]',raw)),
      url_length_category=0 if len(raw)<54 else 1 if len(raw)<75 else 2,
      suspicious_tld=int(e.suffix in ['xyz','top','tk','ml','ga','cf','gq','info','click','link']),
      first_label_brand_similarity=max(SequenceMatcher(None,host.split('.')[0],b).ratio() for b in BRANDS))
    return f

REQUESTS = [r'\b(?:enter|provide|send|share|input|confirm|verify|update|link)\s+(?:us\s+)?your\s+(?:otp|bvn|pin|password)\b',
            r'\bclick\s+(?:here|below)\s+to\s+(?:verify|confirm)\b',r'\b(?:verify|confirm)\s+now\b']
WARNINGS = [r'\b(?:will\s+never|will\s+not|won.t)\s+(?:ask|request)\b',
 r'\b(?:do\s+not|don.t|never)\s+(?:share|disclose|provide|send)\b',
 r'\b(?:never|not)\s+be\s+disclosed\b',r'\bwe\s+do\s+not\s+request\b',
 r'\bbeware\s+of\s+(?:fraudsters|scammers|phishing)\b',r'\bprotect\s+your\s+(?:otp|pin|password|bvn)\b']
REQ=[re.compile(p) for p in REQUESTS]; WARN=[re.compile(p) for p in WARNINGS]
URGENCY={'urgent','immediately','suspended','expire','expired','winner','prize','blocked','restricted','unauthorized'}

def clean_text(text):
    t=str(text).lower()
    t=re.sub(r'https?://\S+|www\.\S+',' urltoken ',t)
    t=re.sub(r'\S+@\S+',' emailtoken ',t)
    return re.sub(r'\s+',' ',re.sub(r'[^a-z\s]',' ',t)).strip()

def email_group(subject, body):
    # Exact normalized body/template family; not a claim of exhaustive near-duplicate discovery.
    s=clean_text(body if str(body).strip() else subject)
    for b in sorted(BRANDS+['access bank','first bank','zenith'],key=len,reverse=True):
        s=re.sub(r'\b'+re.escape(b)+r'\b',' banktoken ',s)
    return re.sub(r'\s+',' ',s).strip()

def extract_email(subject,body):
    raw=(subject or '')+' '+(body or ''); s=raw.lower(); words=s.split(); n=len(words)
    # Requests are counted independently of warnings: copied disclaimers never override malicious asks.
    req=sum(len(p.findall(s)) for p in REQ); warning=sum(len(p.findall(s)) for p in WARN)
    toks=re.findall(r'[a-z]+',s)
    return dict(text_length_log=math.log1p(len(s)),word_count_log=math.log1p(n),
      url_count=len(re.findall(r'https?://\S+|www\.\S+',raw)),credential_request_count=req,
      security_warning_count=warning,credential_request_net_score=req-warning,
      urgency_density=sum(t in URGENCY for t in toks)/(n+1),nigerian_keyword_density=brand_hits(s)/(n+1),
      exclamation_ratio=raw.count('!')/(n+1),caps_ratio=sum(c.isupper() for c in raw)/(len(raw)+1),
      digit_count=sum(c.isdigit() for c in s),avg_word_length=float(np.mean([len(w) for w in words])) if words else 0.,
      has_suspension=int(bool(re.search(r'\b(?:suspend\w*|block\w*|restrict\w*|deactivat\w*)\b',s))))

def url_frame(urls):
    return pd.DataFrame([extract_url(u) for u in urls],columns=URL_ORDER)

def email_frame(subjects,bodies):
    return pd.DataFrame([extract_email(s,b) for s,b in zip(subjects,bodies)],columns=EMAIL_ORDER)

# Context features are learned inputs, never automatic safe overrides.
CONTEXT_ORDER = ['direct_secret_request_count', 'negated_secret_request_count',
                 'educational_secret_request_count', 'otp_notice_count',
                 'transaction_notice_count', 'request_and_warning',
                 'reply_with_secret_count']
EMAIL_ORDER = EMAIL_ORDER + CONTEXT_ORDER
BASE_EXTRACT_EMAIL = extract_email
SECRET = r'(?:otp|one[ -]time\s+(?:password|code)|bvn|pin|password|passcode|cvv|verification\s+code)'
DIRECT = re.compile(r'\b(?:enter|provide|send|share|input|supply|disclose|reply\s+with|respond\s+with|confirm|submit)\s+(?:(?:us|me)\s+)?(?:(?:your|the)\s+)?'+SECRET+r'\b')
NEGATION = re.compile(r'\b(?:(?:do\s+not|don.t|never|not\s+to|avoid)\s+(?:ever\s+)?|(?:never|not|won.t)\s+(?:ask|request)(?:\s+\w+){0,4}\s+to\s+)$')
EDUCATION = re.compile(r'\b(?:fraudsters?|scammers?|criminals?|attackers?|phish(?:ing|ers?))\b|\b(?:may|might)\s+ask\b|\brequests?\s+(?:to|that)\b|\bif\s+(?:anyone|someone|a\s+(?:caller|person)|they|an?\s+agent).{0,60}\b(?:asks?|tells?|instructs?).{0,25}\bto\s*$')
def context_features(subject, body):
    text=((subject or '')+'\n'+(body or '')).lower()
    direct=negated=education=reply=0
    for clause in re.split(r'[.!?;\n]+|\b(?:but|however|nevertheless|instead)\b',text):
        for match in DIRECT.finditer(clause):
            before=clause[:match.start()]
            # Bound context to the current clause and last twelve words.
            nearby=' '.join(before.split()[-12:])+' '
            if NEGATION.search(nearby):negated+=1
            elif EDUCATION.search(nearby):education+=1
            else:
                direct+=1
                reply+=int(bool(re.search(r'reply|respond|send|share|disclose',match.group())))
    return {'direct_secret_request_count':direct,'negated_secret_request_count':negated,
            'educational_secret_request_count':education,
            'otp_notice_count':len(re.findall(r'\b(?:your\s+)?(?:otp|one[ -]time\s+(?:password|code)|verification\s+code)\s+(?:is|:)\s*\d{4,8}\b',text)),
            'transaction_notice_count':int(bool(re.search(r'\b(?:transaction\s+(?:notification|alert|has\s+occurred)|payment\s+(?:received|successful)|(?:credited|debited)\s+(?:with|by)|receipt\s+(?:is|attached))\b',text))),
            'request_and_warning':int(direct>0 and any(p.search(text) for p in WARN)),
            'reply_with_secret_count':reply}
def extract_email(subject, body):
    return {**BASE_EXTRACT_EMAIL(subject,body),**context_features(subject,body)}
