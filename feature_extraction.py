# ============================================================
# feature_extraction.py
# Shared feature logic for URL + Email phishing detection.
# Import this in BOTH training scripts and the deployed app,
# so features computed at inference time exactly match what
# the models were trained on. Never duplicate this logic.
# ============================================================
import re
import math
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# URL FEATURES (v3) — must match url_feature_extraction_v3.py
# ─────────────────────────────────────────────────────────────
from difflib import SequenceMatcher

FREE_HOSTING = [
    'weebly.com', 'wix.com', 'wixsite.com', 'netlify.app',
    'blogspot.com', 'wordpress.com', 'github.io', 'glitch.me',
    'firebaseapp.com', 'web.app', '000webhostapp.com',
    'infinityfreeapp.com', 'byethost', 'atspace.com',
    'freehostia.com', 'freeiz.com', 'myclickfunnels.com',
    'clickfunnels.com', 'carrd.co', 'yolasite.com'
]
NIGERIAN_KEYWORDS = [
    'gtbank','accessbank','access-bank','zenith','opay',
    'palmpay','firstbank','first-bank','uba','fidelity',
    'sterling','fcmb','moniepoint','kuda','flutterwave',
    'paystack','nibss','remita','nigeria','naira','ngn','bvn','ussd'
]
URL_KNOWN_BRANDS = [
    'gtbank', 'opay', 'palmpay', 'zenith', 'uba', 'firstbank',
    'accessbank', 'fcmb', 'sterling', 'fidelity', 'moniepoint',
    'kuda', 'flutterwave', 'paystack', 'wema', 'polaris', 'union',
    'ecobank', 'stanbic', 'providus', 'jaiz', 'heritage'
]
SUSPICIOUS_TLDS = ['.xyz', '.top', '.tk', '.ml', '.ga', '.cf', '.gq', '.info', '.click', '.link']

URL_FEATURE_ORDER = [
    'url_length', 'has_https', 'num_dots', 'num_hyphens',
    'suspicious_path_depth', 'has_ip', 'url_length_cat', 'has_at_symbol',
    'special_char_count', 'nigerian_url_keyword', 'excessive_subdomains',
    'has_port', 'domain_length', 'has_free_hosting', 'digit_ratio',
    'has_redirect', 'suspicious_word_count', 'has_brand_plus_extra',
    'brand_similarity_score', 'is_typosquat_domain', 'domain_entropy',
    'path_entropy', 'vowel_ratio', 'longest_digit_run', 'has_suspicious_tld',
]

SUSPICIOUS_WORDS = [
    'login','signin','verify','secure','account','update',
    'banking','confirm','naira','nigeria','ngn','bvn','transfer',
    'alert','suspended','blocked','urgent','otp','pin','password'
]

def _shannon_entropy(s):
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)

def _fuzzy_brand_match(domain, brands=URL_KNOWN_BRANDS, threshold=0.75):
    core = domain.split('.')[0] if domain else ''
    best = 0.0
    matched_brand = None
    for brand in brands:
        ratio = SequenceMatcher(None, core, brand).ratio()
        if ratio > best:
            best = ratio
            matched_brand = brand
    is_typosquat = 1 if (matched_brand and best >= threshold and core != matched_brand) else 0
    return best, is_typosquat

def extract_url_features(url):
    """Returns a dict of v3 URL features for a single URL string.
    MUST stay identical to url_feature_extraction_v3.py's logic."""
    s = str(url).lower().strip()
    domain_part = re.sub(r'^https?://', '', s)
    domain = domain_part.split('/')[0]
    path = '/'.join(domain_part.split('/')[1:]) if '/' in domain_part else ''
    core_domain = domain.split('.')[0] if domain else ''

    url_length = len(s)
    has_https = 1 if s.startswith('https') else 0
    num_dots = domain.count('.')
    num_hyphens = domain.count('-')
    path_depth = path.count('/') + 1 if path else 0
    suspicious_path_depth = 1 if path_depth >= 4 else 0
    has_ip = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain) else 0
    url_length_cat = 0 if url_length < 54 else (1 if url_length < 75 else 2)
    has_at_symbol = 1 if '@' in s else 0
    special_char_count = len(re.findall(r'[%=&#~?]', s))
    nigerian_url_keyword = 1 if any(w in s for w in NIGERIAN_KEYWORDS) else 0
    parts = domain.split('.')
    num_subdomains = max(0, len(parts) - 2)
    excessive_subdomains = 1 if num_subdomains >= 2 else 0
    has_port = 1 if re.search(r':\d{2,5}', s) else 0
    domain_length = len(domain)
    has_free_hosting = 1 if any(fh in s for fh in FREE_HOSTING) else 0
    digits_in_domain = sum(1 for c in domain if c.isdigit())
    digit_ratio = digits_in_domain / (len(domain) + 1)
    has_redirect = 1 if any(w in s for w in ['redirect', 'return=', 'next=', 'goto=', 'url=']) else 0
    suspicious_word_count = sum(1 for w in SUSPICIOUS_WORDS if w in s)

    has_brand_plus_extra = 0
    for brand in URL_KNOWN_BRANDS:
        if brand in domain and len(domain) > len(brand) + 4:
            has_brand_plus_extra = 1
            break

    brand_similarity_score, is_typosquat_domain = _fuzzy_brand_match(domain)
    domain_entropy = _shannon_entropy(core_domain)
    path_entropy = _shannon_entropy(path)
    vowels = sum(1 for c in core_domain if c in 'aeiou')
    vowel_ratio = vowels / (len(core_domain) + 1)
    digit_runs = re.findall(r'\d+', domain)
    longest_digit_run = max((len(d) for d in digit_runs), default=0)
    has_suspicious_tld = 1 if any(domain.endswith(t) for t in SUSPICIOUS_TLDS) else 0

    return {
        'url_length': url_length, 'has_https': has_https, 'num_dots': num_dots,
        'num_hyphens': num_hyphens, 'suspicious_path_depth': suspicious_path_depth,
        'has_ip': has_ip, 'url_length_cat': url_length_cat, 'has_at_symbol': has_at_symbol,
        'special_char_count': special_char_count, 'nigerian_url_keyword': nigerian_url_keyword,
        'excessive_subdomains': excessive_subdomains, 'has_port': has_port,
        'domain_length': domain_length, 'has_free_hosting': has_free_hosting,
        'digit_ratio': digit_ratio, 'has_redirect': has_redirect,
        'suspicious_word_count': suspicious_word_count, 'has_brand_plus_extra': has_brand_plus_extra,
        'brand_similarity_score': brand_similarity_score, 'is_typosquat_domain': is_typosquat_domain,
        'domain_entropy': domain_entropy, 'path_entropy': path_entropy,
        'vowel_ratio': vowel_ratio, 'longest_digit_run': longest_digit_run,
        'has_suspicious_tld': has_suspicious_tld,
    }

def url_features_to_vector(url):
    """Returns a single-row DataFrame in the EXACT column order the
    URL model expects. Use this at inference time."""
    feats = extract_url_features(url)
    return pd.DataFrame([feats])[URL_FEATURE_ORDER]


# ─────────────────────────────────────────────────────────────
# EMAIL FEATURES (v2) — must match email_feature_extraction_v2.py
# ─────────────────────────────────────────────────────────────
EMAIL_FEATURE_ORDER = [
    'text_length_log', 'word_count_log', 'url_count',
    'credential_request_count', 'security_warning_count',
    'credential_request_net_score', 'urgency_density',
    'nigerian_keyword_density', 'exclamation_ratio', 'caps_ratio',
    'digit_count', 'avg_word_length', 'has_suspension',
    'has_sender_info', 'sender_is_free_email', 'brand_domain_mismatch',
]

REQUEST_PATTERNS = [
    r'enter your otp', r'provide your otp', r'send (us )?your otp',
    r'share your otp', r'input your otp', r'confirm your otp',
    r'otp is', r'your otp code',
    r'enter your bvn', r'provide your bvn', r'send (us )?your bvn',
    r'share your bvn', r'verify your bvn', r'update your bvn',
    r'confirm your bvn', r'link your bvn',
    r'enter your pin', r'provide your pin', r'share your pin',
    r'enter your password', r'confirm your password',
    r'click here to verify', r'click below to verify',
    r'click here to confirm', r'verify now', r'confirm now',
]
WARNING_PATTERNS = [
    r"we (will )?never ask", r"will not ask", r"won'?t ask",
    r'do not share your', r"don'?t share your",
    r'never share your', r'never disclose', r'do not disclose',
    r'never request your', r'we do not request',
    r'please note that we will never',
    r'beware of (fraudsters|scammers|phishing)',
    r'protect your (otp|pin|password|bvn)',
]
URGENCY_WORDS = [
    'urgent', 'immediately', 'suspended', 'expire', 'winner',
    'prize', 'congratulations', 'blocked', 'restricted',
    'unusual', 'unauthorized'
]
EMAIL_NIGERIAN_KW = [
    'gtbank', 'access bank', 'accessbank', 'zenith', 'opay',
    'palmpay', 'firstbank', 'first bank', 'uba', 'fidelity',
    'sterling', 'fcmb', 'moniepoint', 'kuda', 'flutterwave', 'paystack'
]
FREE_EMAIL_DOMAINS = [
    'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'aol.com',
    'icloud.com', 'mail.com', 'yandex.com', 'protonmail.com', 'live.com',
]
BANK_OFFICIAL_DOMAINS = {
    'gtbank': ['gtbank.com', 'gtworld.ng', 'gtcoplc.com'],
    'accessbank': ['accessbankplc.com'],
    'zenith': ['zenithbank.com'],
    'opay': ['opayweb.com', 'opay-inc.com'],
    'palmpay': ['palmpay.com'],
    'firstbank': ['firstbanknigeria.com', 'fbnbank.com'],
    'uba': ['ubagroup.com', 'ubaneoplanet.com'],
    'fidelity': ['fidelitybank.ng'],
    'sterling': ['sterling.ng', 'sterlingbankng.com'],
    'fcmb': ['fcmb.com'],
    'moniepoint': ['moniepoint.com'],
    'kuda': ['kuda.com'],
    'flutterwave': ['flutterwave.com'],
    'paystack': ['paystack.com'],
}

_REQUEST_COMPILED = [re.compile(p) for p in REQUEST_PATTERNS]
_WARNING_COMPILED = [re.compile(p) for p in WARNING_PATTERNS]

def _get_sender_domain(sender):
    if not sender:
        return None
    m = re.search(r'@([\w\.-]+)', str(sender).lower())
    return m.group(1) if m else None

def _brand_domain_mismatch(text_lower, sender_domain):
    if sender_domain is None:
        return 0
    for brand, domains in BANK_OFFICIAL_DOMAINS.items():
        if brand in text_lower:
            if sender_domain in FREE_EMAIL_DOMAINS:
                return 1
            if not any(sender_domain.endswith(d) for d in domains):
                return 1
    return 0

def clean_text_for_tfidf(text):
    """MUST match the cleaning used when the TF-IDF vectorizer was fit."""
    t = str(text).lower()
    t = re.sub(r'http\S+|www\S+', ' URL ', t)
    t = re.sub(r'\S+@\S+', ' EMAIL ', t)
    t = re.sub(r'[^a-z\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def extract_email_features(subject, body, sender=None):
    """Returns a dict of v2 engineered email features for one email.
    MUST stay identical to email_feature_extraction_v2.py's logic."""
    text = (subject or '') + ' ' + (body or '')
    text_lower = text.lower()
    words = text_lower.split()
    word_count = len(words)

    request_count = sum(1 for p in _REQUEST_COMPILED if p.search(text_lower))
    warning_count = sum(1 for p in _WARNING_COMPILED if p.search(text_lower))
    net_score = request_count - warning_count

    urgency_hits = sum(1 for w in URGENCY_WORDS if w in text_lower)
    nigerian_hits = sum(1 for w in EMAIL_NIGERIAN_KW if w in text_lower)

    sender_domain = _get_sender_domain(sender)
    has_sender = 0 if sender_domain is None else 1
    is_free_email = 1 if (sender_domain is not None and sender_domain in FREE_EMAIL_DOMAINS) else 0
    mismatch = _brand_domain_mismatch(text_lower, sender_domain)

    return {
        'text_length_log': math.log1p(len(text_lower)),
        'word_count_log': math.log1p(word_count),
        'url_count': len(re.findall(r'http\S+|www\S+', text)),
        'credential_request_count': request_count,
        'security_warning_count': warning_count,
        'credential_request_net_score': net_score,
        'urgency_density': urgency_hits / (word_count + 1),
        'nigerian_keyword_density': nigerian_hits / (word_count + 1),
        'exclamation_ratio': text.count('!') / (word_count + 1),
        'caps_ratio': sum(1 for c in text if c.isupper()) / (len(text) + 1),
        'digit_count': sum(1 for c in text_lower if c.isdigit()),
        'avg_word_length': (np.mean([len(w) for w in words]) if words else 0),
        'has_suspension': 1 if any(w in text_lower for w in ['suspend', 'block', 'restrict', 'deactivat']) else 0,
        'has_sender_info': has_sender,
        'sender_is_free_email': is_free_email,
        'brand_domain_mismatch': mismatch,
    }

def email_features_to_vector(subject, body, sender=None):
    """Returns a single-row DataFrame in the EXACT column order the
    email model expects. Use this at inference time."""
    feats = extract_email_features(subject, body, sender)
    return pd.DataFrame([feats])[EMAIL_FEATURE_ORDER]


# ─────────────────────────────────────────────────────────────
# RISK SCORING — shared by both URL and Email results
# ─────────────────────────────────────────────────────────────
RISK_BANDS = [
    (0.20, "Safe", "green"),
    (0.40, "Low Risk", "yellow"),
    (0.60, "Medium Risk", "orange"),
    (0.80, "High Risk", "red"),
    (1.01, "Critical Risk", "darkred"),  # 1.01 so prob==1.0 still matches
]

def get_risk_level(probability):
    """Maps a 0-1 phishing probability to a human-readable risk band.
    Returns (label, color) e.g. ('High Risk', 'red')."""
    for threshold, label, color in RISK_BANDS:
        if probability < threshold:
            return label, color
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2]

def get_url_risk_factors(url):
    """Returns a list of human-readable reasons contributing to a URL's
    risk score, for explainability in the app UI (not just a bare number)."""
    feats = extract_url_features(url)
    reasons = []
    if feats['has_ip']:
        reasons.append("Uses a raw IP address instead of a domain name")
    if feats['is_typosquat_domain']:
        reasons.append("Domain closely resembles a known bank/fintech brand but doesn't match exactly")
    if feats['has_free_hosting']:
        reasons.append("Hosted on a free platform commonly abused for phishing")
    if feats['suspicious_path_depth']:
        reasons.append("Unusually deep URL path structure")
    if feats['has_at_symbol']:
        reasons.append("Contains '@' symbol, which can hide the real destination")
    if feats['excessive_subdomains']:
        reasons.append("Multiple nested subdomains")
    if feats['has_suspicious_tld']:
        reasons.append("Uses a top-level domain commonly associated with phishing")
    if feats['domain_entropy'] > 3.5:
        reasons.append("Domain name looks randomly generated")
    if feats['has_redirect']:
        reasons.append("Contains a redirect parameter")
    if not feats['has_https']:
        reasons.append("Not using HTTPS")
    return reasons

def get_email_risk_factors(subject, body, sender=None):
    """Returns a list of human-readable reasons contributing to an
    email's risk score, for explainability in the app UI."""
    feats = extract_email_features(subject, body, sender)
    reasons = []
    if feats['credential_request_net_score'] > 0:
        reasons.append("Directly asks for OTP, BVN, PIN, or password")
    if feats['brand_domain_mismatch']:
        reasons.append("Mentions a bank name but sender's email domain doesn't match that bank")
    if feats['sender_is_free_email']:
        reasons.append("Sent from a free email provider (Gmail/Yahoo/etc.) rather than a corporate domain")
    if feats['has_suspension']:
        reasons.append("Uses account suspension/blocking language")
    if feats['urgency_density'] > 0.05:
        reasons.append("High density of urgency-inducing language")
    if feats['url_count'] > 0:
        reasons.append(f"Contains {feats['url_count']} link(s)")
    if feats['credential_request_net_score'] < 0:
        reasons.append("Contains protective security-warning language (reduces risk)")
    return reasons
