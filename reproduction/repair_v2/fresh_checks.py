"""Additional authored diagnostics; never used for fitting or threshold selection."""
from pathlib import Path
import json,joblib,pandas as pd
from scipy.sparse import hstack,csr_matrix
import features_email as e,features_url as u
R=Path(__file__).resolve().parent
CASES=[
(0,'Credit alert','CR NGN 8200.00. Acct XX4412. Narration SALARY. Available balance NGN 92600.00. This is an automated transaction notification.'),
(0,'Purchase alert','DR NGN 2700.00. A card payment was completed at Example Supermarket. Your account history has been updated.'),
(0,'Fee notification','A monthly account maintenance fee and applicable VAT have been charged. Review the entry in your regular banking dashboard.'),
(0,'Savings update','Interest has been added to your savings account for the month. This is an informational message; no customer reply is requested.'),
(0,'Cash withdrawal','NGN 10000 was withdrawn at an ATM using your card. Keep this alert for your records. If you did not make the withdrawal, contact support through your app.'),
(0,'Account statement','Your statement covering January to March is ready. View it in your normal bank app. This email does not require any information from you.'),
(0,'Security reminder','A caller who asks for the digits on the back of your card is attempting fraud. Refuse the request and end the call.'),
(0,'Warning example','Never send me your OTP is the advice we give customers. Staff should never need your security token, CVV or PIN.'),
(0,'OTP','The one-time password for the payment you initiated is 870231. Do not disclose it. No employee needs to receive this code.'),
(0,'Card replacement notice','Your replacement card is ready for collection at your selected branch. Bring identification and collect it in person during opening hours.'),
(0,'Security advice','If a message says provide your account password to stop a debit, ignore it and contact the bank yourself. Do not follow that instruction.'),
(0,'Meeting confirmation','We will review the software demonstration on Wednesday. Please bring your laptop and your notes on the testing results.'),
(1,'Refund assistance','Kindly email us the three digits on the back of your card together with its number so our agent can process a refund.'),
(1,'Identity issue','To resolve your account verification, send me your card PIN and the code you just received on your phone.'),
(1,'Customer assistance','We need your internet banking password for this support case. Please reply with it so we can continue.'),
(1,'Restricted account','Your bank access has been paused. Enter your password and your token code at https://bank-access.invalid/help to get access back.'),
(1,'Card check','Send your full card number and CVV to this email address. This will allow us to validate your card for continued use.'),
(1,'Account validation','No need to visit a branch. We can complete validation here once you share your PIN and SMS code with our representative.'),
(1,'Service recovery','Please forward the OTP from your phone to our agent. They cannot complete account recovery without that code.'),
(1,'Help desk','For a quick fix, supply the password used to sign into your bank account. Our help desk will then remove the restriction.'),
(1,'Double protection','Do not give passwords to strangers. Our trusted bank team needs your OTP and password in your response to this email.'),
(1,'Profile notice','Go to https://customer-verification.invalid and submit your online banking password to verify ownership of this account.'),
(1,'Oga please','Oga abeg share your PIN with us make we sort the account wahala. Na bank officer dey assist you.'),
(1,'Card security','Reply with your CVV and card expiration date. Your bank account needs this final security check today.')]
def main():
 name=json.loads((R/'results/email_selection.json').read_text())['selected'];b=joblib.load(R/'models'/f'email_{name}.joblib');rows=[]
 for label,s,body in CASES:
  x=hstack([b['transforms']['tfidf'].transform([e.clean_text(s+' '+body)]),csr_matrix(e.email_frame([s],[body]).values)]).tocsr();p=float(b['model'].predict_proba(x)[0,1]);rows.append(dict(label=label,subject=s,body=body,score=p,predicted=int(p>=b['threshold'])))
 (R/'results/email_fresh_authored.json').write_text(json.dumps(rows,indent=2))
 for label in [0,1]:
  q=[x for x in rows if x['label']==label];print('Additional authored emails',label,sum(x['predicted']==label for x in q),'/',len(q))
 print('Errors',[(x['subject'],round(x['score']*100,2)) for x in rows if x['label']!=x['predicted']])
 b=joblib.load(R/'models/url_XGB.joblib');urls=['https://www.gtbank.com/selfservice','https://www.gtbank.com/personal-banking/ways-to-bank/gtworld-app','https://www.gtbank.com/annual-reports','https://www.zenithbank.com/personal-banking/ways-to-bank/ziva/','https://www.zenithbank.com/zerotohero/','https://www.accessbankplc.com/contact-us/branch-atm-locator','https://www.accessbankplc.com/Rates-Guide/','https://www.ubagroup.com/nigeria/help/contact-us/'];p=b['model'].predict_proba(u.url_frame(urls))[:,1];out=[dict(url=s,expected=0,score=float(v),predicted=int(v>=b['threshold'])) for s,v in zip(urls,p)];(R/'results/url_additional_official.json').write_text(json.dumps(out,indent=2));print('Additional official pages cleared',sum(x['predicted']==0 for x in out),'/',len(out))
if __name__=='__main__':main()
