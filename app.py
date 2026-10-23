# Adapted from the uploaded app.py: same URL and email checking workflow.
import streamlit as st
from inference import load_bundle,score,score_message,review_reason
from feature_extraction import get_url_risk_factors,get_email_risk_factors
st.set_page_config(page_title='Phishing Detection System',page_icon='🛡️')
st.title('🛡️ Phishing Detection System')
st.caption('Research prototype for Nigerian financial institutions')
@st.cache_resource
def load_models():return load_bundle('url'),load_bundle('email')
try:url_model,email_model=load_models()
except Exception as ex:
 st.error('Models could not be loaded. Use the matching project files.');st.stop()
def show(result,reason=None):
 st.metric('Model phishing score',f"{result['score']*100:.1f}/100")
 st.progress(result['score'])
 if reason:st.warning('Review needed: '+reason)
 elif result['predicted_label']:st.warning('Potential phishing. Verify through a known official channel.')
 else:st.info('No alert raised. This does not establish that the message or URL is safe.')
 with st.expander('Model decision and limitations'):
  st.write('Binary model alert: '+('phishing' if result['predicted_label'] else 'no phishing alert'))
  st.write(f"Saved decision threshold: {result['threshold']*100:.2f}/100")
  st.write('The score is uncalibrated. Links are not visited, senders are not authenticated, and bank ownership is not verified. Observed indicators are not a complete explanation of the prediction.')
tab_url,tab_email=st.tabs(['🔗 Check a URL','📧 Check an Email'])
with tab_url:
 url_input=st.text_input('Enter a URL to check',placeholder='https://example.com/login')
 if st.button('Analyze URL',key='url_btn'):
  try:
   result=score(url_model,url=url_input);show(result)
   for r in get_url_risk_factors(url_input):st.write('• '+r)
  except ValueError as ex:st.error(str(ex))
with tab_email:
 mode=st.radio('Email input',['Paste text','Upload .eml'],horizontal=True)
 subject_input='';body_input='';raw=None
 if mode=='Paste text':
  subject_input=st.text_input('Email subject (optional)',key='email_subject')
  body_input=st.text_area('Email body',height=220,key='email_body')
 else:
  upload=st.file_uploader('Email file',type=['eml'])
  if upload:raw=upload.getvalue()
 if st.button('Analyze Email',key='email_btn'):
  if mode=='Paste text' and not body_input.strip():st.warning('Please enter the email body.')
  elif mode=='Upload .eml' and raw is None:st.warning('Please upload an .eml file.')
  else:
   try:
    if raw is not None:
     result,msg=score_message(email_model,raw);subject_input=msg['subject'];body_input=msg['model_body']
    else:result=score(email_model,subject=subject_input,body=body_input)
    show(result,review_reason(result,subject_input,body_input))
    risk,protective=get_email_risk_factors(subject_input,body_input)
    if risk or protective:
     st.write('Observed language indicators')
     for r in risk+protective:st.write('• '+r)
   except ValueError as ex:st.error(str(ex))
st.divider()
st.caption('Remove confidential information before submitting text. Images and PDF attachments are not assessed. Verify suspicious requests directly with your bank through a known official channel.')
