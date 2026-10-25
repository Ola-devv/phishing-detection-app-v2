# Adapted from the uploaded app.py: same URL and email checking workflow.
import streamlit as st
from inference import load_bundle,score,score_message,review_reason
from result_explanations import explain
st.set_page_config(page_title='Phishing Detection System',page_icon='🛡️')
st.title('🛡️ Phishing Detection System')
st.caption('Research prototype for Nigerian financial institutions')
@st.cache_resource
def load_models():return load_bundle('url'),load_bundle('email')
try:url_model,email_model=load_models()
except Exception as ex:
 st.error('Models could not be loaded. Use the matching project files.');st.stop()
def show(result,reason=None):
 if result['predicted_label']:st.error('Potential phishing')
 else:st.success('Likely legitimate')
 if reason:st.warning('Review needed: '+reason)
 st.metric('Phishing score',f"{result['score']*100:.1f}/100")
 st.caption('Lower scores indicate fewer signs of phishing; higher scores indicate more.')
 with st.expander('About this result'):
  st.write('This is a model prediction, not a guarantee of safety. The score is not a verified percentage chance of phishing. The app analyses URL patterns and readable email text; it does not authenticate senders or inspect website content.')
def show_explanation(bundle,result,**inputs):
 info=explain(bundle,result,**inputs)
 st.subheader('Why this result?')
 st.write(info['summary'])
 st.write('Detected in this input')
 for item in info['observations'][:3]:st.text('• '+item)
 if info['available']:
  for heading,key in [('Features pushing the score higher','higher'),('Features pushing the score lower','lower')]:
   if info[key]:
    st.write(heading)
    for item in info[key][:2]:st.text('• '+item)
  st.caption('These contributions explain this model score relative to its baseline. They do not verify whether the input is genuine.')
  with st.expander('Model contribution details'):
   st.dataframe(info['details'],hide_index=True)
   st.caption('Positive contributions raise the score; negative contributions lower it. Values are in log-odds, not percentage points. Only the three largest contributions in each direction are shown.')
 else:st.info(info['unavailable_reason'])
 with st.expander('Details detected in this input'):
  for item in info['observations']:st.text('• '+item)
tab_url,tab_email=st.tabs(['🔗 Check a URL','📧 Check an Email'])
with tab_url:
 url_input=st.text_input('Enter a URL to check',placeholder='https://example.com/login')
 if st.button('Analyze URL',key='url_btn'):
  try:
   result=score(url_model,url=url_input);show(result)
   show_explanation(url_model,result,url=url_input)
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
    show_explanation(email_model,result,subject=subject_input,body=body_input)
   except ValueError as ex:st.error(str(ex))
st.divider()
st.caption('Research prototype. Predictions may be incorrect. Verify suspicious requests directly with your bank.')


