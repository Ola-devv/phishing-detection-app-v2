"""Retrain the existing LR/RF/XGB project on fixed historical partitions.
Authored robustness cohorts are explicitly artificial; no automatic deployment.
"""
from pathlib import Path
import json,hashlib,argparse,time
import joblib,numpy as np,pandas as pd
from scipy.sparse import hstack,csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler,StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,roc_curve
from xgboost import XGBClassifier
from urllib.parse import urlsplit,urlunsplit
import features_email as e,features_url as u
from augmentation import examples
from mailbox_reader import readable_message
R=Path(__file__).resolve().parent;P=R.parents[1];O=R/'results';M=R/'models';O.mkdir(exist_ok=True);M.mkdir(exist_ok=True)
def h(s):return hashlib.sha256(s.encode()).hexdigest()
def metric(y,p,t):
 tn,fp,fn,tp=map(int,confusion_matrix(y,np.asarray(p)>=t,labels=[0,1]).ravel());return dict(n=len(y),TN=tn,FP=fp,FN=fn,TP=tp,recall=tp/(tp+fn) if tp+fn else None,FPR=fp/(tn+fp) if tn+fp else None,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,accuracy=(tn+tp)/len(y))
def email_data():
 cache=P/'data/private_email_frame.pkl'
 if cache.exists():return pd.read_pickle(cache)
 raw=pd.read_csv(P/'data/master_email_dataset_v2.csv',nrows=82471,usecols=['subject','body','label'],keep_default_na=False);roles=np.load(P/'data/reproduction_refs/email_roles.npz',allow_pickle=False);d=raw.iloc[roles['raw_index']].copy().reset_index(drop=True)
 d['clean']=[e.clean_text(s+' '+b) for s,b in zip(d.subject,d.body)];d['record_id']=d.clean.map(h)
 assert h('\n'.join(d.record_id))==str(roles['records_digest']);assert np.array_equal(d.label.to_numpy(),roles['label'])
 d['group_id']=roles['group'];d['partition']=pd.Series(roles['partition']).map({0:'train',1:'validation',2:'test'});assert d.groupby('group_id').partition.nunique().max()==1
 d.to_pickle(cache);return d
def url_data():
 cache=P/'data/private_url_cache.joblib'
 if cache.exists():return joblib.load(cache)
 raw=pd.read_csv(P/'data/master_url_dataset.csv');items=[]
 for i,s in enumerate(raw.url):
  try:v=u.url_identity(s);items.append((i,v,h(v)))
  except (ValueError,UnicodeError):pass
 v=pd.DataFrame(items,columns=['i','identity','record_id']);d=raw.iloc[v.i].copy().reset_index(drop=True);d['identity']=v.identity;d['record_id']=v.record_id
 bad=set(d.groupby('identity').label.nunique().loc[lambda x:x>1].index);d=d[~d.identity.isin(bad)].drop_duplicates('identity').reset_index(drop=True)
 roles=np.load(P/'data/reproduction_refs/url_roles.npz',allow_pickle=False);assert h('\n'.join(d.record_id))==str(roles['records_digest']);assert np.array_equal(d.label.to_numpy(),roles['label'])
 d['partition']=pd.Series(roles['partition']).map({0:'train',1:'validation',2:'test'});g=pd.DataFrame({'g':d.url.map(u.url_group),'p':d.partition});assert g.groupby('g').p.nunique().max()==1
 parts=[]
 for i in range(0,len(d),10000):parts.append(u.url_frame(d.url.iloc[i:i+10000]));print('URL extraction',i+len(parts[-1]),flush=True)
 x=pd.concat(parts,ignore_index=True);joblib.dump((d,x),cache,compress=3);return d,x

def main(kind):
 probes=json.loads((R/'probes.json').read_text())['rows'];tf=None
 if kind=='email':
  d=email_data();syn,sv=examples('train'),examples('validation')
  for q in [syn,sv]:q['clean']=[e.clean_text(s+' '+b) for s,b in zip(q.subject,q.body)]
  assert not set(syn.clean)&set(sv.clean);assert not set(syn.clean)&set(d[d.partition.ne('train')].clean)
  inspected={e.clean_text(x.get('subject','')+' '+x.get('body','')) for x in probes if x['kind']=='email'};assert not set(syn.clean)&inspected
  tr=pd.concat([d[d.partition.eq('train')],syn],ignore_index=True);va=d[d.partition.eq('validation')];te=d[d.partition.eq('test')]
  tf=TfidfVectorizer(max_features=8000,ngram_range=(1,2),min_df=2,sublinear_tf=True);tf.fit(tr.clean)
  def mat(q):return hstack([tf.transform(q.clean),csr_matrix(e.email_frame(q.subject,q.body).values)]).tocsr()
  xt,xv,xs,xe=map(mat,[tr,va,sv,te]);yt,yv,ys,ye=tr.label,va.label,sv.label,te.label;ids=te.record_id
  nh=len(tr)-len(syn);balance=float((tr.label.iloc[:nh]==0).sum()/(tr.label.iloc[:nh]==1).sum());weights=np.ones(len(tr));weights[:nh]=np.where(tr.label.iloc[:nh].eq(1),balance,1);weights[nh:]=20
  models={'LR':make_pipeline(MaxAbsScaler(),LogisticRegression(C=1,max_iter=2500,solver='liblinear',random_state=42)),'RF':RandomForestClassifier(n_estimators=150,max_depth=24,min_samples_leaf=2,n_jobs=4,random_state=42),'XGB':XGBClassifier(n_estimators=250,max_depth=4,learning_rate=.08,reg_lambda=4,tree_method='hist',eval_metric='logloss',n_jobs=4,random_state=42)}
  provenance={'historical_train':nh,'authored_train':len(syn),'authored_validation':len(sv),'raw_bank_mailbox_retested':False,'external_phishing_mailbox_retested':False}
 else:
  d,x=url_data();tr=d.partition.eq('train');va=d.partition.eq('validation');te=d.partition.eq('test');base=d[tr&d.label.eq(0)]
  paths=[urlsplit(s if '://' in s else 'https://'+s).path for s in base.url];paths=sorted(set(p for p in paths if 15<len(p)<180 and p.count('/')>=2));assert len(paths)>50;rng=np.random.default_rng(20261023)
  def augment(frame,n):
   vals=[]
   for s in frame.url.sample(n=n,replace=len(frame)<n,random_state=42):
    raw,parsed,host=u.parse_url(s);path=paths[int(rng.integers(len(paths)))];vals.append(urlunsplit((parsed.scheme,parsed.netloc,path,'','')))
   return vals
  au,av=augment(base,10000),augment(d[va&d.label.eq(0)],1500);xa,xs=u.url_frame(au),u.url_frame(av);xt=pd.concat([x[tr],xa],ignore_index=True);xv,xe=x[va],x[te];yt=np.r_[d.loc[tr,'label'],np.zeros(len(xa),int)];yv,ys,ye=d.loc[va,'label'],np.zeros(len(xs),int),d.loc[te,'label'];ids=d.loc[te,'record_id'];weights=np.r_[np.ones(tr.sum()),np.full(len(xa),4.)]
  models={'LR':LogisticRegression(max_iter=2000,random_state=42),'RF':RandomForestClassifier(n_estimators=150,max_depth=24,min_samples_leaf=2,n_jobs=4,random_state=42),'XGB':XGBClassifier(n_estimators=250,max_depth=6,learning_rate=.08,reg_lambda=4,tree_method='hist',eval_metric='logloss',n_jobs=4,random_state=42)}
  provenance={'historical_train':int(tr.sum()),'derived_train':len(au),'derived_validation':len(av),'bank_allowlist':False}
 vals=[];tests=[]
 for name,m in models.items():
  print(kind,'training',name,time.strftime('%H:%M:%S'),flush=True);sc=StandardScaler() if kind=='url' and name=='LR' else None;a=sc.fit_transform(xt) if sc else xt
  if kind=='email' and name=='LR':m.fit(a,yt,logisticregression__sample_weight=weights)
  else:m.fit(a,yt,sample_weight=weights)
  def prob(q):return m.predict_proba(sc.transform(q) if sc else q)[:,1]
  pv,ps,pe=map(prob,[xv,xs,xe]);fps,tps,ts=roc_curve(yv,pv,drop_intermediate=False);eligible=[]
  for i,t in enumerate(ts):
   if not np.isfinite(t) or fps[i]>.02:continue
   sm=metric(ys,ps,float(t))
   if kind=='email':ok=sm['FPR']<=.10 and sm['recall']>=.90 and tps[i]>=.85
   else:ok=sm['FPR']<=.05
   if ok:eligible.append(i)
  if eligible:
   if kind=='email':best=max(eligible,key=lambda i:(metric(yv,pv,float(ts[i]))['F1'],tps[i],-fps[i],float(ts[i])))
   else:best=max(eligible,key=lambda i:(tps[i],-fps[i],float(ts[i])))
   threshold=float(ts[best])
  else:threshold=.5 if kind=='email' else 1.000001
  v=dict(model=name,eligible=bool(eligible),threshold=threshold,historical=metric(yv,pv,threshold),constructed=metric(ys,ps,threshold));vals.append(v);tests.append(dict(model=name,historical=metric(ye,pe,threshold)));print(json.dumps(v),flush=True)
  b=dict(kind=kind,model_name=name,model=m,scaler=sc,threshold=threshold,feature_sha256=hashlib.sha256((R/f'features_{kind}.py').read_bytes()).hexdigest(),transforms={'feature_order':e.EMAIL_ORDER if kind=='email' else u.URL_ORDER},deployment_approved=False)
  if kind=='email':b['reader_sha256']=hashlib.sha256((R/'mailbox_reader.py').read_bytes()).hexdigest();b['transforms']['tfidf']=tf
  joblib.dump(b,M/f'{kind}_{name}.joblib',compress=3);pd.DataFrame({'record_id':ids,'label':ye,'score':pe}).to_csv(O/f'{kind}_{name}_historical.csv',index=False)
  diag=[]
  for q in probes:
   if (kind=='url')!=(q['kind']=='url'):continue
   if not isinstance(q['expected'],int):continue
   if kind=='url':z=u.url_frame([q['url']])
   else:
    if q['kind']=='eml':msg=readable_message(q['raw'].encode());s,body=msg['subject'],msg['model_body']
    else:s,body=q['subject'],q['body']
    inp=pd.DataFrame([{'subject':s,'body':body,'clean':e.clean_text(s+' '+body)}]);z=mat(inp)
   p=float(prob(z)[0]);diag.append(dict(q,score=p,predicted=int(p>=threshold)))
  (O/f'{kind}_{name}_regression.json').write_text(json.dumps(diag,indent=2));print('Errors',[(q['name'],round(q['score']*100,2)) for q in diag if q['expected']!=q['predicted']],flush=True)
  (O/f'{kind}_validation.json').write_text(json.dumps(vals,indent=2));(O/f'{kind}_historical.json').write_text(json.dumps(tests,indent=2))
 valid=[v for v in vals if v['eligible']]
 if kind=='email':key=lambda x:(x['historical']['F1'],x['historical']['recall'],-x['historical']['FPR'])
 else:key=lambda x:(x['historical']['recall'],x['historical']['F1'],-x['historical']['FPR'])
 selected=max(valid,key=key)['model'] if valid else None
 (O/f'{kind}_selection.json').write_text(json.dumps(dict(selected=selected,provenance=provenance,rule='Validation constraints documented in REPAIR_RESULTS.md; historical test and inspected regressions never choose thresholds.',limitations=['Previously inspected historical development partitions, not a fresh final benchmark.','Authored examples/derived paths are not verified Nigerian banking benchmark data.']),indent=2));print('Selected',kind,selected,flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('kind',choices=['email','url']);main(a.parse_args().kind)
