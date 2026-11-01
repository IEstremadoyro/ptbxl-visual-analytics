# Mapa agregado: latido medio de los 256 infartos de prueba frente a plantillas normal e infarto tipico.
# Uso (desde la carpeta del proyecto): python3 scripts/analisis_patron.py  -> escribe patron_resultados.json
import numpy as np, pandas as pd, json, sys
from scipy.signal import find_peaks
from scipy.stats import mannwhitneyu
D='data/'
df=pd.read_csv(D+'ptbxl_predictivo_umap.csv')
SEN=np.load(D+'senales_norm_f16.npy',mmap_mode='r')
ns=np.load(D+'norm_stats.npz'); MU=ns['mu'].reshape(-1); SD=ns['sd'].reshape(-1)
POS=pd.Series(np.arange(len(df)),index=df.ecg_id)
PRE,POST=25,55
def beat(i):
    x=np.asarray(SEN[i],dtype=np.float32)*SD[:,None]+MU[:,None]
    e=np.abs(np.diff(x,axis=1)).sum(0); e=np.convolve(e,np.ones(5)/5,'same')
    pk,_=find_peaks(e,distance=40,height=np.percentile(e,90)*0.5)
    pk=pk[(pk>PRE)&(pk<len(e)-POST)]
    if len(pk)<3: return None
    return np.median(np.stack([x[:,p-PRE:p+POST] for p in pk]),0)
cache={}
def B(i):
    if i not in cache: cache[i]=beat(i)
    return cache[i]
rng=np.random.default_rng(0)
tr=df[df.split=='train']
def tmpl(lbl,n=1500):
    idx=tr.index[tr.label==lbl].to_numpy(); idx=rng.choice(idx,min(n,len(idx)),replace=False)
    bs=[b for b in (B(i) for i in idx) if b is not None]; return np.median(np.stack(bs),0),len(bs)
TN,nN=tmpl('NORM'); TM,nM=tmpl('MI')
print('plantillas',nN,nM)
te=df[(df.split=='test')&(df.label=='MI')]
rows=[]
for i,r in te.iterrows():
    b=B(i)
    if b is None: continue
    vs=[B(int(POS[int(v)])) for v in str(r.vecinos).split(';') if v.strip().isdigit()]
    vs=[v for v in vs if v is not None]
    if not vs: continue
    T=np.median(np.stack(vs),0)
    rows.append(dict(i=i,ecg=int(r.ecg_id),grupo='detectado' if r.prediccion=='MI' else 'perdido',sub=r.mi_subclase,b=b,T=T))
print(len(rows),pd.Series([r['grupo'] for r in rows]).value_counts().to_dict())
def nd(b,T):
    esc=np.maximum(0.2,T.max(1)-T.min(1))[:,None]; return np.abs(b-T)/esc
def cor(a,b): return float(np.corrcoef(a.ravel()-a.mean(),b.ravel()-b.mean())[0,1])
for r in rows:
    r['d_nb']=nd(r['b'],r['T']); r['d_nm']=nd(r['b'],TN); r['d_mi']=nd(r['b'],TM)
    r['c_n']=cor(r['b'],TN); r['c_m']=cor(r['b'],TM)
G={g:[r for r in rows if r['grupo']==g] for g in ('perdido','detectado')}
out={'pre':PRE,'post':POST,'n':{g:len(v) for g,v in G.items()},'n_plantillas':{'NORM':nN,'MI':nM}}
def bh(p):
    p=np.array(p); o=np.argsort(p); q=np.empty_like(p); m=len(p); prev=1
    for k,j in enumerate(o[::-1]): 
        rank=m-k; prev=min(prev,p[j]*m/rank); q[j]=prev
    return q
for key,nm in (('d_nb','vecinos'),('d_nm','normal'),('d_mi','infarto_tipico')):
    out['mapa_'+nm]={g:np.round(np.mean([r[key] for r in v],0),4).tolist() for g,v in G.items()}
    sc={g:np.array([r[key].mean(1) for r in v]) for g,v in G.items()}   # casos x 12
    p=[mannwhitneyu(sc['perdido'][:,l],sc['detectado'][:,l]).pvalue for l in range(12)]
    out['lead_'+nm]={'media_perdido':np.round(sc['perdido'].mean(0),4).tolist(),'media_detectado':np.round(sc['detectado'].mean(0),4).tolist(),
        'mediana_perdido':np.round(np.median(sc['perdido'],0),4).tolist(),'mediana_detectado':np.round(np.median(sc['detectado'],0),4).tolist(),
        'p':np.round(p,5).tolist(),'p_bh':np.round(bh(p),5).tolist()}
# derivacion con mayor diferencia por caso (frente a plantilla de vecinos)
for g,v in G.items():
    top=[int(np.argmax(r['d_nb'].mean(1))) for r in v]
    out.setdefault('top_deriv',{})[g]=np.bincount(top,minlength=12).tolist()
# cercania a plantilla normal vs infarto tipico
for g,v in G.items():
    cn=np.array([r['c_n'] for r in v]); cm=np.array([r['c_m'] for r in v])
    out.setdefault('cercania',{})[g]={'r_normal_mediana':float(np.median(cn)),'r_infarto_mediana':float(np.median(cm)),'frac_mas_cerca_de_normal':float(np.mean(cn>cm))}
p=mannwhitneyu([r['c_n']-r['c_m'] for r in G['perdido']],[r['c_n']-r['c_m'] for r in G['detectado']]).pvalue
out['cercania']['p_diferencia']=float(p)
# subtipo
out['subtipo']={g:pd.Series([r['sub'] for r in v]).value_counts().to_dict() for g,v in G.items()}
json.dump(out,open('patron_resultados.json','w'),indent=1)
print(json.dumps({k:out[k] for k in ('n','cercania','top_deriv','subtipo')},indent=1))
for nm in ('vecinos','normal','infarto_tipico'):
    L=out['lead_'+nm]; print(nm); 
    for l,d in enumerate(['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']): print(f" {d:4s} perd {L['media_perdido'][l]:.3f} det {L['media_detectado'][l]:.3f} p_bh {L['p_bh'][l]:.4f}")

# ---- extras
from scipy.stats import fisher_exact
DER=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
ex={}
mp=np.array(out['mapa_normal']['perdido']); md=np.array(out['mapa_normal']['detectado']); dif=md-mp   # + = detectados se desvian mas
ex['dif_normal_top']=[(DER[l],int((t-PRE)*10),round(float(dif[l,t]),3)) for l,t in zip(*np.unravel_index(np.argsort(-dif.ravel())[:8],dif.shape))]
# por tramo (ms): QRS -40..60, ST 60..160, T >160
ms=(np.arange(PRE+POST)-PRE)*10
for nom,(a,b) in {'previo':(-250,-40),'QRS':(-40,60),'ST':(60,160),'T':(160,550)}.items():
    s=(ms>=a)&(ms<b); ex.setdefault('tramo_normal',{})[nom]={'perdido':float(mp[:,s].mean()),'detectado':float(md[:,s].mean())}
sub={}
for s in ('IMI','AMI'):
    n_p=sum(1 for r in G['perdido'] if r['sub']==s); n_d=sum(1 for r in G['detectado'] if r['sub']==s)
    sub[s]={'perdido':n_p,'detectado':n_d,'recall':n_d/(n_p+n_d)}
tab=[[sub['IMI']['detectado'],sub['IMI']['perdido']],[sub['AMI']['detectado'],sub['AMI']['perdido']]]
sub['fisher_p']=float(fisher_exact(tab)[1]); ex['subtipo']=sub
# por subtipo: desviacion de plantilla normal en derivaciones clave
for s,leads in (('IMI',[1,2,5]),('AMI',[6,7,8,9])):
    for g in ('perdido','detectado'):
        v=[r['d_nm'][leads].mean() for r in G[g] if r['sub']==s]; ex.setdefault('desv_normal_'+s,{})[g]=float(np.mean(v))
    a=[r['d_nm'][leads].mean() for r in G['perdido'] if r['sub']==s]; b=[r['d_nm'][leads].mean() for r in G['detectado'] if r['sub']==s]
    ex['desv_normal_'+s]['p']=float(mannwhitneyu(a,b).pvalue); ex['desv_normal_'+s]['derivaciones']=[DER[l] for l in leads]
print(json.dumps(ex,indent=1))
out['extras']=ex
json.dump(out,open('patron_resultados.json','w'),indent=1)
