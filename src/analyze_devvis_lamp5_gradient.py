"""Lamp5 gradient v2: per-age continuum y-axes via PC AND diffusion pseudotime,
plus a global and a shared-gene 'consensus' axis. All axes are projectable gene
signatures (defined on a reference cell set, applied to every cell) oriented A→B
by the pole contrast and residualised against maturity. Saves lamp5_grad2.npz."""
import anndata as ad, numpy as np, scipy.sparse as sp, re, pandas as pd
from numpy.linalg import svd
import scanpy as sc
from sklearn.decomposition import NMF
np.random.seed(0); sc.settings.verbosity=0
ROOT='/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish'
OUT='/private/tmp/claude-501/-Users-inlebush-cs-lab-green-data-vis/95ef1b74-d552-4696-b236-9181bf853664/scratchpad/lamp5_grad2.npz'

a=ad.read_h5ad(f'{ROOT}/data/devvis_inh_all_ages.h5ad')
lam=a[a.obs['cell_subclass'].astype(str)=='Lamp5'].copy()
X=lam.X; X=(X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32)
vn=np.array(lam.var_names); gi={g:i for i,g in enumerate(vn)}; N,G=X.shape
mu,sd=X.mean(0),X.std(0)+1e-6
def agenum(s):
    m=re.match(r"([EP])([0-9.]+)",str(s)); return (float(m.group(2))-19 if m.group(1)=="E" else float(m.group(2))) if m else np.nan
age=np.array([agenum(x) for x in lam.obs['synchronized_age']],np.float32)
clu=lam.obs['cell_cluster_named'].astype(str).values
imm=['Dcx','Sox11','Sox4','Tubb3','Sox2','Neurod2','Cd24a','Nnat','Igfbpl1','Marcksl1','Ccnd2','Btg1','Tubb2b','Stmn2','Stmn1']
imm=[g for g in imm if g in gi]; immset=[gi[g] for g in imm]
maturity=-((X-mu)/sd)[:,immset].mean(1); maturity=(maturity-maturity.mean())/maturity.std()
poleA=['Cnr1','Cxcl14','Egln3','Nr2f2']; poleB=['Sv2c','Nxph1','Npy']
Zall=(X-mu)/sd

# HVG (exclude maturation) for axis fitting
disp=np.full(G,-np.inf); ok=((X>0).mean(0)>0.05)&(mu>0.1); disp[ok]=(X.var(0)/mu)[ok]
hvg=np.array([h for h in np.argsort(disp)[::-1][:2000] if h not in set(immset)])
def polecon(mask): return np.mean([Zall[mask,gi[g]] for g in poleB],0)-np.mean([Zall[mask,gi[g]] for g in poleA],0)

def signature_from_score(mask, score):
    """regress a per-ref-cell continuum score on genes -> projectable weights."""
    Zr=Zall[mask]; sc_=score-score.mean()
    w=(Zr*sc_[:,None]).mean(0)/(sc_.std()+1e-9)
    w[immset]=0; sel=np.argsort(np.abs(w))[::-1][:300]; ww=np.zeros(G); ww[sel]=w[sel]
    return ww
def project(ww):
    y=Zall@ww
    b=float((y*maturity).sum()/(maturity*maturity).sum()); y=y-b*maturity   # residualise vs maturity
    return ((y-y.mean())/y.std()).astype(np.float32)

# global PCA(30) space for the nonlinear kNN projection across ages
Zg=(X[:,hvg]-X[:,hvg].mean(0))/(X[:,hvg].std(0)+1e-6)
Ug,Sg,_=svd(Zg-Zg.mean(0),full_matrices=False); PCAg=(Ug[:,:30]*Sg[:30]).astype(np.float32)
from sklearn.neighbors import NearestNeighbors
def project_knn(refmask, ref_score, k=15):
    """nonlinear projection: each cell = mean diffusion coord of its k nearest
    REFERENCE cells (in global PCA space). No per-gene linear weights."""
    nn=NearestNeighbors(n_neighbors=min(k,int(refmask.sum()))).fit(PCAg[refmask])
    _,idx=nn.kneighbors(PCAg); y=ref_score[idx].mean(1)
    b=float((y*maturity).sum()/(maturity*maturity).sum()); y=y-b*maturity
    return ((y-y.mean())/y.std()).astype(np.float32)

def pc_axis(mask):   # LINEAR gene-signature axis (one weight per gene)
    Xr=X[np.ix_(mask,hvg)]; Z=(Xr-Xr.mean(0))/(Xr.std(0)+1e-6)
    U,S,Vt=svd(Z-Z.mean(0),full_matrices=False); con=polecon(mask)
    cors=[np.corrcoef(U[:,k],con)[0,1] for k in range(15)]; kb=int(np.argmax(np.abs(cors)))
    return signature_from_score(mask,np.sign(cors[kb])*U[:,kb])
def dpt_score(mask):  # NONLINEAR diffusion coordinate on the reference cells
    Xr=X[np.ix_(mask,hvg)]; Z=(Xr-Xr.mean(0))/(Xr.std(0)+1e-6)
    U,S,Vt=svd(Z-Z.mean(0),full_matrices=False); pca=U[:,:15]*S[:15]
    am=ad.AnnData(Xr); am.obsm['X_pca']=pca
    sc.pp.neighbors(am,use_rep='X_pca',n_neighbors=15,random_state=0); sc.tl.diffmap(am,n_comps=8)
    DC=am.obsm['X_diffmap']; con=polecon(mask)
    cors=[abs(np.corrcoef(DC[:,k],con)[0,1]) for k in range(1,6)]; dk=int(np.argmax(cors))+1
    return np.sign(np.corrcoef(DC[:,dk],con)[0,1])*DC[:,dk]

# reference sets: global + age windows
agebins=[('P3–7',(age>=3)&(age<=7)),('P8–14',(age>=8)&(age<=14)),
         ('P15–28',(age>=15)&(age<=28)),('P56',age>=56)]
refs=[('Global',np.ones(N,bool))]+agebins
axes=[]; names=[]; sigs={}
for rn,rm in refs:
    ww=pc_axis(rm); axes.append(project(ww)); names.append(f'{rn} · PC (linear)'); sigs[(rn,'PC')]=ww
    s=dpt_score(rm); axes.append(project_knn(rm,s)); names.append(f'{rn} · diffusion (nonlinear)')
    print(f'  built axes: {rn}  (ref n={rm.sum()})')
# consensus (shared genes across the 4 age PC signatures)
agepc=[sigs[(rn,'PC')] for rn,_ in agebins]
shared_mask=np.all([np.abs(w)>0 for w in agepc],0)   # nonzero (top-300) in every age
cons_w=np.mean(agepc,0)*shared_mask
axes.append(project(cons_w)); names.append('Consensus (shared genes) · PC')
shared_genes=list(vn[np.where(shared_mask)[0]])
print(f'\nconsensus shared genes across all age PC axes: {len(shared_genes)}')
print('  '+', '.join(sorted(shared_genes, key=lambda g:-abs(cons_w[gi[g]]))[:30]))
Y=np.vstack(axes).T.astype(np.float32)   # N x nAxes
# agreement matrix (Spearman) between axes
from scipy.stats import rankdata
R=np.corrcoef(np.apply_along_axis(rankdata,0,Y).T)
print('\naxis agreement (Spearman) vs Global·PC:')
for i,n in enumerate(names): print(f'  {n:38s} {R[0,i]:+.2f}')
# does the nonlinear axis place mid-peak genes better? peak-bin on mature cells (/9)
matq=maturity>np.quantile(maturity,0.75)
def peakbin(y,g,B=10):
    e=X[matq,gi[g]]; yy=y[matq]; q=np.quantile(yy,np.linspace(0,1,B+1)); bb=np.clip(np.digitize(yy,q[1:-1]),0,B-1)
    m=np.array([e[bb==i].mean() if (bb==i).any() else np.nan for i in range(B)]); return int(np.nanargmax(m))
print('\ngene peak-bin (mature, 0=A-pole .. 9=B-pole):  Global·PC(linear)  vs  Global·diffusion(nonlinear)')
for g in ['Ndnf','Dock5','Cdh7','Igfbp2','Reln','Cxcl14','Nxph1','Krt73']:
    if g in gi: print(f'  {g:8s} linear={peakbin(axes[0],g)}  nonlinear={peakbin(axes[1],g)}')

# ---- NMF (k=12) with clean top genes per component ----
Xh=X[:,hvg]; nmf=NMF(n_components=12,init='nndsvda',random_state=0,max_iter=500)
W=nmf.fit_transform(Xh); H=nmf.components_; Wn=(W/(W.max(0)+1e-9)).astype(np.float32)
hn=vn[hvg]; nmf_top=[list(hn[np.argsort(H[k])[::-1][:15]]) for k in range(12)]
matf=int(np.argmax([sum(g in nmf_top[k] for g in ['Dcx','Sox11','Sox4','Tubb3','Stmn2']) for k in range(12)]))

# ---- gene classification along Global·PC (for the graded-gene list) ----
yG=axes[0]; mature=maturity>np.quantile(maturity,0.75); ym=yG[mature]; B=12
def prof_eta(e,yy):
    q=np.quantile(yy,np.linspace(0,1,B+1)); bb=np.clip(np.digitize(yy,q[1:-1]),0,B-1)
    g=e.mean(); ss=((e-g)**2).sum()+1e-9; sb=0.0; mns=np.zeros(B)
    for i in range(B):
        m=bb==i
        if m.any(): mns[i]=e[m].mean(); sb+=m.sum()*(mns[i]-g)**2
    return sb/ss,mns
from scipy.stats import spearmanr
cand=np.where(((X>0).mean(0)>0.05)&(mu>0.1))[0]; rows=[]
for j in cand:
    e=X[mature,j]; eta,mns=prof_eta(e,ym)
    if eta<0.06: continue
    s=spearmanr(e,ym).correlation or 0; amax=int(np.argmax(mns)); amin=int(np.argmin(mns))
    ppk=(mns[amax]-max(mns[0],mns[-1]))/(np.ptp(mns)+1e-9); pdp=(min(mns[0],mns[-1])-mns[amin])/(np.ptp(mns)+1e-9)
    if abs(s)>=0.35: c='B-pole' if s>0 else 'A-pole'
    elif 2<=amax<=B-3 and ppk>0.35: c='mid-peak'
    elif 2<=amin<=B-3 and pdp>0.35: c='mid-dip'
    else: continue
    rows.append((j,c,round(float(eta),3)))
gdf=pd.DataFrame(rows,columns=['j','cls','eta']).sort_values('eta',ascending=False)

# ---- embed gene set ----
embed=set(int(j) for j in gdf.j)
for g in imm+poleA+poleB+['Ndnf','Krt73','Piezo2','Ralyl','Reln','Id2','Kit','Egfr','Pax6','Egln3','Cnr1','Lamp5']:
    if g in gi: embed.add(gi[g])
for t in nmf_top:
    for g in t:
        if g in gi: embed.add(gi[g])
for j in hvg:
    if len(embed)>=1300: break
    embed.add(int(j))
embed=sorted(embed); jpos={j:i for i,j in enumerate(embed)}
ES=16; u8=np.clip(np.round(X[:,embed]*ES),0,255).astype(np.uint8)
gcls=np.array(['none']*len(embed),dtype=object); geta=np.zeros(len(embed),np.float32)
for _,r in gdf.iterrows():
    i=jpos[int(r.j)]; gcls[i]=r.cls; geta[i]=r.eta
clu_cats=pd.Series(clu).value_counts().index.tolist(); clu_code=np.array([clu_cats.index(c) for c in clu],np.int16)
print(f'\nembed genes: {len(embed)}  ({u8.nbytes/1e6:.1f} MB);  axes: {len(names)}')
np.savez_compressed(OUT, maturity=maturity.astype(np.float32), age=age, Y=Y, axis_names=np.array(names),
    clu_code=clu_code, clu_cats=np.array(clu_cats), embed_genes=vn[embed], expr_u8=u8, expr_scale=ES,
    g_cls=gcls.astype(str), g_eta=geta, nmf_W=Wn, nmf_top=np.array(['/'.join(t) for t in nmf_top]),
    mat_fac=matf, shared_genes=np.array(shared_genes))
print(f'saved {OUT}')
