"""Lamp5 developmental gradient analysis (DevVIS, Gao 2025).
x = maturity (transcriptomic), y = position along the Lamp5 continuum defined in
mature cells as a linear gene signature (so it applies to immature cells too).
Saves lamp5_grad.npz for the HTML builder. This script only computes + validates."""
import anndata as ad, numpy as np, scipy.sparse as sp, re, pandas as pd
from numpy.linalg import svd
np.random.seed(0)
ROOT='/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish'
OUT='/private/tmp/claude-501/-Users-inlebush-cs-lab-green-data-vis/95ef1b74-d552-4696-b236-9181bf853664/scratchpad/lamp5_grad.npz'

a = ad.read_h5ad(f'{ROOT}/data/devvis_inh_all_ages.h5ad')
lam = a[a.obs['cell_subclass'].astype(str)=='Lamp5'].copy()
X = lam.X; X = X.toarray() if sp.issparse(X) else np.asarray(X); X=X.astype(np.float32)
vn = np.array(lam.var_names); gi={g:i for i,g in enumerate(vn)}
N,G = X.shape
def agenum(s):
    m=re.match(r"([EP])([0-9.]+)",str(s)); return (float(m.group(2))-19 if m.group(1)=="E" else float(m.group(2))) if m else np.nan
age = np.array([agenum(x) for x in lam.obs['synchronized_age']], np.float32)
clu = lam.obs['cell_cluster_named'].astype(str).values
print(f'Lamp5: {N} cells x {G} genes; age P3-P56')

# ---- z-score helper on a reference mask ----
def zref(mask):
    mu = X[mask].mean(0); sd = X[mask].std(0)+1e-6
    return mu, sd

# ================= maturity axis (x) =================
IMMATURE=['Dcx','Sox11','Sox4','Tubb3','Sox2','Neurod2','Cd24a','Nnat','Igfbpl1','Marcksl1','Ccnd2','Btg1','Tubb2b','Stmn2','Stmn1']
imm=[g for g in IMMATURE if g in gi]
mu,sd = zref(np.ones(N,bool))
Z = (X-mu)/sd
imm_score = Z[:,[gi[g] for g in imm]].mean(1)
maturity = -imm_score
maturity = (maturity-maturity.mean())/maturity.std()
print(f'\nmaturity: immaturity genes used={imm}')
print(f'  corr(maturity, age) = {np.corrcoef(maturity,age)[0,1]:.3f}   (want strongly +)')

# ================= continuum axis (y) from MOST-MATURE cells =================
mature = maturity > np.quantile(maturity, 0.75)     # top quartile = cleanest adult identity
print(f'\nmature cells for continuum def: {mature.sum()}')
# HVG on mature cells (by normalized dispersion, simple)
Xm = X[mature]
mu_m = Xm.mean(0); var_m = Xm.var(0)
expressed = (Xm>0).mean(0) > 0.05
disp = np.full(G,-np.inf); ok=expressed & (mu_m>0.05)
disp[ok] = var_m[ok]/mu_m[ok]
hvg = np.argsort(disp)[::-1][:2000]
# PCA on mature cells over HVG (z-scored within mature)
mm = Xm[:,hvg].mean(0); ss = Xm[:,hvg].std(0)+1e-6
Zm = (Xm[:,hvg]-mm)/ss
U,S,Vt = svd(Zm - Zm.mean(0), full_matrices=False)
PCm = U[:,:15]*S[:15]
# pole contrast to pick + orient the continuum PC
def zc(g,mask=None):
    v=(X[:,gi[g]]-mu[gi[g]])/sd[gi[g]]; return v[mask] if mask is not None else v
poleA=['Cnr1','Cxcl14','Egln3','Nr2f2']; poleB=['Sv2c','Nxph1']
contrast_m = (np.mean([zc(g,mature) for g in poleB],0) - np.mean([zc(g,mature) for g in poleA],0))
cors=[np.corrcoef(PCm[:,k],contrast_m)[0,1] for k in range(15)]
kbest=int(np.argmax(np.abs(cors))); sign=np.sign(cors[kbest])
print(f'  continuum PC = PC{kbest+1} (corr with poleB-poleA contrast = {cors[kbest]:.3f})')
cont_m = sign*PCm[:,kbest]
# express as a gene signature: corr of each gene with cont_m over mature cells
Zm_all = (Xm-mu)/sd                                  # z all genes on mature cells (global mu/sd)
cont_m_c = cont_m-cont_m.mean()
gene_w = (Zm_all*cont_m_c[:,None]).mean(0) / (cont_m_c.std()+1e-9)   # ~corr*std
# top graded genes define the projection — EXCLUDE maturation genes so y is identity, not age
imm_set=set(gi[g] for g in imm)
gene_w_id = gene_w.copy(); gene_w_id[list(imm_set)] = 0
topK=300; order=np.argsort(np.abs(gene_w_id))[::-1]; sel=order[:topK]
w=np.zeros(G); w[sel]=gene_w_id[sel]
y_all = ((X-mu)/sd) @ w
# orthogonalize continuum against maturity (residualize) so the two axes are independent
b = float((y_all*maturity).sum()/(maturity*maturity).sum())
y_all = y_all - b*maturity
y_all = (y_all-y_all[mature].mean())/y_all[mature].std()
print(f'  corr(continuum y, maturity) = {np.corrcoef(y_all,maturity)[0,1]:.3f}  (want ~0 after residualize)')
print(f'  corr(continuum y, age)      = {np.corrcoef(y_all,age)[0,1]:.3f}')

# validate: cluster means along y
print('\ncluster mean continuum-y (should separate the continuum):')
for c in pd.Series(clu).value_counts().index:
    m=clu==c
    if m.sum()>20: print(f'  {c:16s} n={m.sum():5d}  y={y_all[m].mean():+.2f}  maturity={maturity[m].mean():+.2f}')

# pole genes along y (mature cells): confirm direction
print('\npole/marker genes: corr with continuum-y (mature cells):')
for g in poleA+poleB+['Ndnf','Reln','Id2','Npy','Kit','Egfr']:
    if g in gi: print(f'  {g:8s} {np.corrcoef(zc(g,mature),cont_m)[0,1]:+.3f}')

# ================= NMF factors (k=10) on HVG, all cells =================
from sklearn.decomposition import NMF
Xh = X[:,hvg]                                   # nonneg log2
nmf = NMF(n_components=10, init='nndsvda', random_state=0, max_iter=400)
W = nmf.fit_transform(Xh); H = nmf.components_    # W: N x10 (cell loadings), H: 10 x len(hvg)
Wn = W/ (W.max(0)+1e-9)                            # normalize each factor to [0,1] for colouring
nmf_top=[]; hvg_names=vn[hvg]
for k in range(10):
    top=hvg_names[np.argsort(H[k])[::-1][:8]].tolist(); nmf_top.append(top)
# label the maturation factor (loads Dcx/Sox11) + continuum factors
def fac_score(k, genes): return sum(g in nmf_top[k] for g in genes)
mat_fac=int(np.argmax([fac_score(k,['Dcx','Sox11','Sox4','Tubb3','Stmn2']) for k in range(10)]))
print(f'\nNMF factors (top genes):')
for k in range(10): print(f'  C{k+1}{" [maturation]" if k==mat_fac else "":13s}: {", ".join(nmf_top[k])}')

# ================= gene discovery + shape classification =================
from scipy.stats import spearmanr, rankdata
def spr(e,yv):
    if e.std()<1e-9: return 0.0
    return float(np.corrcoef(rankdata(e),rankdata(yv))[0,1])
B=12
def profile_eta(e, yv):                            # equal-count bins along y, eta^2 + means
    qs=np.quantile(yv,np.linspace(0,1,B+1)); bins=np.clip(np.digitize(yv,qs[1:-1]),0,B-1)
    grand=e.mean(); ss=((e-grand)**2).sum()+1e-9; means=np.zeros(B); ssb=0.0
    for b in range(B):
        m=bins==b
        if m.any(): means[b]=e[m].mean(); ssb+=m.sum()*(means[b]-grand)**2
    return ssb/ss, means
lowm=maturity<np.median(maturity); highm=~lowm
ym=y_all[mature]
expressed_all=(X>0).mean(0)>0.05
cand=np.where(expressed_all & (X.mean(0)>0.1))[0]
rows=[]
for j in cand:
    e=X[:,j]; em=e[mature]
    eta,means=profile_eta(em,ym)
    if eta<0.05: continue
    s=spr(em,ym); mn=means-means.min(); rng=mn.max()+1e-9; mn/=rng
    amax=int(np.argmax(means)); amin=int(np.argmin(means))
    prom_pk=(means[amax]-max(means[0],means[-1]))/(np.ptp(means)+1e-9)
    prom_dp=(min(means[0],means[-1])-means[amin])/(np.ptp(means)+1e-9)
    if abs(s)>=0.35:                 cls='B-pole' if s>0 else 'A-pole'
    elif 2<=amax<=B-3 and prom_pk>0.35: cls='mid-peak'
    elif 2<=amin<=B-3 and prom_dp>0.35: cls='mid-dip'
    else: cls='other'
    if cls=='other': continue
    eta_lo,_=profile_eta(e[lowm],y_all[lowm]); eta_hi,_=profile_eta(e[highm],y_all[highm])
    rows.append((j,cls,round(eta,3),round(s,3),round(float(eta_lo),3),round(float(eta_hi),3),
                 round(float(em.mean()),2)))
gdf=pd.DataFrame(rows,columns=['j','cls','eta','spearman','eta_lowmat','eta_highmat','meanexpr'])
gdf=gdf.sort_values('eta',ascending=False).reset_index(drop=True)
print(f'\nsmoothly-graded genes found: {len(gdf)}')
print(gdf['cls'].value_counts().to_string())
print('\ntop graded genes per class:')
for c in ['A-pole','B-pole','mid-peak','mid-dip']:
    sub=gdf[gdf.cls==c].head(12); print(f'  {c}: '+', '.join(vn[sub.j]))
print('\nmid-peak genes (like Ndnf) — emergence (eta low-mat vs high-mat):')
for _,r in gdf[gdf.cls=='mid-peak'].head(8).iterrows():
    print(f'  {vn[int(r.j)]:10s} eta={r.eta:.2f}  low-mat={r.eta_lowmat:.2f}  high-mat={r.eta_highmat:.2f}')

# ================= choose embed gene set =================
embed=set(gdf.j.tolist())
for g in imm+['Egln3','Cnr1','Cxcl14','Nr2f2','Sv2c','Nxph1','Ndnf','Reln','Id2','Npy','Kit','Egfr','Lamp5']:
    if g in gi: embed.add(gi[g])
for k in range(10):
    for g in nmf_top[k]:
        if g in gi: embed.add(gi[g])
# pad with top-HVG so gene search is useful
for j in hvg:
    if len(embed)>=1200: break
    embed.add(int(j))
embed=sorted(embed)
print(f'\nembed genes: {len(embed)}')

EXPR_SCALE=16
u8=np.clip(np.round(X[:,embed]*EXPR_SCALE),0,255).astype(np.uint8)
# per-embed-gene stats aligned to `embed`
jpos={j:i for i,j in enumerate(embed)}
cls_arr=np.array(['none']*len(embed),dtype=object); eta_a=np.zeros(len(embed),np.float32)
spr_a=np.zeros(len(embed),np.float32); elo=np.zeros(len(embed),np.float32); ehi=np.zeros(len(embed),np.float32)
for _,r in gdf.iterrows():
    i=jpos[int(r.j)]; cls_arr[i]=r.cls; eta_a[i]=r.eta; spr_a[i]=r.spearman; elo[i]=r.eta_lowmat; ehi[i]=r.eta_highmat

# cluster codes
clu_cats=pd.Series(clu).value_counts().index.tolist(); clu_code=np.array([clu_cats.index(c) for c in clu],np.int16)

np.savez_compressed(OUT,
    maturity=maturity.astype(np.float32), y=y_all.astype(np.float32), age=age.astype(np.float32),
    clu_code=clu_code, clu_cats=np.array(clu_cats),
    embed_genes=vn[embed], expr_u8=u8, expr_scale=EXPR_SCALE,
    g_cls=cls_arr.astype(str), g_eta=eta_a, g_spr=spr_a, g_elo=elo, g_ehi=ehi,
    nmf_W=Wn.astype(np.float32), nmf_top=np.array(['/'.join(t) for t in nmf_top]), mat_fac=mat_fac)
print(f'\nsaved {OUT}  ({u8.nbytes/1e6:.1f} MB expr)')
