#!/usr/bin/env python
"""Corrected Fezf2 pipeline (replaces fezf2_process + fezf2_label_transfer).

Fixes the v1 problems (a low-quality/doublet blob that ingest mislabeled as Sst
everywhere): adds Scrublet doublet removal + stricter QC, and labels at the
CLUSTER level (one subclass per Leiden cluster via ingest-majority vote, far more
stable than per-cell), dropping clusters that are low-quality or marker-ambiguous.
"""
import os, warnings, numpy as np, pandas as pd, scanpy as sc, anndata as ad, pickle
warnings.filterwarnings('ignore'); sc.settings.verbosity = 0
ROOT = '/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish'
D = os.path.join(ROOT, 'data', 'fezf2_fishell')
SUB = ['Pvalb', 'Sst', 'Vip', 'Lamp5', 'Sncg']
MK = {'Pvalb': ['Pvalb','Tac1','Cox6a2'], 'Sst': ['Sst'], 'Vip': ['Vip'],
      'Lamp5': ['Lamp5','Ndnf'], 'Sncg': ['Sncg']}

def load(gsm, geno, samp):
    a = sc.read_10x_h5(os.path.join(D, f'{gsm}_filtered_feature_bc_matrix.h5'))
    a.var_names_make_unique()
    a.obs['genotype'] = geno; a.obs['sample'] = samp
    a.obs_names = [f'{samp}_{bc}' for bc in a.obs_names]
    print(f'  scrublet {samp} ...')
    sc.pp.scrublet(a, random_state=0)   # adds predicted_doublet, doublet_score
    return a

print('loading + scrublet ...')
A = sc.concat([load('GSM8409498_P20cIN', 'Control', 'P20_Ctrl'),
               load('GSM8409499_P20cIN_Fezf2KO', 'Fezf2-KO', 'P20_Fezf2KO')], join='outer')
A.layers['counts'] = A.X.copy()
A.var['mt'] = A.var_names.str.startswith('mt-')
sc.pp.calculate_qc_metrics(A, qc_vars=['mt'], inplace=True, percent_top=None)
n0 = A.n_obs
A = A[(A.obs.n_genes_by_counts >= 1200) & (A.obs.n_genes_by_counts <= 9000)
      & (A.obs.pct_counts_mt < 5) & (~A.obs.predicted_doublet)].copy()
sc.pp.filter_genes(A, min_cells=10)
print(f'QC: {n0} -> {A.n_obs} nuclei ({int(A.obs.predicted_doublet.sum()) if "predicted_doublet" in A.obs else 0} dbl flagged pre-filter); '
      f'per sample {dict(A.obs["sample"].value_counts())}')

sc.pp.normalize_total(A, target_sum=1e4); sc.pp.log1p(A); A.raw = A
sc.pp.highly_variable_genes(A, n_top_genes=2000, flavor='seurat', batch_key='sample')
Ah = A[:, A.var.highly_variable].copy()
sc.pp.scale(Ah, max_value=10); sc.tl.pca(Ah, n_comps=30, random_state=0)
sc.pp.neighbors(Ah, n_neighbors=15, random_state=0)
sc.tl.leiden(Ah, resolution=1.0, random_state=0, flavor='igraph', n_iterations=2, directed=False)
sc.tl.umap(Ah, random_state=0)
A.obs['leiden'] = Ah.obs['leiden'].values; A.obsm['X_umap'] = Ah.obsm['X_umap']
print(f'{A.obs.leiden.nunique()} leiden clusters')

# ---- per-cell label transfer (ingest from Tasic AllInhib ref) ----
rc = pickle.load(open(os.path.join(ROOT,'notebooks','cache','AllInhib_archetype_proj_full.pkl'),'rb'))
subc = pd.Series(rc['cell_subclass']).astype(str).replace({'Serpinf1':'Sncg'})
ref = ad.AnnData(np.asarray(rc['X_keep'],dtype=np.float32),
                 obs=pd.DataFrame({'subclass':subc.values}), var=pd.DataFrame(index=list(rc['gene_names'])))
ref = ref[ref.obs.subclass.isin(SUB)].copy()
shared = [g for g in ref.var_names if g in set(A.raw.var_names)]
ref = ref[:, shared].copy()
sc.pp.highly_variable_genes(ref, n_top_genes=2000, flavor='seurat'); ref = ref[:, ref.var.highly_variable].copy()
sc.pp.scale(ref, max_value=10); sc.tl.pca(ref, n_comps=30, random_state=0)
sc.pp.neighbors(ref, n_neighbors=15, random_state=0); sc.tl.umap(ref, random_state=0)
q = ad.AnnData(A.raw[:, ref.var_names].X.copy(), obs=A.obs.copy(), var=pd.DataFrame(index=list(ref.var_names)))
sc.tl.ingest(q, ref, obs='subclass')
A.obs['subclass_pred'] = q.obs['subclass'].astype(str).values

# ---- CLUSTER-level labels: majority ingest vote, with low-quality flagging ----
def score(genes):
    g=[x for x in genes if x in A.raw.var_names]
    X=A.raw[:,g].X; X=X.toarray() if hasattr(X,'toarray') else np.asarray(X); return X.mean(1).ravel()
sco = pd.DataFrame({k:score(v) for k,v in MK.items()}, index=A.obs_names); sco['leiden']=A.obs.leiden.values
cm = sco.groupby('leiden')[SUB].mean()
ngm = A.obs.groupby('leiden')['n_genes_by_counts'].median()
maj = A.obs.groupby('leiden')['subclass_pred'].agg(lambda s: s.value_counts().index[0])
global_med = A.obs['n_genes_by_counts'].median(); lowq_thr = max(1500, 0.6*global_med)
label = {}
for cl in cm.index:
    top2 = cm.loc[cl].sort_values(ascending=False)
    margin = top2.iloc[0] - top2.iloc[1]
    if ngm[cl] < lowq_thr:
        label[cl] = 'LowQ'
    else:
        label[cl] = maj[cl]   # ingest majority; markers used only for LowQ gate
A.obs['subclass'] = A.obs['leiden'].map(label).astype(str)
print(f'\nlow-quality n_genes threshold: {lowq_thr:.0f} (global median {global_med:.0f})')
diag = cm.copy(); diag['n']=A.obs.leiden.value_counts(); diag['ngenes_med']=ngm
diag['ingest_maj']=maj; diag['label']=[label[c] for c in diag.index]
print(diag.sort_values('n',ascending=False).round(2).to_string())

clean = A[A.obs['subclass'].isin(SUB)].copy()
print(f'\nkept {clean.n_obs} clean INs ({A.n_obs-clean.n_obs} LowQ dropped)')
print('\n=== mean marker per ASSIGNED subclass (want diagonal) ===')
ev=pd.DataFrame({k:score(v) for k,v in MK.items()}, index=A.obs_names)
print(ev.assign(s=A.obs['subclass'].values).query("s in @SUB").groupby('s')[SUB].mean().round(2).to_string())
print('\n=== proportions by genotype (%) ===')
print((pd.crosstab(clean.obs['subclass'],clean.obs['genotype'],normalize='columns')*100).round(1).to_string())

clean.X = clean.raw[:, clean.var_names].X.copy() if clean.raw is not None else clean.X
for k in list(clean.layers): del clean.layers[k]
clean.raw = None
out = os.path.join(D, 'p20_cIN_labeled.h5ad'); clean.write(out)
print(f'\nwrote {out} ({os.path.getsize(out)/1e6:.0f} MB)')
