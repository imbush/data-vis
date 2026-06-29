#!/usr/bin/env python
"""Re-label the QC'd Fezf2 cells with an AGE-MATCHED reference.

The adult Tasic ref mislabels P20 Pvalb as Sst (immature Pvalb cells barely
express Pvalb yet, and ambient Sst inflates the floor). Gao 2025 developing-VISp
has validated subclass labels at the matching P-stages, so its P12-P28 inhibitory
neurons are a far better ingest reference. Reuses the cleaned matrix / leiden /
UMAP already in p20_cIN_labeled.h5ad (no re-run of Scrublet/embedding)."""
import os, warnings, numpy as np, pandas as pd, scanpy as sc, anndata as ad
warnings.filterwarnings('ignore'); sc.settings.verbosity = 0
ROOT = '/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish'
H5 = os.path.join(ROOT, 'data', 'fezf2_fishell', 'p20_cIN_labeled.h5ad')
SUB = ['Pvalb', 'Sst', 'Vip', 'Lamp5', 'Sncg']
MK = {'Pvalb': ['Pvalb','Tac1','Cox6a2'], 'Sst': ['Sst'], 'Vip': ['Vip'],
      'Lamp5': ['Lamp5','Ndnf'], 'Sncg': ['Sncg','Cck']}
REMAP = {'Pvalb chandelier':'Pvalb', 'Sst Chodl':'Sst', 'Lamp5 Lhx6':'Lamp5'}

A = sc.read_h5ad(H5)
print(f'query: {A.n_obs} cells, {A.n_vars} genes, leiden {A.obs.leiden.nunique()} clusters')

# ---- age-matched reference: Gao dev VISp, P12-P28 ----
ref = sc.read_h5ad(os.path.join(ROOT, 'data', 'devvis_inh_all_ages.h5ad'))
ages = ref.obs['synchronized_age'].astype(str)
keep_age = ages.str.startswith('P') & ages.str[1:].apply(
    lambda s: s.replace('.','',1).isdigit() and 12 <= float(s) <= 28)
ref = ref[keep_age].copy()
ref.obs['subclass'] = ref.obs['cell_subclass'].astype(str).replace(REMAP)
ref = ref[ref.obs.subclass.isin(SUB)].copy()
print(f'ref (P12-P28): {ref.n_obs} cells | {ref.obs.subclass.value_counts().to_dict()}')

# shared genes, HVG on ref, scale+PCA on ref, ingest query
shared = [g for g in ref.var_names if g in set(A.var_names)]
ref = ref[:, shared].copy()
sc.pp.highly_variable_genes(ref, n_top_genes=2000, flavor='seurat')
ref = ref[:, ref.var.highly_variable].copy()
sc.pp.scale(ref, max_value=10); sc.tl.pca(ref, n_comps=30, random_state=0)
sc.pp.neighbors(ref, n_neighbors=15, random_state=0); sc.tl.umap(ref, random_state=0)
q = ad.AnnData(A[:, ref.var_names].X.copy(), obs=A.obs.copy(),
               var=pd.DataFrame(index=list(ref.var_names)))
sc.pp.scale(q, max_value=10)
sc.tl.ingest(q, ref, obs='subclass')
A.obs['subclass_pred'] = q.obs['subclass'].astype(str).values

# ---- cluster-level majority label + LowQ gate ----
def score(genes):
    g=[x for x in genes if x in A.var_names]
    X=A[:,g].X; X=X.toarray() if hasattr(X,'toarray') else np.asarray(X); return X.mean(1).ravel()
sco = pd.DataFrame({k:score(v) for k,v in MK.items()}, index=A.obs_names)
sco['leiden']=A.obs.leiden.values
cm = sco.groupby('leiden')[SUB].mean()
ngm = A.obs.groupby('leiden')['n_genes_by_counts'].median()
maj = A.obs.groupby('leiden')['subclass_pred'].agg(lambda s: s.value_counts().index[0])
purity = A.obs.groupby('leiden')['subclass_pred'].agg(lambda s: s.value_counts(normalize=True).iloc[0])
global_med = A.obs['n_genes_by_counts'].median(); lowq_thr = max(1800, 0.7*global_med)
label = {cl: ('LowQ' if (ngm[cl] < lowq_thr or purity[cl] < 0.5) else maj[cl]) for cl in cm.index}
A.obs['subclass'] = A.obs['leiden'].map(label).astype(str)

print(f'\nlowq n_genes thr {lowq_thr:.0f} (global median {global_med:.0f})')
diag = cm.copy(); diag['n']=A.obs.leiden.value_counts(); diag['ngenes']=ngm.round(0)
diag['ingest_maj']=maj; diag['purity']=purity.round(2); diag['label']=[label[c] for c in diag.index]
print(diag.sort_values('n',ascending=False).round(2).to_string())

clean = A[A.obs['subclass'].isin(SUB)].copy()
print(f'\nkept {clean.n_obs}/{A.n_obs} clean INs ({A.n_obs-clean.n_obs} LowQ dropped)')
print('\n=== mean marker per ASSIGNED subclass (want diagonal) ===')
print(pd.DataFrame({k:score(v) for k,v in MK.items()}, index=A.obs_names)
      .assign(s=A.obs['subclass'].values).query("s in @SUB").groupby('s')[SUB].mean().round(2).to_string())
print('\n=== proportions by genotype (%) ===')
print((pd.crosstab(clean.obs['subclass'],clean.obs['genotype'],normalize='columns')*100).round(1).to_string())

clean.write(H5)
print(f'\nrewrote {H5} ({os.path.getsize(H5)/1e6:.0f} MB)')
