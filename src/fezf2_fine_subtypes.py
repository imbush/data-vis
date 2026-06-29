#!/usr/bin/env python
"""Add FINE subtype labels (cell_cluster) on top of the trusted subclass calls.

Transfers the age-matched dev-VISp reference's fine clusters (61, incl. Pvalb
chandelier / Sst Chodl) by ingest, then assigns ONE fine label per query Leiden
cluster (majority vote), constrained to be consistent with that cluster's already-
trusted cell_subclass. Result: subtype checkboxes go granular without disturbing
the validated subclass-level proportions. Names are atlas-style (e.g. 'Sst 10')
since the fine identities are still maturing at P20."""
import os, warnings, re, numpy as np, pandas as pd, scanpy as sc, anndata as ad
warnings.filterwarnings('ignore'); sc.settings.verbosity = 0
ROOT = '/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish'
H5 = os.path.join(ROOT, 'data', 'fezf2_fishell', 'p20_cIN_labeled.h5ad')
SUB = ['Pvalb', 'Sst', 'Vip', 'Lamp5', 'Sncg']
# fine subclass (incl chandelier/Chodl) -> parent subclass used in our labels
PARENT = {'Pvalb chandelier':'Pvalb', 'Sst Chodl':'Sst'}

def clean_fine(orig_cluster):
    # '742_Pvalb Gaba_3' -> ('Pvalb', 'Pvalb 3'); '..._Pvalb chandelier Gaba_2'
    parts = str(orig_cluster).split('_')
    mid = parts[1] if len(parts) > 1 else parts[0]
    num = parts[-1]
    sc_name = mid.replace(' Gaba', '').strip()          # 'Pvalb', 'Pvalb chandelier', 'Sst Chodl'
    parent = PARENT.get(sc_name, sc_name)
    return parent, f'{sc_name} {num}'

A = sc.read_h5ad(H5)
print(f'query: {A.n_obs} cells | subclasses {sorted(A.obs.cell_subclass.unique())} '
      f'| {A.obs.leiden.nunique()} leiden clusters')

# ---- reference (P12-P28) with fine labels ----
ref = sc.read_h5ad(os.path.join(ROOT, 'data', 'devvis_inh_all_ages.h5ad'))
ages = ref.obs['synchronized_age'].astype(str)
keep = ages.str.startswith('P') & ages.str[1:].apply(
    lambda s: s.replace('.','',1).isdigit() and 12 <= float(s) <= 28)
ref = ref[keep].copy()
pf = [clean_fine(x) for x in ref.obs['orig_cluster']]
ref.obs['parent'] = [p for p,_ in pf]
ref.obs['fine'] = [f for _,f in pf]
ref = ref[ref.obs.parent.isin(SUB)].copy()
print(f'ref P12-P28: {ref.n_obs} cells | {ref.obs.fine.nunique()} fine clusters')

shared = [g for g in ref.var_names if g in set(A.var_names)]
ref = ref[:, shared].copy()
sc.pp.highly_variable_genes(ref, n_top_genes=2000, flavor='seurat')
ref = ref[:, ref.var.highly_variable].copy()
sc.pp.scale(ref, max_value=10); sc.tl.pca(ref, n_comps=30, random_state=0)
sc.pp.neighbors(ref, n_neighbors=15, random_state=0); sc.tl.umap(ref, random_state=0)
q = ad.AnnData(A[:, ref.var_names].X.copy(), obs=A.obs.copy(),
               var=pd.DataFrame(index=list(ref.var_names)))
sc.pp.scale(q, max_value=10)
sc.tl.ingest(q, ref, obs=['fine','parent'])
A.obs['fine_pred'] = q.obs['fine'].astype(str).values
A.obs['fine_parent'] = q.obs['parent'].astype(str).values

# ---- one fine label per Leiden cluster, consistent with trusted subclass ----
fine_label = {}
for cl, idx in A.obs.groupby('leiden').groups.items():
    sub = A.obs.loc[idx, 'cell_subclass'].iloc[0]           # trusted subclass of this cluster
    cand = A.obs.loc[idx]
    cand = cand[cand['fine_parent'] == sub]                 # keep only fine types of the right subclass
    if len(cand):
        fine_label[cl] = cand['fine_pred'].value_counts().index[0]
    else:                                                   # no consistent fine call -> generic
        fine_label[cl] = f'{sub} (other)'
A.obs['cell_cluster'] = A.obs['leiden'].map(fine_label).astype(str)

# keep only the clean interneurons (subclass already filtered in prior step, but be safe)
A = A[A.obs['cell_subclass'].isin(SUB)].copy()
print(f'\n{A.obs.cell_cluster.nunique()} fine subtypes assigned:')
tab = (A.obs.groupby('cell_subclass')['cell_cluster']
       .agg(lambda s: ', '.join(f'{k}({v})' for k,v in s.value_counts().items())))
for s in SUB:
    if s in tab.index: print(f'  {s}: {tab[s]}')

A.write(H5)
print(f'\nrewrote {H5} ({os.path.getsize(H5)/1e6:.0f} MB)')
