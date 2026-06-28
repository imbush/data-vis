#!/usr/bin/env python
"""Transfer interneuron subclass labels from the Tasic adult V1 reference onto the
Fezf2 query cells via scanpy.ingest (full-transcriptome kNN projection — far more
robust than marker-argmax, which fails at P20 where Pvalb is still immature).
Keeps the joint Ctrl+KO UMAP for the visualizer; slims the saved object.
"""
import os, warnings, numpy as np, pandas as pd, scanpy as sc, anndata as ad
warnings.filterwarnings('ignore'); sc.settings.verbosity = 1
ROOT = '/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish'
D = os.path.join(ROOT, 'data', 'fezf2_fishell')

# ---- reference: Tasic adult V1 GABA from the AllInhib projection cache (pickle:
# lognorm matrix + cell_subclass; avoids the v1_neurons_proc.h5ad uns read bug) ----
import pickle
print('loading Tasic reference (AllInhib proj cache) ...')
rc = pickle.load(open(os.path.join(ROOT, 'notebooks', 'cache',
                                   'AllInhib_archetype_proj_full.pkl'), 'rb'))
subc = pd.Series(rc['cell_subclass']).astype(str).replace({'Serpinf1': 'Sncg'})
ref = ad.AnnData(np.asarray(rc['X_keep'], dtype=np.float32),
                 obs=pd.DataFrame({'subclass': subc.values}),
                 var=pd.DataFrame(index=list(rc['gene_names'])))
ref = ref[ref.obs['subclass'].isin(['Pvalb', 'Sst', 'Vip', 'Lamp5', 'Sncg'])].copy()
print('  ref:', ref.n_obs, 'cells;', dict(pd.Series(ref.obs['subclass']).value_counts()))

# ---- query: processed Fezf2 INs (raw = lognorm) ----
q = sc.read_h5ad(os.path.join(D, 'p20_cIN_processed.h5ad'))
q.X = q.raw.X.copy()           # lognorm
umap_joint = q.obsm['X_umap'].copy()   # preserve the joint Ctrl+KO UMAP for the viz

# ---- shared genes, aligned ----
shared = [g for g in ref.var_names if g in set(q.var_names)]
print(f'  shared genes: {len(shared)}')
ref = ref[:, shared].copy(); q = q[:, shared].copy()

# ---- reference embedding ----
sc.pp.highly_variable_genes(ref, n_top_genes=2000, flavor='seurat')
ref = ref[:, ref.var.highly_variable].copy()
sc.pp.scale(ref, max_value=10); sc.tl.pca(ref, n_comps=30, random_state=0)
sc.pp.neighbors(ref, n_neighbors=15, random_state=0); sc.tl.umap(ref, random_state=0)
q = q[:, ref.var_names].copy()         # match ingest var set

print('ingesting (label transfer) ...')
sc.tl.ingest(q, ref, obs='subclass')
q.obs['subclass'] = q.obs['subclass'].astype(str)
q.obsm['X_umap'] = umap_joint          # restore joint UMAP

print('\n=== predicted subclass proportions by genotype (%) ===')
print((pd.crosstab(q.obs['subclass'], q.obs['genotype'], normalize='columns') * 100).round(1).to_string())
print('\nn per subclass:', dict(q.obs['subclass'].value_counts()))

# ---- write lean object: full-gene lognorm X + obs + umap (for the viz) ----
full = sc.read_h5ad(os.path.join(D, 'p20_cIN_processed.h5ad'))
full.obs['subclass'] = q.obs['subclass'].reindex(full.obs_names).values
full.X = full.raw.X.copy(); del full.raw
if 'counts' in full.layers: del full.layers['counts']
full.obsm['X_umap'] = umap_joint
out = os.path.join(D, 'p20_cIN_labeled.h5ad')
full.write(out)
print(f'\nwrote {out} ({os.path.getsize(out)/1e6:.0f} MB)')
