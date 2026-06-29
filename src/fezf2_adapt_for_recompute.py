#!/usr/bin/env python
"""Add the obs columns the recompute builders' loader expects, in place.

compute_or_load_proj_full() reads: cell_class (=='GABAergic' filter), cell_subclass,
cell_cluster (subtype checkboxes), dissected_region (Region toggle -> we map to
GENOTYPE), and a donor column. The Fezf2 pipeline saves `subclass`/`genotype`/
`sample`; this maps them onto the expected names so GROUPS['Fezf2'] just works.
"""
import os, scanpy as sc
ROOT = '/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish'
p = os.path.join(ROOT, 'data', 'fezf2_fishell', 'p20_cIN_labeled.h5ad')
a = sc.read_h5ad(p)
a.obs['cell_class'] = 'GABAergic'
a.obs['cell_subclass'] = a.obs['subclass'].astype(str)
a.obs['cell_cluster'] = a.obs['subclass'].astype(str)       # subtype = subclass
a.obs['dissected_region'] = a.obs['genotype'].astype(str)   # Region toggle = genotype
a.obs['donor_id'] = a.obs['sample'].astype(str)
a.write(p)
print('adapted', p, '| n=', a.n_obs,
      '| subclasses', sorted(a.obs.cell_subclass.unique()),
      '| genotypes', sorted(a.obs.dissected_region.unique()))
