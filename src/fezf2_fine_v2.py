#!/usr/bin/env python
"""Finer subtypes, marker-renamed. Uses the PER-CELL age-matched dev-VISp fine
labels (31 supported types, not the 15 from per-Leiden-majority), reconciled to
the trusted subclass, with noise types (<MIN_CELLS) merged into the largest type
of their subclass. Each atlas-style fine cluster is renamed by its top reference
marker gene ('Sst 10' -> 'Sst Calb2'), keeping chandelier/Chodl canonical."""
import os, warnings, numpy as np, pandas as pd, scanpy as sc
warnings.filterwarnings('ignore'); sc.settings.verbosity = 0
ROOT = '/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish'
H5 = os.path.join(ROOT, 'data', 'fezf2_fishell', 'p20_cIN_labeled.h5ad')
SUB = ['Pvalb', 'Sst', 'Vip', 'Lamp5', 'Sncg']
PARENT = {'Pvalb chandelier':'Pvalb', 'Sst Chodl':'Sst'}
MIN_CELLS = 40

def clean_fine(orig):
    parts = str(orig).split('_'); mid = parts[1] if len(parts)>1 else parts[0]
    sc_name = mid.replace(' Gaba','').strip()
    return PARENT.get(sc_name, sc_name), f'{sc_name} {parts[-1]}', sc_name

A = sc.read_h5ad(H5)
# ---- reference (P12-P28): compute a top-marker name per fine cluster ----
ref = sc.read_h5ad(os.path.join(ROOT,'data','devvis_inh_all_ages.h5ad'))
ages = ref.obs['synchronized_age'].astype(str)
keep = ages.str.startswith('P') & ages.str[1:].apply(
    lambda s: s.replace('.','',1).isdigit() and 12<=float(s)<=28)
ref = ref[keep].copy()
pf = [clean_fine(x) for x in ref.obs['orig_cluster']]
ref.obs['parent']=[p for p,_,_ in pf]; ref.obs['fine']=[f for _,f,_ in pf]; ref.obs['scname']=[s for _,_,s in pf]
ref = ref[ref.obs.parent.isin(SUB)].copy()
sc.tl.rank_genes_groups(ref, 'fine', method='t-test', n_genes=8)
names = ref.uns['rank_genes_groups']['names']
pretty, used = {}, set()
for fine in names.dtype.names:
    scn = ref.obs.loc[ref.obs.fine==fine,'scname'].iloc[0]
    par = PARENT.get(scn, scn)
    if scn in ('Pvalb chandelier','Sst Chodl'):     # keep canonical, already informative
        nm = scn
    else:
        nm = par
        for g in names[fine]:                        # first top gene that's not the subclass name + unused
            if g not in used and g.lower() != par.lower(): nm = f'{par} {g}'; used.add(g); break
    pretty[fine] = nm
print('rename map:'); [print(f'  {k:18s} -> {v}') for k,v in pretty.items()]

# ---- query: per-cell fine, reconciled to trusted subclass, merge tinies ----
def parent_of(f):                                    # parent subclass implied by the fine NAME itself
    s = ' '.join(str(f).split(' ')[:-1]); return PARENT.get(s, s)
fp, sub = A.obs['fine_pred'].astype(str), A.obs['cell_subclass'].astype(str)
fp_par = fp.map(parent_of)
# dominant per-cell fine type within each subclass (fallback for mismatches)
dom = {s: fp[(fp_par==s)].value_counts().index[0] for s in SUB}
fine = np.where(fp_par.values==sub.values, fp.values, sub.map(dom).values)
A.obs['fine'] = fine
# merge fine types below MIN_CELLS into the largest type of the same subclass
vc = pd.Series(fine).value_counts()
big_by_sub = {s: A.obs.loc[A.obs.cell_subclass==s,'fine'].value_counts().index[0] for s in SUB}
def fix(f, s): return f if vc.get(f,0) >= MIN_CELLS else big_by_sub[s]
A.obs['fine'] = [fix(f,s) for f,s in zip(A.obs['fine'], A.obs['cell_subclass'])]
A.obs['cell_cluster'] = A.obs['fine'].map(lambda f: pretty.get(f, f)).astype(str)

print(f'\n{A.obs.cell_cluster.nunique()} fine subtypes (>= {MIN_CELLS} cells):')
for s in SUB:
    sub_counts = A.obs.loc[A.obs.cell_subclass==s,'cell_cluster'].value_counts()
    print(f'  {s}: ' + ', '.join(f'{k}({v})' for k,v in sub_counts.items()))
A.write(H5)
print(f'\nrewrote {H5} ({os.path.getsize(H5)/1e6:.0f} MB)')
