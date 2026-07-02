#!/usr/bin/env python
"""Marker-name the Gao 2025 dev-VISp inhibitory clusters AND subclusters, the way
the fezf2 pipeline named its fine types.

Adds two obs columns to data/devvis_inh_all_ages.h5ad:
  - cell_cluster_named     : subclass + 1-2 marker genes for the 61-cluster level
  - cell_subcluster_named  : subclass + 1-2 marker genes for the finer 132-subcluster level

Naming rules (per group, within its Tasic-mapped subclass):
  * prefix = cell_subclass (Pvalb / Sst / Vip / Lamp5 / Sncg / Pvalb chandelier /
    Sst Chodl / Lamp5 Lhx6). Chandelier/Chodl/Lhx6 stay canonical.
  * genes = top rank_genes_groups markers, but PREFER a gene from the Tasic 2018
    GABAergic subtype vocabulary when it is genuinely enriched (in this group's
    top-K DE genes) — so names line up with the adult Tasic nomenclature.
  * 1 gene when that uniquely identifies the group within its subclass; a 2nd
    gene is appended only to break ties.
  * the subclass token itself is never used as a marker gene.
Subcluster labels are derived per cell from the taxonomy (cluster_alias -> the
'subcluster' term set), which the base h5ad did not propagate.
"""
import os, re, warnings, numpy as np, pandas as pd, scanpy as sc, anndata as ad
warnings.filterwarnings('ignore'); sc.settings.verbosity = 1

ROOT = '/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish'
H5   = os.path.join(ROOT, 'data', 'devvis_inh_all_ages.h5ad')
TASIC = os.path.join(ROOT, 'data', 'v1_neurons_proc.h5ad')
TAX  = "/Users/Shared/lab-data/allen-v1-dev-atlas/_abc_cache/metadata/Developing-Mouse-Vis-Cortex-taxonomy/20260331"

SUBCLASS_TOKENS = {'Lamp5','Sst','Pvalb','Vip','Sncg','Lhx6','Chodl','chandelier',
                   'Serpinf1','Meis2','Gaba','GABAergic'}
CANON = {'Pvalb chandelier','Sst Chodl','Lamp5 Lhx6'}   # keep subclass as-is; genes still appended
TOPK  = 20      # how many DE genes to scan per group
MIN_CELLS = 10  # groups below this are still named, best-effort


def tasic_marker_vocab():
    """Gene tokens that Tasic 2018 uses in GABAergic subtype names (e.g. Calb2,
    Tpbg, Reln, Cbln4 ...) — the 'accurate Tasic gene' pool to prefer.
    Read via h5py to sidestep an unreadable /uns/log1p element in this file."""
    import h5py
    with h5py.File(TASIC, 'r') as f:
        g = f['obs/cell_cluster']
        if isinstance(g, h5py.Group) and 'categories' in g:      # categorical
            names = [s.decode() if isinstance(s, bytes) else str(s)
                     for s in g['categories'][:]]
        else:                                                     # plain str array
            names = [s.decode() if isinstance(s, bytes) else str(s)
                     for s in g[:]]
        names = list(dict.fromkeys(names))
    vocab = set()
    for nm in names:
        for tok in nm.split():
            if tok in SUBCLASS_TOKENS: continue
            if re.fullmatch(r'[A-Z][A-Za-z0-9]{2,}', tok):   # gene-symbol-ish
                vocab.add(tok)
    return vocab, None


def name_level(a, groupby, tasic_vocab, valid_genes, one_gene_ok=True):
    """Return {group_value: pretty_name} for a categorical obs column."""
    sc.tl.rank_genes_groups(a, groupby, method='t-test', n_genes=TOPK)
    names = a.uns['rank_genes_groups']['names']
    lfc   = a.uns['rank_genes_groups']['logfoldchanges']
    # subclass of each group (majority)
    grp_sub = a.obs.groupby(groupby, observed=True)['cell_subclass'].agg(
        lambda s: s.value_counts().index[0])
    grp_n = a.obs[groupby].value_counts()

    # rank groups big->small so abundant, well-defined types claim their marker first
    order = sorted(names.dtype.names, key=lambda g: -int(grp_n.get(g, 0)))
    pretty, used = {}, {}          # used: subclass -> set(genes already taken)
    for g in order:
        sub = str(grp_sub.get(g, 'GABAergic'))
        used.setdefault(sub, set())
        cand = [(str(gene), float(f)) for gene, f in zip(names[g], lfc[g])
                if str(gene) in valid_genes and str(gene) not in SUBCLASS_TOKENS
                and sub not in str(gene)]              # skip subclass-name genes
        # prefer an enriched Tasic-vocab gene, then any enriched gene, unused first
        def pick(pool):
            tas = [x for x in pool if x[0] in tasic_vocab and x[1] > 0.5]
            for gene, _ in tas:
                if gene not in used[sub]: return gene
            for gene, f in pool:
                if f > 0.25 and gene not in used[sub]: return gene
            for gene, _ in pool:                        # last resort: allow reuse
                return gene
            return None
        g1 = pick(cand)
        genes = [g1] if g1 else []
        if g1: used[sub].add(g1)
        # add a 2nd gene to break a would-be duplicate name
        base_name = f'{sub} {g1}' if g1 else sub
        need2 = (not one_gene_ok) or (base_name in pretty.values())
        if g1 and need2:
            rest = [x for x in cand if x[0] != g1]
            g2 = pick(rest)
            if g2: genes.append(g2); used[sub].add(g2)
        nm = (sub + (' ' + ' '.join(genes) if genes else '')).strip()
        # guarantee uniqueness
        if nm in pretty.values():
            k = 2
            while f'{nm} ({k})' in pretty.values(): k += 1
            nm = f'{nm} ({k})'
        pretty[g] = nm
    return pretty


def main():
    print('loading Tasic vocab ...')
    tvocab, _ = tasic_marker_vocab()
    print(f'  {len(tvocab)} Tasic GABAergic marker tokens, e.g. '
          f'{sorted(list(tvocab))[:12]}')

    print('loading devvis h5ad (to memory) ...')
    a = ad.read_h5ad(H5)
    valid = set(a.var_names.astype(str))
    tvocab &= valid   # only Tasic genes present in this panel

    # ---- derive per-cell subcluster from the taxonomy ----
    print('mapping cluster_alias -> subcluster term ...')
    ann = pd.read_csv(f"{TAX}/cluster_to_cluster_annotation_membership.csv",
                      usecols=['cluster_annotation_term_set_name',
                               'cluster_annotation_term_name', 'cluster_alias'])
    sub_map = ann.loc[ann.cluster_annotation_term_set_name == 'subcluster']\
                 .set_index('cluster_alias')['cluster_annotation_term_name']
    a.obs['orig_subcluster'] = a.obs['cluster_alias'].map(sub_map).astype(str)
    n_cl  = a.obs['orig_cluster'].nunique()
    n_scl = a.obs['orig_subcluster'].nunique()
    print(f'  clusters={n_cl}  subclusters={n_scl}')

    # normalise for DE (X is ABC log2; make it comparable + HVG-free t-test)
    a.obs['orig_cluster'] = a.obs['orig_cluster'].astype('category')
    a.obs['orig_subcluster'] = a.obs['orig_subcluster'].astype('category')

    print('naming CLUSTER level ...')
    cl_map = name_level(a, 'orig_cluster', tvocab, valid, one_gene_ok=True)
    print('naming SUBCLUSTER level ...')
    scl_map = name_level(a, 'orig_subcluster', tvocab, valid, one_gene_ok=False)

    a.obs['cell_cluster_named']    = a.obs['orig_cluster'].map(cl_map).astype(str)
    a.obs['cell_subcluster_named'] = a.obs['orig_subcluster'].map(scl_map).astype(str)

    print('\n===== CLUSTER names (%d) =====' % len(cl_map))
    for k in sorted(cl_map, key=lambda x: cl_map[x]):
        print(f'  {k:24s} -> {cl_map[k]:26s} n={int((a.obs.orig_cluster==k).sum())}')
    print('\n===== SUBCLUSTER names (%d) =====' % len(scl_map))
    for k in sorted(scl_map, key=lambda x: scl_map[x]):
        print(f'  {k:26s} -> {scl_map[k]:30s} n={int((a.obs.orig_subcluster==k).sum())}')

    a.write(H5)
    print(f'\nrewrote {H5} ({os.path.getsize(H5)/1e6:.0f} MB) '
          f'with cell_cluster_named + cell_subcluster_named')


if __name__ == '__main__':
    main()
