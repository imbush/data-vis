#!/usr/bin/env python
"""Process Fishell/Wu (GSE272706) cortical-interneuron snRNA-seq for the Fezf2
visualizer: Control vs Fezf2-KO at P20 (and P7 HET vs KO).

Joint embedding per age (NO cross-genotype batch correction — at P20 there is one
sample per genotype, so genotype is confounded with batch; correcting it would
erase the very effect we want to see). Clusters are annotated to interneuron
classes by canonical markers; non-IN/low-quality clusters are dropped. Saves a
processed AnnData per age with genotype, leiden cluster, IN class, and UMAP.
"""
import os, warnings, numpy as np, scanpy as sc
warnings.filterwarnings('ignore'); sc.settings.verbosity = 1
ROOT = '/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish'
D = os.path.join(ROOT, 'data', 'fezf2_fishell')

# canonical interneuron-class markers (mouse)
CLASS_MARKERS = {
    'Pvalb':  ['Pvalb', 'Tac1'],
    'Sst':    ['Sst', 'Grin3a'],
    'Sst Chodl': ['Sst', 'Chodl', 'Nos1'],
    'Vip':    ['Vip', 'Adarb2'],
    'Lamp5':  ['Lamp5', 'Id2', 'Ndnf'],
    'Sncg':   ['Sncg', 'Cck'],
}
# contamination / non-IN markers to flag clusters for removal
NONIN = {'Excitatory': ['Slc17a7', 'Satb2'], 'Astro': ['Aqp4', 'Gja1'],
         'Oligo': ['Mbp', 'Plp1'], 'OPC': ['Pdgfra'], 'Micro': ['Cx3cr1', 'Csf1r'],
         'Endo': ['Cldn5', 'Flt1']}
GABA = ['Gad1', 'Gad2', 'Slc32a1']


def load(gsm_file, genotype, age, sample):
    a = sc.read_10x_h5(os.path.join(D, f'{gsm_file}_filtered_feature_bc_matrix.h5'))
    a.var_names_make_unique()
    a.obs['genotype'] = genotype; a.obs['age'] = age; a.obs['sample'] = sample
    a.obs_names = [f'{sample}_{bc}' for bc in a.obs_names]
    return a


def process(samples, tag):
    print(f'\n===== {tag} =====')
    A = sc.concat([load(*s) for s in samples], join='outer')
    A.layers['counts'] = A.X.copy()
    print(f'  loaded {A.n_obs} nuclei x {A.n_vars} genes; per sample:',
          dict(A.obs['sample'].value_counts()))

    # QC
    A.var['mt'] = A.var_names.str.startswith('mt-')
    sc.pp.calculate_qc_metrics(A, qc_vars=['mt'], inplace=True, percent_top=None)
    A = A[(A.obs.n_genes_by_counts >= 800) & (A.obs.n_genes_by_counts <= 9000)
          & (A.obs.pct_counts_mt < 5)].copy()
    sc.pp.filter_genes(A, min_cells=10)
    print(f'  after QC: {A.n_obs} nuclei x {A.n_vars} genes')

    # normalize + embed (joint, no genotype correction)
    sc.pp.normalize_total(A, target_sum=1e4); sc.pp.log1p(A)
    A.raw = A
    sc.pp.highly_variable_genes(A, n_top_genes=2000, flavor='seurat', batch_key='sample')
    Ah = A[:, A.var.highly_variable].copy()
    sc.pp.scale(Ah, max_value=10)
    sc.tl.pca(Ah, n_comps=30, random_state=0)
    sc.pp.neighbors(Ah, n_neighbors=15, random_state=0)
    sc.tl.leiden(Ah, resolution=1.2, random_state=0, flavor='igraph', n_iterations=2, directed=False)
    sc.tl.umap(Ah, random_state=0)
    A.obs['leiden'] = Ah.obs['leiden'].values
    A.obsm['X_umap'] = Ah.obsm['X_umap']
    A.obsm['X_pca'] = Ah.obsm['X_pca']
    print(f'  {A.obs.leiden.nunique()} leiden clusters')

    # ---- annotate clusters ----
    def score(genes):
        g = [x for x in genes if x in A.raw.var_names]
        if not g: return np.zeros(A.n_obs)
        X = A.raw[:, g].X
        X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
        return X.mean(1).ravel()
    import pandas as pd
    sc_df = pd.DataFrame({k: score(v) for k, v in {**CLASS_MARKERS, **NONIN, 'GABA': GABA}.items()},
                         index=A.obs_names)
    sc_df['leiden'] = A.obs['leiden'].values
    cm = sc_df.groupby('leiden').mean()
    in_classes = list(CLASS_MARKERS)
    nonin_classes = list(NONIN)
    label, keep_clusters = {}, []
    for cl, row in cm.iterrows():
        gaba = row['GABA']; best_in = row[in_classes].idxmax(); best_nonin = row[nonin_classes].idxmax()
        # keep as interneuron if GABAergic and IN-marker beats non-IN marker
        if gaba > 0.15 and row[best_in] >= row[best_nonin]:
            label[cl] = best_in; keep_clusters.append(cl)
        else:
            label[cl] = f'drop:{best_nonin}'
    A.obs['cell_class'] = A.obs['leiden'].map(label).astype(str)
    print('  cluster -> class:'); print(cm.round(2).assign(label=[label[c] for c in cm.index]).to_string())

    IN = A[A.obs['leiden'].isin(keep_clusters)].copy()
    print(f'  kept {IN.n_obs} interneurons ({A.n_obs-IN.n_obs} dropped non-IN/low-qual)')

    # ---- proportions per genotype ----
    print('\n  === IN class proportions by genotype (%) ===')
    ct = pd.crosstab(IN.obs['cell_class'], IN.obs['genotype'], normalize='columns') * 100
    print(ct.round(1).to_string())

    out = os.path.join(D, f'{tag}_processed.h5ad')
    IN.write(out)
    print(f'  wrote {out} ({os.path.getsize(out)/1e6:.0f} MB)')
    return IN


if __name__ == '__main__':
    process([('GSM8409498_P20cIN', 'Control', 'P20', 'P20_Ctrl'),
             ('GSM8409499_P20cIN_Fezf2KO', 'Fezf2-KO', 'P20', 'P20_Fezf2KO')], 'p20_cIN')
