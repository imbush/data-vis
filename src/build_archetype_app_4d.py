#!/usr/bin/env python
"""Build a self-contained HTML app for exploring a K=4 archetype NMF of a
GABAergic cell-type subset of the Tasic 2018 V1 dataset.

Usage:
  python build_lamp5_archetype_app_4d.py [GROUP]

GROUP is one of: Lamp5 (default), Sst, Pvalb, Vip, Sncg.

The app shows:
  - Cells embedded inside a 3D tetrahedron (W barycentric, 4 archetypes)
  - Genes embedded inside a second 3D tetrahedron (H barycentric)
  - Hover a gene -> cells recolor by that gene's expression
  - Hover a cell -> genes recolor by that cell's expression of each gene
  - Both pyramids are fully rotatable (Plotly built-in trackball / scene controls)

Output: notebooks/{group}_archetype_explorer_4d.html (open in any browser).
"""
import os, sys, json, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.decomposition import non_negative_factorization
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize as l2_normalize
from joblib import Parallel, delayed
import plotly.graph_objects as go
from plotly.io import to_html
from bokeh.palettes import Magma256, Category10, Category20, Set1, Set3

ROOT       = '/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish'
ANNDATA    = os.path.join(ROOT, 'data', 'v1_neurons_proc.h5ad')

# Per-group configuration. `cache` is the pickle that compute_group_cached produced
# in cleaned_cascade_subtype_order.ipynb. `subclasses` selects rows from anndata for
# the broader projection. `cache_subtype_prefix` (if set) further filters the cache
# to that subtype prefix — used to split the joint PV+Sst cache into Sst / Pvalb.
# `cache_outliers` are exclusions already applied at cache build (kept here for the
# "about" panel). `runtime_exclude` are subtype labels dropped right after loading
# the cache (used for late-discovered outliers like Lamp5 Krt73).
GROUPS = {
    'Lamp5': dict(
        cache='Lamp5.pkl',
        subclasses=('Lamp5',),
        cache_subtype_prefix=None,
        cache_outliers=('Lamp5 Lhx6',),
        runtime_exclude=('Lamp5 Krt73',),
    ),
    'Sst': dict(
        cache='PV_Sst.pkl',
        subclasses=('Sst',),
        cache_subtype_prefix=('Sst',),
        cache_outliers=('Sst Chodl',),
        runtime_exclude=(),
    ),
    'Pvalb': dict(
        cache='PV_Sst.pkl',
        subclasses=('Pvalb',),
        cache_subtype_prefix=('Pvalb',),
        cache_outliers=('Pvalb Vipr2', 'Pvalb Gabrg1'),
        runtime_exclude=(),
    ),
    'Vip': dict(
        cache='Vip.pkl',
        subclasses=('Vip',),
        cache_subtype_prefix=None,
        cache_outliers=(),
        runtime_exclude=(),
    ),
    'Sncg': dict(
        cache='Sncg.pkl',
        subclasses=('Sncg', 'Serpinf1'),
        cache_subtype_prefix=None,
        cache_outliers=(),
        runtime_exclude=(),
    ),
}

GROUP_NAME = (sys.argv[1] if len(sys.argv) > 1 else 'Lamp5')
if GROUP_NAME not in GROUPS:
    sys.exit(f'unknown group {GROUP_NAME!r}; choose from {list(GROUPS)}')
GROUP   = GROUPS[GROUP_NAME]
SLUG    = GROUP_NAME.lower()
CACHE      = os.path.join(ROOT, 'notebooks', 'cache', GROUP['cache'])
PROJ_CACHE = os.path.join(ROOT, 'notebooks', 'cache', f'{GROUP_NAME}_archetype_proj.pkl')
OUT        = os.path.join(ROOT, 'notebooks', f'{SLUG}_archetype_explorer_4d.html')

K = 4
N_RESTARTS = 20
SEED = 0

# Broader gene panel: top-N genes by mean log-CPM in Lamp5 (∪ panel HVG so the
# panel is always included). Cap chosen to keep the embedded expression matrix
# (n_cells × n_genes × 1 byte) small enough that the self-contained HTML stays
# in the ~10 MB range and renders quickly.
BROAD_MAX_GENES = 3000


# Curated gene sets — mouse symbols (intersected with the displayed gene pool at
# build time). Liberal so even after HVG-cleaning a few markers land in each bucket.
GENE_SETS = {
    'marker_genes': [
        # pan-GABAergic / pan-glutamatergic contrasts
        'Gad1', 'Gad2', 'Slc32a1', 'Slc6a1',
        'Slc17a6', 'Slc17a7',
        # cardinal interneuron classes
        'Lamp5', 'Vip', 'Sst', 'Pvalb', 'Sncg', 'Ndnf', 'Cck',
        # Lamp5 subtype-distinguishing
        'Krt73', 'Lhx6', 'Vax1', 'Fam19a1', 'Lsp1',
        'Plch2', 'Ntn1', 'Pax6', 'Dock5', 'Tmem182',
        'Pdlim5', 'Spp1', 'Aldh1a3', 'Slc35d3',
        # lineage TFs
        'Adarb2', 'Prox1', 'Nkx2-1', 'Sox6', 'Mafb', 'Maf',
        # other IN markers / contrasts
        'Reln', 'Calb1', 'Calb2', 'Pthlh', 'Tac1', 'Tac2',
        'Cbln1', 'Cbln4', 'Cnr1', 'Nos1', 'Th', 'Crh', 'Crhbp', 'Vipr2',
        # cortical layer / excitatory contrast
        'Cux1', 'Cux2', 'Rorb', 'Tbr1', 'Satb2', 'Fezf2', 'Foxp2',
        # activity-regulated
        'Fos', 'Junb', 'Egr1', 'Npas4', 'Arc', 'Bdnf',
    ],
    'neuromodulators': [
        # cholinergic
        'Chat', 'Slc18a3', 'Slc5a7', 'Ache',
        # catecholamines
        'Th', 'Ddc', 'Dbh', 'Pnmt', 'Slc18a1', 'Slc18a2', 'Slc6a2', 'Slc6a3',
        # serotonin
        'Tph1', 'Tph2', 'Slc6a4',
        # histamine
        'Hdc', 'Hnmt',
        # NO / gas
        'Nos1', 'Nos2', 'Nos3',
        # endocannabinoid
        'Dagla', 'Daglb', 'Faah', 'Mgll', 'Naaa', 'Napepld',
        # GABA / glutamate / glycine machinery
        'Gad1', 'Gad2', 'Slc32a1', 'Slc17a6', 'Slc17a7', 'Slc17a8', 'Slc6a5',
    ],
    'neuropeptides': [
        'Vip', 'Sst', 'Cck', 'Npy', 'Tac1', 'Tac2', 'Tac3',
        'Crh', 'Pdyn', 'Penk', 'Pomc', 'Gal', 'Pnoc', 'Nts',
        'Cort', 'Adcyap1', 'Cartpt', 'Trh', 'Avp', 'Oxt',
        'Ucn', 'Ucn1', 'Ucn2', 'Ucn3',
        'Nppa', 'Nppb', 'Nppc', 'Vgf', 'Grp', 'Pthlh', 'Edn1', 'Edn3',
        'Reln', 'Bdnf', 'Igf1', 'Igf2', 'Ngf', 'Npff', 'Npb', 'Npw',
        'Npvf', 'Gnrh1', 'Gip', 'Glp1', 'Apln', 'Adm', 'Tafa1', 'Tafa2',
        'Tafa3', 'Tafa4', 'Tafa5',
    ],
    'neuromodulator_receptors': [
        # cholinergic — nicotinic (subunits) + muscarinic
        'Chrna1', 'Chrna2', 'Chrna3', 'Chrna4', 'Chrna5', 'Chrna6', 'Chrna7',
        'Chrna9', 'Chrna10',
        'Chrnb1', 'Chrnb2', 'Chrnb3', 'Chrnb4',
        'Chrnd', 'Chrne', 'Chrng',
        'Chrm1', 'Chrm2', 'Chrm3', 'Chrm4', 'Chrm5',
        # adrenergic
        'Adra1a', 'Adra1b', 'Adra1d', 'Adra2a', 'Adra2b', 'Adra2c',
        'Adrb1', 'Adrb2', 'Adrb3',
        # dopamine
        'Drd1', 'Drd2', 'Drd3', 'Drd4', 'Drd5',
        # serotonin (5-HT)
        'Htr1a', 'Htr1b', 'Htr1d', 'Htr1f', 'Htr2a', 'Htr2b', 'Htr2c',
        'Htr3a', 'Htr3b', 'Htr4', 'Htr5a', 'Htr5b', 'Htr6', 'Htr7',
        # histamine
        'Hrh1', 'Hrh2', 'Hrh3', 'Hrh4',
        # cannabinoid
        'Cnr1', 'Cnr2',
        # adenosine
        'Adora1', 'Adora2a', 'Adora2b', 'Adora3',
        # purinergic (P2X + P2Y)
        'P2rx1', 'P2rx2', 'P2rx3', 'P2rx4', 'P2rx5', 'P2rx6', 'P2rx7',
        'P2ry1', 'P2ry2', 'P2ry4', 'P2ry6', 'P2ry10',
        'P2ry12', 'P2ry13', 'P2ry14',
        # trace amine
        'Taar1', 'Taar2', 'Taar3', 'Taar4', 'Taar5', 'Taar6', 'Taar7a',
        'Taar8a', 'Taar9',
    ],
    'neuropeptide_receptors': [
        'Cckar', 'Cckbr',
        'Npy1r', 'Npy2r', 'Npy4r', 'Npy5r', 'Npy6r',
        'Sstr1', 'Sstr2', 'Sstr3', 'Sstr4', 'Sstr5',
        'Vipr1', 'Vipr2',
        'Galr1', 'Galr2', 'Galr3',
        'Crhr1', 'Crhr2',
        'Tacr1', 'Tacr2', 'Tacr3',
        'Oprm1', 'Oprk1', 'Oprd1', 'Oprl1',
        'Adcyap1r1',
        'Mc1r', 'Mc3r', 'Mc4r', 'Mc5r',
        'Mchr1', 'Mchr2',
        'Nmbr', 'Nmur1', 'Nmur2',
        'Hcrtr1', 'Hcrtr2',
        'Trhr',
        'Glp1r', 'Glp2r',
        'Avpr1a', 'Avpr1b', 'Avpr2', 'Oxtr',
        'Prokr1', 'Prokr2',
        'Bdkrb1', 'Bdkrb2',
        'Ednra', 'Ednrb',
    ],
    'synaptic': [
        # vesicle / SNARE / active zone
        'Syp', 'Synpr', 'Sypl1', 'Sypl2',
        'Syn1', 'Syn2', 'Syn3',
        'Snap25', 'Snap47', 'Stx1a', 'Stx1b', 'Stx2', 'Stx3',
        'Vamp1', 'Vamp2', 'Vamp3', 'Vamp4',
        'Sv2a', 'Sv2b', 'Sv2c',
        'Cplx1', 'Cplx2', 'Cplx3', 'Cplx4',
        'Rab3a', 'Rab3b', 'Rab3c', 'Rab3d',
        'Rims1', 'Rims2', 'Rims3', 'Rims4',
        'Rimbp2', 'Bsn', 'Pclo', 'Unc13a', 'Unc13b', 'Unc13c', 'Stxbp1',
        # synaptotagmins (regulated release / Ca sensors)
        'Syt1', 'Syt2', 'Syt3', 'Syt4', 'Syt5', 'Syt6', 'Syt7',
        'Syt9', 'Syt10', 'Syt11', 'Syt12', 'Syt13',
        'Syt14', 'Syt15', 'Syt16', 'Syt17',
        # ionotropic glutamate
        'Gria1', 'Gria2', 'Gria3', 'Gria4',
        'Grik1', 'Grik2', 'Grik3', 'Grik4', 'Grik5',
        'Grin1', 'Grin2a', 'Grin2b', 'Grin2c', 'Grin2d', 'Grin3a', 'Grin3b',
        # metabotropic glutamate
        'Grm1', 'Grm2', 'Grm3', 'Grm4', 'Grm5', 'Grm7', 'Grm8',
        # GABA
        'Gabra1', 'Gabra2', 'Gabra3', 'Gabra4', 'Gabra5', 'Gabra6',
        'Gabrb1', 'Gabrb2', 'Gabrb3',
        'Gabrg1', 'Gabrg2', 'Gabrg3',
        'Gabrd', 'Gabre', 'Gabrp', 'Gabrq',
        'Gabbr1', 'Gabbr2',
        # scaffolds / postsynaptic
        'Dlg1', 'Dlg2', 'Dlg3', 'Dlg4',
        'Shank1', 'Shank2', 'Shank3',
        'Homer1', 'Homer2', 'Homer3',
        'Gphn',
        # adhesion
        'Nrxn1', 'Nrxn2', 'Nrxn3',
        'Nlgn1', 'Nlgn2', 'Nlgn3',
        'Cbln1', 'Cbln2', 'Cbln4',
        'Lrrtm1', 'Lrrtm2', 'Lrrtm3', 'Lrrtm4',
        'Cadm1', 'Cadm2', 'Cadm3',
        'Slitrk1', 'Slitrk2', 'Slitrk3', 'Slitrk4', 'Slitrk5', 'Slitrk6',
    ],
    'axon_guidance': [
        'Slit1', 'Slit2', 'Slit3',
        'Robo1', 'Robo2', 'Robo3', 'Robo4',
        'Sema3a', 'Sema3b', 'Sema3c', 'Sema3d', 'Sema3e', 'Sema3f', 'Sema3g',
        'Sema4a', 'Sema4b', 'Sema4c', 'Sema4d', 'Sema4f', 'Sema4g',
        'Sema5a', 'Sema5b',
        'Sema6a', 'Sema6b', 'Sema6c', 'Sema6d',
        'Sema7a',
        'Plxna1', 'Plxna2', 'Plxna3', 'Plxna4',
        'Plxnb1', 'Plxnb2', 'Plxnb3',
        'Plxnc1', 'Plxnd1',
        'Nrp1', 'Nrp2',
        'Dcc', 'Ntn1', 'Ntn3', 'Ntn4', 'Ntn5',
        'Ntng1', 'Ntng2',
        'Unc5a', 'Unc5b', 'Unc5c', 'Unc5d',
        'Epha1', 'Epha2', 'Epha3', 'Epha4', 'Epha5', 'Epha6', 'Epha7', 'Epha8', 'Epha10',
        'Ephb1', 'Ephb2', 'Ephb3', 'Ephb4', 'Ephb6',
        'Efna1', 'Efna2', 'Efna3', 'Efna4', 'Efna5',
        'Efnb1', 'Efnb2', 'Efnb3',
        'Dscam', 'Dscaml1', 'L1cam',
        'Cntn1', 'Cntn2', 'Cntn3', 'Cntn4', 'Cntn5', 'Cntn6',
        'Cntnap1', 'Cntnap2', 'Cntnap3', 'Cntnap4', 'Cntnap5',
        'Fzd3', 'Wnt5a', 'Wnt5b', 'Wnt7a', 'Wnt7b',
        'Gli3',
    ],
}

GENE_SET_LABELS = {
    'all': 'All',
    'marker_genes': 'Markers',
    'neuromodulators': 'Neuromodulators',
    'neuromodulator_receptors': 'NM receptors',
    'neuropeptides': 'Neuropeptides',
    'neuropeptide_receptors': 'NP receptors',
    'synaptic': 'Synaptic',
    'axon_guidance': 'Axon guidance',
}

GENE_SET_ORDER = ['all', 'marker_genes',
                  'neuromodulators', 'neuromodulator_receptors',
                  'neuropeptides', 'neuropeptide_receptors',
                  'synaptic', 'axon_guidance']


def consensus_nmf(X, K, n_iter, seed):
    def one(seed_r):
        W, H, _ = non_negative_factorization(
            X, n_components=K, beta_loss='frobenius', init='random',
            solver='cd', max_iter=400, tol=1e-4, random_state=seed_r)
        return l2_normalize(H, norm='l2', axis=1)
    H_list = Parallel(n_jobs=-1, backend='loky', verbose=0)(
        delayed(one)(seed + r) for r in range(n_iter))
    H_all = np.vstack(H_list).astype(np.float32)
    km = KMeans(n_clusters=K, n_init=20, random_state=seed).fit(H_all)
    H_med = np.zeros((K, X.shape[1]), dtype=X.dtype)
    for k in range(K):
        m = H_all[km.labels_ == k]
        H_med[k] = np.median(m, axis=0).astype(X.dtype)
    H_med[H_med < 0] = 0
    W, _, _ = non_negative_factorization(
        X, n_components=K, init='custom', H=H_med, beta_loss='frobenius',
        solver='cd', max_iter=400, tol=1e-4, update_H=False, random_state=seed)
    from sklearn.metrics import silhouette_score
    sil = silhouette_score(H_all, km.labels_, sample_size=min(2000, len(H_all)))
    return W, H_med, sil


def compute_or_load_proj(force=False):
    """Fit panel NMF, then NNLS-project broader-transcriptome genes onto the
    panel-derived W. Returns a dict with everything the viz needs.

    Cached because NMF + 2-3k NNLS solves take ~30-60 s; the cache load is <1 s.
    """
    if not force and os.path.exists(PROJ_CACHE):
        with open(PROJ_CACHE, 'rb') as f:
            proj = pickle.load(f)
        print(f'  loaded projection cache: {len(proj["gene_names"])} genes '
              f'({int(np.array(proj["in_panel"]).sum())} panel + '
              f'{len(proj["gene_names"]) - int(np.array(proj["in_panel"]).sum())} imputed)')
        return proj

    import scanpy as sc, scipy.sparse as ss
    from scipy.optimize import nnls
    sc.settings.verbosity = 0

    out_excl = list(GROUP['cache_outliers']) + list(GROUP['runtime_exclude'])
    msg = f'loading {GROUP_NAME} cache'
    if out_excl: msg += f' (excluding {", ".join(out_excl)})'
    print(msg + ' ...')
    with open(CACHE, 'rb') as f: d = pickle.load(f)
    subs_full = np.array(d['subs']).astype(str)
    keep = np.ones(len(subs_full), dtype=bool)
    if GROUP['cache_subtype_prefix']:
        keep &= np.array([s.startswith(GROUP['cache_subtype_prefix'])
                          for s in subs_full])
    for excl in GROUP['runtime_exclude']:
        keep &= subs_full != excl
    X_panel = np.asarray(d['X_cleaned'][keep], dtype=np.float32)
    subs    = subs_full[keep]
    cleaned = list(d['cleaned'])
    n_cells, n_panel = X_panel.shape
    print(f'  {n_cells} cells × {n_panel} panel HVG; '
          f'subtypes: {sorted(set(subs.tolist()))}')

    sd_panel = X_panel.std(0) + 1e-9
    Xs       = X_panel / sd_panel
    print(f'running consensus NMF K={K} ({N_RESTARTS} inits) ...')
    W, H_panel, sil = consensus_nmf(Xs, K, N_RESTARTS, SEED)
    print(f'  W: {W.shape}, H: {H_panel.shape}, stability silhouette: {sil:.3f}')

    print('loading broader anndata (v1_neurons_proc.h5ad) ...')
    a = sc.read_h5ad(ANNDATA)
    a = a[a.obs['cell_class'] == 'GABAergic'].copy()
    a = a[a.obs['cell_subclass'].astype(str).isin(set(GROUP['subclasses']))].copy()
    if GROUP['cache_outliers']:
        a = a[~a.obs['cell_cluster'].astype(str).str.startswith(
              tuple(GROUP['cache_outliers']))].copy()
    if GROUP['cache_subtype_prefix']:
        a = a[a.obs['cell_cluster'].astype(str).str.startswith(
              tuple(GROUP['cache_subtype_prefix']))].copy()
    for excl in GROUP['runtime_exclude']:
        a = a[a.obs['cell_cluster'].astype(str) != excl].copy()
    assert a.n_obs == n_cells, f'cell count mismatch: anndata {a.n_obs} vs cache {n_cells}'
    assert np.array_equal(
        np.array(a.obs['cell_cluster'].astype(str)), subs
    ), 'cell ordering mismatch between cache and anndata after filter'

    # Re-normalize to log-CPM (matching the cache-builder's normalization)
    if 'counts' in a.layers:
        a.X = a.layers['counts'].copy()
        sc.pp.normalize_total(a, target_sum=1e6); sc.pp.log1p(a)

    X_full = a.X.toarray() if ss.issparse(a.X) else np.asarray(a.X)
    X_full = X_full.astype(np.float32)
    gene_names_full = list(a.var_names)
    print(f'  broader matrix: {X_full.shape}')

    mean_full = X_full.mean(0)
    std_full  = X_full.std(0)

    cleaned_set = set(cleaned)
    panel_flag  = np.array([g in cleaned_set for g in gene_names_full])
    # Always include every curated gene-set member (NPRs, neuropeptides, ...)
    # — receptors are typically low-abundance and miss top-N-by-mean thresholds
    # but are exactly the genes the user wants to inspect via the set buttons.
    curated_union = set()
    for gl in GENE_SETS.values():
        curated_union.update(gl)
    curated_mask = np.array([g in curated_union for g in gene_names_full])
    # Take top-N by mean log-CPM ∪ panel ∪ curated sets
    rank_thr_idx = np.argsort(mean_full)[::-1][:BROAD_MAX_GENES]
    top_mask    = np.zeros(len(mean_full), dtype=bool)
    top_mask[rank_thr_idx] = True
    keep_mask   = top_mask | panel_flag | curated_mask
    n_keep      = int(keep_mask.sum())
    print(f'  keep mask: {n_keep} genes '
          f'(top {BROAD_MAX_GENES} by mean log-CPM ∪ panel HVG ∪ curated sets; '
          f'curated forced in: {int((curated_mask & ~top_mask & ~panel_flag).sum())})')

    gene_names = [g for g, k in zip(gene_names_full, keep_mask) if k]
    X_keep     = X_full[:, keep_mask]
    mean_keep  = mean_full[keep_mask]
    std_keep   = std_full[keep_mask]
    in_panel   = np.array([g in cleaned_set for g in gene_names])
    n_keep_panel = int(in_panel.sum())
    n_keep_broad = int((~in_panel).sum())
    print(f'  in panel: {n_keep_panel}; broader (NNLS-imputed): {n_keep_broad}')

    # Build H_all: panel genes inherit H_panel; broader genes get NNLS projection
    # in the same standardized units (X_keep / std_keep), so loadings are comparable.
    panel_lookup = {g: i for i, g in enumerate(cleaned)}
    H_all = np.zeros((K, n_keep), dtype=np.float32)
    nnls_idx = []
    for j, g in enumerate(gene_names):
        if g in panel_lookup:
            H_all[:, j] = H_panel[:, panel_lookup[g]]
        else:
            nnls_idx.append(j)

    print(f'  NNLS-projecting {len(nnls_idx)} broader genes onto W ...')
    import time as _t
    Xs_keep = X_keep / (std_keep + 1e-9)
    t0 = _t.time()
    for ji, j in enumerate(nnls_idx):
        H_all[:, j], _ = nnls(W, Xs_keep[:, j])
        if (ji + 1) % 500 == 0:
            print(f'    {ji+1}/{len(nnls_idx)}  ({(_t.time()-t0):.1f}s)')
    print(f'  NNLS done ({(_t.time()-t0):.1f}s)')

    proj = dict(
        W=W.astype(np.float32),
        H_all=H_all,
        H_panel=H_panel.astype(np.float32),
        sil=float(sil),
        gene_names=gene_names,
        cleaned=cleaned,
        in_panel=in_panel.tolist(),
        mean_expr=mean_keep.astype(np.float32),
        std_expr=std_keep.astype(np.float32),
        X_keep=X_keep,        # log-CPM, used for hover recoloring
        subs=subs,
    )
    with open(PROJ_CACHE, 'wb') as f:
        pickle.dump(proj, f, protocol=4)
    print(f'  cached -> {PROJ_CACHE} ({os.path.getsize(PROJ_CACHE)/1e6:.1f} MB)')
    return proj


def main():
    proj      = compute_or_load_proj()
    W         = proj['W']
    H_all     = proj['H_all']
    H_panel   = proj['H_panel']
    cleaned   = proj['cleaned']
    gene_names = proj['gene_names']
    in_panel  = np.array(proj['in_panel'])
    mean_expr = np.asarray(proj['mean_expr'])
    std_expr  = np.asarray(proj['std_expr'])
    X_keep    = np.asarray(proj['X_keep'])
    subs      = np.array(proj['subs'])
    sil       = proj['sil']
    n_cells   = X_keep.shape[0]
    n_genes   = len(gene_names)

    arch_top = []
    for k in range(K):
        top = pd.Series(H_panel[k], index=cleaned).sort_values(ascending=False).head(8).index.tolist()
        arch_top.append(top)
        print(f'  A{k+1} top: {top[:5]}')

    # Tetrahedron vertices (regular, centered at origin, on unit sphere)
    V = np.array([
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
    ], dtype=float) / np.sqrt(3)

    W_bary    = W / (W.sum(1, keepdims=True) + 1e-9)
    Hcol      = H_all / (H_all.sum(0, keepdims=True) + 1e-9)
    gene_bary = Hcol.T
    cell_xyz  = W_bary @ V
    gene_xyz  = gene_bary @ V

    # Default colors
    cats = sorted(set(subs.tolist()))
    base = list(Category20[20]) + list(Set3[12]) + list(Set1[9]) + list(Category10[10])
    # Cycle if a group has > 51 subtypes (none does, but be defensive)
    subtype_palette = {c: base[i % len(base)] for i, c in enumerate(cats)}
    cell_color_default = [subtype_palette[s] for s in subs]
    archetype_palette  = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']
    gene_dom           = gene_bary.argmax(1)
    gene_color_default = [archetype_palette[k] for k in gene_dom]
    cell_dom           = W_bary.argmax(1)
    cell_dom_color     = [archetype_palette[k] for k in cell_dom]
    gene_dom_color     = gene_color_default     # alias for clarity in JS

    # Expression matrix for cell-hover gene recoloring. Stored as int (= log-CPM × 10)
    # so JSON is ~3× smaller than serializing float32-converted-to-float64 values.
    # JS divides by EXPR_SCALE to recover the original log-CPM value.
    EXPR_SCALE = 10
    expr_matrix = np.round(X_keep * EXPR_SCALE).astype(np.int16).tolist()

    # Loading-dot positions just outside vertices (so they sit between the vertex
    # label and the camera, where the user can see them without going inside the data)
    LOAD_SCALE = 1.22
    load_xyz   = V * LOAD_SCALE

    def build_fig(xyz, colors, hover_text, title, point_label):
        edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        ex, ey, ez = [], [], []
        for i, j in edges:
            ex += [V[i,0], V[j,0], None]
            ey += [V[i,1], V[j,1], None]
            ez += [V[i,2], V[j,2], None]
        edge_trace = go.Scatter3d(
            x=ex, y=ey, z=ez, mode='lines',
            line=dict(color='lightgray', width=2),
            hoverinfo='skip', showlegend=False)
        vertex_labels = [f'A{k+1}<br>({arch_top[k][0]})' for k in range(K)]
        vertex_trace = go.Scatter3d(
            x=V[:,0], y=V[:,1], z=V[:,2],
            mode='markers+text',
            marker=dict(size=6, color='black'),
            text=vertex_labels, textposition='top center',
            textfont=dict(size=12, color='black'),
            hoverinfo='text', hovertext=vertex_labels, showlegend=False)
        points_trace = go.Scatter3d(
            x=xyz[:,0], y=xyz[:,1], z=xyz[:,2],
            mode='markers',
            marker=dict(size=4, color=colors, opacity=0.85, line=dict(width=0)),
            text=hover_text, hoverinfo='text',
            name=point_label, showlegend=False)
        # Loading dots — default light gray; recolored on hover via JS
        loading_trace = go.Scatter3d(
            x=load_xyz[:,0], y=load_xyz[:,1], z=load_xyz[:,2],
            mode='markers',
            marker=dict(size=18, color=['#e0e0e0']*K, opacity=1.0,
                        line=dict(width=1.5, color='#222')),
            hoverinfo='text',
            hovertext=[f'A{k+1} loading' for k in range(K)],
            showlegend=False, name='loadings')
        fig = go.Figure(data=[edge_trace, vertex_trace, points_trace, loading_trace])
        # Widened range so loading dots at V*1.22 are visible
        lim = LOAD_SCALE / np.sqrt(3) * 1.10
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=13)),
            scene=dict(
                xaxis=dict(visible=False, range=[-lim, lim]),
                yaxis=dict(visible=False, range=[-lim, lim]),
                zaxis=dict(visible=False, range=[-lim, lim]),
                aspectmode='cube',
                dragmode='orbit',
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.4),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1),
                ),
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor='white',
            plot_bgcolor='white',
        )
        return fig

    cell_hover_text = [
        f'#{i}<br>subtype: {subs[i]}<br>'
        f'(A1,A2,A3,A4) = ({W_bary[i,0]:.2f}, {W_bary[i,1]:.2f}, {W_bary[i,2]:.2f}, {W_bary[i,3]:.2f})'
        for i in range(n_cells)
    ]
    gene_hover_text = [
        f'<b>{gene_names[j]}</b>'
        + (' (panel HVG)' if in_panel[j] else ' (imputed)')
        + f'<br>dominant: A{gene_dom[j]+1} ({arch_top[gene_dom[j]][0]})<br>'
        + f'mean={mean_expr[j]:.2f}, std={std_expr[j]:.2f}<br>'
        + f'(A1,A2,A3,A4) = ({gene_bary[j,0]:.2f}, {gene_bary[j,1]:.2f}, '
        + f'{gene_bary[j,2]:.2f}, {gene_bary[j,3]:.2f})'
        for j in range(n_genes)
    ]

    n_panel_disp = int(in_panel.sum())
    n_imputed    = n_genes - n_panel_disp

    excluded_subtypes = list(GROUP['cache_outliers']) + list(GROUP['runtime_exclude'])
    excluded_blurb = (f'Outlier subtypes excluded: {", ".join(excluded_subtypes)}.'
                      if excluded_subtypes
                      else 'No subtype-level outliers excluded.')

    fig_cells = build_fig(cell_xyz, cell_color_default, cell_hover_text,
                          f'Cells — W barycentric (n={n_cells})  '
                          f'<i>hover → genes recolor; A1–A4 dots = this cell\'s archetype loadings</i>',
                          'cells')
    fig_genes = build_fig(gene_xyz, gene_color_default, gene_hover_text,
                          f'Genes — H barycentric ({n_panel_disp} panel + {n_imputed} imputed)  '
                          f'<i>hover → cells recolor; A1–A4 dots = this gene\'s archetype loadings</i>',
                          'genes')

    cells_html = to_html(fig_cells, include_plotlyjs='cdn', full_html=False,
                          div_id='cell-plot', config={'displayModeBar': True, 'responsive': True})
    genes_html = to_html(fig_genes, include_plotlyjs=False, full_html=False,
                          div_id='gene-plot', config={'displayModeBar': True, 'responsive': True})

    # Legends
    sub_legend = ''.join(
        f'<span style="display:inline-block;width:10px;height:10px;background:{subtype_palette[s]};'
        f'margin-right:4px;border-radius:50%;"></span> {s} &nbsp;&nbsp;'
        for s in cats)
    arch_legend = ''.join(
        f'<span style="display:inline-block;width:10px;height:10px;background:{archetype_palette[k]};'
        f'margin-right:4px;border-radius:50%;"></span> A{k+1} ({arch_top[k][0]}) &nbsp;&nbsp;'
        for k in range(K))

    # Pass gene xyz separately so the filter can rewrite x/y/z in place
    gx, gy, gz = (np.round(gene_xyz[:, k], 4).tolist() for k in range(3))
    cell_load  = W_bary.round(4).tolist()      # n_cells × K — for hover-dot colors
    gene_load  = gene_bary.round(4).tolist()   # n_genes × K — for hover-dot colors

    # Gene-set masks built against the FULL gene_names list (panel + imputed),
    # plus two special "all" / "panel" pseudo-sets for the toggle.
    panel_mask_list = in_panel.tolist()
    set_masks  = {'all': [True] * n_genes, 'panel': panel_mask_list}
    set_counts = {'all': n_genes,         'panel': n_panel_disp}
    for name, gene_list in GENE_SETS.items():
        gset = set(gene_list)
        mask = [g in gset for g in gene_names]
        set_masks[name] = mask
        set_counts[name] = sum(mask)
        hits = [g for g in gene_names if g in gset]
        print(f'  gene set "{name}": {sum(mask)}/{n_genes} present  '
              f'(e.g. {hits[:6]})')

    # Slider ranges (mean + dispersion = std), padded slightly for headroom
    mean_min, mean_max = float(np.min(mean_expr)), float(np.max(mean_expr))
    std_min,  std_max  = float(np.min(std_expr)),  float(np.max(std_expr))

    js_data = (
        f"const EXPR_SCALE  = {EXPR_SCALE};\n"
        f"const expr_matrix = {json.dumps(expr_matrix)};\n"
        f"const cell_default_colors = {json.dumps(cell_color_default)};\n"
        f"const gene_default_colors = {json.dumps(gene_color_default)};\n"
        f"const cell_subtype = {json.dumps(subs.tolist())};\n"
        f"const gene_name    = {json.dumps(gene_names)};\n"
        f"const gene_in_panel = {json.dumps(panel_mask_list)};\n"
        f"const gene_mean    = {json.dumps([round(float(v), 3) for v in mean_expr])};\n"
        f"const gene_std     = {json.dumps([round(float(v), 3) for v in std_expr])};\n"
        f"const gene_x       = {json.dumps(gx)};\n"
        f"const gene_y       = {json.dumps(gy)};\n"
        f"const gene_z       = {json.dumps(gz)};\n"
        f"const cell_load    = {json.dumps(cell_load)};\n"
        f"const gene_load    = {json.dumps(gene_load)};\n"
        f"const cell_dom_color = {json.dumps(cell_dom_color)};\n"
        f"const gene_dom_color = {json.dumps(gene_dom_color)};\n"
        f"const gene_sets    = {json.dumps(set_masks)};\n"
        f"const gene_set_counts = {json.dumps(set_counts)};\n"
        f"const magma        = {json.dumps(list(Magma256))};\n"
    )

    button_order = ['panel', 'all'] + [n for n in GENE_SET_ORDER if n != 'all']
    set_label = lambda n: 'Panel HVG' if n == 'panel' else GENE_SET_LABELS.get(n, n)
    set_buttons_html = ''.join(
        f'<button class="set-btn{" active" if name == "panel" else ""}" '
        f'data-set="{name}"'
        f'{" disabled" if set_counts.get(name, 0) == 0 else ""}>'
        f'{set_label(name)} ({set_counts.get(name, 0)})</button>'
        for name in button_order
    )

    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{GROUP_NAME} 4-archetype explorer</title>
<style>
html, body {{ height: 100%; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
        display: flex; flex-direction: column; padding: 6px 12px; box-sizing: border-box; }}
h2 {{ margin: 0 0 2px 0; }}
.hint {{ flex: 0 0 auto; font-size: 17px; color: #222; line-height: 1.3;
         margin: 2px 0 6px 0; }}
.hint b {{ color: #1f77b4; }}
.header {{ flex: 0 0 auto; font-size:12px; color:#444; line-height:1.35; }}
.controls {{ flex: 0 0 auto; margin: 4px 0; display: flex; flex-direction: column;
             gap: 4px; font-size: 12px; }}
.controls-row {{ display: flex; align-items: center; flex-wrap: wrap; gap: 6px; }}
.controls-row .label {{ color: #555; font-weight: 600; }}
.set-btn {{ padding: 3px 8px; font-size: 12px; border: 1px solid #bbb;
            background: #f6f6f6; border-radius: 3px; cursor: pointer; }}
.set-btn:hover {{ background: #eee; }}
.set-btn.active {{ background: #1f77b4; color: white; border-color: #1f77b4; }}
#mean-slider, #std-slider {{ width: 180px; }}
.row {{ flex: 1 1 auto; display: flex; flex-direction: row; gap: 8px;
        min-height: 0; }}
.col {{ flex: 1 1 0; min-width: 0; display: flex; flex-direction: column; }}
.col > .plotly-graph-div {{ flex: 1 1 auto; min-height: 0; height: 100% !important; }}
#status {{ color: #555; font-size: 12px; }}
.legend {{ flex: 0 0 auto; font-size: 11px; color: #444; margin-top: 4px; }}
button {{ font-size: 13px; padding: 4px 10px; }}
details summary {{ cursor: pointer; color: #666; font-size: 12px; }}
</style>
</head>
<body>
<h2>{GROUP_NAME} 4-archetype NMF — rotatable tetrahedrons</h2>
<div class="hint">
<b>Hover over a cell on the left</b> to see its expression of the genes on the right.
<b>Hover over a gene on the right</b> to see its expression in the cells on the left.
</div>
<details class="header">
<summary>About this app (click to expand)</summary>
<div style="margin-top:4px;">
Cells ({n_cells}; subtype-coloured by default) embed in a 4-vertex tetrahedron whose
vertices are the 4 NMF archetypes (fit on {n_panel_disp} panel HVG). Consensus stability
silhouette: <b>{sil:.3f}</b>. Genes embed via column-normalised H: the {n_panel_disp} panel
HVG inherit their NMF H; the {n_imputed} broader genes (top by mean log-CPM in {GROUP_NAME})
get their loadings via NNLS projection onto the same W — labelled "imputed" on hover.<br>
<b>Hover</b> a gene → cells recolour by expression; <b>hover</b> a cell → genes recolour.
The 4 large dots outside each tetrahedron's vertices show that hovered point's archetype
loadings (magma: 0 black → 1 white).
Drag inside a pyramid to rotate; right-click + drag to pan; scroll to zoom.
Vertices: A1 = {arch_top[0][0]}, A2 = {arch_top[1][0]}, A3 = {arch_top[2][0]}, A4 = {arch_top[3][0]}.
{excluded_blurb}
</div>
</details>
<div class="controls">
  <div class="controls-row">
    <span class="label">Gene set:</span>
    {set_buttons_html}
  </div>
  <div class="controls-row">
    <span class="label">Min mean expr (log-CPM):</span>
    <input id="mean-slider" type="range" min="{mean_min:.3f}" max="{mean_max:.3f}"
           step="0.01" value="{mean_min:.3f}">
    <span id="mean-value">{mean_min:.2f}</span>
    <span class="label" style="margin-left:14px;">Min dispersion (std):</span>
    <input id="std-slider" type="range" min="{std_min:.3f}" max="{std_max:.3f}"
           step="0.01" value="{std_min:.3f}">
    <span id="std-value">{std_min:.2f}</span>
    <span class="label" style="margin-left:14px;">Visible:</span>
    <span id="visible-count">{n_panel_disp} / {n_genes}</span>
    <button id="reset-btn" style="margin-left:auto;">Reset colours</button>
  </div>
  <div class="controls-row">
    <span id="status">Hover a cell (left) or a gene (right) to colour by expression.
    Big dots outside the vertices show that point's archetype loadings (magma 0→1).</span>
  </div>
</div>
<div class="row">
  <div class="col">{cells_html}</div>
  <div class="col">{genes_html}</div>
</div>
<div class="legend">
<b>Cell default colours</b> (subtype): {sub_legend}<br>
<b>Gene default colours</b> (dominant archetype): {arch_legend}
</div>
<script>
{js_data}

function exprToMagma(values) {{
  // map an array of expression values → magma hex colors with auto range
  let lo = Infinity, hi = -Infinity;
  for (const v of values) {{ if (v < lo) lo = v; if (v > hi) hi = v; }}
  const range = (hi > lo) ? (hi - lo) : 1;
  return values.map(v => {{
    const t = (v - lo) / range;
    const idx = Math.max(0, Math.min(255, Math.round(255 * t)));
    return magma[idx];
  }});
}}

function loadingToMagma(loadings) {{
  // Loadings are barycentric (in [0,1] each, sum to 1). Map directly through magma
  // without auto-range so 0 → black, 1 → white (i.e. absolute scale, not relative).
  return loadings.map(v => {{
    const t = Math.max(0, Math.min(1, v));
    return magma[Math.round(255 * t)];
  }});
}}

const cellPlot = document.getElementById('cell-plot');
const genePlot = document.getElementById('gene-plot');
const status   = document.getElementById('status');

// Trace indices: edges=0, vertices=1, points=2, loading dots=3
const POINTS_TRACE   = 2;
const LOADING_TRACE  = 3;
const DEFAULT_LOAD_COLORS = ['#e0e0e0', '#e0e0e0', '#e0e0e0', '#e0e0e0'];

let lastHoveredCell = null;
let lastHoveredGene = null;

cellPlot.on('plotly_hover', function(data) {{
  const pt = data.points[0];
  if (pt.curveNumber !== POINTS_TRACE) return;
  const i = pt.pointNumber;
  if (lastHoveredCell === i) return;
  lastHoveredCell = i;
  const row = expr_matrix[i];
  const colors = exprToMagma(row);
  Plotly.restyle(genePlot, {{'marker.color': [colors]}}, [POINTS_TRACE]);
  const loadColors = loadingToMagma(cell_load[i]);
  Plotly.restyle(cellPlot, {{'marker.color': [loadColors]}}, [LOADING_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color': [loadColors]}}, [LOADING_TRACE]);
  let lo = Infinity, hi = -Infinity;
  for (const v of row) {{ if (v < lo) lo = v; if (v > hi) hi = v; }}
  const L = cell_load[i];
  const cTag = '<b style="color:' + cell_dom_color[i] + '">Cell #' + i
    + '</b> <span style="color:#555">(' + cell_subtype[i] + ')</span>';
  status.innerHTML = cTag + ' &nbsp; '
    + 'loadings A1..A4 = (' + L[0].toFixed(2) + ', ' + L[1].toFixed(2) + ', '
    + L[2].toFixed(2) + ', ' + L[3].toFixed(2) + ') &nbsp; '
    + 'genes recoloured by expression (range '
    + (lo/EXPR_SCALE).toFixed(2) + '..' + (hi/EXPR_SCALE).toFixed(2) + ', magma)';
}});

genePlot.on('plotly_hover', function(data) {{
  const pt = data.points[0];
  if (pt.curveNumber !== POINTS_TRACE) return;
  const j = pt.pointNumber;
  if (lastHoveredGene === j) return;
  lastHoveredGene = j;
  const n = expr_matrix.length;
  const col = new Array(n);
  for (let i = 0; i < n; i++) col[i] = expr_matrix[i][j];
  const colors = exprToMagma(col);
  Plotly.restyle(cellPlot, {{'marker.color': [colors]}}, [POINTS_TRACE]);
  const loadColors = loadingToMagma(gene_load[j]);
  Plotly.restyle(cellPlot, {{'marker.color': [loadColors]}}, [LOADING_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color': [loadColors]}}, [LOADING_TRACE]);
  let lo = Infinity, hi = -Infinity;
  for (const v of col) {{ if (v < lo) lo = v; if (v > hi) hi = v; }}
  const L = gene_load[j];
  const panelTag = gene_in_panel[j] ? '(panel)' : '(imputed)';
  const gTag = '<b style="color:' + gene_dom_color[j] + '">' + gene_name[j]
    + '</b> <span style="color:#555">' + panelTag + '</span>';
  status.innerHTML = gTag + ' &nbsp; '
    + 'loadings A1..A4 = (' + L[0].toFixed(2) + ', ' + L[1].toFixed(2) + ', '
    + L[2].toFixed(2) + ', ' + L[3].toFixed(2) + ') &nbsp; '
    + 'cells recoloured by expression (range '
    + (lo/EXPR_SCALE).toFixed(2) + '..' + (hi/EXPR_SCALE).toFixed(2) + ', magma)';
}});

document.getElementById('reset-btn').addEventListener('click', function() {{
  Plotly.restyle(cellPlot, {{'marker.color': [cell_default_colors]}}, [POINTS_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color': [gene_default_colors]}}, [POINTS_TRACE]);
  Plotly.restyle(cellPlot, {{'marker.color': [DEFAULT_LOAD_COLORS]}}, [LOADING_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color': [DEFAULT_LOAD_COLORS]}}, [LOADING_TRACE]);
  lastHoveredCell = null; lastHoveredGene = null;
  status.innerHTML = 'Reset. Hover a cell or gene to colour by expression and reveal loadings.';
}});

// --- Gene visibility: gene-set buttons + mean + dispersion sliders ----------
let activeSet = 'panel';
const meanSlider   = document.getElementById('mean-slider');
const stdSlider    = document.getElementById('std-slider');
const meanValueEl  = document.getElementById('mean-value');
const stdValueEl   = document.getElementById('std-value');
const visibleCount = document.getElementById('visible-count');

function applyGeneFilter() {{
  const meanThr = parseFloat(meanSlider.value);
  const stdThr  = parseFloat(stdSlider.value);
  const mask    = gene_sets[activeSet];
  const n       = gene_name.length;
  const xs = new Array(n), ys = new Array(n), zs = new Array(n);
  let visible = 0;
  for (let j = 0; j < n; j++) {{
    if (mask[j] && gene_mean[j] >= meanThr && gene_std[j] >= stdThr) {{
      xs[j] = gene_x[j]; ys[j] = gene_y[j]; zs[j] = gene_z[j];
      visible++;
    }} else {{
      xs[j] = null; ys[j] = null; zs[j] = null;
    }}
  }}
  Plotly.restyle(genePlot, {{x: [xs], y: [ys], z: [zs]}}, [POINTS_TRACE]);
  meanValueEl.textContent = meanThr.toFixed(2);
  stdValueEl.textContent  = stdThr.toFixed(2);
  visibleCount.textContent = visible + ' / ' + n;
}}

meanSlider.addEventListener('input', applyGeneFilter);
stdSlider.addEventListener('input',  applyGeneFilter);

document.querySelectorAll('.set-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    if (btn.disabled) return;
    activeSet = btn.dataset.set;
    document.querySelectorAll('.set-btn').forEach(b =>
      b.classList.toggle('active', b === btn));
    applyGeneFilter();
  }});
}});

function resizePlots() {{
  Plotly.Plots.resize(cellPlot);
  Plotly.Plots.resize(genePlot);
}}
window.addEventListener('resize', resizePlots);
setTimeout(function() {{ resizePlots(); applyGeneFilter(); }}, 50);
</script>
</body>
</html>"""

    with open(OUT, 'w') as f: f.write(page)
    size_mb = os.path.getsize(OUT) / 1e6
    print(f'  done. {size_mb:.1f} MB self-contained HTML.')
    print(f'  open: file://{OUT}')


if __name__ == '__main__':
    main()
