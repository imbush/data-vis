#!/usr/bin/env python
"""KNN-impute Tasic clusters onto the Scala 2020 Tolias mini-atlas Patch-seq
cells + project them onto the AllInhib_V1ALM SVD basis (M1 cells are closer
to ALM than V1; the V1ALM pool gives a more representative manifold).

Reads:
  data/tolias_m1/m1_patchseq_meta_data.csv       (tab-sep, 1329 cells)
  data/tolias_m1/m1_patchseq_exon_counts.csv.gz  (cells × genes raw counts)
  data/tolias_m1/m1_patchseq_ephys_features.csv  (cells × 29 ephys metrics)
  data/tolias_m1/m1_patchseq_morph_features.csv  (646 cells × 64 morph metrics)
  notebooks/cache/AllInhib_V1ALM_archetype_proj_full.pkl  (Tasic basis)
  data-vis/patchseq/morpho_tolias/_manifest.json (cells with morpho PNG)

Filters mini-atlas → inhibitory cells (RNA family ∈ {Pvalb, Sst, Vip, Lamp5,
Sncg}). Of those, we keep the ones whose Cell ID is in the morpho manifest
(everyone else has no image to display).

Writes:
  data/patchseq_met_tolias.pkl   (parallel structure to patchseq_met_gouwens.pkl)
"""
import os, sys, pickle, json, gzip
import numpy as np
import pandas as pd

ROOT = '/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish'
SITE = '/Users/inlebush/cs/lab/green/data-vis'

TOLIAS_DIR = os.path.join(ROOT, 'data', 'tolias_m1')
ALLINHIB_PROJ = os.path.join(ROOT, 'notebooks', 'cache',
                              'AllInhib_V1ALM_archetype_proj_full.pkl')
MORPHO_DIR = os.path.join(SITE, 'patchseq', 'morpho_tolias')
OUT_PKL = os.path.join(ROOT, 'data', 'patchseq_met_tolias.pkl')

INH_FAMILIES = {'Pvalb', 'Sst', 'Vip', 'Lamp5', 'Sncg'}
EXPR_SCALE = 16.0
K_NN = 15
# Abundance-normalization exponent for the KNN vote (see build_patchseq_met_prep.py):
# score(cluster) = sum_sim_weights / N_cluster**ABUND_ALPHA. alpha=0.5 keeps rare
# Tasic clusters (Serpinf1, Sncg) from being swamped by abundant ones.
ABUND_ALPHA = 0.5


def main():
    print('1. loading Tolias M1 mini-atlas data ...')
    meta = pd.read_csv(os.path.join(TOLIAS_DIR, 'm1_patchseq_meta_data.csv'), sep='\t')
    print(f'   meta: {len(meta)} cells')
    print(f'   RNA family: {meta["RNA family"].value_counts().to_dict()}')

    # Filter to inhibitory
    inh = meta[meta['RNA family'].isin(INH_FAMILIES)].copy()
    inh.index = inh['Cell']
    print(f'   inhibitory: {len(inh)} cells')

    # Match to NeuroMorpho morpho availability
    morpho_manifest_path = os.path.join(MORPHO_DIR, '_manifest.json')
    if os.path.exists(morpho_manifest_path):
        with open(morpho_manifest_path) as f:
            morpho_ids = set(json.load(f))
        # NeuroMorpho names like '20190508_sample_2' match mini-atlas 'Cell' column directly
        with_morpho = inh.index.intersection(morpho_ids)
        print(f'   inh × NM morpho intersection: {len(with_morpho)}')
    else:
        morpho_ids = set()
        with_morpho = pd.Index([])
        print(f'   no morpho manifest yet at {morpho_manifest_path}')

    print('2. loading Tasic AllInhib_V1ALM proj_full ...')
    with open(ALLINHIB_PROJ, 'rb') as f:
        proj = pickle.load(f)
    X_tasic = proj['X_keep']
    in_panel = np.array(proj['in_panel'], dtype=bool)
    gene_names = list(proj['gene_names'])
    subs_tasic = np.array(proj['subs'])
    panel_genes = [g for g, k in zip(gene_names, in_panel) if k]
    print(f'   Tasic V1ALM: {X_tasic.shape}, panel HVG {int(in_panel.sum())}')

    Xp_tasic = X_tasic[:, in_panel].astype(np.float32)
    mu = Xp_tasic.mean(0); sd = Xp_tasic.std(0) + 1e-9
    Zp_tasic = (Xp_tasic - mu) / sd

    print('3. loading Tolias exon counts ...')
    exp_path = os.path.join(TOLIAS_DIR, 'm1_patchseq_exon_counts.csv.gz')
    with gzip.open(exp_path, 'rt') as f:
        header = f.readline().rstrip('\n').split(',')
    print(f'   exon counts header (first 5 cols): {header[:5]}')
    print(f'   total cols: {len(header)}')
    # First column is gene name; rest are cells.
    cells_in_exon = header[1:]
    print(f'   cells in exon matrix: {len(cells_in_exon)}')

    # Read selected rows only (panel genes) to save memory? Easier to load full
    # then subset. ~17 MB compressed → ~120 MB uncompressed. Should fit.
    print('   reading full exon matrix into memory ...')
    raw = pd.read_csv(exp_path, sep=',', index_col=0)
    print(f'   shape: {raw.shape} (genes × cells)')
    # raw.columns is cell IDs; raw.index is gene symbols
    # Match panel genes to Tolias gene index
    avail = set(raw.index)
    panel_avail = [g for g in panel_genes if g in avail]
    print(f'   panel genes available in Tolias: {len(panel_avail)} / {len(panel_genes)}')
    broader_avail = [g for g in gene_names if g in avail]
    print(f'   broader genes available: {len(broader_avail)} / {len(gene_names)}')

    # Subset Tolias to inhibitory cells with morpho available (or all inh if none)
    tolias_cells = list(with_morpho) if len(with_morpho) > 50 else list(inh.index)
    keep_cols = [c for c in tolias_cells if c in raw.columns]
    print(f'   keep cells ({len(keep_cols)}/{len(tolias_cells)} present in exon matrix)')
    sub = raw[keep_cols]                          # genes × cells, subset
    sub = sub.loc[panel_avail]                    # only panel genes
    print(f'   subset for projection: {sub.shape}')

    # Build per-cell panel-HVG z-vec (Tolias counts → CPM → log → z by Tasic μ/σ)
    Xp_tolias = sub.values.T.astype(np.float32)   # cells × panel
    # Normalize each cell to CPM
    libsize = Xp_tolias.sum(1) + 1e-9
    Xp_tolias = (Xp_tolias / libsize[:, None]) * 1e6
    Xp_tolias = np.log1p(Xp_tolias)
    # Fill any missing panel gene with Tasic mu (z = 0). panel_avail is a SUBSET
    # of panel_genes, so we need to map sub's rows back to the full panel index.
    panel_idx = {g: i for i, g in enumerate(panel_genes)}
    n_cells = len(keep_cols)
    n_panel = len(panel_genes)
    Xp_full = np.zeros((n_cells, n_panel), dtype=np.float32)
    for k, g in enumerate(panel_avail):
        Xp_full[:, panel_idx[g]] = Xp_tolias[:, k]
    Zp_tolias = (Xp_full - mu) / sd

    from collections import Counter
    ref_clust_counts = Counter(subs_tasic)          # reference cluster sizes for abundance norm
    print('4. KNN imputation K={} cosine (abundance alpha={}) ...'.format(K_NN, ABUND_ALPHA))
    def l2(v):
        n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        return v / n
    A = l2(Zp_tolias); B = l2(Zp_tasic)
    imputed_cluster = np.empty(n_cells, dtype=object)
    knn_dist = np.empty(n_cells, dtype=np.float32)
    CHUNK = 128
    for i0 in range(0, n_cells, CHUNK):
        sims = A[i0:i0+CHUNK] @ B.T
        topK = np.argpartition(-sims, K_NN, axis=1)[:, :K_NN]
        for ki, row in enumerate(topK):
            i = i0 + ki
            top_subs = subs_tasic[row]
            top_sims = sims[ki, row]
            knn_dist[i] = 1.0 - float(np.mean(top_sims))
            c = Counter()
            for s, w in zip(top_subs, top_sims):
                c[s] += float(max(0.0, w))
            imputed_cluster[i] = max(
                c.items(),
                key=lambda kv: kv[1] / (ref_clust_counts[kv[0]] ** ABUND_ALPHA),
            )[0]
    imputed_subclass = np.array([str(c).split()[0] if c else '' for c in imputed_cluster])
    print(f'   imputed subclass dist:')
    from collections import Counter
    for s, n in Counter(imputed_subclass).most_common(10):
        print(f'     {s:20s} {n}')

    print('5. project onto Tasic SVD basis (top-3) ...')
    U, S, Vt = np.linalg.svd(Zp_tasic, full_matrices=False)
    NPC = 3
    U, S, Vt = U[:, :NPC], S[:NPC], Vt[:NPC]
    tasic_scores = U * S
    m = np.max(np.abs(tasic_scores), axis=0) + 1e-9
    tolias_xyz = (Zp_tolias @ Vt.T) / m
    tolias_xyz = np.clip(tolias_xyz, -1.5, 1.5)

    print('6. per-cell broader-gene expression matrix (for hover gene recolor) ...')
    sub2 = raw.loc[broader_avail][keep_cols].values.T.astype(np.float32)
    libsize2 = sub2.sum(1) + 1e-9
    sub2 = (sub2 / libsize2[:, None]) * 1e6
    sub2 = np.log1p(sub2)
    # Pad to full broader-gene order (missing → 0)
    broader_idx = {g: i for i, g in enumerate(gene_names)}
    Xb = np.zeros((n_cells, len(gene_names)), dtype=np.float32)
    for k, g in enumerate(broader_avail):
        Xb[:, broader_idx[g]] = sub2[:, k]
    Xb_q = np.clip(np.round(Xb * EXPR_SCALE), 0, 255).astype(np.uint8)

    print('7. ephys + morph features ...')
    eph = pd.read_csv(os.path.join(TOLIAS_DIR, 'm1_patchseq_ephys_features.csv'))
    eph.index = eph['cell id']
    mor = pd.read_csv(os.path.join(TOLIAS_DIR, 'm1_patchseq_morph_features.csv'))
    mor.index = mor['cell id']
    # Re-index to keep_cols
    eph_aligned = eph.reindex(keep_cols)
    mor_aligned = mor.reindex(keep_cols)

    # Pick a few "headline" ephys + morph features for the side panel.
    EPHYS_KEYS = ['Resting membrane potential (mV)', 'Input resistance (MOhm)',
                   'Rheobase (pA)', 'AP width (ms)', 'AP amplitude (mV)',
                   'Max number of APs', 'Membrane time constant (ms)',
                   'Sag ratio', 'ISI adaptation index']
    MORPH_KEYS = ['axon total length', 'dendrite total length',
                   'axon max path distance to soma', 'soma radius',
                   'normalized depth']

    eph_dict = {}
    for k in EPHYS_KEYS:
        if k in eph_aligned.columns:
            eph_dict[k] = [None if pd.isna(v) else float(v)
                            for v in eph_aligned[k]]
    mor_dict = {}
    for k in MORPH_KEYS:
        if k in mor_aligned.columns:
            mor_dict[k] = [None if pd.isna(v) else float(v)
                            for v in mor_aligned[k]]

    print('8. metadata for hover ...')
    inh_aligned = inh.reindex(keep_cols)
    meta_for_hover = {
        'cell_id':        keep_cols,
        'rna_family':     inh_aligned['RNA family'].fillna('').astype(str).tolist(),
        'rna_type':       inh_aligned['RNA type'].fillna('').astype(str).tolist(),
        'rna_confidence': [None if pd.isna(v) else float(v)
                            for v in inh_aligned['RNA type confidence']],
        'tasic_top3':     inh_aligned['ALM/VISp top-3'].fillna('').astype(str).tolist(),
        'targeted_layer': inh_aligned['Targeted layer'].fillna('').astype(str).tolist(),
        'inferred_layer': inh_aligned['Inferred layer'].fillna('').astype(str).tolist(),
        'cre':            inh_aligned['Cre'].fillna('').astype(str).tolist(),
        'soma_depth_um':  [None if pd.isna(v) else float(v)
                            for v in inh_aligned['Soma depth (µm)']],
        'imputed_cluster': list(imputed_cluster),
        'imputed_subclass': imputed_subclass.tolist(),
        'knn_mean_dist': knn_dist.round(4).tolist(),
        'ephys': eph_dict,
        'morph': mor_dict,
    }

    out = {
        'n_met': n_cells,
        'panel_genes': panel_genes,
        'broader_genes': gene_names,
        'met_meta': meta_for_hover,
        'expr_scale': EXPR_SCALE,
        'met_xyz': tolias_xyz.round(4).tolist(),
    }
    import base64
    out['met_expr_b64'] = base64.b64encode(Xb_q.tobytes()).decode('ascii')
    out['met_expr_shape'] = list(Xb_q.shape)

    with open(OUT_PKL, 'wb') as f:
        pickle.dump(out, f, protocol=4)
    print(f'wrote {OUT_PKL} ({os.path.getsize(OUT_PKL)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
