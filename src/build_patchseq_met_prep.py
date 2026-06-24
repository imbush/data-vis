#!/usr/bin/env python
"""Extend data/patchseq_overlay.pkl with what the MET explorer needs on top of
the SVD-projection coords already there:

  - met_panel_z       : (n_met, n_panel) — z-scored panel-HVG vectors using
                        the Tasic mean/std. Stored as int16 (×100) to keep
                        the HTML small; the JS divides by 100 on use.
  - met_broader_logcpm: (n_met, n_broader) — log-CPM in the broader-gene
                        space the explorer uses for recolouring. uint8
                        quantised to [0, EXPR_SCALE].
  - met_imputed_cluster : (n_met,) majority-vote Tasic cell_cluster among
                          K=15 nearest Tasic cells in cosine distance on
                          the panel-HVG z-space (the same basis as SVD).
  - met_imputed_subclass : derived prefix of the cluster (Sst / Pvalb / …).
  - met_knn_dist        : per-MET-cell mean cosine distance to its 15 NN
                          (used for an "OOD" warning in the UI when high).

Reads `notebooks/cache/AllInhib_archetype_proj_full.pkl` (Tasic source) and
the Gouwens Patch-seq h5ad / metadata, projecting them through the SAME
panel HVG that the AllInhib explorer uses.
"""
import os, sys, pickle, json
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as ss

ROOT = '/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish'

PATCHSEQ_H5AD = '/Users/Shared/lab-data/gouwens2020-patchseq/transcriptome/gouwens_patchseq_cpm.h5ad'
META_CSV = ('/Users/Shared/lab-data/gouwens2020-patchseq/transcriptome/'
            '20200625_patchseq_metadata_mouse/20200625_patchseq_metadata_mouse.csv')

ALLINHIB_PROJ = os.path.join(ROOT, 'notebooks', 'cache', 'AllInhib_archetype_proj_full.pkl')
OUT_PKL = os.path.join(ROOT, 'data', 'patchseq_met_gouwens.pkl')

EXPR_SCALE = 16.0
K_NN = 15
# Abundance-normalization exponent for the KNN vote. score(cluster) =
# sum_sim_weights / N_cluster**ABUND_ALPHA. alpha=0 is plain similarity-weighted
# voting (which lets abundant Tasic clusters swamp rare ones — e.g. Serpinf1,
# with only 26 reference cells, collapses to Vip). alpha=0.5 down-weights
# abundant clusters: validated against the Gouwens AIT2.3.1 ground-truth labels
# it lifts cluster-exact accuracy 53.6%->57.7% and Serpinf1 recall 4%->17%
# (62% precision) while holding subclass accuracy at ~90.5%.
ABUND_ALPHA = 0.5


def main():
    print('1. loading Tasic AllInhib proj_full cache ...')
    with open(ALLINHIB_PROJ, 'rb') as f:
        proj = pickle.load(f)
    X_tasic   = proj['X_keep']                       # cells × genes (log-CPM)
    in_panel  = np.array(proj['in_panel'], dtype=bool)
    gene_names = list(proj['gene_names'])
    subs_tasic = np.array(proj['subs'])              # cluster label per cell
    panel_genes = [g for g, k in zip(gene_names, in_panel) if k]
    n_tasic = X_tasic.shape[0]
    print(f'   Tasic: {n_tasic} cells, {int(in_panel.sum())} panel HVG of {len(gene_names)} broader')

    Xp_tasic = X_tasic[:, in_panel].astype(np.float32)
    mu = Xp_tasic.mean(0)
    sd = Xp_tasic.std(0) + 1e-9
    Zp_tasic = (Xp_tasic - mu) / sd

    print('2. loading Patch-seq + metadata ...')
    ps = ad.read_h5ad(PATCHSEQ_H5AD)
    if ss.issparse(ps.X):
        ps_X = ps.X.toarray()
    else:
        ps_X = np.asarray(ps.X)
    print(f'   Patch-seq: {ps.shape}, X dtype {ps.X.dtype}')
    meta = pd.read_csv(META_CSV, low_memory=False)

    ps_obs = ps.obs.copy()
    ps_obs['_join'] = ps_obs['cell_specimen_id'].astype(str) if 'cell_specimen_id' in ps_obs.columns else ps_obs.index.astype(str)
    meta['_join'] = meta['cell_specimen_id'].astype(str)
    aligned = meta.set_index('_join').reindex(ps_obs['_join'].values)
    has_ait = aligned['corresponding_AIT2.3.1_alias'].notna() & (aligned['corresponding_AIT2.3.1_alias'] != '')
    print(f'   cells with AIT2.3.1: {int(has_ait.sum())}')

    print('3. project Patch-seq into Tasic panel-HVG z-space ...')
    panel_set = {g: i for i, g in enumerate(panel_genes)}
    ps_gene_idx = {g: i for i, g in enumerate(ps.var_names)}
    # map each panel gene → ps column; missing → use Tasic mean (z=0)
    panel_to_ps = np.array([ps_gene_idx.get(g, -1) for g in panel_genes], dtype=np.int32)
    Xp_ps = np.zeros((ps.n_obs, len(panel_genes)), dtype=np.float32)
    for k, j in enumerate(panel_to_ps):
        if j >= 0:
            Xp_ps[:, k] = ps_X[:, j]
        else:
            Xp_ps[:, k] = mu[k]
    Xp_ps = np.log1p(Xp_ps)
    Zp_ps = (Xp_ps - mu) / sd

    from collections import Counter
    ref_clust_counts = Counter(subs_tasic)          # reference cluster sizes for abundance norm
    print('4. KNN imputation: K={} cosine on panel-HVG z-space (abundance alpha={}) ...'.format(K_NN, ABUND_ALPHA))
    # cosine = 1 - normalized dot. Normalize both, then compute big dot.
    def l2(v):
        n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        return v / n
    A = l2(Zp_ps)              # patch-seq, n × n_panel
    B = l2(Zp_tasic)           # tasic
    # similarities: n_ps × n_tasic. Compute in chunks to bound memory.
    n_ps = ps.n_obs
    imputed_cluster = np.empty(n_ps, dtype=object)
    knn_dist = np.empty(n_ps, dtype=np.float32)
    CHUNK = 256
    for i0 in range(0, n_ps, CHUNK):
        sims = A[i0:i0+CHUNK] @ B.T          # CHUNK × n_tasic, cosine sims
        # top-K indices per row
        topK = np.argpartition(-sims, K_NN, axis=1)[:, :K_NN]
        for k_idx, row in enumerate(topK):
            i = i0 + k_idx
            top_subs = subs_tasic[row]
            top_sims = sims[k_idx, row]
            # mean cosine distance
            knn_dist[i] = 1.0 - float(np.mean(top_sims))
            # similarity-weighted vote, abundance-normalized so rare reference
            # clusters aren't swamped by abundant ones (see ABUND_ALPHA note).
            c = Counter()
            for s, w in zip(top_subs, top_sims):
                c[s] += float(max(0.0, w))
            imputed_cluster[i] = max(
                c.items(),
                key=lambda kv: kv[1] / (ref_clust_counts[kv[0]] ** ABUND_ALPHA),
            )[0]
        if (i0 // CHUNK) % 4 == 0:
            print(f'   ...{i0+len(topK):>5}/{n_ps}')

    # Subclass from cluster prefix (first whitespace-separated token).
    imputed_subclass = np.array([str(c).split()[0] if c else '' for c in imputed_cluster])
    print(f'   imputed cluster vocab size: {len(set(imputed_cluster))}')
    print(f'   subclass distribution:')
    from collections import Counter
    for s, n in Counter(imputed_subclass).most_common(10):
        print(f'     {s:20s} {n}')

    print('5. per-MET-cell expression for gene-recolor-on-hover ...')
    # Use the broader gene set the explorer uses (gene_names = panel ∪ top expressed ∪ curated).
    broader_to_ps = np.array([ps_gene_idx.get(g, -1) for g in gene_names], dtype=np.int32)
    Xb = np.zeros((ps.n_obs, len(gene_names)), dtype=np.float32)
    miss = 0
    for k, j in enumerate(broader_to_ps):
        if j >= 0:
            Xb[:, k] = ps_X[:, j]
        else:
            miss += 1
    print(f'   broader-gene matrix: {Xb.shape}; missing-in-patchseq: {miss}/{len(gene_names)}')
    Xb = np.log1p(Xb)
    # Quantise to uint8 matching the explorer's EXPR_SCALE convention.
    Xb_q = np.clip(np.round(Xb * EXPR_SCALE), 0, 255).astype(np.uint8)

    print('6. cell metadata for hover text ...')
    meta_for_hover = {
        'specimen_id': aligned['cell_specimen_id'].fillna(-1).astype('Int64').astype(str).tolist(),
        'specimen_name': aligned['cell_specimen_name'].fillna('').astype(str).tolist(),
        'ait_cluster': aligned['corresponding_AIT2.3.1_alias'].fillna('').astype(str).tolist(),
        'dendrite_type': aligned['dendrite_type'].fillna('').astype(str).tolist(),
        'structure': aligned['structure'].fillna('').astype(str).tolist(),
        'soma_depth': aligned['cell_soma_normalized_depth'].astype(float).tolist(),
        'ephys_session_id': aligned['ephys_session_id'].astype('Int64').astype(str).tolist(),
        'imputed_cluster': list(imputed_cluster),
        'imputed_subclass': imputed_subclass.tolist(),
        'knn_mean_dist': knn_dist.round(4).tolist(),
    }

    out = {
        'n_met':                ps.n_obs,
        'panel_genes':          panel_genes,
        'broader_genes':        gene_names,
        'met_expr_b64':         None,        # filled below
        'met_meta':             meta_for_hover,
        'expr_scale':           EXPR_SCALE,
    }

    # Base64-encode the per-MET-cell uint8 expression matrix (row-major) so the
    # explorer can embed it directly as in the existing AllInhib HTML.
    import base64
    out['met_expr_b64'] = base64.b64encode(Xb_q.tobytes()).decode('ascii')
    out['met_expr_shape'] = list(Xb_q.shape)

    with open(OUT_PKL, 'wb') as f:
        pickle.dump(out, f, protocol=4)
    print(f'wrote {OUT_PKL} ({os.path.getsize(OUT_PKL)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
