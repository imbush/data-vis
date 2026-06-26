#!/usr/bin/env python
"""MOp Interneuron Patch-seq explorer — Tolias / Scala 2020 M1 patch-seq MET
cells overlaid on the AllInhib_V1ALM SVD recompute explorer.

Thin config over scripts/_patchseq_overlay.py (shared with the Gouwens VISp
explorer, so the two stay at feature parity).
"""
import os, json, pickle
import _patchseq_overlay as ov

ROOT = ov.ROOT
SITE = ov.SITE
BASE_HTML  = os.path.join(SITE, 'allinhib_v1alm', 'allinhib_v1alm_svd_recompute_explorer_3d.html')
MET_PKL    = os.path.join(ROOT, 'data', 'patchseq_met_tolias.pkl')
MORPHO_MAN = os.path.join(SITE, 'patchseq', 'morpho_tolias', '_manifest.json')
OUT_HTML   = os.path.join(SITE, 'patchseq', 'patchseq_met_tolias_explorer.html')

TITLE = 'M1 – Interneuron Patch-seq'
CITATION = (
    'Patch-seq data: Scala et al., <i>Nature</i> 2021 — "Phenotypic variation of '
    'transcriptomic cell types in mouse motor cortex" (MOp / M1). '
    '<a href="https://doi.org/10.1038/s41586-020-2907-3" target="_blank" rel="noopener">'
    'doi:10.1038/s41586-020-2907-3</a>. &nbsp; Transcriptomic reference space: '
    'Tasic et al., <i>Nature</i> 2018 (VISp + ALM), '
    '<a href="https://doi.org/10.1038/s41586-018-0654-5" target="_blank" rel="noopener">'
    'doi:10.1038/s41586-018-0654-5</a>.')


def _depth(v):
    return f'{v:.0f} µm' if (v is not None and v == v) else ''


def main():
    with open(MET_PKL, 'rb') as f:
        met = pickle.load(f)
    m = met['met_meta']
    n = met['n_met']
    morpho_ids = []
    if os.path.exists(MORPHO_MAN):
        with open(MORPHO_MAN) as f:
            morpho_ids = list(map(str, json.load(f)))
    print(f'   MET cells: {n}; morpho available: {len(morpho_ids)}')

    def hov(i):
        p = [f"<b>{m['cell_id'][i]}</b>",
             f"family: {m['rna_family'][i] or '—'} · type: {m['rna_type'][i] or '—'}",
             f"imputed Tasic: <b>{m['imputed_cluster'][i]}</b> (KNN d={m['knn_mean_dist'][i]:.3f})"]
        if m['cre'][i]: p.append(f"Cre: {m['cre'][i]}")
        if m['targeted_layer'][i]: p.append(f"layer: {m['targeted_layer'][i]}")
        return '<br>'.join(p)
    met_hover = [hov(i) for i in range(n)]

    conf = m['rna_confidence']
    meta_rows = {
        'Imputed Tasic': [f"{m['imputed_cluster'][i]} (KNN d={m['knn_mean_dist'][i]:.2f})" for i in range(n)],
        'RNA family / type': [
            f"{m['rna_family'][i] or '—'} / {m['rna_type'][i] or '—'}"
            + (f" ({conf[i]*100:.0f}%)" if conf[i] is not None else '') for i in range(n)],
        'ALM/VISp top-3': [m['tasic_top3'][i] or '' for i in range(n)],
        'Layer (target / inferred)': [
            f"{m['targeted_layer'][i] or '?'} / {m['inferred_layer'][i] or '?'}" for i in range(n)],
        'Cre line': [m['cre'][i] or '' for i in range(n)],
        'Soma depth': [_depth(m['soma_depth_um'][i]) for i in range(n)],
    }
    # layer for the Layer colour mode: prefer targeted, fall back to inferred
    met_layer = [m['targeted_layer'][i] or m['inferred_layer'][i] or '' for i in range(n)]

    js_consts = (
        f"const MET_N = {n};\n"
        f"const MET_XYZ = {json.dumps([list(map(float, p)) for p in met['met_xyz']])};\n"
        f"const MET_IMPUTED_CLUSTER = {json.dumps(list(m['imputed_cluster']))};\n"
        f"const MET_IMPUTED_SUBCLASS = {json.dumps(list(m['imputed_subclass']))};\n"
        f"const MET_HOVER = {json.dumps(met_hover)};\n"
        f"const MET_CELL_ID = {json.dumps([str(x) for x in m['cell_id']])};\n"
        f"const MET_LAYER = {json.dumps(met_layer)};\n"
        f"const MET_META = {json.dumps(meta_rows)};\n"
        f"const MET_META_ORDER = {json.dumps(list(meta_rows.keys()))};\n"
        f"const MET_EPHYS = {json.dumps(m['ephys'])};\n"
        f"const MET_MORPH = {json.dumps(m['morph'])};\n"
        f"const MET_EXPR_B64 = {json.dumps(met['met_expr_b64'])};\n"
        f"const MET_EXPR_SHAPE = {json.dumps(met['met_expr_shape'])};\n"
        f"const MET_EXPR_SCALE = {float(met['expr_scale'])};\n"
        f"const MET_MORPHO_IDS = new Set({json.dumps(morpho_ids)});\n"
        f"const MORPHO_DIR = {json.dumps('morpho_tolias')};\n"
        f"const DATASET_LABEL = {json.dumps('MOp Patch-seq (Scala 2020)')};\n"
    )

    size = ov.assemble(BASE_HTML, OUT_HTML, js_consts, TITLE, CITATION,
                       met_gene_names=met['broader_genes'])
    print(f'wrote {OUT_HTML} ({size/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
