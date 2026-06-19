#!/usr/bin/env python
"""VISp Interneuron Patch-seq explorer — Gouwens 2020 patch-seq MET cells
overlaid on the AllInhib SVD recompute explorer.

Thin config over scripts/_patchseq_overlay.py (shared with the Tolias MOp
explorer, so the two stay at feature parity). Note: the Gouwens pkl carries no
per-cell ephys/morph feature grids, so those sub-grids stay empty here; all
other behaviour (colour sync, subset sync, info box, morphology image) matches.
"""
import os, json, pickle
import _patchseq_overlay as ov

ROOT = ov.ROOT
SITE = ov.SITE
BASE_HTML   = os.path.join(SITE, 'allinhib', 'allinhib_svd_recompute_explorer_3d.html')
OVERLAY_PKL = os.path.join(ROOT, 'data', 'patchseq_overlay.pkl')
MET_PKL     = os.path.join(ROOT, 'data', 'patchseq_met_gouwens.pkl')
MORPHO_MAN  = os.path.join(SITE, 'patchseq', 'morpho_t', '_manifest.json')
OUT_HTML    = os.path.join(SITE, 'patchseq', 'patchseq_met_gouwens_explorer.html')

TITLE = 'V1 – Interneuron Patch-seq'
CITATION = (
    'Patch-seq data: Gouwens et al., <i>Cell</i> 2020 — "Integrated Morphoelectric '
    'and Transcriptomic Classification of Cortical GABAergic Cells" (VISp). '
    '<a href="https://doi.org/10.1016/j.cell.2020.09.057" target="_blank" rel="noopener">'
    'doi:10.1016/j.cell.2020.09.057</a>. &nbsp; Transcriptomic reference space: '
    'Tasic et al., <i>Nature</i> 2018 (VISp + ALM), '
    '<a href="https://doi.org/10.1038/s41586-018-0654-5" target="_blank" rel="noopener">'
    'doi:10.1038/s41586-018-0654-5</a>.')


def main():
    with open(OVERLAY_PKL, 'rb') as f:
        overlay = pickle.load(f)
    with open(MET_PKL, 'rb') as f:
        met = pickle.load(f)
    m = met['met_meta']
    n = met['n_met']
    morpho_ids = []
    if os.path.exists(MORPHO_MAN):
        with open(MORPHO_MAN) as f:
            morpho_ids = list(map(str, json.load(f)))
    print(f'   MET cells: {n}; morpho available: {len(morpho_ids)}')

    def knn(i):
        v = m['knn_mean_dist'][i]
        return f'{v:.2f}' if (v is not None and v == v) else '—'

    def hov(i):
        p = [f"<b>{m['specimen_name'][i] or 'cell '+str(i)}</b>",
             f"AIT: {m['ait_cluster'][i] or '—'}",
             f"imputed Tasic: <b>{m['imputed_cluster'][i]}</b> (KNN d={knn(i)})",
             f"dendrite: {m['dendrite_type'][i] or '—'}"]
        if m['structure'][i]: p.append(f"struct: {m['structure'][i]}")
        return '<br>'.join(p)
    met_hover = [hov(i) for i in range(n)]

    def depth(i):
        d = m['soma_depth'][i]
        return f'{d:.2f}' if (d is not None and d == d) else ''

    meta_rows = {
        'Imputed Tasic': [f"{m['imputed_cluster'][i]} (KNN d={knn(i)})" for i in range(n)],
        'AIT 2.3.1':   [m['ait_cluster'][i] or '' for i in range(n)],
        'Dendrite':    [m['dendrite_type'][i] or '' for i in range(n)],
        'Structure':   [m['structure'][i] or '' for i in range(n)],
        'Soma depth':  [depth(i) for i in range(n)],
        'Ephys session': [m['ephys_session_id'][i] if m['ephys_session_id'][i] not in ('', '<NA>') else ''
                          for i in range(n)],
    }
    met_layer = [m['structure'][i] or '' for i in range(n)]   # 'VISp5' etc → digit parsed in JS

    js_consts = (
        f"const MET_N = {n};\n"
        f"const MET_XYZ = {json.dumps([list(map(float, p)) for p in overlay['patchseq_xyz']])};\n"
        f"const MET_IMPUTED_CLUSTER = {json.dumps(list(m['imputed_cluster']))};\n"
        f"const MET_IMPUTED_SUBCLASS = {json.dumps(list(m['imputed_subclass']))};\n"
        f"const MET_HOVER = {json.dumps(met_hover)};\n"
        f"const MET_CELL_ID = {json.dumps([str(x) for x in m['specimen_id']])};\n"
        f"const MET_LAYER = {json.dumps(met_layer)};\n"
        f"const MET_META = {json.dumps(meta_rows)};\n"
        f"const MET_META_ORDER = {json.dumps(list(meta_rows.keys()))};\n"
        f"const MET_EPHYS = {json.dumps({})};\n"
        f"const MET_MORPH = {json.dumps({})};\n"
        f"const MET_EXPR_B64 = {json.dumps(met['met_expr_b64'])};\n"
        f"const MET_EXPR_SHAPE = {json.dumps(met['met_expr_shape'])};\n"
        f"const MET_EXPR_SCALE = {float(met['expr_scale'])};\n"
        f"const MET_MORPHO_IDS = new Set({json.dumps(morpho_ids)});\n"
        f"const MORPHO_DIR = {json.dumps('morpho_t')};\n"
        f"const DATASET_LABEL = {json.dumps('VISp Patch-seq (Gouwens 2020)')};\n"
    )

    size = ov.assemble(BASE_HTML, OUT_HTML, js_consts, TITLE, CITATION)
    print(f'wrote {OUT_HTML} ({size/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
