#!/usr/bin/env python
"""Marker-gene explorer.

A two-population differential-expression view built on the same SVD biplot as
the recompute explorers. The user:
  - assigns cell subtypes to Group A or Group B (each subtype A / B / neither),
  - filters the gene pool (gene set + name search + min mean / dispersion),
  - sees both populations on the cell biplot (colour-by-group A vs B),
  - clicks "Find marker genes" to rank the shown genes by how cleanly they
    separate A from B (AUROC), listed vertically with an aligned dot plot
    (mean expression = colour, fraction expressing = size, per group),
  - hovers a gene row to recolour the cells by that gene and ring it in the
    gene biplot.

Self-contained single HTML (no server). Usage:
    python build_marker_app_3d.py [GROUP]   # default: base GROUP_NAME
Output: notebooks/{group}_marker_explorer_3d.html
"""
import os, json, base64, warnings
warnings.filterwarnings('ignore')
import numpy as np
from plotly.io import to_html
import plotly.graph_objects as go

import build_lamp5_archetype_app_4d as base
import build_lamp5_svd_app_3d as svdmod

GROUP_NAME  = base.GROUP_NAME
SLUG        = base.SLUG
OUT         = os.path.join(base.ROOT, 'notebooks', f'{SLUG}_marker_explorer_3d.html')
NPC         = 3
POLE_COLORS = svdmod.POLE_COLORS
POLE_NAMES  = svdmod.POLE_NAMES
prep_cols   = svdmod.prep_cols

GRP_A_COLOR = '#2c6fbb'   # blue
GRP_B_COLOR = '#cc4125'   # red
GREY_OFF    = 'rgba(0,0,0,0)'   # cells in neither group are fully invisible

from bokeh.palettes import Magma256, Viridis256


def main():
    proj       = base.compute_or_load_proj_full()
    gene_names = list(proj['gene_names'])
    in_panel   = np.array(proj['in_panel'])
    mean_expr  = np.asarray(proj['mean_expr'])
    std_expr   = np.asarray(proj['std_expr'])
    X_keep     = np.asarray(proj['X_keep'], dtype=np.float64)
    subs       = np.array(proj['subs'])
    cell_subclass = np.array(proj.get('cell_subclass', [GROUP_NAME] * len(subs)))
    cell_layer_arr = proj.get('cell_layer')
    if cell_layer_arr is None:
        cell_layer_list = None
    else:
        cell_layer_list = [str(x) for x in cell_layer_arr]
        if len(set(cell_layer_list)) <= 1:
            cell_layer_list = None
    cell_age_arr = proj.get('cell_age')
    cell_age_list = ([str(x) for x in cell_age_arr] if cell_age_arr is not None else None)
    n_cells = X_keep.shape[0]
    n_genes = len(gene_names)

    qc = base.compute_or_load_qc_full()
    assert np.array_equal(np.array(qc['subs']), subs), 'QC/proj cell-order mismatch'
    qc_total  = np.asarray(qc['total_counts'], dtype=np.float64)
    qc_ngenes = np.asarray(qc['n_genes'], dtype=np.float64)
    qc_ribo   = np.asarray(qc['pct_ribo'], dtype=np.float64)

    # ---- initial SVD (full cohort) -----------------------------------------
    Xp = X_keep[:, in_panel]
    Zp = prep_cols(Xp)
    U, S, Vt = np.linalg.svd(Zp, full_matrices=False)
    U, S, Vt = U[:, :NPC], S[:NPC], Vt[:NPC]
    cell_scores = U * S
    Zall = prep_cols(X_keep)
    gene_load3 = (Zall.T @ U) / S
    print(f'  marker explorer: SVD on {Xp.shape[0]} cells × {int(in_panel.sum())} panel genes')

    # ---- 2D UMAP of the cohort cells (cleaner DR for the lasso overlay) -----
    # Computed from the same standardized panel-HVG matrix the SVD uses (Zp),
    # so the lasso overlay shows a proper non-linear embedding rather than the
    # linear PC1×PC2 plane. Cell-index aligned with X_keep / subs / cell_xyz.
    import anndata as _ad
    import scanpy as sc
    _adata = _ad.AnnData(np.asarray(Zp, dtype=np.float32))
    sc.pp.pca(_adata, n_comps=min(30, Zp.shape[1] - 1, Zp.shape[0] - 1))
    sc.pp.neighbors(_adata, random_state=0)
    sc.tl.umap(_adata, random_state=0)
    _umap = np.asarray(_adata.obsm['X_umap'], dtype=np.float64)
    umap_x = np.round(_umap[:, 0], 4).tolist()
    umap_y = np.round(_umap[:, 1], 4).tolist()
    print(f'  UMAP computed: 2D UMAP on {Zp.shape[0]} cells × {Zp.shape[1]} panel HVGs '
          f'(PCA n_comps={min(30, Zp.shape[1] - 1, Zp.shape[0] - 1)})')

    def fill_cube(M, robust=False):
        # robust: normalise by the 99.5th percentile of |score| per axis (not the
        # max) so a handful of outlier subtypes don't compress the main cloud,
        # then clip into the cube. Plain max-normalisation for the gene loadings.
        if robust:
            m = np.percentile(np.abs(M), 99.5, axis=0) + 1e-9
            return np.clip(M / m, -1.0, 1.0)
        m = np.max(np.abs(M), axis=0) + 1e-9
        return M / m
    # centre each cloud on its median so the scene origin sits in the middle of
    # the points (rotation orbits the cloud instead of swinging it off-screen)
    def _center(M):
        M = M - np.median(M, axis=0)
        return np.clip(M, -1.0, 1.0)
    cell_xyz = _center(fill_cube(cell_scores, robust=True))
    gene_xyz = _center(fill_cube(gene_load3))

    Vpanel = Vt.T
    panel_genes_list = [g for g, k in zip(gene_names, in_panel) if k]
    pole_top = []
    for k in range(NPC):
        order = np.argsort(Vpanel[:, k])
        pole_top.append(panel_genes_list[order[-1]])
        pole_top.append(panel_genes_list[order[0]])

    cats = sorted(set(subs.tolist()))
    subtype_palette = base.build_subtype_palette(cats)
    sub_idx_of = {c: i for i, c in enumerate(cats)}
    cell_sub_idx = np.array([sub_idx_of[s] for s in subs],
                            dtype=(np.uint16 if len(cats) > 255 else np.uint8))

    # subclass grouping + sensible default A/B populations (computed up front so
    # the initial cell figure can be baked with only the A∪B subset shown — a
    # restyle of a gl3d trace doesn't always repaint on first load).
    subtype_to_subclass = {}
    for s, csc in zip(subs, cell_subclass):
        subtype_to_subclass.setdefault(s, csc)
    subclass_order = sorted(set(cell_subclass.tolist()))
    by_subclass = {csc: [c for c in cats if subtype_to_subclass.get(c) == csc]
                   for csc in subclass_order}
    if 'Pvalb' in subclass_order and 'Sst' in subclass_order:
        a_subs = by_subclass['Pvalb']; b_subs = by_subclass['Sst']
    elif len(subclass_order) >= 2:
        a_subs = by_subclass[subclass_order[0]]; b_subs = by_subclass[subclass_order[1]]
    else:
        half = max(1, len(cats) // 2)
        a_subs = cats[:half]; b_subs = cats[half:]
    default_a, default_b = list(a_subs), list(b_subs)
    da_set, db_set = set(default_a), set(default_b)

    def signed_pole(v3):
        k = int(np.argmax(np.abs(v3)))
        return 2 * k + (0 if v3[k] >= 0 else 1)
    gene_pole = np.array([signed_pole(gene_load3[j]) for j in range(n_genes)])
    cell_pole = np.array([signed_pole(cell_scores[i]) for i in range(n_cells)])
    gene_color_default = [POLE_COLORS[p] for p in gene_pole]
    cell_dom_color = [POLE_COLORS[p] for p in cell_pole]

    # expression as base64 uint8 (EXPR_SCALE=16 → log-CPM ∈ [0,16])
    EXPR_SCALE = 16
    _expr_q = np.clip(np.round(X_keep * EXPR_SCALE), 0, 255).astype(np.uint8)
    expr_b64 = base64.b64encode(_expr_q.tobytes()).decode('ascii')
    cell_sub_idx_b64 = base64.b64encode(cell_sub_idx.tobytes()).decode('ascii')

    # ---- figure construction (mirror of the SVD biplot) --------------------
    LIM, POLE = 1.0, 1.22
    axis_ends = np.array([[POLE, 0, 0], [-POLE, 0, 0], [0, POLE, 0],
                          [0, -POLE, 0], [0, 0, POLE], [0, 0, -POLE]])

    def build_fig(xyz, colors, hover_text, sizes=4):
        ax_x, ax_y, ax_z = [], [], []
        for a, b in [((-LIM, 0, 0), (LIM, 0, 0)), ((0, -LIM, 0), (0, LIM, 0)),
                     ((0, 0, -LIM), (0, 0, LIM))]:
            ax_x += [a[0], b[0], None]; ax_y += [a[1], b[1], None]; ax_z += [a[2], b[2], None]
        edge_trace = go.Scatter3d(x=ax_x, y=ax_y, z=ax_z, mode='lines',
                                  line=dict(color='lightgray', width=2),
                                  hoverinfo='skip', showlegend=False)
        pole_lab = [POLE_NAMES[p] for p in range(6)]
        vertex_trace = go.Scatter3d(
            x=axis_ends[:, 0], y=axis_ends[:, 1], z=axis_ends[:, 2],
            mode='markers+text', marker=dict(size=4, color=POLE_COLORS),
            text=pole_lab, textposition='top center',
            textfont=dict(size=11, color='black'),
            hoverinfo='text', hovertext=pole_lab, showlegend=False)
        points_trace = go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers',
            marker=dict(size=sizes, color=colors, opacity=0.85, line=dict(width=0)),
            text=hover_text, hoverinfo='text', showlegend=False)
        loading_trace = go.Scatter3d(
            x=axis_ends[:, 0], y=axis_ends[:, 1], z=axis_ends[:, 2], mode='markers',
            marker=dict(size=0, color=['#e0e0e0'] * 6, opacity=0.0, line=dict(width=0)),
            hoverinfo='text', hovertext=POLE_NAMES, showlegend=False)
        highlight_trace = go.Scatter3d(
            x=[None], y=[None], z=[None], mode='markers',
            marker=dict(size=15, color='rgba(0,0,0,0)', line=dict(width=4, color='#00e5ff')),
            hoverinfo='skip', showlegend=False, name='search')
        fig = go.Figure(data=[edge_trace, vertex_trace, points_trace,
                              loading_trace, highlight_trace])
        lim = POLE * 1.12
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False, range=[-lim, lim], autorange=False),
                yaxis=dict(visible=False, range=[-lim, lim], autorange=False),
                zaxis=dict(visible=False, range=[-lim, lim], autorange=False),
                aspectmode='cube', dragmode='orbit',
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.4),
                            center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1))),
            margin=dict(l=0, r=0, t=4, b=0),
            paper_bgcolor='white', plot_bgcolor='white')
        return fig

    def _age_tag(i):
        return '' if cell_age_list is None else f' · age {cell_age_list[i]}'
    cell_hover_text = [
        f'#{i}<br>subtype: {subs[i]}{_age_tag(i)}<br>'
        f'PC1,2,3 = ({cell_scores[i,0]:.2f}, {cell_scores[i,1]:.2f}, {cell_scores[i,2]:.2f})'
        for i in range(n_cells)]
    gene_hover_text = [
        f'<b>{gene_names[j]}</b>' + (' (panel HVG)' if in_panel[j] else ' (projected)')
        + f'<br>mean={mean_expr[j]:.2f}, std={std_expr[j]:.2f}'
        for j in range(n_genes)]

    # bake the default A∪B view into the initial cell figure. Every cell keeps
    # valid coords + a uniform marker size (gl3d won't reliably honour a
    # per-point size array or NaN coords on first paint); cells outside both
    # groups recede to a light grey instead of being hidden.
    init_colors = [GRP_A_COLOR if s in da_set else (GRP_B_COLOR if s in db_set else GREY_OFF)
                   for s in subs]
    fig_cells = build_fig(cell_xyz, init_colors, cell_hover_text)
    fig_genes = build_fig(gene_xyz, gene_color_default, gene_hover_text)
    cells_html = to_html(fig_cells, include_plotlyjs='cdn', full_html=False,
                         div_id='cell-plot', config={'displayModeBar': True, 'responsive': True})
    genes_html = to_html(fig_genes, include_plotlyjs=False, full_html=False,
                         div_id='gene-plot', config={'displayModeBar': True, 'responsive': True})

    gx, gy, gz = (np.round(gene_xyz[:, k], 4).tolist() for k in range(3))
    cx, cy, cz = (np.round(cell_xyz[:, k], 4).tolist() for k in range(3))

    panel_idx = [j for j, p in enumerate(in_panel.tolist()) if p]

    # gene-set masks
    panel_mask_list = in_panel.tolist()
    set_masks = {'all': [True] * n_genes, 'panel': panel_mask_list}
    set_counts = {'all': n_genes, 'panel': int(in_panel.sum())}
    for name, gene_list in base.GENE_SETS.items():
        gset = set(gene_list)
        mask = [g in gset for g in gene_names]
        set_masks[name] = mask; set_counts[name] = int(sum(mask))

    mean_min, mean_max = float(np.min(mean_expr)), float(np.max(mean_expr))
    std_min, std_max = float(np.min(std_expr)), float(np.max(std_expr))

    # ---- subtype A/B group selector, grouped by subclass -------------------
    subtype_counts = {c: int(np.sum(subs == c)) for c in cats}
    grp_rows = []
    for csc in subclass_order:
        sub_subs = by_subclass[csc]
        if not sub_subs:
            continue
        total = sum(subtype_counts[c] for c in sub_subs)
        chips = ''.join(
            f'<div class="mk-subt" data-sub="{c}">'
            f'<button class="mk-ab mk-a" data-sub="{c}" data-grp="A" title="Add to Group A">A</button>'
            f'<button class="mk-ab mk-b" data-sub="{c}" data-grp="B" title="Add to Group B">B</button>'
            f'<span class="dot" style="color:{subtype_palette[c]}">●</span> {c} '
            f'<span class="ct">({subtype_counts[c]})</span></div>'
            for c in sub_subs)
        grp_rows.append(
            f'<div class="mk-subclass"><div class="mk-subclass-head"><b>{csc}</b> '
            f'<span class="ct">({total} cells, {len(sub_subs)} subtypes)</span> '
            f'<button class="mk-bulk" data-grp="A" data-subclass="{csc}">all→A</button>'
            f'<button class="mk-bulk" data-grp="B" data-subclass="{csc}">all→B</button>'
            f'<button class="mk-bulk" data-grp="none" data-subclass="{csc}">clear</button></div>'
            f'<div class="mk-subclass-subs">{chips}</div></div>')
    group_selector_html = ''.join(grp_rows)

    set_buttons_html = ''.join(
        f'<button class="set-btn{" active" if name == "panel" else ""}" data-set="{name}">'
        f'{name} <span class="ct">({set_counts[name]})</span></button>'
        for name in (['panel', 'all'] + list(base.GENE_SETS.keys())))
    gene_datalist = ('<datalist id="gene-datalist">'
                     + ''.join(f'<option value="{g}">' for g in gene_names) + '</datalist>')

    # ---- JS data block -----------------------------------------------------
    js_data = (
        f"const N_CELLS = {n_cells};\n"
        f"const N_GENES = {n_genes};\n"
        f"const EXPR_SCALE = {EXPR_SCALE};\n"
        f"const GRP_A_COLOR = {json.dumps(GRP_A_COLOR)};\n"
        f"const GRP_B_COLOR = {json.dumps(GRP_B_COLOR)};\n"
        f"const GREY_OFF = {json.dumps(GREY_OFF)};\n"
        f"const EXPR_B64 = {json.dumps(expr_b64)};\n"
        "const expr_matrix = (function() {\n"
        "  const bin = atob(EXPR_B64); const a = new Uint8Array(bin.length);\n"
        "  for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i); return a; })();\n"
        f"const SUBTYPE_CATS = {json.dumps(cats)};\n"
        f"const SUBTYPE_PALETTE = {json.dumps([subtype_palette[c] for c in cats])};\n"
        f"const _sub_idx_b64 = {json.dumps(cell_sub_idx_b64)};\n"
        f"const _SUB_IDX_BYTES = {2 if len(cats) > 255 else 1};\n"
        "const cell_sub_idx = (function() {\n"
        "  const bin = atob(_sub_idx_b64); const n = bin.length / _SUB_IDX_BYTES;\n"
        "  const out = new Array(n);\n"
        "  if (_SUB_IDX_BYTES === 1) { for (let i=0;i<n;i++) out[i] = bin.charCodeAt(i); }\n"
        "  else { for (let i=0;i<n;i++) out[i] = bin.charCodeAt(2*i) | (bin.charCodeAt(2*i+1)<<8); }\n"
        "  return out; })();\n"
        f"const gene_name = {json.dumps(gene_names)};\n"
        f"const gene_in_panel = {json.dumps([bool(x) for x in in_panel.tolist()])};\n"
        f"const gene_x = {json.dumps(gx)};\n"
        f"const gene_y = {json.dumps(gy)};\n"
        f"const gene_z = {json.dumps(gz)};\n"
        f"const cell_x = {json.dumps(cx)};\n"
        f"const cell_y = {json.dumps(cy)};\n"
        f"const cell_z = {json.dumps(cz)};\n"
        f"const umap_x = {json.dumps(umap_x)};\n"
        f"const umap_y = {json.dumps(umap_y)};\n"
        f"const gene_mean = {json.dumps([round(float(v),4) for v in mean_expr])};\n"
        f"const gene_std = {json.dumps([round(float(v),4) for v in std_expr])};\n"
        f"const gene_default_colors = {json.dumps(gene_color_default)};\n"
        f"const cell_dom_color = {json.dumps(cell_dom_color)};\n"
        f"const SET_MASKS = {json.dumps(set_masks)};\n"
        f"const panel_idx = {json.dumps(panel_idx)};\n"
        f"const qc_total = {json.dumps([round(float(v),2) for v in qc_total])};\n"
        f"const qc_ngenes = {json.dumps([int(v) for v in qc_ngenes])};\n"
        f"const qc_ribo = {json.dumps([round(float(v),3) for v in qc_ribo])};\n"
        f"const cell_layer = {json.dumps(cell_layer_list)};\n"
        f"const cell_age = {json.dumps(cell_age_list)};\n"
        f"const DEFAULT_A = {json.dumps(default_a)};\n"
        f"const DEFAULT_B = {json.dumps(default_b)};\n"
        f"const magma = {json.dumps(list(Magma256))};\n"
        f"const viridis = {json.dumps(list(Viridis256))};\n"
    )

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{base.cohort_title(GROUP_NAME).replace(' Atlas', '')} Marker Gene Finder</title>
<style>
{base.UNIFIED_DESIGN_CSS}
html, body {{ margin: 0; padding: 0; }}
button {{ font-size: 13px; padding: 4px 10px; }}
{base.LAYOUT_CSS}
{base.VIZ_NAV_CSS}
{base.BUTTON_CSS}
{base.COLOR_KEY_CSS}
{base.HOME_LINK_CSS}
/* big, centered Find-marker-genes control */
#find-btn {{ font-size: 16px; font-weight: 700; padding: 10px 26px;
             background: var(--earth, #a9794f); color: #fff; border: none;
             border-radius: 9px; cursor: pointer; box-shadow: 0 1px 4px rgba(0,0,0,.12); }}
#find-btn:hover {{ background: var(--earth-dark, #875e38); }}
#marker-results {{ display: flex; flex-direction: column; align-items: center; }}
.mk-grid {{ margin: 0 auto; }}
/* group A/B selector */
.mk-subclass {{ margin: 4px 0; }}
.mk-subclass-head {{ font-size: 12px; margin-bottom: 2px; }}
.mk-subclass-head .ct {{ color: #888; font-weight: 400; }}
.mk-bulk {{ font-size: 11px; padding: 1px 6px; margin-left: 4px; }}
.mk-subclass-subs {{ display: flex; flex-wrap: wrap; gap: 4px; }}
.mk-subt {{ display: inline-flex; align-items: center; gap: 3px; padding: 1px 6px;
            border: 1px solid var(--line); border-radius: 12px; font-size: 12px; background: var(--card); }}
.mk-subt .ct {{ color: var(--muted); }}
.mk-ab {{ font-size: 10px; font-weight: 700; padding: 0 5px; border-radius: 8px;
          border: 1px solid var(--btn-border); background: var(--btn-bg); color: var(--btn-fg); cursor: pointer; }}
.mk-ab.mk-a.on {{ background: {GRP_A_COLOR}; color: #fff; border-color: {GRP_A_COLOR}; }}
.mk-ab.mk-b.on {{ background: {GRP_B_COLOR}; color: #fff; border-color: {GRP_B_COLOR}; }}
/* marker results: gene list + dot plot */
#marker-results {{ width: 100%; }}
.mk-legend {{ font-size: 12px; color: #555; margin: 6px 0; display: flex; gap: 18px;
              flex-wrap: wrap; align-items: center; }}
.mk-grid {{ display: grid; grid-template-columns: minmax(120px, max-content) 44px 44px 1fr;
            align-items: center; gap: 2px 10px; max-width: 560px; }}
.mk-grid .mk-h {{ font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: .04em;
                  padding-bottom: 4px; }}
.mk-row {{ display: contents; cursor: pointer; }}
.mk-row > * {{ padding: 2px 0; }}
.mk-row:hover .mk-gname {{ color: {GRP_A_COLOR}; font-weight: 700; }}
.mk-row.active .mk-gname {{ color: #008b9a; font-weight: 700; }}
.mk-gname {{ font-size: 13px; white-space: nowrap; }}
.mk-cell {{ display: flex; justify-content: center; align-items: center; height: 22px; }}
.mk-dot {{ border-radius: 50%; border: 1px solid rgba(0,0,0,.18); }}
.mk-sep {{ display: flex; align-items: center; gap: 6px; font-size: 11px; color: #555; }}
.mk-sepbar {{ height: 7px; background: #6a8f3c; border-radius: 4px; min-width: 1px; }}
.mk-rule {{ height: 1px; background: #eee; grid-column: 1 / -1; }}
.genestruct-box {{ display: flex; align-items: center; flex-wrap: wrap; gap: 6px;
                   border: 1px solid #cdd6e0; background: #f4f8fc; border-radius: 6px;
                   padding: 6px 10px; margin: 4px 0; font-size: 12px; }}
.genestruct-box .gs-title {{ font-weight: 700; color: #1f5d8c; margin-right: 6px; }}
#genestruct-status {{ color: #555; font-size: 12px; }}
/* lasso selection overlay */
#lasso-overlay {{ display: none; position: fixed; inset: 0; z-index: 9999;
                  background: rgba(0,0,0,.45); align-items: center; justify-content: center; }}
#lasso-panel {{ width: 70vw; height: 70vh; background: #fff; border-radius: 10px;
                box-shadow: 0 8px 40px rgba(0,0,0,.4); display: flex; flex-direction: column;
                padding: 10px 14px; }}
#lasso-head {{ display: flex; align-items: center; justify-content: space-between;
               margin-bottom: 6px; }}
#lasso-title {{ font-size: 15px; font-weight: 700; }}
#lasso-close {{ font-size: 13px; padding: 4px 12px; cursor: pointer; }}
#lasso-plot {{ flex: 1; min-height: 0; }}
#lasso-foot {{ font-size: 12px; color: #444; min-height: 18px; padding-top: 4px; }}
</style>
</head>
<body>
{base.HOME_LINK_HTML}
<div class="wrap">
<h2>{base.cohort_title(GROUP_NAME).replace(' Atlas', '')} Marker Gene Finder</h2>
<div class="hint">Assign subtypes to <b style="color:{GRP_A_COLOR}">Group A</b> and
<b style="color:{GRP_B_COLOR}">Group B</b>, filter the gene pool, then
<b>Find marker genes</b> to rank the genes that most cleanly separate the two groups.</div>

<div class="plot-pair">
  <div class="plot-box"><div class="plot-box-title" id="cell-plot-title"></div>{cells_html}</div>
  <div class="plot-box"><div class="plot-box-title" id="gene-plot-title"></div>{genes_html}</div>
</div>

<div class="controls-row" style="justify-content:center;"><span id="status">Hover a gene (right plot) or a marker row to colour cells by expression.</span></div>

<div class="ctrl-box">
  <div class="ctrl-box-title">Plotting Method</div>
  <div class="controls-row">{base.viz_nav_html(SLUG, 'marker')}</div>
</div>

<div class="ctrl-box">
  <div class="ctrl-box-title">Color Scheme</div>
  <div class="controls-row">
    <button id="c-group" class="qc-btn active" title="Colour cells by group membership (A vs B).">Group (A / B)</button>
    <button id="c-subtype" class="qc-btn">Subtype</button>
    <button id="qc-counts" class="qc-btn">Counts</button>
    <button id="qc-genes" class="qc-btn">Genes</button>
    <button id="qc-ribo" class="qc-btn">% ribo</button>
    <button id="qc-layer" class="qc-btn" title="Colour each cell by its dissected cortical layer (L1 → L6b) on viridis. Cells from multi-layer or pan (e.g. L1-L6) dissections are not layer-specific and shown faint grey. Hidden for cohorts without layer info.">Layer of Microdissection</button>
    <button id="reset-btn">Reset colours (group)</button>
  </div>
</div>

<div class="ctrl-box">
  <div class="ctrl-box-title">Pick the Two Populations to Compare (by Cell Type)</div>
  <div class="controls-row" style="gap:14px;">
    <span id="grp-a-summary" class="label" style="color:{GRP_A_COLOR};font-weight:700;"></span>
    <span id="grp-b-summary" class="label" style="color:{GRP_B_COLOR};font-weight:700;"></span>
    <button id="grp-clear">clear both</button>
  </div>
  <div class="controls-row" style="gap:10px;">
    <button id="replot-panel" title="Re-fit the SVD embedding using only the two selected groups (panel HVG genes), so the axes capture what separates A from B.">Replot on these groups (gene panel)</button>
    <button id="replot-all" title="Re-fit the SVD embedding on the two groups using every gene currently shown in the gene filter.">Replot (shown genes)</button>
  </div>
  <div class="controls-row" style="display:block;">{group_selector_html}</div>
</div>

<div class="ctrl-box">
  <div class="ctrl-box-title">Pick the Two Populations to Compare (by lasso on umap)</div>
  <div class="hint" style="margin-bottom:6px;">Opens a 2D UMAP of the currently-selected cell types (those assigned to Group A or B); draw a loop around cells to reassign them. Lasso overrides layer on top of the subtype-based groups.</div>
  <div class="controls-row" style="gap:10px;">
    <button id="lasso-a-btn" title="Open a 2D UMAP of the selected cell types and lasso (draw a loop around) cells to add them to Group A.">Lasso → A</button>
    <button id="lasso-b-btn" title="Open a 2D UMAP of the selected cell types and lasso (draw a loop around) cells to add them to Group B.">Lasso → B</button>
    <button id="lasso-clear" title="Remove all per-cell lasso overrides (subtype-based groups remain).">Clear lasso selection</button>
    <span id="lasso-box-status" class="label" style="margin-left:8px; color:#666;"></span>
  </div>
</div>

<div class="ctrl-box">
  <div class="ctrl-box-title">Filter Which Genes are Considered</div>
  <div class="controls-row"><span class="label">Gene set:</span>{set_buttons_html}
    <span class="label" style="margin-left:14px;">Find gene:</span>
    <input id="gene-search" list="gene-datalist" placeholder="e.g. Cnr1" autocomplete="off"
           style="width:130px; font-size:12px; padding:2px 6px;">
    <button id="search-clear">clear</button>{gene_datalist}</div>
  <div class="controls-row">
    <span class="label">Min mean expr (log-CPM):</span>
    <input id="mean-slider" type="range" min="{mean_min:.3f}" max="{mean_max:.3f}" step="0.01" value="{mean_min:.3f}">
    <span id="mean-value">{mean_min:.2f}</span>
    <span class="label" style="margin-left:14px;">Min dispersion (std):</span>
    <input id="std-slider" type="range" min="{std_min:.3f}" max="{std_max:.3f}" step="0.01" value="{std_min:.3f}">
    <span id="std-value">{std_min:.2f}</span>
    <span class="label" style="margin-left:14px;">Considered:</span>
    <span id="visible-count">0 / {n_genes}</span>
  </div>
  <div class="genestruct-box" title="Embed the active cells into a K-dim PCA latent space, then score each gene by how much of its variance is recoverable from that embedding (reconstruction R^2). High = the gene tracks the shared cell-state structure of the active cells; low = private/independent variation. The Min recoverability R² slider feeds the considered-gene filter.">
    <span class="gs-title">Gene structure (recoverability)</span>
    <button id="genestruct-btn" class="qc-btn">Score genes</button>
    <button id="genestruct-color" class="qc-btn" disabled>Colour by recoverability</button>
    <span class="label" style="margin-left:8px;">latent dims K:</span>
    <input id="genestruct-k" type="range" min="2" max="100" step="1" value="30" style="width:110px; vertical-align:middle;">
    <span id="genestruct-k-val" style="color:#1f77b4; font-weight:600;">30</span>
    <span class="label" style="margin-left:10px;">Min recoverability R²:</span>
    <input id="genestruct-slider" type="range" min="0" max="1" step="0.01" value="0" style="width:110px; vertical-align:middle;" disabled>
    <span id="genestruct-thr-val">off</span>
    <span id="genestruct-status" style="margin-left:8px;"></span>
  </div>
</div>

<div class="ctrl-box">
  <div class="ctrl-box-title">Marker Genes</div>
  <div class="controls-row">
    <button id="find-btn" title="Rank the considered genes by how cleanly they separate Group A from Group B (AUROC).">Find marker genes</button>
    <span class="label" style="margin-left:10px;">show top</span>
    <input id="topn-input" type="number" min="5" max="100" value="25" step="5" style="width:60px;">
    <span id="find-status" style="margin-left:10px; color:#666;"></span>
  </div>
  <div id="marker-results"></div>
</div>

<footer class="cite">{base.cohort_citation(GROUP_NAME)}</footer>
</div>

<div id="lasso-overlay">
  <div id="lasso-panel">
    <div id="lasso-head">
      <span id="lasso-title"></span>
      <button id="lasso-close" title="Close the lasso overlay.">✕ close</button>
    </div>
    <div id="lasso-plot"></div>
    <div id="lasso-foot"><span id="lasso-status"></span></div>
  </div>
</div>
<script>
{js_data}
{base.COLOR_KEY_JS}
{JS_LOGIC}
</script>
</body></html>
"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write(page)
    sz = os.path.getsize(OUT) / 1e6
    print(f'wrote {OUT} ({sz:.1f} MB)')


# ---- client-side logic (plain string: single braces, no f-string) ----------
JS_LOGIC = r"""
const cellPlot = document.getElementById('cell-plot');
const genePlot = document.getElementById('gene-plot');
const status   = document.getElementById('status');
const POINTS_TRACE = 2, LOADING_TRACE = 3, HIGHLIGHT_TRACE = 4;

// per-cell default subtype colour
const cell_default_colors = new Array(N_CELLS);
const cell_subtype = new Array(N_CELLS);
for (let i = 0; i < N_CELLS; i++) {
  cell_default_colors[i] = SUBTYPE_PALETTE[cell_sub_idx[i]];
  cell_subtype[i] = SUBTYPE_CATS[cell_sub_idx[i]];
}

// state
let groupA = new Set(DEFAULT_A);
let groupB = new Set(DEFAULT_B);
// per-cell override layered on top of the subtype groups (cellIndex -> 'A'|'B'|'none')
let cellGroupOverride = new Map();
let colorMode = 'group';      // group | subtype | counts | genes | ribo | layer | gene
let activeGene = -1;          // gene index when colorMode === 'gene'
let geneActive = new Array(N_GENES).fill(false);
let curMarkers = [];
// readVal(i,j): log-CPM expression the gene-structure PCA needs.
function readVal(i, j){ return expr_matrix[i*N_GENES+j] / EXPR_SCALE; }

// ---- group helpers ---------------------------------------------------------
function cellGroup(i) {
  // a lassoed cell override wins over its subtype-based membership
  if (cellGroupOverride.has(i)) {
    const o = cellGroupOverride.get(i);
    return (o === 'A' || o === 'B') ? o : null;
  }
  const s = cell_subtype[i];
  if (groupA.has(s)) return 'A';
  if (groupB.has(s)) return 'B';
  return null;
}
function cellActiveAt(i) { return cellGroup(i) !== null; }

// ---- colour palettes -------------------------------------------------------
function valuesToViridis(vals, activeMask) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < vals.length; i++) { if (!activeMask[i]) continue;
    const v = vals[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
  const range = (hi > lo) ? (hi - lo) : 1;
  return { lo, hi, colorAt: v => viridis[Math.max(0, Math.min(255, Math.round(255*(v-lo)/range)))] };
}
function exprToMagmaArr(values) {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const range = (hi > lo) ? (hi - lo) : 1;
  return Array.from(values, v => magma[Math.max(0, Math.min(255, Math.round(255*(v-lo)/range)))]);
}

// layer coloring (matches the recompute explorers: single layers only)
function layerToDepth(s) {
  if (s == null) return NaN;
  s = String(s).trim().toUpperCase();
  if (s.indexOf('-') >= 0) return NaN;   // multi-layer / pan dissection: not layer-specific
  if (s === 'L1') return 1.0; if (s === 'L2/3') return 2.5; if (s === 'L4') return 4.0;
  if (s === 'L5') return 5.0; if (s === 'L6') return 6.0; if (s === 'L6B') return 6.5;
  return NaN;
}
const GREY_NO_LAYER = 'rgba(0,0,0,0)';

// ---- titles ----------------------------------------------------------------
let titleGeneRef = 'strongest PC';
function activeCellCount() { let n = 0; for (let i = 0; i < N_CELLS; i++) if (cellActiveAt(i)) n++; return n; }
function colorModeLabel() {
  if (colorMode === 'group') return 'group (A vs B)';
  if (colorMode === 'subtype') return 'subtype';
  if (colorMode === 'counts') return 'total counts';
  if (colorMode === 'genes') return 'genes detected';
  if (colorMode === 'ribo') return '% ribosomal';
  if (colorMode === 'layer') return 'layer of microdissection';
  if (colorMode === 'gene') return gene_name[activeGene] + ' expression';
  return colorMode;
}
function refreshTitles() {
  const ct = document.getElementById('cell-plot-title');
  const gt = document.getElementById('gene-plot-title');
  if (ct) ct.innerHTML = 'Cells on <b>SVD</b> axes · coloured by <b>' + colorModeLabel()
    + '</b> · n=' + activeCellCount().toLocaleString();
  if (gt) gt.innerHTML = 'Genes on <b>SVD</b> axes · coloured by <b>' + titleGeneRef + '</b>';
}

// ---- render the cell biplot (A∪B coloured; other cells recede to grey) ------
// Every cell stays plotted at a uniform marker size — only marker.color is
// restyled (gl3d repaints colour reliably; per-point size/coords do not).
function renderCells() {
  const col = new Array(N_CELLS);
  const active = new Array(N_CELLS);
  for (let i = 0; i < N_CELLS; i++) active[i] = cellActiveAt(i);

  if (colorMode === 'group') {
    for (let i = 0; i < N_CELLS; i++) { const g = cellGroup(i); col[i] = g === 'A' ? GRP_A_COLOR : (g === 'B' ? GRP_B_COLOR : GREY_OFF); }
    setColorKeyCats('', [{color: GRP_A_COLOR, label: 'Group A'}, {color: GRP_B_COLOR, label: 'Group B'}]);
  } else if (colorMode === 'subtype') {
    for (let i = 0; i < N_CELLS; i++) col[i] = active[i] ? cell_default_colors[i] : GREY_OFF;
    clearColorKey();
  } else if (colorMode === 'counts' || colorMode === 'genes' || colorMode === 'ribo') {
    const src = colorMode === 'counts' ? qc_total : (colorMode === 'genes' ? qc_ngenes : qc_ribo);
    const vs = valuesToViridis(src, active);
    for (let i = 0; i < N_CELLS; i++) col[i] = active[i] ? vs.colorAt(src[i]) : GREY_OFF;
    setColorKeyGradient(colorModeLabel(), 'viridis', vs.lo, vs.hi, v => (colorMode === 'ribo' ? v.toFixed(1) + '%' : Math.round(v).toLocaleString()));
  } else if (colorMode === 'layer') {
    if (!cell_layer) { for (let i = 0; i < N_CELLS; i++) col[i] = active[i] ? GREY_NO_LAYER : GREY_OFF; clearColorKey(); }
    else {
      const depths = new Array(N_CELLS);
      for (let i = 0; i < N_CELLS; i++) depths[i] = active[i] ? layerToDepth(cell_layer[i]) : NaN;
      const valid = depths.filter(v => !isNaN(v));
      let lo = 1, hi = 6.5; if (valid.length) { lo = Math.min.apply(null, valid); hi = Math.max.apply(null, valid); }
      const range = (hi > lo) ? (hi - lo) : 1;
      for (let i = 0; i < N_CELLS; i++) {
        if (!active[i]) col[i] = GREY_OFF;
        else if (isNaN(depths[i])) col[i] = GREY_NO_LAYER;
        else col[i] = viridis[Math.max(0, Math.min(255, Math.round(255*(depths[i]-lo)/range)))];
      }
      setColorKeyGradient('layer of microdissection', 'viridis', lo, hi, d => 'L' + Math.round(d));
    }
  } else if (colorMode === 'gene' && activeGene >= 0) {
    const j = activeGene;
    const vals = [], idx = [];
    for (let i = 0; i < N_CELLS; i++) if (active[i]) { vals.push(expr_matrix[i*N_GENES+j]); idx.push(i); }
    const cols = exprToMagmaArr(vals);
    for (let i = 0; i < N_CELLS; i++) col[i] = GREY_OFF;
    for (let k = 0; k < idx.length; k++) col[idx[k]] = cols[k];
    let glo = Infinity, ghi = -Infinity; for (const v of vals) { if (v < glo) glo = v; if (v > ghi) ghi = v; }
    if (glo === Infinity) { glo = 0; ghi = 0; }
    setColorKeyGradient(gene_name[j] + ' expression', 'magma', glo / EXPR_SCALE, ghi / EXPR_SCALE, v => v.toFixed(2));
  }
  Plotly.restyle(cellPlot, {'marker.color': [col]}, [POINTS_TRACE]);
  refreshTitles();
}

// gene biplot coords (for the highlight ring)
const gene_xyz = (function(){ const a = new Array(N_GENES); for (let j=0;j<N_GENES;j++) a[j]=[gene_x[j],gene_y[j],gene_z[j]]; return a; })();

// ---- group selector UI -----------------------------------------------------
function syncGroupButtons() {
  document.querySelectorAll('.mk-ab').forEach(b => {
    const s = b.dataset.sub, g = b.dataset.grp;
    const on = (g === 'A' && groupA.has(s)) || (g === 'B' && groupB.has(s));
    b.classList.toggle('on', on);
  });
  // effective cell counts honour the lasso overrides (cellGroup), not just subtype sets
  let aN = 0, bN = 0;
  for (let i = 0; i < N_CELLS; i++) { const g = cellGroup(i); if (g === 'A') aN++; else if (g === 'B') bN++; }
  let lassoNote = '';
  if (cellGroupOverride.size) {
    let la = 0, lb = 0;
    cellGroupOverride.forEach(v => { if (v === 'A') la++; else if (v === 'B') lb++; });
    lassoNote = ' (incl. ' + la + '→A / ' + lb + '→B lassoed)';
  }
  document.getElementById('grp-a-summary').textContent = 'A: ' + groupA.size + ' subtypes · ' + aN.toLocaleString() + ' cells';
  document.getElementById('grp-b-summary').textContent = 'B: ' + groupB.size + ' subtypes · ' + bN.toLocaleString() + ' cells' + lassoNote;
}
function countCells(set) { let n = 0; for (let i = 0; i < N_CELLS; i++) if (set.has(cell_subtype[i])) n++; return n; }
function assign(sub, grp) {
  groupA.delete(sub); groupB.delete(sub);
  if (grp === 'A') groupA.add(sub); else if (grp === 'B') groupB.add(sub);
}
function onGroupsChanged() {
  syncGroupButtons();
  clearMarkers();
  invalidateGeneStruct();
  renderCells();
}
document.querySelectorAll('.mk-ab').forEach(b => b.addEventListener('click', () => {
  const s = b.dataset.sub, g = b.dataset.grp;
  const already = (g === 'A' && groupA.has(s)) || (g === 'B' && groupB.has(s));
  assign(s, already ? 'none' : g);
  onGroupsChanged();
}));
document.querySelectorAll('.mk-bulk').forEach(b => b.addEventListener('click', () => {
  const grp = b.dataset.grp, csc = b.dataset.subclass;
  document.querySelectorAll('.mk-subclass').forEach(block => {
    if (block.querySelector('.mk-subclass-head b').textContent !== csc) return;
    block.querySelectorAll('.mk-subt').forEach(chip => assign(chip.dataset.sub, grp === 'none' ? 'none' : grp));
  });
  onGroupsChanged();
}));
document.getElementById('grp-clear').addEventListener('click', () => {
  groupA.clear(); groupB.clear(); onGroupsChanged();
});

// ---- lasso overlay: paint cells into A or B by enclosing them ---------------
// 3D Plotly has no native lasso, so we open an overlay with a 2D Plotly scatter
// of a SUBSET of the cells — only the currently-selected cell types (those whose
// subtype is assigned to Group A or B, i.e. cellActiveAt) — laid out on the 2D
// UMAP (umap_x × umap_y) computed at build time. Plotly 2D supports lasso
// reliably. Because the scatter is a subset, one trace holds the selected cells
// in order and `lassoIdxMap[k]` = the global cell index of the k-th plotted
// point; a selection's pointNumber must be mapped back through lassoIdxMap.
let lassoTarget = 'A';   // which group the lassoed cells get assigned to
let lassoIdxMap = [];    // plotted-point-order → global cell index
const lassoOverlay = document.getElementById('lasso-overlay');
const lassoPlotDiv = document.getElementById('lasso-plot');
const lassoTitle   = document.getElementById('lasso-title');
const lassoStatus  = document.getElementById('lasso-status');
const lassoBoxStatus = document.getElementById('lasso-box-status');

// colours for the subset scatter, in lassoIdxMap order (by current group A/B/none)
function lassoSubsetColors() {
  const col = new Array(lassoIdxMap.length);
  for (let k = 0; k < lassoIdxMap.length; k++) {
    const g = cellGroup(lassoIdxMap[k]);
    col[k] = g === 'A' ? GRP_A_COLOR : (g === 'B' ? GRP_B_COLOR : 'rgba(150,150,150,0.45)');
  }
  return col;
}
function openLasso(target) {
  lassoTarget = target;
  // rebuild the subset (selection may have changed since last open): include
  // only cells of the currently-selected cell types (assigned to a group).
  lassoIdxMap = [];
  for (let i = 0; i < N_CELLS; i++) if (cellActiveAt(i)) lassoIdxMap.push(i);
  if (!lassoIdxMap.length) {
    if (lassoBoxStatus) lassoBoxStatus.textContent =
      'No cell types selected — assign subtypes to Group A or B first.';
    return;
  }
  if (lassoBoxStatus) lassoBoxStatus.textContent = '';
  const tcol = target === 'A' ? GRP_A_COLOR : GRP_B_COLOR;
  lassoTitle.innerHTML = 'Lasso cells → <b style="color:' + tcol + '">Group ' + target + '</b>'
    + ' &nbsp;<span style="font-weight:400;color:#666;font-size:12px">'
    + '(UMAP of the ' + lassoIdxMap.length.toLocaleString()
    + ' selected cells; draw a loop around cells to assign them)</span>';
  lassoStatus.textContent = '';
  lassoOverlay.style.display = 'flex';
  const sx = new Array(lassoIdxMap.length), sy = new Array(lassoIdxMap.length);
  for (let k = 0; k < lassoIdxMap.length; k++) { const i = lassoIdxMap[k]; sx[k] = umap_x[i]; sy[k] = umap_y[i]; }
  const data = [{
    x: sx, y: sy, mode: 'markers', type: 'scattergl',
    marker: { size: 5, color: lassoSubsetColors(), line: { width: 0 } },
    hoverinfo: 'skip', showlegend: false
  }];
  const layout = {
    dragmode: 'lasso',
    margin: { l: 30, r: 10, t: 6, b: 30 },
    xaxis: { title: 'UMAP1', zeroline: false },
    yaxis: { title: 'UMAP2', zeroline: false },
    paper_bgcolor: 'white', plot_bgcolor: 'white'
  };
  const config = { displayModeBar: true, responsive: true,
    modeBarButtonsToAdd: ['lasso2d', 'select2d'], displaylogo: false };
  Plotly.newPlot(lassoPlotDiv, data, layout, config).then(() => {
    Plotly.Plots.resize(lassoPlotDiv);
    lassoPlotDiv.on('plotly_selected', function(eventData) {
      if (!eventData || !eventData.points || !eventData.points.length) return;
      let n = 0;
      eventData.points.forEach(pt => {
        const k = pt.pointNumber;        // index into the SUBSET, not the global cell index
        if (k == null || k < 0 || k >= lassoIdxMap.length) return;
        const i = lassoIdxMap[k];        // → global cell index
        if (i != null && i >= 0 && i < N_CELLS) { cellGroupOverride.set(i, lassoTarget); n++; }
      });
      lassoStatus.textContent = 'Assigned ' + n + ' cell' + (n === 1 ? '' : 's')
        + ' → Group ' + lassoTarget + '.';
      // recolour the 2D scatter to reflect the new assignment, then refresh main UI
      Plotly.restyle(lassoPlotDiv, { 'marker.color': [lassoSubsetColors()] }, [0]);
      onGroupsChanged();
    });
  });
}
function closeLasso() {
  lassoOverlay.style.display = 'none';
  try { Plotly.purge(lassoPlotDiv); } catch (e) {}
}
document.getElementById('lasso-a-btn').addEventListener('click', () => openLasso('A'));
document.getElementById('lasso-b-btn').addEventListener('click', () => openLasso('B'));
document.getElementById('lasso-close').addEventListener('click', closeLasso);
document.getElementById('lasso-clear').addEventListener('click', () => {
  if (!cellGroupOverride.size) return;
  cellGroupOverride.clear();
  onGroupsChanged();
});

// ---- colour-scheme buttons -------------------------------------------------
const QC_BTN_IDS = ['c-group','c-subtype','qc-counts','qc-genes','qc-ribo','qc-layer'];
function setActiveBtn(id) { QC_BTN_IDS.forEach(b => { const el = document.getElementById(b); if (el) el.classList.toggle('active', b === id); }); }
function setMode(mode, btnId) { colorMode = mode; activeGene = -1; setActiveBtn(btnId); ringGene(-1); renderCells(); }
document.getElementById('c-group').addEventListener('click', () => setMode('group','c-group'));
document.getElementById('c-subtype').addEventListener('click', () => setMode('subtype','c-subtype'));
document.getElementById('qc-counts').addEventListener('click', () => setMode('counts','qc-counts'));
document.getElementById('qc-genes').addEventListener('click', () => setMode('genes','qc-genes'));
document.getElementById('qc-ribo').addEventListener('click', () => setMode('ribo','qc-ribo'));
document.getElementById('qc-layer').addEventListener('click', () => setMode('layer','qc-layer'));
document.getElementById('reset-btn').addEventListener('click', () => setMode('group','c-group'));
if (!cell_layer) { const lb = document.getElementById('qc-layer'); if (lb) { lb.disabled = true; lb.title = 'No microdissection layer info for this cohort.'; } }

// ---- gene filter -----------------------------------------------------------
let curSet = 'panel';
const meanSlider = document.getElementById('mean-slider');
const stdSlider = document.getElementById('std-slider');
function applyGeneFilter() {
  const mask = SET_MASKS[curSet];
  const mmin = parseFloat(meanSlider.value), smin = parseFloat(stdSlider.value);
  const gsSlider = document.getElementById('genestruct-slider');
  const structThr = (gene_struct && gsSlider) ? parseFloat(gsSlider.value) : 0;
  document.getElementById('mean-value').textContent = mmin.toFixed(2);
  document.getElementById('std-value').textContent = smin.toFixed(2);
  let n = 0;
  const gx = new Array(N_GENES), gy = new Array(N_GENES), gz = new Array(N_GENES);
  for (let j = 0; j < N_GENES; j++) {
    const ok = mask[j] && gene_mean[j] >= mmin && gene_std[j] >= smin
      && (!gene_struct || gene_struct[j] >= structThr);
    geneActive[j] = ok;
    if (ok) { gx[j] = gene_x[j]; gy[j] = gene_y[j]; gz[j] = gene_z[j]; n++; }
    else { gx[j] = null; gy[j] = null; gz[j] = null; }
  }
  Plotly.restyle(genePlot, {x: [gx], y: [gy], z: [gz]}, [POINTS_TRACE]);
  document.getElementById('visible-count').textContent = n + ' / ' + N_GENES;
}
document.querySelectorAll('.set-btn').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('.set-btn').forEach(x => x.classList.remove('active'));
  b.classList.add('active'); curSet = b.dataset.set; applyGeneFilter();
}));
meanSlider.addEventListener('input', applyGeneFilter);
stdSlider.addEventListener('input', applyGeneFilter);

const geneSearch = document.getElementById('gene-search');
geneSearch.addEventListener('change', () => {
  const name = geneSearch.value.trim();
  const j = gene_name.indexOf(name);
  if (j >= 0) { colorByGene(j); }
});
document.getElementById('search-clear').addEventListener('click', () => { geneSearch.value = ''; ringGene(-1); setMode('group','c-group'); });

// ---- gene hover / colour-by-gene + ring ------------------------------------
function ringGene(j) {
  if (j < 0) { Plotly.restyle(genePlot, {x: [[null]], y: [[null]], z: [[null]]}, [HIGHLIGHT_TRACE]); return; }
  Plotly.restyle(genePlot, {x: [[gene_xyz[j][0]]], y: [[gene_xyz[j][1]]], z: [[gene_xyz[j][2]]]}, [HIGHLIGHT_TRACE]);
}
function colorByGene(j) {
  activeGene = j; colorMode = 'gene'; setActiveBtn(null);
  renderCells(); ringGene(j);
  titleGeneRef = gene_name[j]; refreshTitles();
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < N_CELLS; i++) if (cellActiveAt(i)) { const v = expr_matrix[i*N_GENES+j]; if (v<lo) lo=v; if (v>hi) hi=v; }
  if (lo === Infinity) { lo = 0; hi = 0; }
  status.innerHTML = '<b style="color:' + gene_default_colors[j] + '">' + gene_name[j] + '</b>'
    + (gene_in_panel[j] ? ' (panel)' : ' (projected)')
    + ' &nbsp; cells (A∪B) recoloured by expression (range ' + (lo/EXPR_SCALE).toFixed(2) + '..' + (hi/EXPR_SCALE).toFixed(2) + ', magma)';
  // also highlight the matching marker row, if present
  document.querySelectorAll('.mk-row').forEach(r => r.classList.toggle('active', parseInt(r.dataset.gene,10) === j));
}
let lastHoveredGene = null;
genePlot.on('plotly_hover', function(data) {
  const pt = data.points[0]; if (pt.curveNumber !== POINTS_TRACE) return;
  const j = pt.pointNumber; if (!geneActive[j]) return; if (lastHoveredGene === j) return; lastHoveredGene = j;
  colorByGene(j);
});

// ---- marker finding (AUROC) ------------------------------------------------
function auroc(j, aIdx, bIdx) {
  const nA = aIdx.length, nB = bIdx.length, n = nA + nB;
  const v = new Float32Array(n), g = new Uint8Array(n);
  for (let k = 0; k < nA; k++) { v[k] = expr_matrix[aIdx[k]*N_GENES+j]; g[k] = 1; }
  for (let k = 0; k < nB; k++) { v[nA+k] = expr_matrix[bIdx[k]*N_GENES+j]; }
  const ord = Array.from({length: n}, (_, k) => k).sort((p, q) => v[p] - v[q]);
  let rankSumA = 0, i = 0;
  while (i < n) {
    let jx = i; while (jx + 1 < n && v[ord[jx+1]] === v[ord[i]]) jx++;
    const avgRank = (i + jx) / 2 + 1;
    for (let t = i; t <= jx; t++) if (g[ord[t]] === 1) rankSumA += avgRank;
    i = jx + 1;
  }
  const U = rankSumA - nA * (nA + 1) / 2;
  return U / (nA * nB);   // P(A > B)
}
function groupStats(j, idx) {
  let s = 0, nz = 0; const n = idx.length;
  for (let k = 0; k < n; k++) { const e = expr_matrix[idx[k]*N_GENES+j]; s += e; if (e > 0) nz++; }
  return { mean: n ? s / n / EXPR_SCALE : 0, frac: n ? nz / n : 0 };
}
function clearMarkers() { curMarkers = []; document.getElementById('marker-results').innerHTML = ''; document.getElementById('find-status').textContent = ''; }

document.getElementById('find-btn').addEventListener('click', findMarkers);
document.getElementById('replot-panel').addEventListener('click', () => replot(false));
document.getElementById('replot-all').addEventListener('click', () => replot(true));
function findMarkers() {
  const aIdx = [], bIdx = [];
  for (let i = 0; i < N_CELLS; i++) { const g = cellGroup(i); if (g === 'A') aIdx.push(i); else if (g === 'B') bIdx.push(i); }
  const fs = document.getElementById('find-status');
  if (aIdx.length < 2 || bIdx.length < 2) { fs.textContent = 'Assign at least 2 cells to each of Group A and Group B.'; return; }
  const cand = []; for (let j = 0; j < N_GENES; j++) if (geneActive[j]) cand.push(j);
  if (!cand.length) { fs.textContent = 'No genes pass the current filter.'; return; }
  fs.textContent = 'scoring ' + cand.length + ' genes on ' + aIdx.length + ' vs ' + bIdx.length + ' cells …';
  // defer so the status paints first
  setTimeout(() => {
    const scored = cand.map(j => {
      const auc = auroc(j, aIdx, bIdx);
      const sep = Math.max(auc, 1 - auc);            // 0.5 = no separation, 1 = perfect
      return { j: j, auc: auc, sep: sep, up: auc >= 0.5 ? 'A' : 'B' };
    });
    scored.sort((p, q) => q.sep - p.sep);
    const topN = Math.max(1, Math.min(100, parseInt(document.getElementById('topn-input').value, 10) || 25));
    const top = scored.slice(0, topN);
    top.forEach(m => { const a = groupStats(m.j, aIdx), b = groupStats(m.j, bIdx);
      m.meanA = a.mean; m.fracA = a.frac; m.meanB = b.mean; m.fracB = b.frac; });
    curMarkers = top;
    renderMarkerList(top);
    fs.textContent = top.length + ' markers (top ' + topN + ' of ' + cand.length + ' by AUROC).';
  }, 20);
}

function dotSpan(mean, frac, maxMean) {
  const size = 6 + Math.round(13 * Math.max(0, Math.min(1, frac)));
  const t = maxMean > 0 ? Math.max(0, Math.min(1, mean / maxMean)) : 0;
  const c = viridis[Math.round(255 * t)];
  return '<span class="mk-cell"><span class="mk-dot" style="width:' + size + 'px;height:' + size + 'px;background:' + c + '"></span></span>';
}
function renderMarkerList(top) {
  let maxMean = 1e-9; top.forEach(m => { maxMean = Math.max(maxMean, m.meanA, m.meanB); });
  let html = '<div class="mk-legend">'
    + '<span>dot <b>colour</b> = mean expression (viridis), <b>size</b> = fraction expressing</span>'
    + '<span><span class="mk-dot" style="width:14px;height:14px;background:#440154;display:inline-block;vertical-align:middle"></span> low '
    + '<span class="mk-dot" style="width:14px;height:14px;background:#fde725;display:inline-block;vertical-align:middle"></span> high</span>'
    + '<span>bar = separation (AUROC), ▲ marks the higher group</span></div>';
  html += '<div class="mk-grid">'
    + '<div class="mk-h">gene</div>'
    + '<div class="mk-h" style="text-align:center;color:' + GRP_A_COLOR + '">A</div>'
    + '<div class="mk-h" style="text-align:center;color:' + GRP_B_COLOR + '">B</div>'
    + '<div class="mk-h">separation</div>';
  top.forEach(m => {
    const sepPct = Math.round((m.sep - 0.5) / 0.5 * 100);
    const upcol = m.up === 'A' ? GRP_A_COLOR : GRP_B_COLOR;
    html += '<div class="mk-row" data-gene="' + m.j + '">'
      + '<div class="mk-gname">' + gene_name[m.j] + (gene_in_panel[m.j] ? '' : ' <span style="color:#aaa;font-size:10px">(proj)</span>') + '</div>'
      + dotSpan(m.meanA, m.fracA, maxMean)
      + dotSpan(m.meanB, m.fracB, maxMean)
      + '<div class="mk-sep"><span class="mk-sepbar" style="width:' + Math.max(2, sepPct) + '%"></span>'
      + '<span style="color:' + upcol + '">▲' + m.up + '</span> ' + m.sep.toFixed(3) + '</div>'
      + '</div>';
  });
  html += '</div>';
  document.getElementById('marker-results').innerHTML = html;
  document.querySelectorAll('.mk-row').forEach(r => {
    r.addEventListener('mouseenter', () => colorByGene(parseInt(r.dataset.gene, 10)));
    r.addEventListener('click', () => colorByGene(parseInt(r.dataset.gene, 10)));
  });
}

// ---- replot: refit the SVD on just the two groups --------------------------
function powerIterTopK(A, K, maxIter, tol) {
  maxIter = maxIter || 80; tol = tol || 1e-7;
  const m = A.length, n = A[0].length;
  const W = new Array(m);
  for (let i = 0; i < m; i++) W[i] = Float64Array.from(A[i]);
  const U = [], V = []; const S = new Float64Array(K);
  for (let k = 0; k < K; k++) {
    let v = new Float64Array(n);
    for (let j = 0; j < n; j++) v[j] = Math.sin((j + 1) * (k + 1) * 0.13) + 0.1;
    let vn = 0; for (let j = 0; j < n; j++) vn += v[j]*v[j]; vn = Math.sqrt(vn);
    for (let j = 0; j < n; j++) v[j] /= vn;
    let s_prev = 0, s = 0; let u = new Float64Array(m);
    for (let iter = 0; iter < maxIter; iter++) {
      for (let i = 0; i < m; i++) { let acc = 0; const Wi = W[i]; for (let j = 0; j < n; j++) acc += Wi[j]*v[j]; u[i] = acc; }
      let un = 0; for (let i = 0; i < m; i++) un += u[i]*u[i]; un = Math.sqrt(un);
      if (un < 1e-14) break;
      for (let i = 0; i < m; i++) u[i] /= un;
      let v_new = new Float64Array(n);
      for (let i = 0; i < m; i++) { const ui = u[i], Wi = W[i]; for (let j = 0; j < n; j++) v_new[j] += ui*Wi[j]; }
      let sn = 0; for (let j = 0; j < n; j++) sn += v_new[j]*v_new[j]; sn = Math.sqrt(sn);
      if (sn < 1e-14) { s = 0; break; }
      for (let j = 0; j < n; j++) v_new[j] /= sn;
      s = sn; v = v_new;
      if (iter > 1 && Math.abs(s - s_prev) < tol * s) break;
      s_prev = s;
    }
    S[k] = s; U.push(Float64Array.from(u)); V.push(Float64Array.from(v));
    for (let i = 0; i < m; i++) { const sui = s*u[i]; const Wi = W[i]; for (let j = 0; j < n; j++) Wi[j] -= sui*v[j]; }
  }
  return { U, S, V };
}
function _robustMax(absVals) {
  const v = absVals.slice().sort((a, b) => a - b);
  return v[Math.floor(0.995 * (v.length - 1))] || 1;
}
function replot(useAllGenes) {
  const sel = []; for (let i = 0; i < N_CELLS; i++) if (cellActiveAt(i)) sel.push(i);
  const fs = document.getElementById('find-status');
  if (sel.length < 4) { if (fs) fs.textContent = 'Assign cells to both groups first.'; return; }
  const basisIdx = [];
  for (let j = 0; j < N_GENES; j++) if (geneActive[j] && (useAllGenes || gene_in_panel[j])) basisIdx.push(j);
  if (basisIdx.length < 3) { if (fs) fs.textContent = 'Too few genes in the basis — relax the gene filter.'; return; }
  const m = sel.length, n = basisIdx.length;
  // z-score basis genes over the selected cells, build Zp (m × n)
  const bmean = new Float64Array(n), bstd = new Float64Array(n);
  for (let g = 0; g < n; g++) {
    const j = basisIdx[g]; let s = 0;
    for (let a = 0; a < m; a++) s += expr_matrix[sel[a]*N_GENES + j] / EXPR_SCALE;
    const mu = s / m; let vv = 0;
    for (let a = 0; a < m; a++) { const x = expr_matrix[sel[a]*N_GENES + j]/EXPR_SCALE - mu; vv += x*x; }
    bmean[g] = mu; bstd[g] = Math.sqrt(vv / Math.max(1, m-1)) || 1;
  }
  const Zp = new Array(m);
  for (let a = 0; a < m; a++) {
    const row = new Float64Array(n), i = sel[a];
    for (let g = 0; g < n; g++) row[g] = (expr_matrix[i*N_GENES + basisIdx[g]]/EXPR_SCALE - bmean[g]) / bstd[g];
    Zp[a] = row;
  }
  const { U, S } = powerIterTopK(Zp, 3);
  // cell scores → robust-normalised cube coords
  const cmax = [1,1,1];
  for (let k = 0; k < 3; k++) cmax[k] = _robustMax(Array.from(U[k], (u,a) => Math.abs(u*S[k])));
  for (let a = 0; a < m; a++) {
    const i = sel[a];
    cell_x[i] = Math.max(-1, Math.min(1, U[0][a]*S[0]/cmax[0]));
    cell_y[i] = Math.max(-1, Math.min(1, U[1][a]*S[1]/cmax[1]));
    cell_z[i] = Math.max(-1, Math.min(1, U[2][a]*S[2]/cmax[2]));
  }
  // gene loadings on the new basis (all genes), z-scored over the selected cells
  const gmean = new Float64Array(N_GENES), gstd = new Float64Array(N_GENES);
  for (let j = 0; j < N_GENES; j++) {
    let s = 0; for (let a = 0; a < m; a++) s += expr_matrix[sel[a]*N_GENES + j]/EXPR_SCALE;
    const mu = s / m; let vv = 0;
    for (let a = 0; a < m; a++) { const x = expr_matrix[sel[a]*N_GENES + j]/EXPR_SCALE - mu; vv += x*x; }
    gmean[j] = mu; gstd[j] = Math.sqrt(vv / Math.max(1, m-1)) || 1;
  }
  const gl = [new Float64Array(N_GENES), new Float64Array(N_GENES), new Float64Array(N_GENES)];
  for (let a = 0; a < m; a++) {
    const i = sel[a], u0 = U[0][a], u1 = U[1][a], u2 = U[2][a], base = i*N_GENES;
    for (let j = 0; j < N_GENES; j++) {
      const z = (expr_matrix[base + j]/EXPR_SCALE - gmean[j]) / gstd[j];
      gl[0][j] += z*u0; gl[1][j] += z*u1; gl[2][j] += z*u2;
    }
  }
  for (let k = 0; k < 3; k++) { const inv = S[k] > 1e-9 ? 1/S[k] : 0; for (let j = 0; j < N_GENES; j++) gl[k][j] *= inv; }
  const gmaxA = [_robustMax(Array.from(gl[0], Math.abs)), _robustMax(Array.from(gl[1], Math.abs)), _robustMax(Array.from(gl[2], Math.abs))];
  for (let j = 0; j < N_GENES; j++) {
    gene_x[j] = Math.max(-1, Math.min(1, gl[0][j]/gmaxA[0]));
    gene_y[j] = Math.max(-1, Math.min(1, gl[1][j]/gmaxA[1]));
    gene_z[j] = Math.max(-1, Math.min(1, gl[2][j]/gmaxA[2]));
    gene_xyz[j] = [gene_x[j], gene_y[j], gene_z[j]];
  }
  // push new coords to both plots, then recolour + refilter
  const cx = new Array(N_CELLS), cy = new Array(N_CELLS), cz = new Array(N_CELLS);
  for (let i = 0; i < N_CELLS; i++) { if (cellActiveAt(i)) { cx[i]=cell_x[i]; cy[i]=cell_y[i]; cz[i]=cell_z[i]; } else { cx[i]=null; cy[i]=null; cz[i]=null; } }
  Plotly.restyle(cellPlot, {x:[cx], y:[cy], z:[cz]}, [POINTS_TRACE]);
  renderCells();
  applyGeneFilter();
  ringGene(-1);
  if (fs) fs.textContent = 'Re-fit SVD on ' + m + ' cells × ' + n + (useAllGenes ? ' shown' : ' panel') + ' genes.';
}

// ---- Gene-structure (recoverability R^2) ----------------------------------
let gene_struct = null;
const gsBtn    = document.getElementById('genestruct-btn');
const gsColor  = document.getElementById('genestruct-color');
const gsK      = document.getElementById('genestruct-k');
const gsKval   = document.getElementById('genestruct-k-val');
const gsSlider2= document.getElementById('genestruct-slider');
const gsThrVal = document.getElementById('genestruct-thr-val');
const gsStatus = document.getElementById('genestruct-status');
gsK.addEventListener('input', () => { gsKval.textContent = gsK.value; });
function invalidateGeneStruct() {
  if (!gene_struct) return;
  gene_struct = null;
  gsSlider2.disabled = true; gsColor.disabled = true;
  gsStatus.innerHTML = '<span style="color:#999">cell selection changed — rescore</span>';
  applyGeneFilter();
}
function computeGeneStructure() {
  const cellSel = [];
  for (let i = 0; i < N_CELLS; i++) if (cellActiveAt(i)) cellSel.push(i);
  const m = cellSel.length;
  if (m < 5) { gsStatus.innerHTML = '<span style="color:#c00">need &ge;5 cells</span>'; return; }
  const np = panel_idx.length;
  let K = Math.max(2, Math.min(parseInt(gsK.value) || 30, np - 1, m - 1));
  const Zp = new Array(m); for (let i = 0; i < m; i++) Zp[i] = new Float64Array(np);
  for (let k = 0; k < np; k++) {
    const j = panel_idx[k]; let s = 0, ss = 0;
    for (let ii = 0; ii < m; ii++) { const v = readVal(cellSel[ii], j); Zp[ii][k] = v; s += v; ss += v*v; }
    const mean = s/m, sd = Math.sqrt(Math.max(ss/m - mean*mean, 1e-18));
    for (let ii = 0; ii < m; ii++) Zp[ii][k] = (Zp[ii][k] - mean) / sd;
  }
  const C = new Array(np);
  for (let a = 0; a < np; a++) { const Ca = new Float64Array(np);
    for (let ii = 0; ii < m; ii++) { const za = Zp[ii][a]; if (za === 0) continue;
      const Zi = Zp[ii]; for (let b = a; b < np; b++) Ca[b] += za * Zi[b]; }
    C[a] = Ca; }
  for (let a = 0; a < np; a++) for (let b = a+1; b < np; b++) C[b][a] = C[a][b];
  const {U: Vc, S: eig} = powerIterTopK(C, K, 60);
  const Uk = [];
  for (let k = 0; k < K; k++) {
    const sk = Math.sqrt(Math.max(eig[k], 1e-18));
    const uk = new Float64Array(m); const vk = Vc[k];
    for (let ii = 0; ii < m; ii++) { let acc = 0; const Zi = Zp[ii];
      for (let a = 0; a < np; a++) acc += Zi[a] * vk[a]; uk[ii] = acc / sk; }
    Uk.push(uk);
  }
  const n_all = gene_name.length;
  const r2 = new Float32Array(n_all);
  for (let j = 0; j < n_all; j++) {
    let s = 0, ss = 0;
    for (let ii = 0; ii < m; ii++) { const v = readVal(cellSel[ii], j); s += v; ss += v*v; }
    const mean = s/m, varj = Math.max(ss/m - mean*mean, 1e-18);
    let energy = 0;
    for (let k = 0; k < K; k++) { const uk = Uk[k]; let dot = 0;
      for (let ii = 0; ii < m; ii++) dot += (readVal(cellSel[ii], j) - mean) * uk[ii];
      energy += dot * dot; }
    r2[j] = Math.max(0, Math.min(1, energy / (varj * m)));
  }
  gene_struct = r2;
  gsSlider2.disabled = false; gsColor.disabled = false;
  gsThrVal.textContent = parseFloat(gsSlider2.value) > 0 ? (+gsSlider2.value).toFixed(2) : 'off';
  colorGenesByStruct();
  applyGeneFilter();
  const order = Array.from(r2.keys()).sort((a,b) => r2[b] - r2[a]);
  const top = order.slice(0, 8).map(j => gene_name[j] + ' (' + r2[j].toFixed(2) + ')');
  const panelMean = panel_idx.reduce((a,j)=>a+r2[j],0)/np;
  gsStatus.innerHTML = '<b>K=' + K + '</b>, ' + m + ' cells · top: ' + top.join(', ')
    + ' · panel mean R²=' + panelMean.toFixed(2);
}
function colorGenesByStruct() {
  if (!gene_struct) return;
  // recolour only the GENE biplot by recoverability; leave the cell plot's
  // current colour mode untouched (marker has no 'qc' cell mode).
  const colors = Array.from(gene_struct, v => viridis[Math.max(0, Math.min(255, Math.round(255*v)))]);
  Plotly.restyle(genePlot, {'marker.color': [colors]}, [POINTS_TRACE]);
  setColorKeyGradient('gene recoverability R²', 'viridis', 0, 1, v => v.toFixed(2));
}
gsBtn.addEventListener('click', () => {
  gsBtn.disabled = true; gsStatus.textContent = 'scoring…';
  setTimeout(() => { try { computeGeneStructure(); } finally { gsBtn.disabled = false; } }, 30);
});
gsColor.addEventListener('click', colorGenesByStruct);
gsSlider2.addEventListener('input', () => {
  gsThrVal.textContent = parseFloat(gsSlider2.value) > 0 ? (+gsSlider2.value).toFixed(2) : 'off';
  applyGeneFilter();
});

// ---- boot ------------------------------------------------------------------
function boot() {
  applyGeneFilter();
  syncGroupButtons();
  renderCells();
  refreshTitles();
  // Some WebGL backends skip the first paint of a gl3d scene until a
  // resize/restyle nudges it. Force a resize + repaint shortly after load.
  setTimeout(function() {
    try {
      Plotly.Plots.resize(cellPlot);
      Plotly.Plots.resize(genePlot);
      renderCells();
    } catch (e) {}
  }, 250);
}
if (cellPlot.data) boot(); else cellPlot.on('plotly_afterplot', function once() { boot(); cellPlot.removeListener('plotly_afterplot', once); });
"""


if __name__ == '__main__':
    # cohort is selected by base from sys.argv[1] at import time
    main()
