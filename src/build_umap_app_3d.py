#!/usr/bin/env python
"""UMAP variant of the archetype-explorer family — cells & genes on a shared 3D UMAP.

Fourth member alongside the NMF tetrahedron, SVD biplot, and diffmap viewers. The
latent space is a 3D UMAP of the cells (scanpy pipeline: PCA → kNN → UMAP). A UMAP
is a *cell* embedding, so genes have no native dual — each gene is placed at the
**expression-weighted centroid** of the cell UMAP coordinates (the centre of mass
of the cells expressing it):

  cells   : UMAP1..UMAP3                                  (random_state=0)
  genes   : centroid_k(g) = Σ_i w_ig·UMAP_ik / Σ_i w_ig    with w_ig = expr of g in cell i

Both panel and broader genes are placed the same way. Each panel is scaled per-axis
to fill the cube; the 6 signed poles are ±UMAP1, ±UMAP2, ±UMAP3.

This explorer is *static* — UMAP fitting depends on stochastic optimization, so the
recompute pattern used by the SVD/diffmap explorers is not applied here. For
subtype-subset refits, hop over to the SVD or diffmap viewers via the nav.

Usage:  python build_umap_app_3d.py [GROUP]   # Lamp5 (default), Sst, Pvalb, Vip, Sncg
Output: notebooks/{group}_umap_explorer_3d.html
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import anndata as ad_mod
import scanpy as sc
import plotly.graph_objects as go
from plotly.io import to_html
from bokeh.palettes import Magma256, Viridis256, Category20, Set3, Set1, Category10

import build_lamp5_archetype_app_4d as base
sc.settings.verbosity = 0

GROUP_NAME = base.GROUP_NAME
SLUG       = base.SLUG
OUT        = os.path.join(base.ROOT, 'notebooks', f'{SLUG}_umap_explorer_3d.html')
NUC        = 3   # number of UMAP components

POLE_COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf']
POLE_NAMES  = ['UMAP1+', 'UMAP1-', 'UMAP2+', 'UMAP2-', 'UMAP3+', 'UMAP3-']


def main():
    proj      = base.compute_or_load_proj_full()
    gene_names = proj['gene_names']
    in_panel  = np.array(proj['in_panel'])
    mean_expr = np.asarray(proj['mean_expr'])
    std_expr  = np.asarray(proj['std_expr'])
    X_keep    = np.asarray(proj['X_keep'], dtype=np.float64)
    subs      = np.array(proj['subs'])
    n_cells   = X_keep.shape[0]
    n_genes   = len(gene_names)

    qc = base.compute_or_load_qc_full()
    assert np.array_equal(np.array(qc['subs']), subs), 'QC/proj cell-order mismatch'
    qc_total, qc_ngenes, qc_ribo = (np.asarray(qc['total_counts']),
                                    np.asarray(qc['n_genes']),
                                    np.asarray(qc['pct_ribo']))

    # ---- 3D UMAP on the z-scored PANEL matrix ----------------------------------
    Xp = X_keep[:, in_panel]
    Zp = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-9)
    adp = ad_mod.AnnData(Zp.astype(np.float32))
    sc.pp.pca(adp, n_comps=min(30, n_cells - 1), random_state=0)
    sc.pp.neighbors(adp, n_neighbors=15, random_state=0)
    sc.tl.umap(adp, n_components=NUC, random_state=0, min_dist=0.3)
    UC = np.asarray(adp.obsm['X_umap'], dtype=np.float64)
    print(f'  UMAP fitted: n_cells={n_cells}, panel HVG={int(in_panel.sum())}')

    # ---- genes = expression-weighted centroid of cell UMAP coords --------------
    Wn = X_keep.copy()
    Wn[Wn < 0] = 0
    Wn = Wn / (Wn.sum(0, keepdims=True) + 1e-9)
    gene_uc = Wn.T @ UC

    def fill_cube(M):
        return M / (np.max(np.abs(M - M.mean(0)), axis=0) + 1e-9) * 0.95 \
            if False else (M - M.mean(0)) / (np.max(np.abs(M - M.mean(0)), axis=0) + 1e-9)
    # UMAP coordinates are arbitrary; centre then per-axis max-abs scale to [-1, 1].
    cell_xyz = fill_cube(UC)
    gene_xyz = fill_cube(gene_uc)

    panel_idx = np.where(in_panel)[0]
    panel_names = [gene_names[i] for i in panel_idx]
    pole_top = []
    for k in range(NUC):
        col = gene_xyz[panel_idx, k]
        pole_top.append(panel_names[int(np.argmax(col))])
        pole_top.append(panel_names[int(np.argmin(col))])

    cats = sorted(set(subs.tolist()))
    base_pal = list(Category20[20]) + list(Set3[12]) + list(Set1[9]) + list(Category10[10])
    subtype_palette = {c: base_pal[i % len(base_pal)] for i, c in enumerate(cats)}
    cell_color_default = [subtype_palette[s] for s in subs]

    def signed_pole(v3):
        k = int(np.argmax(np.abs(v3)))
        return 2 * k + (0 if v3[k] >= 0 else 1)
    gene_pole = np.array([signed_pole(gene_xyz[j]) for j in range(n_genes)])
    cell_pole = np.array([signed_pole(cell_xyz[i]) for i in range(n_cells)])
    gene_color_default = [POLE_COLORS[p] for p in gene_pole]
    cell_dom_color     = [POLE_COLORS[p] for p in cell_pole]
    gene_dom_color     = gene_color_default

    def pole_loads(M):
        out = np.zeros((M.shape[0], 6))
        for k in range(NUC):
            out[:, 2*k]   = np.clip(M[:, k], 0, None)
            out[:, 2*k+1] = np.clip(-M[:, k], 0, None)
        return out
    cell_load = pole_loads(cell_xyz).round(4).tolist()
    gene_load = pole_loads(gene_xyz).round(4).tolist()

    EXPR_SCALE = 10
    expr_matrix = np.round(X_keep * EXPR_SCALE).astype(np.int16).tolist()

    LIM, POLE = 1.0, 1.22
    axis_ends = np.array([[ POLE,0,0],[-POLE,0,0],[0, POLE,0],
                          [0,-POLE,0],[0,0, POLE],[0,0,-POLE]])

    def build_fig(xyz, colors, hover_text, title):
        ax_x, ax_y, ax_z = [], [], []
        for a, b in [((-LIM,0,0),(LIM,0,0)), ((0,-LIM,0),(0,LIM,0)), ((0,0,-LIM),(0,0,LIM))]:
            ax_x += [a[0], b[0], None]; ax_y += [a[1], b[1], None]; ax_z += [a[2], b[2], None]
        edge_trace = go.Scatter3d(x=ax_x, y=ax_y, z=ax_z, mode='lines',
                                  line=dict(color='lightgray', width=2),
                                  hoverinfo='skip', showlegend=False)
        pole_lab = [f'{POLE_NAMES[p]}<br>({pole_top[p]})' for p in range(6)]
        vertex_trace = go.Scatter3d(
            x=axis_ends[:,0], y=axis_ends[:,1], z=axis_ends[:,2],
            mode='markers+text', marker=dict(size=4, color=POLE_COLORS),
            text=pole_lab, textposition='top center',
            textfont=dict(size=11, color='black'),
            hoverinfo='text', hovertext=pole_lab, showlegend=False)
        points_trace = go.Scatter3d(
            x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], mode='markers',
            marker=dict(size=4, color=colors, opacity=0.85, line=dict(width=0)),
            text=hover_text, hoverinfo='text', showlegend=False)
        loading_trace = go.Scatter3d(
            x=axis_ends[:,0], y=axis_ends[:,1], z=axis_ends[:,2], mode='markers',
            marker=dict(size=16, color=['#e0e0e0']*6, opacity=1.0,
                        line=dict(width=1.5, color='#222')),
            hoverinfo='text', hovertext=POLE_NAMES, showlegend=False)
        highlight_trace = go.Scatter3d(
            x=[None], y=[None], z=[None], mode='markers',
            marker=dict(size=15, color='rgba(0,0,0,0)',
                        line=dict(width=4, color='#00e5ff')),
            hoverinfo='skip', showlegend=False, name='search')
        fig = go.Figure(data=[edge_trace, vertex_trace, points_trace,
                              loading_trace, highlight_trace])
        lim = POLE * 1.12
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=13)),
            scene=dict(
                xaxis=dict(visible=False, range=[-lim, lim]),
                yaxis=dict(visible=False, range=[-lim, lim]),
                zaxis=dict(visible=False, range=[-lim, lim]),
                aspectmode='cube', dragmode='orbit',
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.4),
                            center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1))),
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor='white', plot_bgcolor='white')
        return fig

    cell_hover_text = [
        f'#{i}<br>subtype: {subs[i]}<br>'
        f'UMAP1,2,3 = ({UC[i,0]:.2f}, {UC[i,1]:.2f}, {UC[i,2]:.2f})'
        for i in range(n_cells)]
    gene_hover_text = [
        f'<b>{gene_names[j]}</b>'
        + (' (panel HVG)' if in_panel[j] else ' (broader)')
        + f'<br>centroid pole: {POLE_NAMES[gene_pole[j]]} ({pole_top[gene_pole[j]]})<br>'
        + f'mean={mean_expr[j]:.2f}, std={std_expr[j]:.2f}<br>'
        + f'centroid UMAP1,2,3 = ({gene_uc[j,0]:.2f}, {gene_uc[j,1]:.2f}, {gene_uc[j,2]:.2f})'
        for j in range(n_genes)]

    n_panel_disp = int(in_panel.sum())
    n_imputed    = n_genes - n_panel_disp
    historical_outliers = list(base.GROUP['cache_outliers']) + list(base.GROUP['runtime_exclude'])
    excluded_blurb = (
        f'Historically-flagged outlier subtypes ({", ".join(historical_outliers)}) are '
        f'<b>included</b> here.' if historical_outliers
        else 'No subtypes are flagged as outliers.')

    fig_cells = build_fig(cell_xyz, cell_color_default, cell_hover_text,
                          f'Cells — 3D UMAP (n={n_cells})  '
                          f'<i>hover → genes recolor; ±UMAP dots = this cell\'s coords</i>')
    fig_genes = build_fig(gene_xyz, gene_color_default, gene_hover_text,
                          f'Genes — expression-weighted UMAP centroids '
                          f'({n_panel_disp} panel + {n_imputed} broader)')

    cells_html = to_html(fig_cells, include_plotlyjs='cdn', full_html=False,
                          div_id='cell-plot', config={'displayModeBar': True, 'responsive': True})
    genes_html = to_html(fig_genes, include_plotlyjs=False, full_html=False,
                          div_id='gene-plot', config={'displayModeBar': True, 'responsive': True})

    sub_legend = ''.join(
        f'<span style="display:inline-block;width:10px;height:10px;background:{subtype_palette[s]};'
        f'margin-right:4px;border-radius:50%;"></span> {s} &nbsp;&nbsp;' for s in cats)
    pole_legend = ''.join(
        f'<span style="display:inline-block;width:10px;height:10px;background:{POLE_COLORS[p]};'
        f'margin-right:4px;border-radius:50%;"></span> {POLE_NAMES[p]} ({pole_top[p]}) &nbsp;&nbsp;'
        for p in range(6))

    gx, gy, gz = (np.round(gene_xyz[:, k], 4).tolist() for k in range(3))

    panel_mask_list = in_panel.tolist()
    set_masks  = {'all': [True]*n_genes, 'panel': panel_mask_list}
    set_counts = {'all': n_genes,        'panel': n_panel_disp}
    for name, gene_list in base.GENE_SETS.items():
        gset = set(gene_list)
        mask = [g in gset for g in gene_names]
        set_masks[name] = mask; set_counts[name] = sum(mask)

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
        f"const gene_mean    = {json.dumps([round(float(v),3) for v in mean_expr])};\n"
        f"const gene_std     = {json.dumps([round(float(v),3) for v in std_expr])};\n"
        f"const gene_x       = {json.dumps(gx)};\n"
        f"const gene_y       = {json.dumps(gy)};\n"
        f"const gene_z       = {json.dumps(gz)};\n"
        f"const cell_load    = {json.dumps(cell_load)};\n"
        f"const gene_load    = {json.dumps(gene_load)};\n"
        f"const cell_score   = {json.dumps(UC.round(3).tolist())};\n"
        f"const gene_loading = {json.dumps(gene_uc.round(3).tolist())};\n"
        f"const cell_dom_color = {json.dumps(cell_dom_color)};\n"
        f"const gene_dom_color = {json.dumps(gene_dom_color)};\n"
        f"const gene_sets    = {json.dumps(set_masks)};\n"
        f"const gene_set_counts = {json.dumps(set_counts)};\n"
        f"const magma        = {json.dumps(list(Magma256))};\n"
        f"const viridis      = {json.dumps(list(Viridis256))};\n"
        f"const qc_total     = {json.dumps([round(float(v), 1) for v in qc_total])};\n"
        f"const qc_ngenes    = {json.dumps([int(v) for v in qc_ngenes])};\n"
        f"const qc_ribo      = {json.dumps([round(float(v), 2) for v in qc_ribo])};\n"
    )

    button_order = ['panel', 'all'] + [n for n in base.GENE_SET_ORDER if n != 'all']
    set_label = lambda n: 'Panel HVG' if n == 'panel' else base.GENE_SET_LABELS.get(n, n)
    set_buttons_html = ''.join(
        f'<button class="set-btn{" active" if name == "panel" else ""}" data-set="{name}"'
        f'{" disabled" if set_counts.get(name, 0) == 0 else ""}>'
        f'{set_label(name)} ({set_counts.get(name, 0)})</button>'
        for name in button_order)
    gene_datalist = ('<datalist id="gene-datalist">'
                     + ''.join(f'<option value="{g}">' for g in gene_names)
                     + '</datalist>')

    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{GROUP_NAME} 3D UMAP explorer</title>
<style>
html, body {{ height: 100%; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
        display: flex; flex-direction: column; padding: 6px 12px; box-sizing: border-box; }}
h2 {{ margin: 0 0 2px 0; }}
.hint {{ flex: 0 0 auto; font-size: 17px; color: #222; line-height: 1.3; margin: 2px 0 6px 0; }}
.hint b {{ color: #1f77b4; }}
.header {{ flex: 0 0 auto; font-size:12px; color:#444; line-height:1.35; }}
.controls {{ flex: 0 0 auto; margin: 4px 0; display: flex; flex-direction: column; gap: 4px; font-size: 12px; }}
.controls-row {{ display: flex; align-items: center; flex-wrap: wrap; gap: 6px; }}
.controls-row .label {{ color: #555; font-weight: 600; }}
.set-btn {{ padding: 3px 8px; font-size: 12px; border: 1px solid #bbb; background: #f6f6f6;
            border-radius: 3px; cursor: pointer; }}
.set-btn:hover {{ background: #eee; }}
.set-btn.active {{ background: #1f77b4; color: white; border-color: #1f77b4; }}
#mean-slider, #std-slider {{ width: 180px; }}
.row {{ flex: 1 1 auto; display: flex; flex-direction: row; gap: 8px; min-height: 0; }}
.col {{ flex: 1 1 0; min-width: 0; display: flex; flex-direction: column; }}
.col > .plotly-graph-div {{ flex: 1 1 auto; min-height: 0; height: 100% !important; }}
#status {{ color: #555; font-size: 12px; }}
.legend {{ flex: 0 0 auto; font-size: 11px; color: #444; margin-top: 4px; }}
button {{ font-size: 13px; padding: 4px 10px; }}
details summary {{ cursor: pointer; color: #666; font-size: 12px; }}
{base.VIZ_NAV_CSS}
</style>
</head>
<body>
<h2>{GROUP_NAME} 3D UMAP — rotatable cells & gene centroids</h2>
{base.viz_nav_html(SLUG, 'umap')}
<div class="hint">
<b>Hover over a cell on the left</b> to see its expression of the genes on the right.
<b>Hover over a gene on the right</b> to see its expression in the cells on the left.
</div>
<details class="header">
<summary>About this app (click to expand)</summary>
<div style="margin-top:4px;">
This is the <b>3D UMAP</b> member of the explorer family. Cells are embedded by scanpy's
UMAP (PCA → 15-NN → UMAP, <code>n_components=3, min_dist=0.3, random_state=0</code>) on the
{n_panel_disp}-HVG z-scored panel. UMAP is a <i>cell</i> embedding, so genes have no native
dual — each gene is placed at the <b>expression-weighted centroid</b> of the cell UMAP
coordinates. The {n_imputed} broader genes are placed the same way. Each panel is mean-
centred then per-axis max-abs scaled to fill the [-1, 1] cube. Poles are ±UMAP1, ±UMAP2,
±UMAP3 with labels = most-extreme panel-gene centroid on each axis.<br>
UMAP fitting is stochastic, so this explorer is <i>static</i>. For subtype-subset refits
hop over to the SVD or diffmap viewers via the nav row. {excluded_blurb}
</div>
</details>
<div class="controls">
  <div class="controls-row"><span class="label">Gene set:</span>{set_buttons_html}
    <span class="label" style="margin-left:14px;">Find gene:</span>
    <input id="gene-search" list="gene-datalist" placeholder="e.g. Cnr1" autocomplete="off"
           style="width:130px; font-size:12px; padding:2px 6px;">
    <button id="search-clear">clear</button>
    {gene_datalist}</div>
  <div class="controls-row">
    <span class="label">Min mean expr (log-CPM):</span>
    <input id="mean-slider" type="range" min="{mean_min:.3f}" max="{mean_max:.3f}" step="0.01" value="{mean_min:.3f}">
    <span id="mean-value">{mean_min:.2f}</span>
    <span class="label" style="margin-left:14px;">Min dispersion (std):</span>
    <input id="std-slider" type="range" min="{std_min:.3f}" max="{std_max:.3f}" step="0.01" value="{std_min:.3f}">
    <span id="std-value">{std_min:.2f}</span>
    <span class="label" style="margin-left:14px;">Visible:</span>
    <span id="visible-count">{n_panel_disp} / {n_genes}</span>
    <span style="margin-left:auto; display:flex; gap:6px; align-items:center;">
      <span class="label">QC colour:</span>
      <button id="qc-counts" class="qc-btn">Counts</button>
      <button id="qc-genes" class="qc-btn">Genes</button>
      <button id="qc-ribo" class="qc-btn">% ribo</button>
      <button id="reset-btn">Reset colours</button>
    </span>
  </div>
  <div class="controls-row"><span id="status">Hover a cell (left) or a gene (right) to colour by expression.</span></div>
</div>
<div class="row">
  <div class="col">{cells_html}</div>
  <div class="col">{genes_html}</div>
</div>
<div class="legend">
<b>Cell default colours</b> (subtype): {sub_legend}<br>
<b>Gene default colours</b> (strongest signed UMAP axis): {pole_legend}
</div>
<script>
{js_data}

function exprToMagma(values) {{
  let lo = Infinity, hi = -Infinity;
  for (const v of values) {{ if (v < lo) lo = v; if (v > hi) hi = v; }}
  const range = (hi > lo) ? (hi - lo) : 1;
  return values.map(v => magma[Math.max(0, Math.min(255, Math.round(255*(v-lo)/range)))]);
}}
function loadingToMagma(loadings) {{
  return loadings.map(v => magma[Math.round(255 * Math.max(0, Math.min(1, v)))]);
}}

const cellPlot = document.getElementById('cell-plot');
const genePlot = document.getElementById('gene-plot');
const status   = document.getElementById('status');
const POINTS_TRACE = 2, LOADING_TRACE = 3, HIGHLIGHT_TRACE = 4;
const DEFAULT_LOAD_COLORS = ['#e0e0e0','#e0e0e0','#e0e0e0','#e0e0e0','#e0e0e0','#e0e0e0'];
let lastHoveredCell = null, lastHoveredGene = null;

cellPlot.on('plotly_hover', function(data) {{
  const pt = data.points[0]; if (pt.curveNumber !== POINTS_TRACE) return;
  const i = pt.pointNumber; if (lastHoveredCell === i) return; lastHoveredCell = i;
  const row = expr_matrix[i];
  Plotly.restyle(genePlot, {{'marker.color': [exprToMagma(row)]}}, [POINTS_TRACE]);
  const lc = loadingToMagma(cell_load[i]);
  Plotly.restyle(cellPlot, {{'marker.color': [lc]}}, [LOADING_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color': [lc]}}, [LOADING_TRACE]);
  let lo = Infinity, hi = -Infinity; for (const v of row) {{ if (v<lo) lo=v; if (v>hi) hi=v; }}
  const s = cell_score[i];
  status.innerHTML = '<b style="color:' + cell_dom_color[i] + '">Cell #' + i
    + '</b> <span style="color:#555">(' + cell_subtype[i] + ')</span> &nbsp; '
    + 'UMAP1,2,3 = (' + s[0].toFixed(2) + ', ' + s[1].toFixed(2) + ', ' + s[2].toFixed(2) + ') &nbsp; '
    + 'genes recoloured by expression (range ' + (lo/EXPR_SCALE).toFixed(2) + '..' + (hi/EXPR_SCALE).toFixed(2) + ', magma)';
}});

genePlot.on('plotly_hover', function(data) {{
  const pt = data.points[0]; if (pt.curveNumber !== POINTS_TRACE) return;
  const j = pt.pointNumber; if (lastHoveredGene === j) return; lastHoveredGene = j;
  const n = expr_matrix.length; const col = new Array(n);
  for (let i = 0; i < n; i++) col[i] = expr_matrix[i][j];
  Plotly.restyle(cellPlot, {{'marker.color': [exprToMagma(col)]}}, [POINTS_TRACE]);
  const lc = loadingToMagma(gene_load[j]);
  Plotly.restyle(cellPlot, {{'marker.color': [lc]}}, [LOADING_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color': [lc]}}, [LOADING_TRACE]);
  let lo = Infinity, hi = -Infinity; for (const v of col) {{ if (v<lo) lo=v; if (v>hi) hi=v; }}
  const L = gene_loading[j];
  const tag = gene_in_panel[j] ? '(panel)' : '(broader)';
  status.innerHTML = '<b style="color:' + gene_dom_color[j] + '">' + gene_name[j]
    + '</b> <span style="color:#555">' + tag + '</span> &nbsp; '
    + 'centroid UMAP1,2,3 = (' + L[0].toFixed(2) + ', ' + L[1].toFixed(2) + ', ' + L[2].toFixed(2) + ') &nbsp; '
    + 'cells recoloured by expression (range ' + (lo/EXPR_SCALE).toFixed(2) + '..' + (hi/EXPR_SCALE).toFixed(2) + ', magma)';
}});

document.getElementById('reset-btn').addEventListener('click', function() {{
  Plotly.restyle(cellPlot, {{'marker.color': [cell_default_colors]}}, [POINTS_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color': [gene_default_colors]}}, [POINTS_TRACE]);
  Plotly.restyle(cellPlot, {{'marker.color': [DEFAULT_LOAD_COLORS]}}, [LOADING_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color': [DEFAULT_LOAD_COLORS]}}, [LOADING_TRACE]);
  lastHoveredCell = null; lastHoveredGene = null;
  status.innerHTML = 'Reset. Hover a cell or gene to colour by expression and reveal UMAP coords.';
}});

let activeSet = 'panel';
const meanSlider = document.getElementById('mean-slider'), stdSlider = document.getElementById('std-slider');
const meanValueEl = document.getElementById('mean-value'), stdValueEl = document.getElementById('std-value');
const visibleCount = document.getElementById('visible-count');
function applyGeneFilter() {{
  const meanThr = parseFloat(meanSlider.value), stdThr = parseFloat(stdSlider.value);
  const mask = gene_sets[activeSet], n = gene_name.length;
  const xs = new Array(n), ys = new Array(n), zs = new Array(n); let visible = 0;
  for (let j = 0; j < n; j++) {{
    if (mask[j] && gene_mean[j] >= meanThr && gene_std[j] >= stdThr) {{
      xs[j]=gene_x[j]; ys[j]=gene_y[j]; zs[j]=gene_z[j]; visible++;
    }} else {{ xs[j]=null; ys[j]=null; zs[j]=null; }}
  }}
  Plotly.restyle(genePlot, {{x:[xs], y:[ys], z:[zs]}}, [POINTS_TRACE]);
  meanValueEl.textContent = meanThr.toFixed(2); stdValueEl.textContent = stdThr.toFixed(2);
  visibleCount.textContent = visible + ' / ' + n;
}}
meanSlider.addEventListener('input', applyGeneFilter);
stdSlider.addEventListener('input', applyGeneFilter);
document.querySelectorAll('.set-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    if (btn.disabled) return; activeSet = btn.dataset.set;
    document.querySelectorAll('.set-btn').forEach(b => b.classList.toggle('active', b === btn));
    applyGeneFilter();
  }});
}});

const geneSearch = document.getElementById('gene-search');
function clearSearch() {{
  Plotly.restyle(genePlot, {{x: [[null]], y: [[null]], z: [[null]]}}, [HIGHLIGHT_TRACE]);
}}
function runSearch() {{
  const q = geneSearch.value.trim().toLowerCase();
  if (!q) {{ clearSearch(); return; }}
  let j = gene_name.findIndex(g => g.toLowerCase() === q);
  if (j < 0) j = gene_name.findIndex(g => g.toLowerCase().startsWith(q));
  if (j < 0) {{ status.innerHTML = 'Gene <b>' + geneSearch.value + '</b> not in this gene pool.'; clearSearch(); return; }}
  const n = expr_matrix.length; const col = new Array(n);
  for (let i = 0; i < n; i++) col[i] = expr_matrix[i][j];
  Plotly.restyle(cellPlot, {{'marker.color': [exprToMagma(col)]}}, [POINTS_TRACE]);
  const lc = loadingToMagma(gene_load[j]);
  Plotly.restyle(cellPlot, {{'marker.color': [lc]}}, [LOADING_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color': [lc]}}, [LOADING_TRACE]);
  Plotly.restyle(genePlot, {{x: [[gene_x[j]]], y: [[gene_y[j]]], z: [[gene_z[j]]]}}, [HIGHLIGHT_TRACE]);
  lastHoveredGene = j;
  const hidden = !(gene_sets[activeSet][j]
                   && gene_mean[j] >= parseFloat(meanSlider.value)
                   && gene_std[j] >= parseFloat(stdSlider.value));
  status.innerHTML = '<b style="color:' + gene_dom_color[j] + '">' + gene_name[j] + '</b> '
    + (gene_in_panel[j] ? '(panel)' : '(broader)')
    + (hidden ? ' <span style="color:#c00">[hidden by current filter — ring still shows its position]</span>' : '')
    + ' — cells recoloured by its expression (magma).';
}}
geneSearch.addEventListener('change', runSearch);
geneSearch.addEventListener('keydown', e => {{ if (e.key === 'Enter') runSearch(); }});
document.getElementById('search-clear').addEventListener('click', function() {{
  geneSearch.value = ''; clearSearch();
}});

function valuesToViridis(values) {{
  let lo = Infinity, hi = -Infinity;
  for (const v of values) {{ if (v < lo) lo = v; if (v > hi) hi = v; }}
  const range = (hi > lo) ? (hi - lo) : 1;
  return values.map(v => viridis[Math.max(0, Math.min(255, Math.round(255*(v-lo)/range)))]);
}}
function colorByQC(arr, label, fmt) {{
  Plotly.restyle(cellPlot, {{'marker.color': [valuesToViridis(arr)]}}, [POINTS_TRACE]);
  let lo = Infinity, hi = -Infinity;
  for (const v of arr) {{ if (v < lo) lo = v; if (v > hi) hi = v; }}
  status.innerHTML = 'Cells coloured by <b>' + label + '</b> (viridis; ' + fmt(lo) + ' → ' + fmt(hi) + ')';
}}
const fmtInt = v => Math.round(v).toLocaleString();
const fmtPct = v => v.toFixed(1) + '%';
document.getElementById('qc-counts').addEventListener('click', () => colorByQC(qc_total, 'total counts', fmtInt));
document.getElementById('qc-genes').addEventListener('click', () => colorByQC(qc_ngenes, 'genes detected', fmtInt));
document.getElementById('qc-ribo').addEventListener('click', () => colorByQC(qc_ribo, '% ribosomal', fmtPct));

function resizePlots() {{ Plotly.Plots.resize(cellPlot); Plotly.Plots.resize(genePlot); }}
window.addEventListener('resize', resizePlots);
setTimeout(function() {{ resizePlots(); applyGeneFilter(); }}, 50);
</script>
</body>
</html>"""

    with open(OUT, 'w') as f: f.write(page)
    print(f'  done. {os.path.getsize(OUT)/1e6:.1f} MB self-contained HTML.')
    print(f'  open: file://{OUT}')


if __name__ == '__main__':
    main()
