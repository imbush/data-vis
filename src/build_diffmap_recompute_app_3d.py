#!/usr/bin/env python
"""Diffmap biplot explorer with in-browser recompute on a selected subtype subset.

Same biplot UI as build_lamp5_diffmap_app_3d.py plus a subtype-checkbox row and
a "Recompute diffmap" button that re-fits a diffusion-map-like embedding to
just the checked cells, in the browser.

Server-side initial fit uses scanpy's diffmap (PCA → kNN → diffmap eigenvalues).
The in-browser recompute uses a *Laplacian eigenmap* (a close cousin of
diffmap) — adaptive Gaussian kernel on a kNN graph in the standardized panel
space, symmetric Laplacian normalization, top-4 eigenvectors via power
iteration with deflation, skip the trivial eigenvector. Genes are placed at
the expression-weighted centroid of the cell coordinates, same as the base
diffmap script.

Usage:  python build_diffmap_recompute_app_3d.py [GROUP]
Output: notebooks/{group}_diffmap_recompute_explorer_3d.html
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

GROUP_NAME  = base.GROUP_NAME
SLUG        = base.SLUG
OUT         = os.path.join(base.ROOT, 'notebooks',
                           f'{SLUG}_diffmap_recompute_explorer_3d.html')
NDC         = 3

POLE_COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf']
POLE_NAMES  = ['DC1+', 'DC1-', 'DC2+', 'DC2-', 'DC3+', 'DC3-']


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

    # ---- initial server-side diffmap (full cohort) ----------------------------
    Xp = X_keep[:, in_panel]
    Zp = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-9)
    adp = ad_mod.AnnData(Zp.astype(np.float32))
    sc.pp.pca(adp, n_comps=min(30, n_cells - 1), random_state=0)
    sc.pp.neighbors(adp, n_neighbors=15, random_state=0)
    sc.tl.diffmap(adp, n_comps=NDC + 2)
    DC = np.asarray(adp.obsm['X_diffmap'][:, 1:NDC + 1], dtype=np.float64)
    evals = np.asarray(adp.uns['diffmap_evals'])[1:NDC + 1]
    print(f'  initial diffmap evals DC1-3: {np.round(evals, 4)}')

    Wn = X_keep.copy(); Wn[Wn < 0] = 0
    Wn = Wn / (Wn.sum(0, keepdims=True) + 1e-9)
    gene_dc = Wn.T @ DC

    def fill_cube(M):
        return M / (np.max(np.abs(M), axis=0) + 1e-9)
    cell_xyz = fill_cube(DC)
    gene_xyz = fill_cube(gene_dc)

    panel_idx = np.where(in_panel)[0]
    panel_genes_list = [gene_names[i] for i in panel_idx]
    pole_top = []
    for k in range(NDC):
        col = gene_xyz[panel_idx, k]
        pole_top.append(panel_genes_list[int(np.argmax(col))])
        pole_top.append(panel_genes_list[int(np.argmin(col))])

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

    def pole_loads(M):
        out = np.zeros((M.shape[0], 6))
        for k in range(NDC):
            out[:, 2*k]   = np.clip(M[:, k], 0, None)
            out[:, 2*k+1] = np.clip(-M[:, k], 0, None)
        return out
    cell_load = pole_loads(cell_xyz).round(4).tolist()
    gene_load = pole_loads(gene_xyz).round(4).tolist()

    EXPR_SCALE = 10
    expr_matrix = np.round(X_keep * EXPR_SCALE).astype(np.int16).tolist()
    panel_idx_list = [j for j, p in enumerate(in_panel.tolist()) if p]

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
        f'DC1,2,3 = ({DC[i,0]:.3f}, {DC[i,1]:.3f}, {DC[i,2]:.3f})'
        for i in range(n_cells)]
    gene_hover_text = [
        f'<b>{gene_names[j]}</b>'
        + (' (panel HVG)' if in_panel[j] else ' (projected)')
        + f'<br>strongest: {POLE_NAMES[gene_pole[j]]} ({pole_top[gene_pole[j]]})<br>'
        + f'mean={mean_expr[j]:.2f}, std={std_expr[j]:.2f}<br>'
        + f'centroid DC1,2,3 = ({gene_dc[j,0]:.3f}, {gene_dc[j,1]:.3f}, {gene_dc[j,2]:.3f})'
        for j in range(n_genes)]

    n_panel_disp = int(in_panel.sum())
    n_imputed    = n_genes - n_panel_disp
    historical_outliers = list(base.GROUP['cache_outliers']) + list(base.GROUP['runtime_exclude'])
    excluded_blurb = (
        f'Historically-flagged outlier subtypes ({", ".join(historical_outliers)}) are '
        f'<b>included</b> here — uncheck them in the subtype row and recompute to drop them.'
        if historical_outliers else 'No subtypes are flagged as outliers.')

    fig_cells = build_fig(cell_xyz, cell_color_default, cell_hover_text,
                          f'Cells — diffmap (n={n_cells})  '
                          f'<i>recompute on the subtype subset to re-fit the diffusion embedding</i>')
    fig_genes = build_fig(gene_xyz, gene_color_default, gene_hover_text,
                          f'Genes — expression-weighted centroid '
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

    subtype_counts = {c: int(np.sum(subs == c)) for c in cats}
    subtype_checkbox_html = ''.join(
        f'<label class="subt-chk">'
        f'<input type="checkbox" data-sub="{c}" checked> '
        f'<span style="color:{subtype_palette[c]}; font-weight:700;">●</span> '
        f'{c} <span class="ct">({subtype_counts[c]})</span></label>'
        for c in cats)

    js_data = (
        f"const EXPR_SCALE  = {EXPR_SCALE};\n"
        f"const expr_matrix = {json.dumps(expr_matrix)};\n"
        f"const cell_default_colors = {json.dumps(cell_color_default)};\n"
        f"let gene_default_colors = {json.dumps(gene_color_default)};\n"
        f"const cell_subtype = {json.dumps(subs.tolist())};\n"
        f"const subtype_palette = {json.dumps(subtype_palette)};\n"
        f"const gene_name    = {json.dumps(gene_names)};\n"
        f"const gene_in_panel = {json.dumps(panel_mask_list)};\n"
        f"const gene_mean    = {json.dumps([round(float(v),3) for v in mean_expr])};\n"
        f"const gene_std     = {json.dumps([round(float(v),3) for v in std_expr])};\n"
        f"let gene_x = {json.dumps(gx)};\n"
        f"let gene_y = {json.dumps(gy)};\n"
        f"let gene_z = {json.dumps(gz)};\n"
        f"let cell_load = {json.dumps(cell_load)};\n"
        f"let gene_load = {json.dumps(gene_load)};\n"
        f"let cell_dc = {json.dumps(DC.round(4).tolist())};\n"
        f"let gene_centroid = {json.dumps(gene_dc.round(4).tolist())};\n"
        f"let cell_dom_color = {json.dumps(cell_dom_color)};\n"
        f"let gene_dom_color = {json.dumps(gene_color_default)};\n"
        f"let cell_active = Array({n_cells}).fill(true);\n"
        f"let pole_top = {json.dumps(pole_top)};\n"
        f"const panel_idx = {json.dumps(panel_idx_list)};\n"
        f"const POLE_NAMES_  = {json.dumps(POLE_NAMES)};\n"
        f"const POLE_COLORS_ = {json.dumps(POLE_COLORS)};\n"
        f"const gene_sets    = {json.dumps(set_masks)};\n"
        f"const gene_set_counts = {json.dumps(set_counts)};\n"
        f"const magma        = {json.dumps(list(Magma256))};\n"
        f"const viridis      = {json.dumps(list(Viridis256))};\n"
        f"const qc_total     = {json.dumps([round(float(v), 1) for v in qc_total])};\n"
        f"const qc_ngenes    = {json.dumps([int(v) for v in qc_ngenes])};\n"
        f"const qc_ribo      = {json.dumps([round(float(v), 2) for v in qc_ribo])};\n"
    )

    button_order = ['panel', 'all'] + [n for n in base.GENE_SET_ORDER if n != 'all']
    set_label_fn = lambda n: 'Panel HVG' if n == 'panel' else base.GENE_SET_LABELS.get(n, n)
    set_buttons_html = ''.join(
        f'<button class="set-btn{" active" if name == "panel" else ""}" data-set="{name}"'
        f'{" disabled" if set_counts.get(name, 0) == 0 else ""}>'
        f'{set_label_fn(name)} ({set_counts.get(name, 0)})</button>'
        for name in button_order)
    gene_datalist = ('<datalist id="gene-datalist">'
                     + ''.join(f'<option value="{g}">' for g in gene_names)
                     + '</datalist>')

    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{GROUP_NAME} diffmap recompute explorer</title>
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
.subt-chk {{ display:inline-flex; align-items:center; gap:3px; padding:1px 6px;
             border:1px solid #ddd; border-radius:3px; background:#fafafa;
             font-size:12px; cursor:pointer; user-select:none; }}
.subt-chk .ct {{ color:#888; }}
#recompute-btn {{ font-weight:600; background:#ff7f0e; color:white;
                  border:1px solid #cc6510; padding:4px 12px; border-radius:3px; cursor:pointer; }}
#recompute-btn:hover {{ background:#ec6a00; }}
#recompute-btn:disabled {{ background:#aaa; border-color:#888; cursor:not-allowed; }}
#mean-slider, #std-slider {{ width: 180px; }}
.row {{ flex: 1 1 auto; display: flex; flex-direction: row; gap: 8px; min-height: 0; }}
.col {{ flex: 1 1 0; min-width: 0; display: flex; flex-direction: column; }}
.col > .plotly-graph-div {{ flex: 1 1 auto; min-height: 0; height: 100% !important; }}
#status, #recompute-status {{ color: #555; font-size: 12px; }}
.legend {{ flex: 0 0 auto; font-size: 11px; color: #444; margin-top: 4px; }}
button {{ font-size: 13px; padding: 4px 10px; }}
details summary {{ cursor: pointer; color: #666; font-size: 12px; }}
{base.VIZ_NAV_CSS}
</style>
</head>
<body>
<h2>{GROUP_NAME} diffmap — subtype-subset recompute</h2>
{base.viz_nav_html(SLUG, 'diffmap')}
<div class="hint">
<b>Pick which subtypes you want to fit the embedding to</b>, then click <b>Recompute diffmap</b>.
The diffusion embedding re-fits to those cells; every gene re-projects as a centroid.
</div>
<details class="header">
<summary>About this app (click to expand)</summary>
<div style="margin-top:4px;">
Initial view shows the diffusion-map embedding fit on all <b>{n_cells}</b> {GROUP_NAME} cells
via scanpy (panel: {n_panel_disp} HVG z-scored, PCA→kNN→diffmap; DC1/2/3 eigenvalues
{evals[0]:.3f}, {evals[1]:.3f}, {evals[2]:.3f}). Genes sit at the expression-weighted centroid
of cell DCs. {excluded_blurb}<br>
<b>Recompute</b> runs a <i>Laplacian eigenmap</i> client-side (adaptive Gaussian kernel on a
15-NN graph in the standardized panel space; symmetric normalization; top-4 eigenvectors of
A = D⁻¹ᐟ² W D⁻¹ᐟ² via power iteration with deflation; drop the trivial one). Laplacian
eigenmaps and diffusion maps share the same eigenstructure on the same kernel — the visual
geometry will be very close to scanpy's diffmap on the same subset. Gene centroids are
re-computed against the new cell DCs. Recompute takes ~1–5 s for a few-hundred to ~1k cells.
</div>
</details>

<div class="controls">
  <div class="controls-row" style="background:#fff3e0; border:1px solid #ffcc80; border-radius:4px; padding:4px 6px;">
    <span class="label">Subtypes:</span>
    {subtype_checkbox_html}
    <button id="subt-all">all</button>
    <button id="subt-none">none</button>
    <span style="flex: 1 1 0;"></span>
    <button id="recompute-btn">Recompute diffmap on selected →</button>
    <span id="recompute-status" style="margin-left:8px;"></span>
  </div>
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
<b>Gene default colours</b> (strongest signed DC): <span id="pole-legend">{pole_legend}</span>
</div>
<script>
{js_data}

const cellPlot = document.getElementById('cell-plot');
const genePlot = document.getElementById('gene-plot');
const status   = document.getElementById('status');
const recomputeStatus = document.getElementById('recompute-status');
const POINTS_TRACE = 2, VERTEX_TRACE = 1, LOADING_TRACE = 3, HIGHLIGHT_TRACE = 4;
const DEFAULT_LOAD_COLORS = ['#e0e0e0','#e0e0e0','#e0e0e0','#e0e0e0','#e0e0e0','#e0e0e0'];
const KNN_K = 15;
let lastHoveredCell = null, lastHoveredGene = null;

function exprToMagma(values) {{
  let lo = Infinity, hi = -Infinity;
  for (const v of values) {{ if (v < lo) lo = v; if (v > hi) hi = v; }}
  const range = (hi > lo) ? (hi - lo) : 1;
  return values.map(v => magma[Math.max(0, Math.min(255, Math.round(255*(v-lo)/range)))]);
}}
function loadingToMagma(loadings) {{
  return loadings.map(v => magma[Math.round(255 * Math.max(0, Math.min(1, v)))]);
}}

cellPlot.on('plotly_hover', function(data) {{
  const pt = data.points[0]; if (pt.curveNumber !== POINTS_TRACE) return;
  const i = pt.pointNumber; if (lastHoveredCell === i) return; lastHoveredCell = i;
  if (!cell_active[i]) return;
  const row = expr_matrix[i];
  Plotly.restyle(genePlot, {{'marker.color': [exprToMagma(row)]}}, [POINTS_TRACE]);
  const lc = loadingToMagma(cell_load[i]);
  Plotly.restyle(cellPlot, {{'marker.color': [lc]}}, [LOADING_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color': [lc]}}, [LOADING_TRACE]);
  let lo = Infinity, hi = -Infinity; for (const v of row) {{ if (v<lo) lo=v; if (v>hi) hi=v; }}
  const d = cell_dc[i];
  status.innerHTML = '<b style="color:' + cell_dom_color[i] + '">Cell #' + i
    + '</b> <span style="color:#555">(' + cell_subtype[i] + ')</span> &nbsp; '
    + 'DC1,2,3 = (' + d[0].toFixed(3) + ', ' + d[1].toFixed(3) + ', ' + d[2].toFixed(3) + ') &nbsp; '
    + 'genes recoloured by expression (range ' + (lo/EXPR_SCALE).toFixed(2) + '..' + (hi/EXPR_SCALE).toFixed(2) + ', magma)';
}});

genePlot.on('plotly_hover', function(data) {{
  const pt = data.points[0]; if (pt.curveNumber !== POINTS_TRACE) return;
  const j = pt.pointNumber; if (lastHoveredGene === j) return; lastHoveredGene = j;
  const n = expr_matrix.length; const col = new Array(n);
  for (let i = 0; i < n; i++) col[i] = cell_active[i] ? expr_matrix[i][j] : null;
  const visible = col.filter(v => v !== null);
  const colors = exprToMagma(visible);
  const cellColors = new Array(n);
  let vi = 0;
  for (let i = 0; i < n; i++) cellColors[i] = (col[i] === null) ? '#bbbbbb' : colors[vi++];
  Plotly.restyle(cellPlot, {{'marker.color': [cellColors]}}, [POINTS_TRACE]);
  const lc = loadingToMagma(gene_load[j]);
  Plotly.restyle(cellPlot, {{'marker.color': [lc]}}, [LOADING_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color': [lc]}}, [LOADING_TRACE]);
  let lo = Infinity, hi = -Infinity; for (const v of visible) {{ if (v<lo) lo=v; if (v>hi) hi=v; }}
  const c = gene_centroid[j];
  const tag = gene_in_panel[j] ? '(panel)' : '(broader)';
  status.innerHTML = '<b style="color:' + gene_dom_color[j] + '">' + gene_name[j]
    + '</b> <span style="color:#555">' + tag + '</span> &nbsp; '
    + 'centroid DC1,2,3 = (' + c[0].toFixed(3) + ', ' + c[1].toFixed(3) + ', ' + c[2].toFixed(3) + ') &nbsp; '
    + 'cells recoloured by expression (range ' + (lo/EXPR_SCALE).toFixed(2) + '..' + (hi/EXPR_SCALE).toFixed(2) + ', magma)';
}});

document.getElementById('reset-btn').addEventListener('click', function() {{
  const cells = cell_default_colors.map((c, i) => cell_active[i] ? c : '#dddddd');
  Plotly.restyle(cellPlot, {{'marker.color': [cells]}}, [POINTS_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color': [gene_default_colors]}}, [POINTS_TRACE]);
  Plotly.restyle(cellPlot, {{'marker.color': [DEFAULT_LOAD_COLORS]}}, [LOADING_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color': [DEFAULT_LOAD_COLORS]}}, [LOADING_TRACE]);
  lastHoveredCell = null; lastHoveredGene = null;
  status.innerHTML = 'Reset. Hover a cell or gene to colour by expression.';
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
  for (let i = 0; i < n; i++) col[i] = cell_active[i] ? expr_matrix[i][j] : null;
  const visible = col.filter(v => v !== null);
  const colors = exprToMagma(visible);
  const cellColors = new Array(n);
  let vi = 0;
  for (let i = 0; i < n; i++) cellColors[i] = (col[i] === null) ? '#bbbbbb' : colors[vi++];
  Plotly.restyle(cellPlot, {{'marker.color': [cellColors]}}, [POINTS_TRACE]);
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
  const valid = arr.filter((v, i) => cell_active[i]);
  const palette = valuesToViridis(valid);
  let vi = 0;
  const cellColors = arr.map((v, i) => cell_active[i] ? palette[vi++] : '#dddddd');
  Plotly.restyle(cellPlot, {{'marker.color': [cellColors]}}, [POINTS_TRACE]);
  let lo = Infinity, hi = -Infinity;
  for (const v of valid) {{ if (v < lo) lo = v; if (v > hi) hi = v; }}
  status.innerHTML = 'Cells coloured by <b>' + label + '</b> (viridis; ' + fmt(lo) + ' → ' + fmt(hi) + ')';
}}
const fmtInt = v => Math.round(v).toLocaleString();
const fmtPct = v => v.toFixed(1) + '%';
document.getElementById('qc-counts').addEventListener('click', () => colorByQC(qc_total, 'total counts', fmtInt));
document.getElementById('qc-genes').addEventListener('click', () => colorByQC(qc_ngenes, 'genes detected', fmtInt));
document.getElementById('qc-ribo').addEventListener('click', () => colorByQC(qc_ribo, '% ribosomal', fmtPct));

// ============================================================================
// Subtype-subset diffmap recompute (Laplacian eigenmap)
// ============================================================================
const subtypeCheckboxes = Array.from(document.querySelectorAll('.subt-chk input[type="checkbox"]'));
document.getElementById('subt-all').addEventListener('click', () => {{
  subtypeCheckboxes.forEach(cb => cb.checked = true);
}});
document.getElementById('subt-none').addEventListener('click', () => {{
  subtypeCheckboxes.forEach(cb => cb.checked = false);
}});
function selectedSubtypes() {{
  const out = new Set();
  subtypeCheckboxes.forEach(cb => {{ if (cb.checked) out.add(cb.dataset.sub); }});
  return out;
}}

// Build z-scored (subset rows, panel columns) matrix as Array of Float64Array rows.
function buildSubsetPanelZ(cellSel) {{
  const m = cellSel.length, n = panel_idx.length;
  // Per-column mean/std on the subset
  const mean = new Float64Array(n), std = new Float64Array(n);
  for (let k = 0; k < n; k++) {{
    const j = panel_idx[k];
    let s = 0, ss = 0;
    for (let ii = 0; ii < m; ii++) {{
      const v = expr_matrix[cellSel[ii]][j] / EXPR_SCALE;
      s += v; ss += v*v;
    }}
    mean[k] = s / m;
    std[k]  = Math.sqrt(Math.max(ss / m - mean[k]*mean[k], 1e-18));
  }}
  const Z = new Array(m);
  for (let ii = 0; ii < m; ii++) {{
    const Zi = new Float64Array(n);
    for (let k = 0; k < n; k++) {{
      const v = expr_matrix[cellSel[ii]][panel_idx[k]] / EXPR_SCALE;
      Zi[k] = (v - mean[k]) / std[k];
    }}
    Z[ii] = Zi;
  }}
  return Z;
}}

// Build sparse symmetric affinity matrix W via adaptive Gaussian kernel on a kNN graph.
// Returns {{rowPtr, colIdx, val}} in CSR format, plus the degrees D.
function buildKnnAffinity(Z, K) {{
  const m = Z.length, n = Z[0].length;
  // 1) Pairwise distances (squared) — only need each row to find top-K nearest.
  //    For m up to ~2000 this is O(m^2 n) ≈ 1500^2 × 150 = 340M ops; ~5-10s in JS.
  //    Use Float64Array buffers for tight inner loop.
  const knnIdx = new Int32Array(m * K);
  const knnD2  = new Float64Array(m * K);
  for (let i = 0; i < m; i++) {{
    const Zi = Z[i];
    // Track top-K smallest (excluding self): use insertion-sort over a K-buffer.
    const buf = new Float64Array(K); buf.fill(Infinity);
    const idx = new Int32Array(K); idx.fill(-1);
    for (let j = 0; j < m; j++) {{
      if (j === i) continue;
      const Zj = Z[j];
      let d2 = 0;
      for (let k = 0; k < n; k++) {{ const dd = Zi[k] - Zj[k]; d2 += dd*dd; }}
      // Insert if d2 < max(buf). buf is unsorted; we'll keep it as max-heap-ish unsorted.
      // Use largest-index scan.
      let maxPos = 0, maxVal = buf[0];
      for (let t = 1; t < K; t++) if (buf[t] > maxVal) {{ maxVal = buf[t]; maxPos = t; }}
      if (d2 < maxVal) {{ buf[maxPos] = d2; idx[maxPos] = j; }}
    }}
    // Sort the K kept by distance ascending (small K, small cost)
    for (let a = 1; a < K; a++) {{
      const bv = buf[a], bi = idx[a];
      let p = a - 1;
      while (p >= 0 && buf[p] > bv) {{ buf[p+1] = buf[p]; idx[p+1] = idx[p]; p--; }}
      buf[p+1] = bv; idx[p+1] = bi;
    }}
    for (let t = 0; t < K; t++) {{
      knnIdx[i*K + t] = idx[t];
      knnD2[i*K + t]  = buf[t];
    }}
  }}
  // 2) Adaptive sigma: distance to the (K//2)-th nearest neighbor (gives a
  //    median-ish bandwidth).
  const halfK = Math.max(1, Math.floor(K / 2));
  const sigma = new Float64Array(m);
  for (let i = 0; i < m; i++) sigma[i] = Math.sqrt(knnD2[i*K + halfK]) + 1e-12;
  // 3) Build directed-graph entries (i, j, w) for j in kNN(i)
  const triplets = new Array(m * K);
  let tp = 0;
  for (let i = 0; i < m; i++) {{
    for (let t = 0; t < K; t++) {{
      const j = knnIdx[i*K + t];
      const d2 = knnD2[i*K + t];
      const w = Math.exp(-d2 / (sigma[i] * sigma[j]));
      triplets[tp++] = [i, j, w];
    }}
  }}
  // 4) Symmetrize: W[i,j] := max(W[i,j], W[j,i]) (or sum/avg; max gives a sparser graph
  //    that closely mirrors scanpy's "symmetric kNN" handling).
  const tmap = new Map();      // key "i,j" with i<=j → max weight
  for (const [i, j, w] of triplets) {{
    const a = Math.min(i, j), b = Math.max(i, j);
    const k = a * m + b;
    const prev = tmap.get(k);
    if (prev === undefined || w > prev) tmap.set(k, w);
  }}
  // 5) Build CSR sparse symmetric matrix
  const rowEntries = Array.from({{length: m}}, () => []);
  for (const [k, w] of tmap) {{
    const i = Math.floor(k / m), j = k % m;
    rowEntries[i].push([j, w]);
    if (i !== j) rowEntries[j].push([i, w]);
  }}
  // count
  let nnz = 0; for (let i = 0; i < m; i++) nnz += rowEntries[i].length;
  const rowPtr = new Int32Array(m + 1);
  const colIdx = new Int32Array(nnz);
  const val    = new Float64Array(nnz);
  let p = 0;
  for (let i = 0; i < m; i++) {{
    rowPtr[i] = p;
    for (const [j, w] of rowEntries[i]) {{ colIdx[p] = j; val[p] = w; p++; }}
  }}
  rowPtr[m] = p;
  // 6) Degrees D
  const D = new Float64Array(m);
  for (let i = 0; i < m; i++) {{
    let s = 0;
    for (let q = rowPtr[i]; q < rowPtr[i+1]; q++) s += val[q];
    D[i] = s;
  }}
  return {{rowPtr, colIdx, val, D}};
}}

// Top-K eigenvectors of a symmetric sparse matrix via power iteration with deflation.
// A is implicit: applyA(v) computes A @ v. Returns {{evals: Float64Array(K), evecs: K Float64Array(m)}}.
function topKEigSym(applyA, m, K, maxIter, tol) {{
  maxIter = maxIter || 100; tol = tol || 1e-7;
  const evals = new Float64Array(K), evecs = new Array(K);
  // We deflate by subtracting λ_k v_k v_k^T after each component.
  const found = [];
  for (let k = 0; k < K; k++) {{
    // initialize v deterministically per k (sine of index)
    let v = new Float64Array(m);
    for (let i = 0; i < m; i++) v[i] = Math.sin((i + 1) * (k + 1) * 0.21) + 0.05;
    // orthogonalize against found vectors
    for (let f = 0; f < found.length; f++) {{
      const u = found[f].vec; let dot = 0;
      for (let i = 0; i < m; i++) dot += v[i] * u[i];
      for (let i = 0; i < m; i++) v[i] -= dot * u[i];
    }}
    let vn = 0; for (let i = 0; i < m; i++) vn += v[i]*v[i]; vn = Math.sqrt(vn);
    if (vn < 1e-14) {{
      // fallback: random-ish unit vector
      for (let i = 0; i < m; i++) v[i] = Math.cos((i + 1) * (k + 17) * 0.07);
      vn = 0; for (let i = 0; i < m; i++) vn += v[i]*v[i]; vn = Math.sqrt(vn);
    }}
    for (let i = 0; i < m; i++) v[i] /= vn;
    let lambda = 0, lambda_prev = 0;
    for (let iter = 0; iter < maxIter; iter++) {{
      // u = A v
      let u = applyA(v);
      // re-orthogonalize against found vectors (to keep deflation honest)
      for (let f = 0; f < found.length; f++) {{
        const uf = found[f].vec; let dot = 0;
        for (let i = 0; i < m; i++) dot += u[i] * uf[i];
        for (let i = 0; i < m; i++) u[i] -= dot * uf[i];
      }}
      // Rayleigh quotient ~ ||u|| since v already unit-norm
      let un = 0; for (let i = 0; i < m; i++) un += u[i]*u[i]; un = Math.sqrt(un);
      if (un < 1e-14) {{ lambda = 0; break; }}
      // estimate eigenvalue as v · (A v) (Rayleigh)
      let rq = 0; for (let i = 0; i < m; i++) rq += u[i] * v[i];
      // normalize
      for (let i = 0; i < m; i++) v[i] = u[i] / un;
      lambda = rq;
      if (iter > 1 && Math.abs(lambda - lambda_prev) < tol * (Math.abs(lambda) + 1e-12)) break;
      lambda_prev = lambda;
    }}
    evals[k] = lambda;
    evecs[k] = v;
    found.push({{val: lambda, vec: v}});
  }}
  return {{evals, evecs}};
}}

function recomputeDiffmap() {{
  const t0 = performance.now();
  const sel = selectedSubtypes();
  const cellSel = [];
  for (let i = 0; i < cell_subtype.length; i++) if (sel.has(cell_subtype[i])) cellSel.push(i);
  const m = cellSel.length;
  if (m < KNN_K + 1) {{
    recomputeStatus.innerHTML = '<span style="color:#c00">need ≥' + (KNN_K + 1)
      + ' cells (got ' + m + ')</span>';
    return;
  }}

  recomputeStatus.textContent = 'building kNN graph…';
  const Z = buildSubsetPanelZ(cellSel);
  const {{rowPtr, colIdx, val, D}} = buildKnnAffinity(Z, KNN_K);

  // Symmetric normalized: A = D^{{-1/2}} W D^{{-1/2}}
  const Dinv2 = new Float64Array(m);
  for (let i = 0; i < m; i++) Dinv2[i] = 1.0 / Math.sqrt(D[i] + 1e-18);
  function applyA(v) {{
    const out = new Float64Array(m);
    for (let i = 0; i < m; i++) {{
      let s = 0;
      const di = Dinv2[i];
      for (let q = rowPtr[i]; q < rowPtr[i+1]; q++) {{
        s += val[q] * Dinv2[colIdx[q]] * v[colIdx[q]];
      }}
      out[i] = di * s;
    }}
    return out;
  }}
  recomputeStatus.textContent = 'eigendecomposing…';
  // Top-4 eigenvectors; the largest (≈1) is trivial (proportional to sqrt(D)), drop it.
  const K_EIG = 4;
  const {{evals, evecs}} = topKEigSym(applyA, m, K_EIG);

  // For diffmap-style eigenvectors of the random-walk transition, we lift
  // back: ψ_k = D^{{-1/2}} φ_k where φ_k is the eigenvector of A. (The 0-th
  // trivial component is proportional to sqrt(D); we drop it.)
  const DC = new Array(m);
  for (let ii = 0; ii < m; ii++) {{
    DC[ii] = [
      Dinv2[ii] * evecs[1][ii],
      Dinv2[ii] * evecs[2][ii],
      Dinv2[ii] * evecs[3][ii],
    ];
  }}

  // Gene centroid in new DC space: gene_dc[j, k] = Σ_i w[i,g] * DC[i, k] / Σ_i w[i,g]
  // where w[i, g] = max(expression_value, 0) for cell i in the selected set, gene g.
  // For non-selected cells we contribute 0 (they're not in the DC array).
  const n_all = gene_name.length;
  const newGeneCentroid = new Array(n_all);
  // Pre-extract DC arrays per axis for vectorized inner loop
  const dc0 = new Float64Array(m), dc1 = new Float64Array(m), dc2 = new Float64Array(m);
  for (let ii = 0; ii < m; ii++) {{ dc0[ii] = DC[ii][0]; dc1[ii] = DC[ii][1]; dc2[ii] = DC[ii][2]; }}
  for (let j = 0; j < n_all; j++) {{
    let s0 = 0, s1 = 0, s2 = 0, sw = 0;
    for (let ii = 0; ii < m; ii++) {{
      const w = Math.max(0, expr_matrix[cellSel[ii]][j] / EXPR_SCALE);
      sw += w;
      s0 += w * dc0[ii]; s1 += w * dc1[ii]; s2 += w * dc2[ii];
    }}
    if (sw < 1e-12) newGeneCentroid[j] = [0, 0, 0];
    else newGeneCentroid[j] = [s0/sw, s1/sw, s2/sw];
  }}

  // Fill-cube scaling for both cells and genes
  let cmax = [1e-12, 1e-12, 1e-12], gmax = [1e-12, 1e-12, 1e-12];
  for (let ii = 0; ii < m; ii++) for (let k = 0; k < 3; k++)
    if (Math.abs(DC[ii][k]) > cmax[k]) cmax[k] = Math.abs(DC[ii][k]);
  for (let j = 0; j < n_all; j++) for (let k = 0; k < 3; k++)
    if (Math.abs(newGeneCentroid[j][k]) > gmax[k]) gmax[k] = Math.abs(newGeneCentroid[j][k]);

  // Build full-length cell xyz (null for non-selected)
  const n_cells_total = cell_subtype.length;
  const newCellX = new Array(n_cells_total).fill(null);
  const newCellY = new Array(n_cells_total).fill(null);
  const newCellZ = new Array(n_cells_total).fill(null);
  const newCellLoad = cell_load.slice();
  const newCellDC = cell_dc.slice();
  const newCellActive = new Array(n_cells_total).fill(false);
  cellSel.forEach((i, ii) => {{
    newCellActive[i] = true;
    newCellX[i] = DC[ii][0] / cmax[0];
    newCellY[i] = DC[ii][1] / cmax[1];
    newCellZ[i] = DC[ii][2] / cmax[2];
    newCellDC[i] = [+DC[ii][0].toFixed(4), +DC[ii][1].toFixed(4), +DC[ii][2].toFixed(4)];
    const x = newCellX[i], y = newCellY[i], z = newCellZ[i];
    newCellLoad[i] = [Math.max(x,0), Math.max(-x,0), Math.max(y,0),
                       Math.max(-y,0), Math.max(z,0), Math.max(-z,0)].map(v => +v.toFixed(4));
  }});

  // Gene xyz + 6-pole loads from scaled centroids
  const newGeneX = new Array(n_all), newGeneY = new Array(n_all), newGeneZ = new Array(n_all);
  const newGeneLoad = new Array(n_all);
  for (let j = 0; j < n_all; j++) {{
    const c = newGeneCentroid[j];
    newGeneX[j] = +(c[0] / gmax[0]).toFixed(4);
    newGeneY[j] = +(c[1] / gmax[1]).toFixed(4);
    newGeneZ[j] = +(c[2] / gmax[2]).toFixed(4);
    newGeneLoad[j] = [Math.max(newGeneX[j],0), Math.max(-newGeneX[j],0),
                      Math.max(newGeneY[j],0), Math.max(-newGeneY[j],0),
                      Math.max(newGeneZ[j],0), Math.max(-newGeneZ[j],0)].map(v => +v.toFixed(4));
  }}

  function domPole(x, y, z) {{
    const ax = Math.abs(x), ay = Math.abs(y), az = Math.abs(z);
    let k = 0, mx = ax; if (ay > mx) {{ k = 1; mx = ay; }} if (az > mx) {{ k = 2; }}
    const vals = [x, y, z];
    return 2*k + (vals[k] >= 0 ? 0 : 1);
  }}
  const newCellDom = new Array(n_cells_total);
  for (let i = 0; i < n_cells_total; i++) {{
    if (newCellActive[i]) newCellDom[i] = POLE_COLORS_[domPole(newCellX[i], newCellY[i], newCellZ[i])];
    else newCellDom[i] = '#cccccc';
  }}
  const newGeneDom = new Array(n_all);
  for (let j = 0; j < n_all; j++) newGeneDom[j] = POLE_COLORS_[domPole(newGeneX[j], newGeneY[j], newGeneZ[j])];

  // Pole-label: most-extreme PANEL gene centroid per ±axis
  const newPoleTop = new Array(6);
  for (let k = 0; k < 3; k++) {{
    let bestPos = -Infinity, bestPosJ = panel_idx[0];
    let bestNeg =  Infinity, bestNegJ = panel_idx[0];
    for (let pi = 0; pi < panel_idx.length; pi++) {{
      const j = panel_idx[pi];
      const c = newGeneCentroid[j][k];
      if (c > bestPos) {{ bestPos = c; bestPosJ = j; }}
      if (c < bestNeg) {{ bestNeg = c; bestNegJ = j; }}
    }}
    newPoleTop[2*k]   = gene_name[bestPosJ];
    newPoleTop[2*k+1] = gene_name[bestNegJ];
  }}

  // ---- commit globals ----
  gene_x = newGeneX; gene_y = newGeneY; gene_z = newGeneZ;
  cell_load = newCellLoad; gene_load = newGeneLoad;
  cell_dc = newCellDC;
  gene_centroid = newGeneCentroid.map(c => [+c[0].toFixed(4), +c[1].toFixed(4), +c[2].toFixed(4)]);
  cell_dom_color = newCellDom; gene_dom_color = newGeneDom;
  gene_default_colors = newGeneDom.slice();
  cell_active = newCellActive;
  pole_top = newPoleTop;

  // ---- redraw ----
  const cellColors = cell_default_colors.map((c, i) => newCellActive[i] ? c : '#dddddd');
  Plotly.restyle(cellPlot, {{x:[newCellX], y:[newCellY], z:[newCellZ],
                              'marker.color':[cellColors]}}, [POINTS_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color':[gene_default_colors]}}, [POINTS_TRACE]);
  applyGeneFilter();
  const newPoleLab = [];
  for (let p = 0; p < 6; p++) newPoleLab.push(POLE_NAMES_[p] + '<br>(' + newPoleTop[p] + ')');
  Plotly.restyle(cellPlot, {{text:[newPoleLab], hovertext:[newPoleLab]}}, [VERTEX_TRACE]);
  Plotly.restyle(genePlot, {{text:[newPoleLab], hovertext:[newPoleLab]}}, [VERTEX_TRACE]);
  Plotly.restyle(cellPlot, {{'marker.color':[DEFAULT_LOAD_COLORS]}}, [LOADING_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color':[DEFAULT_LOAD_COLORS]}}, [LOADING_TRACE]);
  Plotly.restyle(genePlot, {{x:[[null]], y:[[null]], z:[[null]]}}, [HIGHLIGHT_TRACE]);
  document.getElementById('pole-legend').innerHTML = newPoleTop.map((g, p) =>
    `<span style="display:inline-block;width:10px;height:10px;background:${{POLE_COLORS_[p]}};` +
    `margin-right:4px;border-radius:50%;"></span> ${{POLE_NAMES_[p]}} (${{g}}) &nbsp;&nbsp;`).join('');

  const dt = ((performance.now() - t0) / 1000).toFixed(2);
  recomputeStatus.innerHTML = '<b>recomputed</b> on ' + m + ' / ' + n_cells_total + ' cells (' + dt + 's) — '
    + 'DC1 eval=' + evals[1].toFixed(3) + ', DC2=' + evals[2].toFixed(3) + ', DC3=' + evals[3].toFixed(3)
    + ' &nbsp; poles: ' + newPoleTop.map((g, p) => POLE_NAMES_[p] + '=' + g).join(', ');
  lastHoveredCell = null; lastHoveredGene = null;
}}

document.getElementById('recompute-btn').addEventListener('click', () => {{
  const btn = document.getElementById('recompute-btn');
  btn.disabled = true; recomputeStatus.textContent = 'computing…';
  setTimeout(() => {{ try {{ recomputeDiffmap(); }} finally {{ btn.disabled = false; }} }}, 30);
}});

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
