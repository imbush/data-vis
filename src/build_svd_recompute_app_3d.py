#!/usr/bin/env python
"""SVD biplot explorer with in-browser SVD recompute on a selected subtype subset.

Same biplot UI as build_lamp5_svd_app_3d.py plus:
  - A row of subtype checkboxes (default: all checked).
  - A "Recompute SVD on selected" button. Pressing it filters cells to the
    checked subtypes, re-z-scores the panel matrix on that subset, runs
    power-iteration SVD (top-3) in the browser, projects every gene onto the
    new basis, and updates both biplots in place (positions, vertex labels,
    default colours).

The initial page is exactly the full-cohort SVD; recompute is a one-click
mutation that stays entirely client-side, so the HTML remains a single
self-contained file (no server / Pyodide).

Usage:  python build_svd_recompute_app_3d.py [GROUP]   # default: base GROUP_NAME
Output: notebooks/{group}_svd_recompute_explorer_3d.html
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import plotly.graph_objects as go
from plotly.io import to_html
from bokeh.palettes import Magma256, Viridis256, Category20, Set3, Set1, Category10

import build_lamp5_archetype_app_4d as base
import build_lamp5_svd_app_3d as svdmod

GROUP_NAME  = base.GROUP_NAME
SLUG        = base.SLUG
OUT         = os.path.join(base.ROOT, 'notebooks',
                           f'{SLUG}_svd_recompute_explorer_3d.html')
NPC         = 3
POLE_COLORS = svdmod.POLE_COLORS
POLE_NAMES  = svdmod.POLE_NAMES
prep_cols   = svdmod.prep_cols


def main():
    # Use the FULL-cohort proj (no cache_outliers / runtime_exclude filtering)
    # so the user can include outlier subtypes via the checkbox UI.
    proj      = base.compute_or_load_proj_full()
    cleaned   = proj['cleaned']
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

    # ---- initial SVD: all cells. Top-3 used for embedding; top-10 retained
    # for the live variance-explained bar chart.
    BARS_K = 10
    Xp = X_keep[:, in_panel]
    Zp = prep_cols(Xp)
    S_all = np.linalg.svd(Zp, compute_uv=False)
    U, S, Vt = np.linalg.svd(Zp, full_matrices=False)
    K_bars   = min(BARS_K, len(S))
    var_ratio_bars = ((S[:K_bars]**2) / (S_all**2).sum()).tolist()
    U, S, Vt = U[:, :NPC], S[:NPC], Vt[:NPC]
    var_ratio = (S**2) / (S_all**2).sum()
    cell_scores = U * S
    Zall = prep_cols(X_keep)
    gene_load3 = (Zall.T @ U) / S
    print(f'  initial SVD var explained PC1-{K_bars}: '
          f'{np.round(var_ratio_bars, 3)} (cum {sum(var_ratio_bars):.3f})')

    def fill_cube(M):
        m = np.max(np.abs(M), axis=0) + 1e-9
        return M / m
    cell_xyz = fill_cube(cell_scores)
    gene_xyz = fill_cube(gene_load3)

    Vpanel = Vt.T
    panel_genes_list = [g for g, k in zip(gene_names, in_panel) if k]
    pole_top = []
    for k in range(NPC):
        order = np.argsort(Vpanel[:, k])
        pole_top.append(panel_genes_list[order[-1]])
        pole_top.append(panel_genes_list[order[0]])

    cats = sorted(set(subs.tolist()))
    base_pal = list(Category20[20]) + list(Set3[12]) + list(Set1[9]) + list(Category10[10])
    subtype_palette = {c: base_pal[i % len(base_pal)] for i, c in enumerate(cats)}
    cell_color_default = [subtype_palette[s] for s in subs]

    def signed_pole(v3):
        k = int(np.argmax(np.abs(v3)))
        return 2*k + (0 if v3[k] >= 0 else 1)
    gene_pole = np.array([signed_pole(gene_load3[j]) for j in range(n_genes)])
    cell_pole = np.array([signed_pole(cell_scores[i]) for i in range(n_cells)])
    gene_color_default = [POLE_COLORS[p] for p in gene_pole]
    cell_dom_color     = [POLE_COLORS[p] for p in cell_pole]
    gene_dom_color     = gene_color_default

    def pole_loads(M):
        out = np.zeros((M.shape[0], 6))
        for k in range(NPC):
            out[:, 2*k]   = np.clip(M[:, k], 0, None)
            out[:, 2*k+1] = np.clip(-M[:, k], 0, None)
        return out
    cell_load = pole_loads(cell_xyz).round(4).tolist()
    gene_load = pole_loads(gene_xyz).round(4).tolist()

    EXPR_SCALE = 10
    expr_matrix = np.round(X_keep * EXPR_SCALE).astype(np.int16).tolist()
    panel_idx = [j for j, p in enumerate(in_panel.tolist()) if p]

    # ---- figure construction (same as SVD script) ----------------------------
    LIM = 1.0
    POLE = 1.22
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
        f'PC1,2,3 = ({cell_scores[i,0]:.2f}, {cell_scores[i,1]:.2f}, {cell_scores[i,2]:.2f})'
        for i in range(n_cells)]
    gene_hover_text = [
        f'<b>{gene_names[j]}</b>'
        + (' (panel HVG)' if in_panel[j] else ' (projected)')
        + f'<br>strongest: {POLE_NAMES[gene_pole[j]]} ({pole_top[gene_pole[j]]})<br>'
        + f'mean={mean_expr[j]:.2f}, std={std_expr[j]:.2f}<br>'
        + f'loadings PC1,2,3 = ({gene_load3[j,0]:.2f}, {gene_load3[j,1]:.2f}, {gene_load3[j,2]:.2f})'
        for j in range(n_genes)]

    n_panel_disp = int(in_panel.sum())
    n_imputed    = n_genes - n_panel_disp
    historical_outliers = list(base.GROUP['cache_outliers']) + list(base.GROUP['runtime_exclude'])
    excluded_blurb = (
        f'Historically-flagged outlier subtypes ({", ".join(historical_outliers)}) are '
        f'<b>included</b> here — uncheck them in the subtype row and recompute to drop them.'
        if historical_outliers else 'No subtypes are flagged as outliers.')

    fig_cells = build_fig(cell_xyz, cell_color_default, cell_hover_text,
                          f'Cells — SVD scores (n={n_cells})  '
                          f'<i>recompute on the subtype subset to fit PCs to it</i>')
    fig_genes = build_fig(gene_xyz, gene_color_default, gene_hover_text,
                          f'Genes — SVD loadings ({n_panel_disp} panel + {n_imputed} projected)  '
                          f'<i>updated by recompute as panel & projected genes</i>')

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

    # gene-set masks (same as the SVD script)
    panel_mask_list = in_panel.tolist()
    set_masks  = {'all': [True]*n_genes, 'panel': panel_mask_list}
    set_counts = {'all': n_genes,        'panel': n_panel_disp}
    for name, gene_list in base.GENE_SETS.items():
        gset = set(gene_list)
        mask = [g in gset for g in gene_names]
        set_masks[name] = mask; set_counts[name] = sum(mask)

    mean_min, mean_max = float(np.min(mean_expr)), float(np.max(mean_expr))
    std_min,  std_max  = float(np.min(std_expr)),  float(np.max(std_expr))

    # subtype checkbox row
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
        f"let cell_score = {json.dumps(cell_scores.round(3).tolist())};\n"
        f"let gene_loading = {json.dumps(gene_load3.round(3).tolist())};\n"
        f"let cell_xyz_arr = {json.dumps(cell_xyz.round(4).tolist())};\n"
        f"let cell_dom_color = {json.dumps(cell_dom_color)};\n"
        f"let gene_dom_color = {json.dumps(gene_dom_color)};\n"
        f"let cell_active = Array({n_cells}).fill(true);\n"
        f"let pole_top = {json.dumps(pole_top)};\n"
        f"const panel_idx = {json.dumps(panel_idx)};\n"
        f"const POLE_NAMES_  = {json.dumps(POLE_NAMES)};\n"
        f"const POLE_COLORS_ = {json.dumps(POLE_COLORS)};\n"
        f"const gene_sets    = {json.dumps(set_masks)};\n"
        f"const gene_set_counts = {json.dumps(set_counts)};\n"
        f"const BARS_K = {K_bars};\n"
        f"let var_ratio_bars = {json.dumps([round(float(v), 4) for v in var_ratio_bars])};\n"
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

    vr = [f'{100*v:.1f}%' for v in var_ratio]
    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{GROUP_NAME} SVD recompute explorer</title>
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
#recompute-btn, #recompute-genes-btn {{ font-weight:600; color:white;
                  padding:4px 12px; border-radius:3px; cursor:pointer; }}
#recompute-btn          {{ background:#ff7f0e; border:1px solid #cc6510; }}
#recompute-btn:hover    {{ background:#ec6a00; }}
#recompute-genes-btn    {{ background:#1f77b4; border:1px solid #145a86; }}
#recompute-genes-btn:hover {{ background:#1565a0; }}
.rank-label {{ font-size: 12px; color: #555; font-weight: 600;
                display: inline-flex; align-items: center; gap: 4px; }}
#rank-input {{ width: 42px; padding: 2px 4px; font-size: 12px;
                border: 1px solid #bbb; border-radius: 3px;
                text-align: center; }}
#recompute-btn:disabled, #recompute-genes-btn:disabled {{ background:#aaa; border-color:#888; cursor:not-allowed; }}
#mean-slider, #std-slider {{ width: 180px; }}
.var-bars {{ display:flex; align-items:flex-end; gap:5px;
             height: 76px; padding: 4px 8px; background:#fafafa;
             border:1px solid #e0e0e0; border-radius:3px; }}
.var-bars .bar-col {{ display:flex; flex-direction:column; align-items:center;
                       justify-content:flex-end; width:38px; height:100%;
                       font-size:10px; color:#444; line-height:1.1; }}
.var-bars .bar-pct  {{ font-weight:600; color:#222; }}
.var-bars .bar-fill {{ width:22px; background:#888; border-radius:1px 1px 0 0;
                        margin: 1px 0; min-height:1px; }}
.var-bars .bar-fill.top1 {{ background:#d62728; }}
.var-bars .bar-fill.top2 {{ background:#1f77b4; }}
.var-bars .bar-fill.top3 {{ background:#2ca02c; }}
.var-bars .bar-name {{ color:#666; }}
.var-bars-label {{ font-size:11px; color:#555; font-weight:600;
                    align-self:center; margin-right:4px; }}
.viz-stack {{ flex: 1 1 auto; display: flex; flex-direction: column;
              min-height: 0; gap: 6px; }}
.viz-stack > .row {{ flex: 2 1 0; min-height: 200px; }}
.row {{ display: flex; flex-direction: row; gap: 8px; min-height: 0; }}
.col {{ flex: 1 1 0; min-width: 0; display: flex; flex-direction: column; }}
.col > .plotly-graph-div {{ flex: 1 1 auto; min-height: 0; height: 100% !important; }}
.heatmap-row {{ flex: 1 1 0; min-height: 150px; display: flex; flex-direction: column;
                background: #fafafa; border: 1px solid #e0e0e0; border-radius: 3px;
                padding: 4px 6px; }}
.heatmap-caption {{ font-size: 11px; color: #555; line-height: 1.2;
                    margin-bottom: 2px; display: flex; justify-content: space-between; }}
.line-strip-wrap {{ flex: 0 0 96px; position: relative; min-height: 72px;
                     border-bottom: 1px solid #c0c0c0; background: #ffffff; }}
#line-canvas {{ position: absolute; left: 0; top: 0;
                 width: 100%; height: 100%; }}
.heatmap-canvas-wrap {{ flex: 1 1 auto; min-height: 0; position: relative; }}
#heatmap-canvas {{ position: absolute; left: 0; top: 0;
                    width: 100%; height: 100%;
                    image-rendering: -moz-crisp-edges;
                    image-rendering: pixelated; }}
#heatmap-overlay {{ position: absolute; left: 0; top: 0;
                    width: 100%; height: 100%; pointer-events: none; }}
#status, #recompute-status {{ color: #555; font-size: 12px; }}
.legend {{ flex: 0 0 auto; font-size: 11px; color: #444; margin-top: 4px; }}
button {{ font-size: 13px; padding: 4px 10px; }}
details summary {{ cursor: pointer; color: #666; font-size: 12px; }}
{base.VIZ_NAV_CSS}
</style>
</head>
<body>
<h2>{GROUP_NAME} SVD biplot — subtype-subset recompute</h2>
{base.viz_nav_html(SLUG, 'svd')}
<div class="hint">
<b>Pick which subtypes you want to fit PCs to</b>, then click <b>Recompute SVD</b>.
The PC basis re-fits to the selected cells; every gene re-projects onto it.
</div>
<details class="header">
<summary>About this app (click to expand)</summary>
<div style="margin-top:4px;">
Initial view shows the SVD fit on all <b>{n_cells}</b> cells from the {GROUP_NAME} cohort
(panel: {n_panel_disp} HVG, z-scored across cells; PC1 {vr[0]}, PC2 {vr[1]}, PC3 {vr[2]}).
The {n_imputed} broader genes are projected onto the same singular vectors. {excluded_blurb}<br>
<b>Recompute</b> reruns the SVD client-side (block power iteration with deflation, top-3
components) on whichever cells are checked. After recompute, the PC basis, pole-label genes,
default colours, and every gene's projection update in place. Non-selected cells are hidden
(coordinates set to null) but stay in the data so you can re-check them later. The original
hover machinery (gene set buttons, mean/std sliders, gene search, QC overlays) all keep
working against the new basis.
</div>
</details>

<div class="controls">
  <div class="controls-row" style="background:#fff3e0; border:1px solid #ffcc80; border-radius:4px; padding:4px 6px;">
    <span class="label">Subtypes:</span>
    {subtype_checkbox_html}
    <button id="subt-all">all</button>
    <button id="subt-none">none</button>
    <span style="flex: 1 1 0;"></span>
    <label class="rank-label" title="Number of singular components to keep in the recompute (1–3). Lower rank collapses axes: rank=2 puts all points on the PC1×PC2 plane (z=0); rank=1 puts them on the PC1 axis.">rank
      <input id="rank-input" type="number" min="1" max="3" value="3" step="1"></label>
    <button id="recompute-btn" title="Refit SVD on the panel HVG, using only the checked-subtype cells.">Recompute on panel HVG →</button>
    <button id="recompute-genes-btn" title="Refit SVD using only the genes currently visible in the right biplot (gene set ∩ mean/std sliders).">Recompute on shown genes →</button>
    <span id="recompute-status" style="margin-left:8px;"></span>
  </div>
  <div class="controls-row" style="gap:8px;">
    <span class="var-bars-label">Variance explained:</span>
    <div id="svd-bars" class="var-bars"></div>
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
      <button id="qc-pt" class="qc-btn">Pseudotime</button>
      <button id="reset-btn">Reset colours</button>
    </span>
  </div>
  <div class="controls-row"><span id="status">Hover a cell (left) or a gene (right) to colour by expression.</span></div>
</div>
<div class="viz-stack">
  <div class="row">
    <div class="col">{cells_html}</div>
    <div class="col">{genes_html}</div>
  </div>
  <div class="heatmap-row">
    <div class="heatmap-caption">
      <span>Panel HVG z-expression — cells ordered by PC1 (pseudotime), genes by argmax of smoothed expression. <span id="heatmap-info" style="color:#888;"></span></span>
      <span>← early &nbsp; PC1 &nbsp; late →</span>
    </div>
    <div class="line-strip-wrap"><canvas id="line-canvas"></canvas></div>
    <div class="heatmap-canvas-wrap">
      <canvas id="heatmap-canvas"></canvas>
      <canvas id="heatmap-overlay"></canvas>
    </div>
  </div>
</div>
<div class="legend">
<b>Cell default colours</b> (subtype): {sub_legend}<br>
<b>Gene default colours</b> (strongest signed PC): <span id="pole-legend">{pole_legend}</span>
</div>
<script>
{js_data}

const cellPlot = document.getElementById('cell-plot');
const genePlot = document.getElementById('gene-plot');
const status   = document.getElementById('status');
const recomputeStatus = document.getElementById('recompute-status');
const POINTS_TRACE = 2, VERTEX_TRACE = 1, LOADING_TRACE = 3, HIGHLIGHT_TRACE = 4;
const DEFAULT_LOAD_COLORS = ['#e0e0e0','#e0e0e0','#e0e0e0','#e0e0e0','#e0e0e0','#e0e0e0'];
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
  const s = cell_score[i];
  status.innerHTML = '<b style="color:' + cell_dom_color[i] + '">Cell #' + i
    + '</b> <span style="color:#555">(' + cell_subtype[i] + ')</span> &nbsp; '
    + 'PC1,2,3 = (' + s[0].toFixed(2) + ', ' + s[1].toFixed(2) + ', ' + s[2].toFixed(2) + ') &nbsp; '
    + 'genes recoloured by expression (range ' + (lo/EXPR_SCALE).toFixed(2) + '..' + (hi/EXPR_SCALE).toFixed(2) + ', magma)';
}});

genePlot.on('plotly_hover', function(data) {{
  const pt = data.points[0]; if (pt.curveNumber !== POINTS_TRACE) return;
  const j = pt.pointNumber; if (lastHoveredGene === j) return; lastHoveredGene = j;
  const n = expr_matrix.length; const col = new Array(n);
  for (let i = 0; i < n; i++) col[i] = cell_active[i] ? expr_matrix[i][j] : null;
  // strip nulls for the magma range (so hidden cells stay dark)
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
  const L = gene_loading[j];
  const tag = gene_in_panel[j] ? '(panel)' : '(projected)';
  status.innerHTML = '<b style="color:' + gene_dom_color[j] + '">' + gene_name[j]
    + '</b> <span style="color:#555">' + tag + '</span> &nbsp; '
    + 'loadings PC1,2,3 = (' + L[0].toFixed(2) + ', ' + L[1].toFixed(2) + ', ' + L[2].toFixed(2) + ') &nbsp; '
    + 'cells recoloured by expression (range ' + (lo/EXPR_SCALE).toFixed(2) + '..' + (hi/EXPR_SCALE).toFixed(2) + ', magma)';
  drawHeatmapOverlay(j);
  drawLineGraph(j);
}});

document.getElementById('reset-btn').addEventListener('click', function() {{
  // Repaint cells using current default colors, dimmed for inactive cells.
  const cells = cell_default_colors.map((c, i) => cell_active[i] ? c : '#dddddd');
  Plotly.restyle(cellPlot, {{'marker.color': [cells]}}, [POINTS_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color': [gene_default_colors]}}, [POINTS_TRACE]);
  Plotly.restyle(cellPlot, {{'marker.color': [DEFAULT_LOAD_COLORS]}}, [LOADING_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color': [DEFAULT_LOAD_COLORS]}}, [LOADING_TRACE]);
  lastHoveredCell = null; lastHoveredGene = null;
  clearHeatmapOverlay();
  clearLineGraph();
  status.innerHTML = 'Reset. Hover a cell or gene to colour by expression and reveal PC projection.';
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
    + (gene_in_panel[j] ? '(panel)' : '(projected)')
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
document.getElementById('qc-pt').addEventListener('click', () => {{
  // Pseudotime colour = PC1 (cell_score[:,0]) ascending → viridis on active cells.
  const arr = cell_score.map(s => s[0]);
  colorByQC(arr, 'pseudotime (PC1)', v => v.toFixed(2));
}});

// ============================================================================
// Subtype-subset SVD recompute
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

function powerIterTopK(A, K, maxIter, tol) {{
  // A: m x n (array of Float64Array rows). Returns {{U, S, V}} for top-K SVD via
  // power iteration with deflation. U: K arrays of length m, V: K arrays length n.
  maxIter = maxIter || 80; tol = tol || 1e-7;
  const m = A.length, n = A[0].length;
  // Work on a deep copy so we can deflate without mutating the input
  const W = new Array(m);
  for (let i = 0; i < m; i++) W[i] = Float64Array.from(A[i]);
  const U = [], V = []; const S = new Float64Array(K);
  for (let k = 0; k < K; k++) {{
    // init v with non-zero deterministic-ish vector (sin-of-index) — avoids
    // the degenerate "all-zeros" power-iter trap and is reproducible.
    let v = new Float64Array(n);
    for (let j = 0; j < n; j++) v[j] = Math.sin((j + 1) * (k + 1) * 0.13) + 0.1;
    let vn = 0; for (let j = 0; j < n; j++) vn += v[j]*v[j]; vn = Math.sqrt(vn);
    for (let j = 0; j < n; j++) v[j] /= vn;
    let s_prev = 0, s = 0;
    let u = new Float64Array(m);
    for (let iter = 0; iter < maxIter; iter++) {{
      // u = W v
      for (let i = 0; i < m; i++) {{
        let acc = 0; const Wi = W[i];
        for (let j = 0; j < n; j++) acc += Wi[j] * v[j];
        u[i] = acc;
      }}
      let un = 0; for (let i = 0; i < m; i++) un += u[i]*u[i]; un = Math.sqrt(un);
      if (un < 1e-14) {{ break; }}
      for (let i = 0; i < m; i++) u[i] /= un;
      // v = W^T u
      let v_new = new Float64Array(n);
      for (let i = 0; i < m; i++) {{
        const ui = u[i], Wi = W[i];
        for (let j = 0; j < n; j++) v_new[j] += ui * Wi[j];
      }}
      let sn = 0; for (let j = 0; j < n; j++) sn += v_new[j]*v_new[j]; sn = Math.sqrt(sn);
      if (sn < 1e-14) {{ s = 0; break; }}
      for (let j = 0; j < n; j++) v_new[j] /= sn;
      s = sn;
      v = v_new;
      if (iter > 1 && Math.abs(s - s_prev) < tol * s) break;
      s_prev = s;
    }}
    S[k] = s;
    U.push(Float64Array.from(u));
    V.push(Float64Array.from(v));
    // Deflate: W -= s * u * v^T
    for (let i = 0; i < m; i++) {{
      const sui = s * u[i]; const Wi = W[i];
      for (let j = 0; j < n; j++) Wi[j] -= sui * v[j];
    }}
  }}
  return {{U, S, V}};
}}

function recomputeSVD(basisIdx, basisLabel) {{
  // basisIdx is the list of gene indices used as the SVD's basis ("panel" of
  // genes the SVD is *fit on*). Default = the panel HVG. Pass the currently-
  // visible gene indices to refit on whatever the gene biplot is showing.
  basisIdx = basisIdx || panel_idx;
  basisLabel = basisLabel || 'panel HVG';
  const t0 = performance.now();
  const sel = selectedSubtypes();
  // Cell index list of selected cells
  const cellSel = [];
  for (let i = 0; i < cell_subtype.length; i++) if (sel.has(cell_subtype[i])) cellSel.push(i);
  const m = cellSel.length;
  if (m < 4) {{
    recomputeStatus.innerHTML = '<span style="color:#c00">need ≥4 cells (got ' + m + ')</span>';
    return;
  }}
  const n_panel = basisIdx.length;
  const n_all   = gene_name.length;
  if (n_panel < 3) {{
    recomputeStatus.innerHTML = '<span style="color:#c00">need ≥3 basis genes (got ' + n_panel + ')</span>';
    return;
  }}

  // Build standardized basis matrix Zp on selected cells (m × n_panel)
  // Compute per-column mean & std on the subset, then z-score.
  const Zp = new Array(m);
  for (let i = 0; i < m; i++) Zp[i] = new Float64Array(n_panel);
  // Pass 1: collect raw values per panel gene
  const panelMean = new Float64Array(n_panel);
  const panelStd  = new Float64Array(n_panel);
  for (let k = 0; k < n_panel; k++) {{
    const j = basisIdx[k];
    let s = 0, ss = 0;
    for (let ii = 0; ii < m; ii++) {{
      const v = expr_matrix[cellSel[ii]][j] / EXPR_SCALE;
      Zp[ii][k] = v; s += v; ss += v*v;
    }}
    const mean = s / m;
    const var_ = Math.max(ss / m - mean*mean, 1e-18);
    panelMean[k] = mean; panelStd[k] = Math.sqrt(var_);
  }}
  for (let ii = 0; ii < m; ii++) {{
    const Zi = Zp[ii];
    for (let k = 0; k < n_panel; k++) Zi[k] = (Zi[k] - panelMean[k]) / panelStd[k];
  }}
  // Sum-of-squares of Zp (for var-explained ratio later)
  let frob2 = 0;
  for (let i = 0; i < m; i++) {{
    const Zi = Zp[i];
    for (let k = 0; k < n_panel; k++) frob2 += Zi[k] * Zi[k];
  }}

  // Power-iteration SVD top-BARS_K on Zp (the bar chart always shows top-10,
  // even when the user asked for a lower-rank embedding).
  const K_compute = Math.min(BARS_K, n_panel);
  const {{U, S, V}} = powerIterTopK(Zp, K_compute);
  // Update the variance-explained bars on the *full* spectrum sum (frob2).
  const newVarBars = new Array(BARS_K).fill(0);
  for (let k = 0; k < K_compute; k++) newVarBars[k] = (S[k]*S[k]) / (frob2 + 1e-12);
  var_ratio_bars = newVarBars;
  renderSvdBars(var_ratio_bars);

  // Embedding rank: user-chosen 1..3. Lower rank collapses unused axes to 0,
  // so rank=2 puts everything on the PC1×PC2 plane (z=0), rank=1 onto PC1.
  const rankReq = Math.max(1, Math.min(3, parseInt(document.getElementById('rank-input').value) || 3));
  const K_emb   = Math.min(rankReq, K_compute);

  // Cell scores for selected cells (m × 3 with k≥K_emb zeroed)
  const scoreSel = new Array(m);
  for (let ii = 0; ii < m; ii++) {{
    scoreSel[ii] = [0, 0, 0];
    for (let k = 0; k < K_emb; k++) scoreSel[ii][k] = U[k][ii] * S[k];
  }}

  // Project every gene onto the rank-K_emb basis: loading[j][k] = (z_j · u_k)/S_k
  const newGeneLoading = new Array(n_all);
  for (let j = 0; j < n_all; j++) {{
    let s = 0, ss = 0;
    for (let ii = 0; ii < m; ii++) {{
      const v = expr_matrix[cellSel[ii]][j] / EXPR_SCALE;
      s += v; ss += v*v;
    }}
    const mean = s / m;
    const stdv = Math.sqrt(Math.max(ss / m - mean*mean, 1e-18));
    const accs = [0, 0, 0];
    for (let ii = 0; ii < m; ii++) {{
      const z = (expr_matrix[cellSel[ii]][j] / EXPR_SCALE - mean) / stdv;
      for (let k = 0; k < K_emb; k++) accs[k] += z * U[k][ii];
    }}
    const out = [0, 0, 0];
    for (let k = 0; k < K_emb; k++) out[k] = accs[k] / S[k];
    newGeneLoading[j] = out;
  }}

  // Per-axis max-abs scaling to fill the [-1, 1] cube
  let cmax = [1e-9, 1e-9, 1e-9], gmax = [1e-9, 1e-9, 1e-9];
  for (let ii = 0; ii < m; ii++) for (let k = 0; k < 3; k++)
    if (Math.abs(scoreSel[ii][k]) > cmax[k]) cmax[k] = Math.abs(scoreSel[ii][k]);
  for (let j = 0; j < n_all; j++) for (let k = 0; k < 3; k++)
    if (Math.abs(newGeneLoading[j][k]) > gmax[k]) gmax[k] = Math.abs(newGeneLoading[j][k]);

  // Build new full-length cell_xyz arrays (nulls for non-selected)
  const n_cells_total = cell_subtype.length;
  const newCellX = new Array(n_cells_total).fill(null);
  const newCellY = new Array(n_cells_total).fill(null);
  const newCellZ = new Array(n_cells_total).fill(null);
  const newCellScore = cell_score.slice();          // shallow copy (will overwrite selected)
  const newCellLoad = new Array(n_cells_total);
  for (let i = 0; i < n_cells_total; i++) newCellLoad[i] = cell_load[i];   // preserve old for non-sel
  const newCellActive = new Array(n_cells_total).fill(false);
  cellSel.forEach((i, ii) => {{
    newCellActive[i] = true;
    newCellX[i] = scoreSel[ii][0] / cmax[0];
    newCellY[i] = scoreSel[ii][1] / cmax[1];
    newCellZ[i] = scoreSel[ii][2] / cmax[2];
    newCellScore[i] = [+scoreSel[ii][0].toFixed(3), +scoreSel[ii][1].toFixed(3),
                       +scoreSel[ii][2].toFixed(3)];
    // 6-pole loads for hover dots: relu(±xyz) on the cube-scaled xyz
    const x = newCellX[i], y = newCellY[i], z = newCellZ[i];
    newCellLoad[i] = [Math.max(x,0), Math.max(-x,0),
                       Math.max(y,0), Math.max(-y,0),
                       Math.max(z,0), Math.max(-z,0)].map(v => +v.toFixed(4));
  }});

  const newGeneX = new Array(n_all), newGeneY = new Array(n_all), newGeneZ = new Array(n_all);
  const newGeneLoad = new Array(n_all);
  for (let j = 0; j < n_all; j++) {{
    const L = newGeneLoading[j];
    newGeneX[j] = +(L[0] / gmax[0]).toFixed(4);
    newGeneY[j] = +(L[1] / gmax[1]).toFixed(4);
    newGeneZ[j] = +(L[2] / gmax[2]).toFixed(4);
    newGeneLoad[j] = [Math.max(newGeneX[j],0), Math.max(-newGeneX[j],0),
                      Math.max(newGeneY[j],0), Math.max(-newGeneY[j],0),
                      Math.max(newGeneZ[j],0), Math.max(-newGeneZ[j],0)].map(v => +v.toFixed(4));
  }}

  // Dominant-PC color per cell/gene (signed pole = argmax|coord| × 2 + sign)
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
  for (let j = 0; j < n_all; j++) {{
    newGeneDom[j] = POLE_COLORS_[domPole(newGeneX[j], newGeneY[j], newGeneZ[j])];
  }}

  // Update pole-label genes: top + and - basis gene on each PC, from V
  const newPoleTop = new Array(6);
  for (let k = 0; k < 3; k++) {{
    let bestPosIdx = 0, bestPos = -Infinity, bestNegIdx = 0, bestNeg = Infinity;
    for (let i = 0; i < n_panel; i++) {{
      const w = V[k][i];
      if (w > bestPos) {{ bestPos = w; bestPosIdx = i; }}
      if (w < bestNeg) {{ bestNeg = w; bestNegIdx = i; }}
    }}
    newPoleTop[2*k]   = gene_name[basisIdx[bestPosIdx]];
    newPoleTop[2*k+1] = gene_name[basisIdx[bestNegIdx]];
  }}
  // var explained ratio
  const totVar2 = frob2;
  const vr = [S[0]*S[0]/totVar2, S[1]*S[1]/totVar2, S[2]*S[2]/totVar2];

  // ---- commit new state to globals ----
  gene_x = newGeneX; gene_y = newGeneY; gene_z = newGeneZ;
  cell_load = newCellLoad; gene_load = newGeneLoad;
  cell_score = newCellScore;
  gene_loading = newGeneLoading.map(L => [+L[0].toFixed(3), +L[1].toFixed(3), +L[2].toFixed(3)]);
  cell_dom_color = newCellDom; gene_dom_color = newGeneDom;
  cell_active = newCellActive;
  pole_top = newPoleTop;

  // New gene_default_colors (by dominant signed PC)
  gene_default_colors = newGeneDom.slice();

  // ---- update both plots ----
  // 1. cell points: positions + colors (subtype-coloured, dimmed for hidden)
  const cellColors = cell_default_colors.map((c, i) => newCellActive[i] ? c : '#dddddd');
  Plotly.restyle(cellPlot, {{x:[newCellX], y:[newCellY], z:[newCellZ],
                              'marker.color':[cellColors]}}, [POINTS_TRACE]);
  // 2. gene points: positions + colors
  Plotly.restyle(genePlot, {{'marker.color':[gene_default_colors]}}, [POINTS_TRACE]);
  applyGeneFilter();   // applies the current mean/std sliders → positions for visible genes
  // 3. vertex labels (poles): update text on both plots
  const newPoleLab = [];
  for (let p = 0; p < 6; p++) newPoleLab.push(POLE_NAMES_[p] + '<br>(' + newPoleTop[p] + ')');
  Plotly.restyle(cellPlot, {{text:[newPoleLab], hovertext:[newPoleLab]}}, [VERTEX_TRACE]);
  Plotly.restyle(genePlot, {{text:[newPoleLab], hovertext:[newPoleLab]}}, [VERTEX_TRACE]);
  // 4. loading dots: reset to gray (hovered state is stale)
  Plotly.restyle(cellPlot, {{'marker.color':[DEFAULT_LOAD_COLORS]}}, [LOADING_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color':[DEFAULT_LOAD_COLORS]}}, [LOADING_TRACE]);
  // 5. clear gene search highlight
  Plotly.restyle(genePlot, {{x:[[null]], y:[[null]], z:[[null]]}}, [HIGHLIGHT_TRACE]);
  // 6. refresh the pole-colour legend in the page footer
  document.getElementById('pole-legend').innerHTML = newPoleTop.map((g, p) =>
    `<span style="display:inline-block;width:10px;height:10px;background:${{POLE_COLORS_[p]}};` +
    `margin-right:4px;border-radius:50%;"></span> ${{POLE_NAMES_[p]}} (${{g}}) &nbsp;&nbsp;`).join('');

  // 7. status
  const dt = ((performance.now() - t0) / 1000).toFixed(2);
  recomputeStatus.innerHTML = '<b>recomputed</b> on ' + m + ' / ' + n_cells_total
    + ' cells, ' + n_panel + ' ' + basisLabel + ' (rank ' + K_emb + ', ' + dt + 's) — '
    + 'PC1=' + (100*vr[0]).toFixed(1) + '%, PC2=' + (100*vr[1]).toFixed(1)
    + '%, PC3=' + (100*vr[2]).toFixed(1) + '% &nbsp; '
    + 'poles: ' + newPoleTop.map((g, p) => POLE_NAMES_[p] + '=' + g).join(', ');
  lastHoveredCell = null; lastHoveredGene = null;
  // 8. heatmap reflects the new PC1 ordering on the new active cell set
  renderHeatmap();
}}

document.getElementById('recompute-btn').addEventListener('click', () => {{
  const btn = document.getElementById('recompute-btn');
  btn.disabled = true; recomputeStatus.textContent = 'computing…';
  // Defer to next frame so the disabled state actually renders
  setTimeout(() => {{ try {{ recomputeSVD(panel_idx, 'panel HVG'); }} finally {{ btn.disabled = false; }} }}, 30);
}});

function visibleGeneIdx() {{
  const meanThr = parseFloat(meanSlider.value);
  const stdThr  = parseFloat(stdSlider.value);
  const mask = gene_sets[activeSet];
  const out = [];
  for (let j = 0; j < gene_name.length; j++) {{
    if (mask[j] && gene_mean[j] >= meanThr && gene_std[j] >= stdThr) out.push(j);
  }}
  return out;
}}
document.getElementById('recompute-genes-btn').addEventListener('click', () => {{
  const visible = visibleGeneIdx();
  if (visible.length < 3) {{
    recomputeStatus.innerHTML = '<span style="color:#c00">need ≥3 visible genes (got '
      + visible.length + '). Loosen the filters.</span>';
    return;
  }}
  const btn = document.getElementById('recompute-genes-btn');
  btn.disabled = true;
  recomputeStatus.textContent = 'computing on ' + visible.length + ' shown genes…';
  setTimeout(() => {{
    try {{ recomputeSVD(visible, 'shown genes'); }}
    finally {{ btn.disabled = false; }}
  }}, 30);
}});

// --- Pseudotime heatmap (cells × panel HVG, z-score, magma) ----------------
const heatCanvas    = document.getElementById('heatmap-canvas');
const heatOverlay   = document.getElementById('heatmap-overlay');
const lineCanvas    = document.getElementById('line-canvas');
const heatInfo      = document.getElementById('heatmap-info');
const Z_CLIP = 2.5;       // clip z-scores to ±Z_CLIP for the magma mapping
// State saved by renderHeatmap so the hover overlay can draw consistently.
let heatmapActiveIdx = null;            // length-m_active cell indices in PC1 order
let heatmapPanelToRow = null;           // length-n_panel: row index in heatmap, or -1
let heatmapGeneToPanelPos = null;       // length-n_genes: panel idx in panel_idx[], or -1
let heatmapSmoothWin = 0;
function ensureGeneToPanelPos() {{
  if (heatmapGeneToPanelPos) return;
  heatmapGeneToPanelPos = new Int32Array(gene_name.length).fill(-1);
  for (let p = 0; p < panel_idx.length; p++) heatmapGeneToPanelPos[panel_idx[p]] = p;
}}
function sizeHeatmapCanvas() {{
  const dpr = window.devicePixelRatio || 1;
  const hr = heatCanvas.getBoundingClientRect();
  const W = Math.max(1, Math.floor(hr.width  * dpr));
  const H = Math.max(1, Math.floor(hr.height * dpr));
  heatCanvas.width  = W; heatCanvas.height = H;
  heatOverlay.width = W; heatOverlay.height = H;
}}
function sizeLineCanvas() {{
  const dpr = window.devicePixelRatio || 1;
  const lr = lineCanvas.getBoundingClientRect();
  lineCanvas.width  = Math.max(1, Math.floor(lr.width  * dpr));
  lineCanvas.height = Math.max(1, Math.floor(lr.height * dpr));
}}
function renderHeatmap() {{
  sizeHeatmapCanvas();
  const ctx = heatCanvas.getContext('2d', {{alpha: false}});
  const W = heatCanvas.width, H = heatCanvas.height;
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, W, H);

  // 1. Active cells ordered by PC1 ascending
  const activeIdx = [];
  for (let i = 0; i < cell_active.length; i++) if (cell_active[i]) activeIdx.push(i);
  activeIdx.sort((a, b) => cell_score[a][0] - cell_score[b][0]);
  const m = activeIdx.length;
  if (m < 4) {{
    heatInfo.textContent = '— need ≥4 active cells';
    return;
  }}
  const nP = panel_idx.length;

  // 2. Per-gene z-score across active cells, in the new cell order
  const Zg = new Array(nP);
  for (let p = 0; p < nP; p++) {{
    const j = panel_idx[p];
    let s = 0, ss = 0;
    for (let ii = 0; ii < m; ii++) {{
      const v = expr_matrix[activeIdx[ii]][j] / EXPR_SCALE;
      s += v; ss += v*v;
    }}
    const mean = s / m;
    const stdv = Math.sqrt(Math.max(ss/m - mean*mean, 1e-12));
    const row = new Float32Array(m);
    for (let ii = 0; ii < m; ii++) {{
      row[ii] = (expr_matrix[activeIdx[ii]][j] / EXPR_SCALE - mean) / stdv;
    }}
    Zg[p] = row;
  }}

  // 3. Smooth each gene row with rolling mean for argmax detection
  const win = Math.max(7, Math.min(m, Math.floor(m / 50) | 0));
  const halfW = win >> 1;
  const argmaxOf = new Int32Array(nP);
  for (let p = 0; p < nP; p++) {{
    const row = Zg[p];
    let bestI = 0, bestV = -Infinity, s = 0, cnt = 0;
    // initialize running sum over [0, win)
    const initN = Math.min(win, m);
    for (let i = 0; i < initN; i++) s += row[i];
    cnt = initN;
    for (let i = 0; i < m; i++) {{
      // window edges centered on i: [i-halfW, i+halfW]
      // Add right edge entering
      if (i + halfW < m && i > 0) {{ s += row[i + halfW]; cnt++; }}
      // Remove left edge leaving
      if (i - halfW - 1 >= 0) {{ s -= row[i - halfW - 1]; cnt--; }}
      const sm = s / Math.max(1, cnt);
      if (sm > bestV) {{ bestV = sm; bestI = i; }}
    }}
    argmaxOf[p] = bestI;
  }}

  // 4. Sort genes by argmax (early → late)
  const order = [];
  for (let p = 0; p < nP; p++) order.push(p);
  order.sort((a, b) => argmaxOf[a] - argmaxOf[b]);

  // 5. Paint canvas: each gene row × cell column → magma(z clipped to ±Z_CLIP)
  const cellW = W / m;
  const rowH  = H / nP;
  // Use ImageData for speed when many tiny rects
  if (cellW < 1.5 && rowH < 2) {{
    const img = ctx.createImageData(W, H);
    const buf = img.data;
    for (let p = 0; p < nP; p++) {{
      const row = Zg[order[p]];
      const y0 = Math.floor(p * rowH);
      const y1 = Math.floor((p + 1) * rowH);
      for (let i = 0; i < m; i++) {{
        const z = row[i];
        const t = Math.max(0, Math.min(1, (z + Z_CLIP) / (2 * Z_CLIP)));
        const idx = Math.round(255 * t);
        const hex = magma[idx]; // "#rrggbb"
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        const x0 = Math.floor(i * cellW);
        const x1 = Math.floor((i + 1) * cellW);
        for (let y = y0; y < y1; y++) {{
          let pi = (y * W + x0) * 4;
          for (let x = x0; x < x1; x++) {{
            buf[pi++] = r; buf[pi++] = g; buf[pi++] = b; buf[pi++] = 255;
          }}
        }}
      }}
    }}
    ctx.putImageData(img, 0, 0);
  }} else {{
    // Fewer cells / rows: fillRect is fine and avoids the ImageData copy.
    for (let p = 0; p < nP; p++) {{
      const row = Zg[order[p]];
      const y = p * rowH;
      const h = rowH + 1;
      for (let i = 0; i < m; i++) {{
        const z = row[i];
        const t = Math.max(0, Math.min(1, (z + Z_CLIP) / (2 * Z_CLIP)));
        ctx.fillStyle = magma[Math.round(255 * t)];
        ctx.fillRect(i * cellW, y, cellW + 1, h);
      }}
    }}
  }}
  heatInfo.textContent = '— ' + m + ' cells × ' + nP + ' panel HVG (z-clipped ±' + Z_CLIP + ', smoothing window=' + win + ')';
  // Save state so the hover overlay can render consistently with this draw.
  heatmapActiveIdx = activeIdx;
  heatmapSmoothWin = win;
  ensureGeneToPanelPos();
  heatmapPanelToRow = new Int32Array(nP);
  for (let rowPos = 0; rowPos < nP; rowPos++) heatmapPanelToRow[order[rowPos]] = rowPos;
  // After a redraw the prior overlay/line is stale — repaint if a gene is still hovered.
  clearHeatmapOverlay();
  if (lastHoveredGene !== null && lastHoveredGene >= 0) {{
    drawHeatmapOverlay(lastHoveredGene);
    drawLineGraph(lastHoveredGene);
  }} else {{
    clearLineGraph();
  }}
}}

function clearHeatmapOverlay() {{
  if (!heatOverlay.getContext) return;
  const ctx = heatOverlay.getContext('2d');
  ctx.clearRect(0, 0, heatOverlay.width, heatOverlay.height);
}}

// Thin white outline around the gene's row in the heatmap (only for panel HVG).
function drawHeatmapOverlay(j) {{
  clearHeatmapOverlay();
  if (!heatmapActiveIdx || heatmapActiveIdx.length < 4) return;
  if (j < 0 || j >= gene_name.length) return;
  ensureGeneToPanelPos();
  const panelPos = heatmapGeneToPanelPos[j];
  if (panelPos < 0) return;            // not panel HVG — no row to outline
  const ctx = heatOverlay.getContext('2d');
  const W = heatOverlay.width, H = heatOverlay.height;
  const dpr = window.devicePixelRatio || 1;
  const nP = heatmapPanelToRow.length;
  const rowH = H / nP;
  const rowPos = heatmapPanelToRow[panelPos];
  const y0 = rowPos * rowH;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = Math.max(1, dpr);
  // Inset by 0.5 device-px so the stroke sits crisply on pixel boundaries
  ctx.strokeRect(0.5, y0 + 0.5, W - 1, rowH - 1);
}}

// The smoothed-expression line plot lives in its own strip above the heatmap,
// with its own y-axis (z-score). The strip's x-axis is shared with the heatmap
// (cells in PC1 order) but it never overlaps the gene rows below.
function clearLineGraph() {{
  if (!lineCanvas.getContext) return;
  sizeLineCanvas();
  const ctx = lineCanvas.getContext('2d');
  const W = lineCanvas.width, H = lineCanvas.height;
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, W, H);
  drawLineAxes(ctx, W, H);
  // placeholder when no gene is hovered
  const dpr = window.devicePixelRatio || 1;
  ctx.font = (11 * dpr) + 'px sans-serif';
  ctx.fillStyle = '#999';
  ctx.textAlign = 'center';
  ctx.fillText('hover a gene → smoothed expression curve appears here',
               W / 2, H / 2 + 4 * dpr);
}}

function drawLineAxes(ctx, W, H) {{
  const dpr = window.devicePixelRatio || 1;
  const gutter = Math.round(40 * dpr);
  // y-axis line
  ctx.strokeStyle = '#aaa';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(gutter + 0.5, 0); ctx.lineTo(gutter + 0.5, H);
  ctx.stroke();
  // ticks at +Z, 0, -Z
  function yFromZ(z) {{
    const t = Math.max(0, Math.min(1, (z + Z_CLIP) / (2 * Z_CLIP)));
    return H - t * H;
  }}
  ctx.strokeStyle = '#aaa';
  ctx.beginPath();
  [+Z_CLIP, 0, -Z_CLIP].forEach(z => {{
    const y = yFromZ(z);
    ctx.moveTo(gutter - 4 * dpr, y); ctx.lineTo(gutter, y);
  }});
  ctx.stroke();
  // labels
  ctx.font = (13 * dpr) + 'px sans-serif';
  ctx.fillStyle = '#444';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText('+' + Z_CLIP, gutter - 6 * dpr, yFromZ(+Z_CLIP) + 7 * dpr);
  ctx.fillText('0',          gutter - 6 * dpr, yFromZ(0));
  ctx.fillText('-' + Z_CLIP, gutter - 6 * dpr, yFromZ(-Z_CLIP) - 7 * dpr);
  // y-axis title
  ctx.save();
  ctx.translate(10 * dpr, H / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#444';
  ctx.font = (12 * dpr) + 'px sans-serif';
  ctx.fillText('z-score', 0, 0);
  ctx.restore();
  // light zero baseline across plot area
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.15)';
  ctx.beginPath();
  ctx.moveTo(gutter, yFromZ(0)); ctx.lineTo(W, yFromZ(0));
  ctx.stroke();
}}

function drawLineGraph(j) {{
  if (!heatmapActiveIdx || heatmapActiveIdx.length < 4) return;
  sizeLineCanvas();
  const ctx = lineCanvas.getContext('2d');
  const W = lineCanvas.width, H = lineCanvas.height;
  const dpr = window.devicePixelRatio || 1;
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, W, H);
  drawLineAxes(ctx, W, H);

  const m = heatmapActiveIdx.length;
  // Compute smoothed z-score of gene j across active cells in pseudotime order
  let s = 0, ss = 0;
  for (let ii = 0; ii < m; ii++) {{
    const v = expr_matrix[heatmapActiveIdx[ii]][j] / EXPR_SCALE;
    s += v; ss += v*v;
  }}
  const mean = s / m;
  const stdv = Math.sqrt(Math.max(ss/m - mean*mean, 1e-12));
  const raw = new Float32Array(m);
  for (let ii = 0; ii < m; ii++) raw[ii] = (expr_matrix[heatmapActiveIdx[ii]][j] / EXPR_SCALE - mean) / stdv;
  const win = heatmapSmoothWin > 0 ? heatmapSmoothWin : Math.max(7, Math.floor(m / 50) | 0);
  const halfW = win >> 1;
  const sm = new Float32Array(m);
  let acc = 0, cnt = 0;
  const initN = Math.min(win, m);
  for (let i = 0; i < initN; i++) acc += raw[i];
  cnt = initN;
  for (let i = 0; i < m; i++) {{
    if (i + halfW < m && i > 0) {{ acc += raw[i + halfW]; cnt++; }}
    if (i - halfW - 1 >= 0) {{ acc -= raw[i - halfW - 1]; cnt--; }}
    sm[i] = acc / Math.max(1, cnt);
  }}
  const gutter = Math.round(40 * dpr);
  const plotW = W - gutter;
  const cellW = plotW / m;
  function yFromZ(z) {{
    const t = Math.max(0, Math.min(1, (z + Z_CLIP) / (2 * Z_CLIP)));
    return H - t * H;
  }}
  // Blue smoothed line
  ctx.strokeStyle = '#1f77b4';
  ctx.lineWidth = Math.max(1.5 * dpr, 1.5);
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(gutter + cellW * 0.5, yFromZ(sm[0]));
  for (let i = 1; i < m; i++) ctx.lineTo(gutter + i * cellW + cellW * 0.5, yFromZ(sm[i]));
  ctx.stroke();
  // Gene label top-right
  ctx.font = 'bold ' + (11 * dpr) + 'px sans-serif';
  ctx.fillStyle = gene_dom_color[j] || '#222';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'top';
  ctx.fillText(gene_name[j] + (gene_in_panel[j] ? '' : '  (projected)'),
               W - 4 * dpr, 3 * dpr);
}}

// --- Variance-explained bar chart -------------------------------------------
function renderSvdBars(vr) {{
  // vr is a length-BARS_K array of var ratios in [0,1].
  // Bar height proportional to ratio, normalized to the largest bar.
  const maxR = Math.max(...vr, 1e-9);
  const barsEl = document.getElementById('svd-bars');
  const barH = 46;
  barsEl.innerHTML = vr.map((r, k) => {{
    const h = Math.max(1, (r / maxR) * barH);
    const cls = (k < 3) ? `top${{k+1}}` : '';
    return '<div class="bar-col">'
      + '<span class="bar-pct">' + (100*r).toFixed(1) + '%</span>'
      + '<div class="bar-fill ' + cls + '" style="height:' + h.toFixed(1) + 'px"></div>'
      + '<span class="bar-name">PC' + (k+1) + '</span>'
      + '</div>';
  }}).join('');
}}
renderSvdBars(var_ratio_bars);

function resizePlots() {{ Plotly.Plots.resize(cellPlot); Plotly.Plots.resize(genePlot); }}
let heatResizeTimer = null;
function scheduleHeatmapRedraw() {{
  if (heatResizeTimer) clearTimeout(heatResizeTimer);
  heatResizeTimer = setTimeout(renderHeatmap, 80);
}}
window.addEventListener('resize', () => {{ resizePlots(); scheduleHeatmapRedraw(); }});
setTimeout(function() {{ resizePlots(); applyGeneFilter(); renderHeatmap(); clearLineGraph(); }}, 80);
</script>
</body>
</html>"""

    with open(OUT, 'w') as f: f.write(page)
    print(f'  done. {os.path.getsize(OUT)/1e6:.1f} MB self-contained HTML.')
    print(f'  open: file://{OUT}')


if __name__ == '__main__':
    main()
