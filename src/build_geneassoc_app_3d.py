#!/usr/bin/env python
"""Gene-association explorer.

Same SVD biplot as the marker / recompute explorers, but built around a single
focus gene instead of two cell groups. Pick (search) a gene and the tool ranks
every other gene by how strongly it co-varies with it across cells:
  - Pearson correlation (signed) — positive / negative linear association
  - Mutual information (binned) — ANY dependence, including non-linear / non-
    monotonic "function of the rough value" relationships.
The cells recolour by the focus gene's expression; the gene biplot recolours by
each gene's association to the focus (diverging red/blue for correlation, magma
for MI); the ranked list lets you hover any associated gene to recolour the
cells by it and ring it in the gene biplot.

Self-contained single HTML. Usage:
    python build_geneassoc_app_3d.py [GROUP]
Output: notebooks/{group}_geneassoc_explorer_3d.html
"""
import os, json, base64, warnings
warnings.filterwarnings('ignore')
import numpy as np
from plotly.io import to_html
import plotly.graph_objects as go

import build_lamp5_archetype_app_4d as base
import build_lamp5_svd_app_3d as svdmod
from bokeh.palettes import Magma256, Viridis256

GROUP_NAME  = base.GROUP_NAME
SLUG        = base.SLUG
OUT         = os.path.join(base.ROOT, 'notebooks', f'{SLUG}_geneassoc_explorer_3d.html')
NPC         = 3
POLE_COLORS = svdmod.POLE_COLORS
POLE_NAMES  = svdmod.POLE_NAMES
prep_cols   = svdmod.prep_cols


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
    cell_layer_list = None if cell_layer_arr is None else [str(x) for x in cell_layer_arr]
    if cell_layer_list is not None and len(set(cell_layer_list)) <= 1:
        cell_layer_list = None
    cell_age_arr = proj.get('cell_age')
    cell_age_list = ([str(x) for x in cell_age_arr] if cell_age_arr is not None else None)
    # Optional per-cell region / genotype (only set when the cohort spans
    # multiple regions or genotypes). When present we expose a Region toggle.
    cell_region_arr = proj.get('cell_region')
    if cell_region_arr is None:
        cell_region_list = None
        region_options = []
    else:
        cell_region_list = [str(x) for x in cell_region_arr]
        region_options = sorted(set(cell_region_list))
        if len(region_options) <= 1:
            cell_region_list = None
            region_options = []
    n_cells = X_keep.shape[0]
    n_genes = len(gene_names)

    qc = base.compute_or_load_qc_full()
    assert np.array_equal(np.array(qc['subs']), subs), 'QC/proj cell-order mismatch'
    qc_total  = np.asarray(qc['total_counts'], dtype=np.float64)
    qc_ngenes = np.asarray(qc['n_genes'], dtype=np.float64)
    qc_ribo   = np.asarray(qc['pct_ribo'], dtype=np.float64)

    # ---- SVD biplot (full cohort) ------------------------------------------
    Xp = X_keep[:, in_panel]
    Zp = prep_cols(Xp)
    U, S, Vt = np.linalg.svd(Zp, full_matrices=False)
    U, S, Vt = U[:, :NPC], S[:NPC], Vt[:NPC]
    cell_scores = U * S
    Zall = prep_cols(X_keep)
    gene_load3 = (Zall.T @ U) / S
    print(f'  geneassoc: SVD on {Xp.shape[0]} cells × {int(in_panel.sum())} panel genes')

    def center_cube(M):
        # centre the cloud on its (robust) median so the scene origin sits in the
        # middle of the points — orbiting then rotates the cloud in place instead
        # of swinging it off-screen — then robust-scale to fill the [-1,1] cube.
        M = M - np.median(M, axis=0)
        s = np.percentile(np.abs(M), 99.5, axis=0) + 1e-9
        return np.clip(M / s, -1.0, 1.0)
    cell_xyz = center_cube(cell_scores)
    gene_xyz = center_cube(gene_load3)

    cats = sorted(set(subs.tolist()))
    subtype_palette = base.build_subtype_palette(cats)
    sub_idx_of = {c: i for i, c in enumerate(cats)}
    cell_sub_idx = np.array([sub_idx_of[s] for s in subs],
                            dtype=(np.uint16 if len(cats) > 255 else np.uint8))

    def signed_pole(v3):
        k = int(np.argmax(np.abs(v3)))
        return 2 * k + (0 if v3[k] >= 0 else 1)
    gene_pole = np.array([signed_pole(gene_load3[j]) for j in range(n_genes)])
    cell_pole = np.array([signed_pole(cell_scores[i]) for i in range(n_cells)])
    gene_color_default = [POLE_COLORS[p] for p in gene_pole]
    cell_dom_color = [POLE_COLORS[p] for p in cell_pole]
    gene_dom_color = gene_color_default

    # Pole-label genes: top + and - panel gene on each PC (from Vt).
    Vpanel = Vt.T
    panel_genes_list = [g for g, k in zip(gene_names, in_panel.tolist()) if k]
    pole_top = []
    for k in range(NPC):
        order = np.argsort(Vpanel[:, k])
        pole_top.append(panel_genes_list[order[-1]])
        pole_top.append(panel_genes_list[order[0]])

    def pole_loads(M):
        out = np.zeros((M.shape[0], 6))
        for k in range(NPC):
            out[:, 2 * k]     = np.clip(M[:, k], 0, None)
            out[:, 2 * k + 1] = np.clip(-M[:, k], 0, None)
        return out
    cell_load = pole_loads(cell_xyz).round(4).tolist()
    gene_load = pole_loads(gene_xyz).round(4).tolist()
    panel_idx = [j for j, p in enumerate(in_panel.tolist()) if p]

    EXPR_SCALE = 16
    _expr_q = np.clip(np.round(X_keep * EXPR_SCALE), 0, 255).astype(np.uint8)
    expr_b64 = base64.b64encode(_expr_q.tobytes()).decode('ascii')
    cell_sub_idx_b64 = base64.b64encode(cell_sub_idx.tobytes()).decode('ascii')

    LIM, POLE = 1.0, 1.22
    axis_ends = np.array([[POLE, 0, 0], [-POLE, 0, 0], [0, POLE, 0],
                          [0, -POLE, 0], [0, 0, POLE], [0, 0, -POLE]])

    def build_fig(xyz, colors, hover_text):
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
            marker=dict(size=4, color=colors, opacity=0.85, line=dict(width=0)),
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

    cell_hover_text = [
        f'#{i}<br>subtype: {subs[i]}<br>'
        f'PC1,2,3 = ({cell_scores[i,0]:.2f}, {cell_scores[i,1]:.2f}, {cell_scores[i,2]:.2f})'
        for i in range(n_cells)]
    gene_hover_text = [
        f'<b>{gene_names[j]}</b>' + (' (panel HVG)' if in_panel[j] else ' (projected)')
        + f'<br>mean={mean_expr[j]:.2f}, std={std_expr[j]:.2f}'
        for j in range(n_genes)]

    fig_cells = build_fig(cell_xyz, [subtype_palette[s] for s in subs], cell_hover_text)
    fig_genes = build_fig(gene_xyz, gene_color_default, gene_hover_text)
    cells_html = to_html(fig_cells, include_plotlyjs='cdn', full_html=False,
                         div_id='cell-plot', config={'displayModeBar': True, 'responsive': True})
    genes_html = to_html(fig_genes, include_plotlyjs=False, full_html=False,
                         div_id='gene-plot', config={'displayModeBar': True, 'responsive': True})

    gx, gy, gz = (np.round(gene_xyz[:, k], 4).tolist() for k in range(3))
    cx, cy, cz = (np.round(cell_xyz[:, k], 4).tolist() for k in range(3))

    panel_mask_list = in_panel.tolist()
    set_masks = {'all': [True] * n_genes, 'panel': panel_mask_list}
    set_counts = {'all': n_genes, 'panel': int(in_panel.sum())}
    for name, gene_list in base.GENE_SETS.items():
        gset = set(gene_list)
        mask = [g in gset for g in gene_names]
        set_masks[name] = mask; set_counts[name] = int(sum(mask))

    mean_min, mean_max = float(np.min(mean_expr)), float(np.max(mean_expr))
    std_min, std_max = float(np.min(std_expr)), float(np.max(std_expr))

    set_buttons_html = ''.join(
        f'<button class="set-btn{" active" if name == "all" else ""}" data-set="{name}">'
        f'{name} <span class="ct">({set_counts[name]})</span></button>'
        for name in (['all', 'panel'] + list(base.GENE_SETS.keys())))
    gene_datalist = ('<datalist id="gene-datalist">'
                     + ''.join(f'<option value="{g}">' for g in gene_names) + '</datalist>')

    # subtype subset checkboxes, grouped by subclass (all checked by default)
    subtype_counts = {c: int(np.sum(subs == c)) for c in cats}
    subtype_to_subclass = {}
    for s, csc in zip(subs, cell_subclass):
        subtype_to_subclass.setdefault(s, csc)
    subclass_order = sorted(set(cell_subclass.tolist()))
    by_subclass = {csc: [c for c in cats if subtype_to_subclass.get(c) == csc]
                   for csc in subclass_order}
    sub_rows = []
    for csc in subclass_order:
        ss = by_subclass[csc]
        if not ss:
            continue
        total = sum(subtype_counts[c] for c in ss)
        chips = ''.join(
            f'<label class="ga-sub"><input type="checkbox" class="ga-sub-cb" data-sub="{c}" checked> '
            f'<span style="color:{subtype_palette[c]}">●</span> {c} '
            f'<span class="ct">({subtype_counts[c]})</span></label>' for c in ss)
        sub_rows.append(
            f'<div class="ga-subclass"><div class="ga-subclass-head"><b>{csc}</b> '
            f'<span class="ct">({total} cells)</span> '
            f'<button class="ga-bulk" data-subclass="{csc}" data-on="1">all</button>'
            f'<button class="ga-bulk" data-subclass="{csc}" data-on="0">none</button></div>'
            f'<div class="ga-subclass-subs">{chips}</div></div>')
    subtype_checkbox_html = ''.join(sub_rows)

    # Region/Genotype toggle — only when the cohort spans multiple regions.
    if region_options:
        region_count = {r: cell_region_list.count(r) for r in region_options}
        region_toggle_html = (
            '<span class="region-toggle">'
            '<span class="label">Region:</span>'
            '<button class="rg-btn active" data-region="both" '
            f'title="Use all cells regardless of region">Both ({len(cell_region_list)})</button>'
            + ''.join(
                f'<button class="rg-btn" data-region="{r}" '
                f'title="Use only cells from {r}">{r} ({region_count[r]})</button>'
                for r in region_options)
            + '</span>')
    else:
        region_toggle_html = ''

    # Age toggle — only when the cohort spans multiple developmental ages.
    def _age_sort_key(s):
        s = str(s).strip()
        if s.startswith(('E', 'e')):
            try: return -float(s[1:])
            except ValueError: return 0
        if s.startswith(('P', 'p')):
            try: return float(s[1:])
            except ValueError: return 999
        return 1000
    if cell_age_list is not None:
        unique_ages = sorted(set(cell_age_list), key=_age_sort_key)
        age_count = {a: cell_age_list.count(a) for a in unique_ages}
        if len(unique_ages) > 1:
            age_toggle_html = (
                '<span class="age-toggle"><span class="label">Age:</span>'
                '<button class="ag-btn-all" data-act="all" title="Enable all ages">all</button>'
                '<button class="ag-btn-all" data-act="none" title="Disable all ages">none</button>'
                + ''.join(
                    f'<button class="ag-btn active" data-age="{a}" '
                    f'title="Include {a} cells ({age_count[a]})">{a}</button>'
                    for a in unique_ages)
                + '</span>')
        else:
            age_toggle_html = ''
    else:
        age_toggle_html = ''

    title = base.cohort_title(GROUP_NAME).replace(' Atlas', '')

    js_data = (
        f"const N_CELLS = {n_cells};\n"
        f"const N_GENES = {n_genes};\n"
        f"const EXPR_SCALE = {EXPR_SCALE};\n"
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
        f"let gene_x = {json.dumps(gx)};\n"
        f"let gene_y = {json.dumps(gy)};\n"
        f"let gene_z = {json.dumps(gz)};\n"
        f"let cell_x = {json.dumps(cx)};\n"
        f"let cell_y = {json.dumps(cy)};\n"
        f"let cell_z = {json.dumps(cz)};\n"
        f"const gene_mean = {json.dumps([round(float(v),4) for v in mean_expr])};\n"
        f"const gene_std = {json.dumps([round(float(v),4) for v in std_expr])};\n"
        f"let gene_default_colors = {json.dumps(gene_color_default)};\n"
        f"let cell_dom_color = {json.dumps(cell_dom_color)};\n"
        f"let gene_dom_color = {json.dumps(gene_dom_color)};\n"
        f"let cell_load = {json.dumps(cell_load)};\n"
        f"let gene_load = {json.dumps(gene_load)};\n"
        f"let cell_score = {json.dumps(cell_scores.round(3).tolist())};\n"
        f"let gene_loading = {json.dumps(gene_load3.round(3).tolist())};\n"
        f"let pole_top = {json.dumps(pole_top)};\n"
        f"const panel_idx = {json.dumps(panel_idx)};\n"
        f"const POLE_NAMES_ = {json.dumps(POLE_NAMES)};\n"
        f"const POLE_COLORS_ = {json.dumps(POLE_COLORS)};\n"
        f"const SET_MASKS = {json.dumps(set_masks)};\n"
        f"const qc_total = {json.dumps([round(float(v),2) for v in qc_total])};\n"
        f"const qc_ngenes = {json.dumps([int(v) for v in qc_ngenes])};\n"
        f"const qc_ribo = {json.dumps([round(float(v),3) for v in qc_ribo])};\n"
        f"const cell_layer = {json.dumps(cell_layer_list)};\n"
        f"const cell_region = {json.dumps(cell_region_list)};\n"
        f"const region_options = {json.dumps(region_options)};\n"
        f"const cell_age = {json.dumps(cell_age_list)};\n"
        f"const magma = {json.dumps(list(Magma256))};\n"
        f"const viridis = {json.dumps(list(Viridis256))};\n"
    )

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title} Gene Association Finder</title>
<style>
{base.UNIFIED_DESIGN_CSS}
html, body {{ margin: 0; padding: 0; }}
button {{ font-size: 13px; padding: 4px 10px; }}
{base.LAYOUT_CSS}
{base.VIZ_NAV_CSS}
{base.BUTTON_CSS}
{base.COLOR_KEY_CSS}
{base.HOME_LINK_CSS}
#focus-gene {{ font-size: 15px; font-weight: 600; padding: 7px 12px; width: 180px;
               border: 1px solid var(--btn-border); border-radius: 8px; }}
#ga-metric-row {{ margin-top: 6px; }}
#ga-results {{ display: flex; flex-direction: column; align-items: center; }}
.ga-grid {{ display: grid; grid-template-columns: minmax(110px,max-content) 1fr 64px;
            align-items: center; gap: 2px 12px; max-width: 540px; margin: 0 auto; }}
.ga-grid .ga-h {{ font-size: 11px; color: var(--muted); text-transform: uppercase;
                  letter-spacing: .04em; padding-bottom: 4px; }}
.ga-row {{ display: contents; cursor: pointer; }}
.ga-row > * {{ padding: 2px 0; }}
.ga-row:hover .ga-name, .ga-row.active .ga-name {{ color: var(--earth-dark); font-weight: 700; }}
.ga-name {{ font-size: 13px; white-space: nowrap; }}
.ga-bar-wrap {{ display: flex; align-items: center; gap: 6px; }}
.ga-bar {{ height: 9px; border-radius: 4px; min-width: 1px; }}
.ga-val {{ font-size: 11px; color: var(--muted); text-align: right;
           font-variant-numeric: tabular-nums; }}
.ga-foc {{ color: var(--accent); font-weight: 700; }}
.ga-subclass {{ margin: 4px 0; width: 100%; }}
.ga-subclass-head {{ font-size: 12px; margin-bottom: 2px; text-align: center; }}
.ga-subclass-head .ct {{ color: var(--muted); font-weight: 400; }}
.ga-bulk {{ font-size: 11px; padding: 1px 6px; margin-left: 4px; }}
.ga-subclass-subs {{ display: flex; flex-wrap: wrap; gap: 4px; justify-content: center; }}
.ga-sub {{ display: inline-flex; align-items: center; gap: 3px; padding: 1px 6px;
           border: 1px solid var(--line); border-radius: 12px; font-size: 12px; background: var(--card); }}
.ga-sub .ct {{ color: var(--muted); }}
.region-toggle {{ display: inline-flex; align-items: center; gap: 4px;
                    padding-right: 8px; margin-right: 6px; border-right: 1px dashed #d0a060; }}
.region-toggle .rg-btn {{ padding: 2px 8px; font-size: 12px; border: 1px solid #ccc;
                            background: #fff; border-radius: 3px; cursor: pointer; font-weight: 600; }}
.region-toggle .rg-btn:hover {{ background: #f6f6f6; }}
.region-toggle .rg-btn.active {{ background: #6b46c1; color: white; border-color: #553c9a; cursor: default; }}
.age-toggle {{ display: inline-flex; align-items: center; gap: 3px; padding-right: 8px;
               margin-right: 6px; border-right: 1px dashed #5fa86a; flex-wrap: wrap; max-width: 100%; }}
.age-toggle .ag-btn, .age-toggle .ag-btn-all {{ padding: 1px 6px; font-size: 11px;
  border: 1px solid #b0c4a0; background: #fff; border-radius: 3px; cursor: pointer; font-weight: 500; }}
.age-toggle .ag-btn-all {{ font-size: 10px; padding: 1px 5px; background: #f3f7f0; border-color: #c0d0b0; }}
.age-toggle .ag-btn:hover {{ background: #eef5ea; }}
.age-toggle .ag-btn.active {{ background: #2e7d32; color: white; border-color: #1b5e20; }}
.age-toggle .ag-btn:not(.active) {{ opacity: 0.5; text-decoration: line-through; }}
#recompute-btn, #recompute-genes-btn {{ font-weight: 600; color: white; padding: 4px 12px;
                  border-radius: 3px; cursor: pointer; }}
#recompute-btn       {{ background: #ff7f0e; border: 1px solid #cc6510; }}
#recompute-btn:hover {{ background: #ec6a00; }}
#recompute-genes-btn {{ background: #1f77b4; border: 1px solid #145a86; }}
#recompute-genes-btn:hover {{ background: #1565a0; }}
#recompute-btn:disabled, #recompute-genes-btn:disabled {{ background: #aaa; border-color: #888; cursor: not-allowed; }}
.rank-label {{ font-size: 12px; color: #555; font-weight: 600; display: inline-flex; align-items: center; gap: 4px; }}
#rank-input {{ width: 42px; padding: 2px 4px; font-size: 12px; border: 1px solid #bbb;
                border-radius: 3px; text-align: center; }}
.genestruct-box {{ display: flex; align-items: center; flex-wrap: wrap; gap: 6px;
                   border: 1px solid #cdd6e0; background: #f4f8fc; border-radius: 6px;
                   padding: 6px 10px; margin: 4px 0; font-size: 12px; }}
.genestruct-box .gs-title {{ font-weight: 700; color: #1f5d8c; margin-right: 6px; }}
#recompute-status, #genestruct-status {{ color: #555; font-size: 12px; }}
</style>
</head>
<body>
{base.HOME_LINK_HTML}
<div class="wrap">
<h2>{title} Gene Association Finder</h2>
<div class="hint">Search a <b>focus gene</b>; the tool ranks every other gene by how
strongly it co-varies with it across cells — signed <b>correlation</b> (positive / negative)
or <b>mutual information</b> (any dependence). Hover a result to recolour the cells by it.</div>

<div class="plot-pair">
  <div class="plot-box"><div class="plot-box-title" id="cell-plot-title"></div>{cells_html}</div>
  <div class="plot-box"><div class="plot-box-title" id="gene-plot-title"></div>{genes_html}</div>
</div>

<div class="controls-row" style="justify-content:center;"><span id="status">Pick a focus gene to begin.</span></div>

<div class="ctrl-box">
  <div class="ctrl-box-title">Plotting Method</div>
  <div class="controls-row">{base.viz_nav_html(SLUG, 'geneassoc')}</div>
</div>

<div class="ctrl-box">
  <div class="ctrl-box-title">Focus Gene &amp; Association</div>
  <div class="controls-row" style="gap:10px;">
    <span class="label">Focus gene:</span>
    <input id="focus-gene" list="gene-datalist" placeholder="e.g. Pvalb" autocomplete="off">
    <button id="focus-clear">clear</button>{gene_datalist}
  </div>
  <div class="controls-row" id="ga-metric-row" style="gap:8px;">
    <span class="label">rank by:</span>
    <button id="metric-corr" class="qc-btn active" title="Signed Pearson correlation across cells — positive and negative linear association.">correlation (signed)</button>
    <button id="metric-mi" class="qc-btn" title="Mutual information (8-bin) across cells — captures any dependence, including non-linear / non-monotonic.">mutual information</button>
    <span class="label" style="margin-left:10px;">top</span>
    <input id="ga-topn" type="number" min="5" max="60" value="20" step="5" style="width:56px;">
    <span id="ga-status" style="margin-left:10px; color:var(--muted);"></span>
  </div>
  <div id="ga-results"></div>
</div>

<div class="ctrl-box">
  <div class="ctrl-box-title">Color Scheme</div>
  <div class="controls-row">
    <button id="c-subtype" class="qc-btn active">Subtype</button>
    <button id="qc-counts" class="qc-btn">Counts</button>
    <button id="qc-genes" class="qc-btn">Genes</button>
    <button id="qc-ribo" class="qc-btn">% ribo</button>
    <button id="qc-density" class="qc-btn" title="Number of similar cells: local density of each cell in the CURRENT gene space (the considered genes). Genes are first reduced to a K-dim PCA latent so neighbour counts stay meaningful despite the curse of dimensionality, then each cell gets a Gaussian-kernel effective count of nearby cells (median-heuristic bandwidth, anchor-subsampled). Recompute-aware.">&asymp; similar cells</button>
    <button id="qc-pt" class="qc-btn" title="Colour cells by pseudotime (PC1 of the current embedding).">Pseudotime</button>
    <button id="qc-region" class="qc-btn" title="Colour each cell by its region / genotype. Greyed out for single-region cohorts.">Region/Genotype</button>
    <button id="qc-age" class="qc-btn" title="Colour each cell by its developmental stage. Greyed out for cohorts without age info.">Age</button>
    <button id="qc-layer" class="qc-btn" title="Colour each cell by dissected cortical layer (single-layer dissections only). Hidden for cohorts without layer info.">Layer of Microdissection</button>
  </div>
</div>

<div class="ctrl-box">
  <div class="ctrl-box-title">Recompute Embedding</div>
  <div class="controls-row">
    <label class="rank-label" title="Number of singular components to keep in the recompute (1–3). Lower rank collapses axes: rank=2 puts all points on the PC1×PC2 plane (z=0); rank=1 puts them on the PC1 axis.">rank
      <input id="rank-input" type="number" min="1" max="3" value="3" step="1"></label>
    <button id="recompute-btn" title="Refit SVD on the panel HVG, using only the active (checked-subtype, region/age-allowed) cells.">Replot on subtypes (panel HVG)</button>
    <button id="recompute-genes-btn" title="Refit SVD using only the genes currently shown in the right biplot (gene set ∩ mean/std ∩ recoverability filters), on the active cells.">Replot with shown genes</button>
    <span id="recompute-status" style="margin-left:8px;"></span>
  </div>
</div>

<div class="ctrl-box">
  <div class="ctrl-box-title">Filter Which Cell-types are Included</div>
  <div class="controls-row">
    {region_toggle_html}
    {age_toggle_html}
    <button id="ga-sub-all">all</button>
    <button id="ga-sub-none">none</button>
    <span class="label" id="ga-sub-count"></span>
  </div>
  <div class="controls-row" style="display:block;">{subtype_checkbox_html}</div>
</div>

<div class="ctrl-box">
  <div class="ctrl-box-title">Filter Which Genes are Considered</div>
  <div class="controls-row"><span class="label">Gene set:</span>{set_buttons_html}
    <span class="label" style="margin-left:14px;">Considered:</span>
    <span id="visible-count">0 / {n_genes}</span></div>
  <div class="controls-row">
    <span class="label">Min mean expr (log-CPM):</span>
    <input id="mean-slider" type="range" min="{mean_min:.3f}" max="{mean_max:.3f}" step="0.01" value="{mean_min:.3f}">
    <span id="mean-value">{mean_min:.2f}</span>
    <span class="label" style="margin-left:14px;">Min dispersion (std):</span>
    <input id="std-slider" type="range" min="{std_min:.3f}" max="{std_max:.3f}" step="0.01" value="{std_min:.3f}">
    <span id="std-value">{std_min:.2f}</span>
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

<span id="pole-legend" style="display:none"></span>

<footer class="cite">{base.cohort_citation(GROUP_NAME)}</footer>
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
    print(f'wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)')


JS_LOGIC = r"""
const cellPlot = document.getElementById('cell-plot');
const genePlot = document.getElementById('gene-plot');
const status   = document.getElementById('status');
const recomputeStatus = document.getElementById('recompute-status');
const POINTS_TRACE = 2, VERTEX_TRACE = 1, LOADING_TRACE = 3, HIGHLIGHT_TRACE = 4;
const DEFAULT_LOAD_COLORS = ['#e0e0e0','#e0e0e0','#e0e0e0','#e0e0e0','#e0e0e0','#e0e0e0'];

const cell_default_colors = new Array(N_CELLS);
const cell_subtype = new Array(N_CELLS);
for (let i = 0; i < N_CELLS; i++) {
  cell_default_colors[i] = SUBTYPE_PALETTE[cell_sub_idx[i]];
  cell_subtype[i] = SUBTYPE_CATS[cell_sub_idx[i]];
}
let gene_xyz = (function(){ const a=new Array(N_GENES); for(let j=0;j<N_GENES;j++) a[j]=[gene_x[j],gene_y[j],gene_z[j]]; return a; })();
function rebuildGeneXYZ(){ const a=new Array(N_GENES); for(let j=0;j<N_GENES;j++) a[j]=[gene_x[j],gene_y[j],gene_z[j]]; gene_xyz=a; }

// cell_active[] mirrors cellActive(i) so the ported recompute/density code (which
// reads a cell_active array) and the geneassoc predicate stay in sync.
let cell_active = new Array(N_CELLS).fill(true);
function syncCellActive(){ for(let i=0;i<N_CELLS;i++) cell_active[i]=cellActive(i); }

let colorMode = 'subtype';   // subtype | counts | genes | ribo | layer | gene | qc
let activeGene = -1;         // gene index when colorMode === 'gene'
let focusGene = -1;
let selectedSubs = new Set(SUBTYPE_CATS);   // which cell-types are included
// Region/genotype + age subset filters (extend the active-cell predicate).
let activeRegion = 'both';
const activeAges = new Set((typeof cell_age !== 'undefined' && cell_age) ? Array.from(new Set(cell_age)) : []);
function regionAllowed(i){ if(activeRegion==='both' || !cell_region) return true; return cell_region[i]===activeRegion; }
function ageAllowed(i){ if(typeof cell_age==='undefined' || !cell_age) return true; return activeAges.has(cell_age[i]); }
function cellActive(i) { return selectedSubs.has(cell_subtype[i]) && regionAllowed(i) && ageAllowed(i); }
function getSel() { const s = []; for (let i = 0; i < N_CELLS; i++) if (cellActive(i)) s.push(i); return s; }
function selectedSubtypes(){ return selectedSubs; }
const INVIS = 'rgba(0,0,0,0)';
let metric = 'corr';         // corr | mi
let geneActive = new Array(N_GENES).fill(true);
let assoc = null;            // Float64Array of association values to focusGene
// readVal(i,j): log-CPM expression the SVD / PCA needs.
function readVal(i, j){ return expr_matrix[i*N_GENES+j] / EXPR_SCALE; }

// ---- palettes --------------------------------------------------------------
function valuesToViridis(vals, mask) {
  let lo=Infinity, hi=-Infinity;
  for (let i=0;i<vals.length;i++){ if(mask && !mask[i]) continue; const v=vals[i]; if(v<lo)lo=v; if(v>hi)hi=v; }
  const range=(hi>lo)?(hi-lo):1;
  return { lo, hi, colorAt: v => viridis[Math.max(0,Math.min(255,Math.round(255*(v-lo)/range)))] };
}
function exprToMagmaArr(values){
  let lo=Infinity, hi=-Infinity;
  for (const v of values){ if(v<lo)lo=v; if(v>hi)hi=v; }
  const range=(hi>lo)?(hi-lo):1;
  return Array.from(values, v => magma[Math.max(0,Math.min(255,Math.round(255*(v-lo)/range)))]);
}
// diverging red(+)/white(0)/blue(-) for signed correlation
function divColor(r){
  const t=Math.max(-1,Math.min(1,r));
  if (t>=0){ const a=t; return 'rgb('+Math.round(255)+','+Math.round(255-175*a)+','+Math.round(255-205*a)+')'; }
  const a=-t; return 'rgb('+Math.round(255-215*a)+','+Math.round(255-175*a)+','+Math.round(255)+')';
}
function layerToDepth(s){
  if (s==null) return NaN; s=String(s).trim().toUpperCase();
  if (s.indexOf('-')>=0) return NaN;
  if (s==='L1')return 1; if(s==='L2/3')return 2.5; if(s==='L4')return 4; if(s==='L5')return 5; if(s==='L6')return 6; if(s==='L6B')return 6.5;
  return NaN;
}
const GREY_NO_LAYER='rgba(170,170,170,0.18)';

let qcLabel = '';           // label for the active QC color-by, shown in the title
function refreshTitles(){
  const ct=document.getElementById('cell-plot-title'), gt=document.getElementById('gene-plot-title');
  let cl = colorMode==='qc' ? qcLabel
        : colorMode==='gene' ? (gene_name[activeGene]+' expression')
        : (colorMode==='subtype'?'subtype':(colorMode==='counts'?'total counts':(colorMode==='genes'?'genes detected':(colorMode==='ribo'?'% ribosomal':'layer'))));
  if (ct) ct.innerHTML='Cells on <b>SVD</b> axes · coloured by <b>'+cl+'</b> · n='+getSel().length.toLocaleString();
  const gl = focusGene>=0 ? (metric==='corr'?'correlation to ':'mutual info with ')+gene_name[focusGene] : 'strongest PC';
  if (gt) gt.innerHTML='Genes on <b>SVD</b> axes · coloured by <b>'+gl+'</b>';
}

// ---- cell colouring --------------------------------------------------------
let qcColors = null;        // cached colors for the active QC color-by (density/pt/region/age)
function renderCells(){
  const col=new Array(N_CELLS);
  const act=new Array(N_CELLS); for(let i=0;i<N_CELLS;i++) act[i]=cellActive(i);
  if (colorMode==='qc' && qcColors){ for(let i=0;i<N_CELLS;i++) col[i]=act[i]?qcColors[i]:INVIS;
    Plotly.restyle(cellPlot,{'marker.color':[col]},[POINTS_TRACE]); refreshTitles(); return; }
  if (colorMode==='subtype'){ for(let i=0;i<N_CELLS;i++) col[i]=cell_default_colors[i]; clearColorKey(); }
  else if (colorMode==='counts'||colorMode==='genes'||colorMode==='ribo'){
    const src=colorMode==='counts'?qc_total:(colorMode==='genes'?qc_ngenes:qc_ribo);
    const vs=valuesToViridis(src,act);
    for(let i=0;i<N_CELLS;i++) col[i]=vs.colorAt(src[i]);
    setColorKeyGradient(colorMode==='counts'?'total counts':(colorMode==='genes'?'genes detected':'% ribosomal'),'viridis',vs.lo,vs.hi,v=>(colorMode==='ribo'?v.toFixed(1)+'%':Math.round(v).toLocaleString()));
  } else if (colorMode==='layer'){
    if(!cell_layer){ for(let i=0;i<N_CELLS;i++) col[i]=GREY_NO_LAYER; clearColorKey(); }
    else { const d=cell_layer.map(layerToDepth); const valid=d.filter((v,i)=>act[i]&&!isNaN(v));
      let lo=1,hi=6.5; if(valid.length){lo=Math.min.apply(null,valid);hi=Math.max.apply(null,valid);}
      const range=(hi>lo)?(hi-lo):1;
      for(let i=0;i<N_CELLS;i++) col[i]=isNaN(d[i])?GREY_NO_LAYER:viridis[Math.max(0,Math.min(255,Math.round(255*(d[i]-lo)/range)))];
      setColorKeyGradient('layer of microdissection','viridis',lo,hi,x=>'L'+Math.round(x)); }
  } else if (colorMode==='gene' && activeGene>=0){
    const j=activeGene; const vals=[],idx=[];
    for(let i=0;i<N_CELLS;i++) if(act[i]){ vals.push(expr_matrix[i*N_GENES+j]); idx.push(i); }
    const cols=exprToMagmaArr(vals);
    for(let i=0;i<N_CELLS;i++) col[i]=INVIS;
    for(let k=0;k<idx.length;k++) col[idx[k]]=cols[k];
    let lo=Infinity,hi=-Infinity; for(const v of vals){if(v<lo)lo=v;if(v>hi)hi=v;}
    if(lo===Infinity){lo=0;hi=0;}
    setColorKeyGradient(gene_name[j]+' expression','magma',lo/EXPR_SCALE,hi/EXPR_SCALE,v=>v.toFixed(2));
  }
  for(let i=0;i<N_CELLS;i++) if(!act[i]) col[i]=INVIS;
  Plotly.restyle(cellPlot,{'marker.color':[col]},[POINTS_TRACE]);
  refreshTitles();
}
function setMode(mode,btnId){ colorMode=mode; if(mode!=='gene') activeGene=-1;
  ['c-subtype','qc-counts','qc-genes','qc-ribo','qc-density','qc-pt','qc-region','qc-age','qc-layer'].forEach(b=>{const e=document.getElementById(b); if(e) e.classList.toggle('active',b===btnId);});
  if(mode!=='gene') ringGene(focusGene); renderCells(); }
document.getElementById('c-subtype').addEventListener('click',()=>setMode('subtype','c-subtype'));
document.getElementById('qc-counts').addEventListener('click',()=>setMode('counts','qc-counts'));
document.getElementById('qc-genes').addEventListener('click',()=>setMode('genes','qc-genes'));
document.getElementById('qc-ribo').addEventListener('click',()=>setMode('ribo','qc-ribo'));
document.getElementById('qc-layer').addEventListener('click',()=>setMode('layer','qc-layer'));
if(!cell_layer){ const lb=document.getElementById('qc-layer'); if(lb){lb.disabled=true; lb.title='No layer info for this cohort.';} }

// ---- QC color-by (density / pseudotime / region / age) ---------------------
// colorByQC paints active cells on viridis from an arbitrary per-cell scalar,
// caches the colors, switches into the 'qc' color mode, and sets the color key.
function valuesToViridisArr(values){
  let lo=Infinity, hi=-Infinity;
  for(const v of values){ if(v<lo)lo=v; if(v>hi)hi=v; }
  const range=(hi>lo)?(hi-lo):1;
  return values.map(v=>viridis[Math.max(0,Math.min(255,Math.round(255*(v-lo)/range)))]);
}
function setQCMode(btnId){ colorMode='qc'; activeGene=-1;
  ['c-subtype','qc-counts','qc-genes','qc-ribo','qc-density','qc-pt','qc-region','qc-age','qc-layer'].forEach(b=>{const e=document.getElementById(b);if(e)e.classList.toggle('active',b===btnId);});
}
function colorByQC(arr, label, fmt, btnId){
  const cols=new Array(N_CELLS);
  const valid=[]; for(let i=0;i<N_CELLS;i++) if(cellActive(i)) valid.push(arr[i]);
  const pal=valuesToViridisArr(valid);
  let vi=0; for(let i=0;i<N_CELLS;i++) cols[i]=cellActive(i)?pal[vi++]:INVIS;
  qcColors=cols; qcLabel=label; setQCMode(btnId); ringGene(focusGene);
  Plotly.restyle(cellPlot,{'marker.color':[cols]},[POINTS_TRACE]);
  let lo=Infinity,hi=-Infinity; for(const v of valid){ if(v<lo)lo=v; if(v>hi)hi=v; }
  if(lo===Infinity){lo=0;hi=0;}
  setColorKeyGradient(label,'viridis',lo,hi,fmt);
  refreshTitles();
}
function colorByQCCats(arr, label, colorOf, cats, btnId){
  const cols=new Array(N_CELLS);
  for(let i=0;i<N_CELLS;i++) cols[i]=cellActive(i)?(colorOf(arr[i])||'#888888'):INVIS;
  qcColors=cols; qcLabel=label; setQCMode(btnId); ringGene(focusGene);
  Plotly.restyle(cellPlot,{'marker.color':[cols]},[POINTS_TRACE]);
  setColorKeyCats(label, cats);
  refreshTitles();
}
const fmtInt = v => Math.round(v).toLocaleString();

// "≈ similar cells" density (recompute-aware; basis = considered genes).
document.getElementById('qc-density').addEventListener('click',()=>{
  const b=document.getElementById('qc-density'); b.disabled=true; status.innerHTML='computing density…';
  setTimeout(()=>{ try{ colorBySimilarCells(); }finally{ b.disabled=false; } },30);
});
// Pseudotime = PC1 of the current embedding (cell_score[:,0]).
document.getElementById('qc-pt').addEventListener('click',()=>{
  const arr=cell_score.map(s=>s[0]);
  colorByQC(arr,'pseudotime (PC1)',v=>v.toFixed(2),'qc-pt');
});
// Region / genotype color-by. Hidden for single-region cohorts.
const REGION_COLORS = { 'VISp':'#ff7f0e','V1':'#ff7f0e','ALM':'#9467bd' };
(function(){ const rb=document.getElementById('qc-region');
  const regs=(typeof cell_region!=='undefined'&&cell_region)?cell_region:[];
  const uniq=Array.from(new Set(regs));
  if(uniq.length<2){ if(rb) rb.remove(); return; }
  const palette={}; uniq.forEach((r,k)=>palette[r]=REGION_COLORS[r]||['#1f77b4','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2'][k%6]);
  rb.addEventListener('click',()=>{
    colorByQCCats(cell_region,'region/genotype',r=>palette[r],uniq.slice().sort().map(r=>({color:palette[r],label:r})),'qc-region');
  });
})();
// Age color-by. Hidden unless multiple ages.
function ageToNumber(s){ if(s==null) return NaN; s=String(s).trim();
  if(s[0]==='E'||s[0]==='e') return -(parseFloat(s.slice(1))||0);
  if(s[0]==='P'||s[0]==='p') return (parseFloat(s.slice(1))||0); return parseFloat(s); }
(function(){ const ab=document.getElementById('qc-age');
  const ages=(typeof cell_age!=='undefined'&&cell_age)?cell_age:[];
  const uniq=Array.from(new Set(ages));
  if(uniq.length<2){ if(ab) ab.remove(); return; }
  const ageNum=cell_age.map(ageToNumber);
  ab.addEventListener('click',()=>{
    colorByQC(ageNum,'age (E neg / P pos days)',v=>(v<0?'E'+(-v):'P'+v),'qc-age');
  });
})();

function colorByGene(j){
  activeGene=j; colorMode='gene';
  ['c-subtype','qc-counts','qc-genes','qc-ribo','qc-density','qc-pt','qc-region','qc-age','qc-layer'].forEach(b=>{const e=document.getElementById(b);if(e)e.classList.remove('active');});
  renderCells();
  document.querySelectorAll('.ga-row').forEach(r=>r.classList.toggle('active',parseInt(r.dataset.gene,10)===j));
}
function ringGene(j){
  if(j<0){ Plotly.restyle(genePlot,{x:[[null]],y:[[null]],z:[[null]]},[HIGHLIGHT_TRACE]); return; }
  Plotly.restyle(genePlot,{x:[[gene_xyz[j][0]]],y:[[gene_xyz[j][1]]],z:[[gene_xyz[j][2]]]},[HIGHLIGHT_TRACE]);
}

// ---- precompute binned matrix for mutual information -----------------------
const MI_BINS = 8;
const _edges = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 100.0];   // log-CPM bin edges (bin 0 = zero)
const binMat = new Uint8Array(N_CELLS * N_GENES);
(function(){
  for (let k = 0; k < binMat.length; k++) {
    const v = expr_matrix[k] / EXPR_SCALE;
    if (v <= 0) { binMat[k] = 0; continue; }
    let b = 1; while (b < MI_BINS - 1 && v >= _edges[b]) b++;
    binMat[k] = b;
  }
})();

// ---- gene filter -----------------------------------------------------------
let curSet='all';
const meanSlider=document.getElementById('mean-slider'), stdSlider=document.getElementById('std-slider');
function applyGeneFilter(){
  const mask=SET_MASKS[curSet], mmin=parseFloat(meanSlider.value), smin=parseFloat(stdSlider.value);
  const gsSlider=document.getElementById('genestruct-slider');
  const structThr=(gene_struct&&gsSlider)?parseFloat(gsSlider.value):0;
  document.getElementById('mean-value').textContent=mmin.toFixed(2);
  document.getElementById('std-value').textContent=smin.toFixed(2);
  let n=0; const gx=new Array(N_GENES),gy=new Array(N_GENES),gz=new Array(N_GENES);
  for(let j=0;j<N_GENES;j++){ const ok=mask[j]&&gene_mean[j]>=mmin&&gene_std[j]>=smin
      &&(!gene_struct||gene_struct[j]>=structThr); geneActive[j]=ok;
    if(ok){gx[j]=gene_x[j];gy[j]=gene_y[j];gz[j]=gene_z[j];n++;} else {gx[j]=null;gy[j]=null;gz[j]=null;} }
  Plotly.restyle(genePlot,{x:[gx],y:[gy],z:[gz]},[POINTS_TRACE]);
  document.getElementById('visible-count').textContent=n+' / '+N_GENES;
}
// Indices of the currently-considered (shown) genes. Used by "Replot with shown
// genes", the density basis, and the association candidate set.
function visibleGeneIdx(){ const out=[]; for(let j=0;j<N_GENES;j++) if(geneActive[j]) out.push(j); return out; }
document.querySelectorAll('.set-btn').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('.set-btn').forEach(x=>x.classList.remove('active'));
  b.classList.add('active'); curSet=b.dataset.set; applyGeneFilter(); if(focusGene>=0) computeAssoc();
}));
meanSlider.addEventListener('input',applyGeneFilter);
stdSlider.addEventListener('input',applyGeneFilter);

// ---- association compute ---------------------------------------------------
function computeCorr(f){
  // Pearson r of focus gene f vs every gene across the SELECTED cells
  const sel=getSel(), N=sel.length;
  const r=new Float64Array(N_GENES);
  let sf=0; for(let a=0;a<N;a++) sf+=expr_matrix[sel[a]*N_GENES+f]; const mf=sf/N;
  const fz=new Float64Array(N); let vf=0;
  for(let a=0;a<N;a++){ const x=expr_matrix[sel[a]*N_GENES+f]-mf; fz[a]=x; vf+=x*x; }
  const sdf=Math.sqrt(vf)||1;
  const sumj=new Float64Array(N_GENES), sumjj=new Float64Array(N_GENES), cross=new Float64Array(N_GENES);
  for(let a=0;a<N;a++){ const base=sel[a]*N_GENES, fza=fz[a];
    for(let j=0;j<N_GENES;j++){ const e=expr_matrix[base+j]; sumj[j]+=e; sumjj[j]+=e*e; cross[j]+=fza*e; } }
  for(let j=0;j<N_GENES;j++){ const mj=sumj[j]/N; const vj=sumjj[j]-N*mj*mj;
    const sdj=Math.sqrt(vj)||1; r[j]=cross[j]/(sdf*sdj); }
  return r;
}
function computeMI(f){
  const sel=getSel(), N=sel.length;
  const mi=new Float64Array(N_GENES); const B=MI_BINS;
  const pf=new Float64Array(B); for(let a=0;a<N;a++) pf[binMat[sel[a]*N_GENES+f]]++;
  const joint=new Float64Array(B*B);
  for(let j=0;j<N_GENES;j++){
    if(!geneActive[j] && j!==f){ mi[j]=0; continue; }
    joint.fill(0); const pj=new Float64Array(B);
    for(let a=0;a<N;a++){ const i=sel[a]; const bf=binMat[i*N_GENES+f], bj=binMat[i*N_GENES+j]; joint[bf*B+bj]++; pj[bj]++; }
    let m=0;
    for(let a=0;a<B;a++){ if(pf[a]===0) continue;
      for(let b=0;b<B;b++){ const c=joint[a*B+b]; if(c===0||pj[b]===0) continue;
        const pab=c/N; m += pab*Math.log2(pab/((pf[a]/N)*(pj[b]/N))); } }
    mi[j]=m;
  }
  return mi;
}
function onSubsChange(){
  selectedSubs=new Set(); document.querySelectorAll('.ga-sub-cb:checked').forEach(cb=>selectedSubs.add(cb.dataset.sub));
  syncCellActive(); invalidateGeneStruct();
  const e=document.getElementById('ga-sub-count'); if(e) e.textContent=getSel().length.toLocaleString()+' cells included';
  renderCells(); if(focusGene>=0) computeAssoc();
}
// Region / genotype toggle.
document.querySelectorAll('.rg-btn').forEach(btn=>btn.addEventListener('click',()=>{
  activeRegion=btn.dataset.region;
  document.querySelectorAll('.rg-btn').forEach(b=>b.classList.toggle('active',b===btn));
  onSubsChange();
}));
// Age toggle — per-age chips (AND filter on top of subtype + region).
document.querySelectorAll('.ag-btn').forEach(btn=>btn.addEventListener('click',()=>{
  const a=btn.dataset.age;
  if(activeAges.has(a)){ activeAges.delete(a); btn.classList.remove('active'); }
  else { activeAges.add(a); btn.classList.add('active'); }
  onSubsChange();
}));
document.querySelectorAll('.ag-btn-all').forEach(btn=>btn.addEventListener('click',()=>{
  const want=btn.dataset.act==='all';
  document.querySelectorAll('.ag-btn').forEach(b=>{ b.classList.toggle('active',want);
    if(want) activeAges.add(b.dataset.age); else activeAges.delete(b.dataset.age); });
  onSubsChange();
}));
document.querySelectorAll('.ga-sub-cb').forEach(cb=>cb.addEventListener('change',onSubsChange));
document.querySelectorAll('.ga-bulk').forEach(b=>b.addEventListener('click',()=>{ const on=b.dataset.on==='1', csc=b.dataset.subclass;
  document.querySelectorAll('.ga-subclass').forEach(block=>{ if(block.querySelector('.ga-subclass-head b').textContent!==csc) return;
    block.querySelectorAll('.ga-sub-cb').forEach(cb=>cb.checked=on); }); onSubsChange(); }));
document.getElementById('ga-sub-all').addEventListener('click',()=>{ document.querySelectorAll('.ga-sub-cb').forEach(cb=>cb.checked=true); onSubsChange(); });
document.getElementById('ga-sub-none').addEventListener('click',()=>{ document.querySelectorAll('.ga-sub-cb').forEach(cb=>cb.checked=false); onSubsChange(); });
function computeAssoc(){
  if(focusGene<0) return;
  const gs=document.getElementById('ga-status');
  gs.textContent='computing…';
  setTimeout(()=>{
    assoc = (metric==='corr') ? computeCorr(focusGene) : computeMI(focusGene);
    paintGenePlotByAssoc();
    renderAssocList();
    colorByGene(focusGene);     // cells coloured by the focus gene
    ringGene(focusGene);
    gs.textContent='';
  },20);
}
function paintGenePlotByAssoc(){
  const col=new Array(N_GENES);
  if(metric==='corr'){ for(let j=0;j<N_GENES;j++) col[j]=geneActive[j]?divColor(assoc[j]):'#e6e6e6';
    setColorKeyGradient('correlation to '+gene_name[focusGene],'rdbu',-1,1,v=>v.toFixed(1)); }
  else { let hi=1e-9; for(let j=0;j<N_GENES;j++) if(geneActive[j]&&assoc[j]>hi) hi=assoc[j];
    for(let j=0;j<N_GENES;j++){ const t=geneActive[j]?Math.max(0,Math.min(1,assoc[j]/hi)):0;
      col[j]=geneActive[j]?magma[Math.round(255*t)]:'#e6e6e6'; }
    setColorKeyGradient('mutual info with '+gene_name[focusGene],'magma',0,hi,v=>v.toFixed(2)); }
  col[focusGene]='#00e5ff';
  Plotly.restyle(genePlot,{'marker.color':[col]},[POINTS_TRACE]);
}
function renderAssocList(){
  const topN=Math.max(5,Math.min(60,parseInt(document.getElementById('ga-topn').value,10)||20));
  const cand=[]; for(let j=0;j<N_GENES;j++) if(geneActive[j]&&j!==focusGene) cand.push(j);
  let rows;
  if(metric==='corr'){
    cand.sort((a,b)=>assoc[b]-assoc[a]);
    const pos=cand.filter(j=>assoc[j]>0).slice(0,topN);
    const neg=cand.filter(j=>assoc[j]<0).slice(-topN).reverse();
    rows=pos.concat(neg);
  } else { cand.sort((a,b)=>assoc[b]-assoc[a]); rows=cand.slice(0,topN); }
  let maxAbs=1e-9; rows.forEach(j=>maxAbs=Math.max(maxAbs,Math.abs(assoc[j])));
  let html='<div class="ga-grid"><div class="ga-h">gene</div><div class="ga-h">'
    +(metric==='corr'?'correlation (red +, blue −)':'mutual information')+'</div><div class="ga-h"></div>';
  html+='<div class="ga-row" data-gene="'+focusGene+'"><div class="ga-name ga-foc">'+gene_name[focusGene]
    +' (focus)</div><div></div><div></div></div>';
  rows.forEach(j=>{ const v=assoc[j], w=Math.round(100*Math.abs(v)/maxAbs);
    const c=(metric==='corr')?divColor(v):magma[Math.round(255*Math.max(0,Math.min(1,v/maxAbs)))];
    html+='<div class="ga-row" data-gene="'+j+'"><div class="ga-name">'+gene_name[j]
      +(gene_in_panel[j]?'':' <span style="color:#aaa;font-size:10px">(proj)</span>')+'</div>'
      +'<div class="ga-bar-wrap"><span class="ga-bar" style="width:'+Math.max(2,w)+'%;background:'+c+'"></span></div>'
      +'<div class="ga-val">'+(metric==='corr'?(v>=0?'+':'')+v.toFixed(2):v.toFixed(3))+'</div></div>'; });
  html+='</div>';
  document.getElementById('ga-results').innerHTML=html;
  document.querySelectorAll('.ga-row').forEach(r=>{
    r.addEventListener('mouseenter',()=>{const j=parseInt(r.dataset.gene,10); colorByGene(j); ringGene(j);});
    r.addEventListener('click',()=>{const j=parseInt(r.dataset.gene,10); colorByGene(j); ringGene(j);});
  });
}

// ============================================================================
// In-browser SVD refit + density + gene-structure (ported from the recompute app)
// ============================================================================
function powerIterTopK(A, K, maxIter, tol) {
  // A: m x n (array of Float64Array rows). Returns {U, S, V} for top-K SVD via
  // power iteration with deflation. U: K arrays of length m, V: K arrays length n.
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
    let s_prev = 0, s = 0;
    let u = new Float64Array(m);
    for (let iter = 0; iter < maxIter; iter++) {
      for (let i = 0; i < m; i++) {
        let acc = 0; const Wi = W[i];
        for (let j = 0; j < n; j++) acc += Wi[j] * v[j];
        u[i] = acc;
      }
      let un = 0; for (let i = 0; i < m; i++) un += u[i]*u[i]; un = Math.sqrt(un);
      if (un < 1e-14) { break; }
      for (let i = 0; i < m; i++) u[i] /= un;
      let v_new = new Float64Array(n);
      for (let i = 0; i < m; i++) {
        const ui = u[i], Wi = W[i];
        for (let j = 0; j < n; j++) v_new[j] += ui * Wi[j];
      }
      let sn = 0; for (let j = 0; j < n; j++) sn += v_new[j]*v_new[j]; sn = Math.sqrt(sn);
      if (sn < 1e-14) { s = 0; break; }
      for (let j = 0; j < n; j++) v_new[j] /= sn;
      s = sn; v = v_new;
      if (iter > 1 && Math.abs(s - s_prev) < tol * s) break;
      s_prev = s;
    }
    S[k] = s;
    U.push(Float64Array.from(u));
    V.push(Float64Array.from(v));
    for (let i = 0; i < m; i++) {
      const sui = s * u[i]; const Wi = W[i];
      for (let j = 0; j < n; j++) Wi[j] -= sui * v[j];
    }
  }
  return {U, S, V};
}

function recomputeSVD(basisIdx, basisLabel) {
  // Refit the biplot on the active cells × a gene basis (panel HVG by default,
  // or the currently-shown genes). Updates both plots' positions, the per-cell /
  // per-gene dominant-pole colors and the 6 pole labels. Re-runs the association
  // afterwards if a focus gene is set.
  basisIdx = basisIdx || panel_idx;
  basisLabel = basisLabel || 'panel HVG';
  const t0 = performance.now();
  syncCellActive();
  const cellSel = [];
  for (let i = 0; i < cell_subtype.length; i++) if (cellActive(i)) cellSel.push(i);
  const m = cellSel.length;
  if (m < 4) { recomputeStatus.innerHTML = '<span style="color:#c00">need &ge;4 cells (got '+m+')</span>'; return; }
  const n_panel = basisIdx.length, n_all = gene_name.length;
  if (n_panel < 3) { recomputeStatus.innerHTML = '<span style="color:#c00">need &ge;3 basis genes (got '+n_panel+')</span>'; return; }

  const Zp = new Array(m);
  for (let i = 0; i < m; i++) Zp[i] = new Float64Array(n_panel);
  const panelMean = new Float64Array(n_panel), panelStd = new Float64Array(n_panel);
  for (let k = 0; k < n_panel; k++) {
    const j = basisIdx[k]; let s = 0, ss = 0;
    for (let ii = 0; ii < m; ii++) { const v = readVal(cellSel[ii], j); Zp[ii][k] = v; s += v; ss += v*v; }
    const mean = s / m; const var_ = Math.max(ss / m - mean*mean, 1e-18);
    panelMean[k] = mean; panelStd[k] = Math.sqrt(var_);
  }
  for (let ii = 0; ii < m; ii++) { const Zi = Zp[ii];
    for (let k = 0; k < n_panel; k++) Zi[k] = (Zi[k] - panelMean[k]) / panelStd[k]; }

  const K_compute = Math.min(3, n_panel);
  const {U, S, V} = powerIterTopK(Zp, K_compute);
  const rankReq = Math.max(1, Math.min(3, parseInt(document.getElementById('rank-input').value) || 3));
  const K_emb = Math.min(rankReq, K_compute);

  const scoreSel = new Array(m);
  for (let ii = 0; ii < m; ii++) { scoreSel[ii] = [0,0,0];
    for (let k = 0; k < K_emb; k++) scoreSel[ii][k] = U[k][ii] * S[k]; }

  const newGeneLoading = new Array(n_all);
  for (let j = 0; j < n_all; j++) {
    let s = 0, ss = 0;
    for (let ii = 0; ii < m; ii++) { const v = readVal(cellSel[ii], j); s += v; ss += v*v; }
    const mean = s / m; const stdv = Math.sqrt(Math.max(ss / m - mean*mean, 1e-18));
    const accs = [0,0,0];
    for (let ii = 0; ii < m; ii++) { const z = (readVal(cellSel[ii], j) - mean) / stdv;
      for (let k = 0; k < K_emb; k++) accs[k] += z * U[k][ii]; }
    const out = [0,0,0];
    for (let k = 0; k < K_emb; k++) out[k] = accs[k] / S[k];
    newGeneLoading[j] = out;
  }

  let cmax = [1e-9,1e-9,1e-9], gmax = [1e-9,1e-9,1e-9];
  for (let ii = 0; ii < m; ii++) for (let k = 0; k < 3; k++)
    if (Math.abs(scoreSel[ii][k]) > cmax[k]) cmax[k] = Math.abs(scoreSel[ii][k]);
  for (let j = 0; j < n_all; j++) for (let k = 0; k < 3; k++)
    if (Math.abs(newGeneLoading[j][k]) > gmax[k]) gmax[k] = Math.abs(newGeneLoading[j][k]);

  const n_cells_total = cell_subtype.length;
  const newCellX = new Array(n_cells_total).fill(null);
  const newCellY = new Array(n_cells_total).fill(null);
  const newCellZ = new Array(n_cells_total).fill(null);
  const newCellScore = cell_score.slice();
  const newCellLoad = new Array(n_cells_total);
  for (let i = 0; i < n_cells_total; i++) newCellLoad[i] = cell_load[i];
  cellSel.forEach((i, ii) => {
    newCellX[i] = scoreSel[ii][0] / cmax[0];
    newCellY[i] = scoreSel[ii][1] / cmax[1];
    newCellZ[i] = scoreSel[ii][2] / cmax[2];
    newCellScore[i] = [+scoreSel[ii][0].toFixed(3), +scoreSel[ii][1].toFixed(3), +scoreSel[ii][2].toFixed(3)];
    const x = newCellX[i], y = newCellY[i], z = newCellZ[i];
    newCellLoad[i] = [Math.max(x,0),Math.max(-x,0),Math.max(y,0),Math.max(-y,0),Math.max(z,0),Math.max(-z,0)].map(v=>+v.toFixed(4));
  });

  const newGeneX = new Array(n_all), newGeneY = new Array(n_all), newGeneZ = new Array(n_all);
  const newGeneLoad = new Array(n_all);
  for (let j = 0; j < n_all; j++) {
    const L = newGeneLoading[j];
    newGeneX[j] = +(L[0] / gmax[0]).toFixed(4);
    newGeneY[j] = +(L[1] / gmax[1]).toFixed(4);
    newGeneZ[j] = +(L[2] / gmax[2]).toFixed(4);
    newGeneLoad[j] = [Math.max(newGeneX[j],0),Math.max(-newGeneX[j],0),Math.max(newGeneY[j],0),Math.max(-newGeneY[j],0),Math.max(newGeneZ[j],0),Math.max(-newGeneZ[j],0)].map(v=>+v.toFixed(4));
  }

  function domPole(x, y, z) {
    const ax = Math.abs(x), ay = Math.abs(y), az = Math.abs(z);
    let k = 0, mx = ax; if (ay > mx) { k = 1; mx = ay; } if (az > mx) { k = 2; }
    const vals = [x, y, z];
    return 2*k + (vals[k] >= 0 ? 0 : 1);
  }
  const newCellDom = new Array(n_cells_total);
  for (let i = 0; i < n_cells_total; i++) {
    if (cellActive(i)) newCellDom[i] = POLE_COLORS_[domPole(newCellX[i], newCellY[i], newCellZ[i])];
    else newCellDom[i] = '#cccccc';
  }
  const newGeneDom = new Array(n_all);
  for (let j = 0; j < n_all; j++) newGeneDom[j] = POLE_COLORS_[domPole(newGeneX[j], newGeneY[j], newGeneZ[j])];

  const newPoleTop = new Array(6);
  for (let k = 0; k < 3; k++) {
    let bestPosIdx = 0, bestPos = -Infinity, bestNegIdx = 0, bestNeg = Infinity;
    for (let i = 0; i < n_panel; i++) { const w = V[k][i];
      if (w > bestPos) { bestPos = w; bestPosIdx = i; }
      if (w < bestNeg) { bestNeg = w; bestNegIdx = i; } }
    newPoleTop[2*k]   = gene_name[basisIdx[bestPosIdx]];
    newPoleTop[2*k+1] = gene_name[basisIdx[bestNegIdx]];
  }
  // ---- commit new state to globals ----
  gene_x = newGeneX; gene_y = newGeneY; gene_z = newGeneZ; rebuildGeneXYZ();
  cell_x = newCellX; cell_y = newCellY; cell_z = newCellZ;
  cell_load = newCellLoad; gene_load = newGeneLoad;
  cell_score = newCellScore;
  gene_loading = newGeneLoading.map(L => [+L[0].toFixed(3), +L[1].toFixed(3), +L[2].toFixed(3)]);
  cell_dom_color = newCellDom; gene_dom_color = newGeneDom;
  pole_top = newPoleTop;
  gene_default_colors = newGeneDom.slice();
  invalidateGeneStruct();

  // ---- update both plots ----
  Plotly.restyle(cellPlot, {x:[newCellX], y:[newCellY], z:[newCellZ]}, [POINTS_TRACE]);
  Plotly.restyle(genePlot, {'marker.color':[gene_default_colors]}, [POINTS_TRACE]);
  applyGeneFilter();
  const newPoleLab = POLE_NAMES_.slice();
  Plotly.restyle(cellPlot, {text:[newPoleLab], hovertext:[newPoleLab]}, [VERTEX_TRACE]);
  Plotly.restyle(genePlot, {text:[newPoleLab], hovertext:[newPoleLab]}, [VERTEX_TRACE]);
  Plotly.restyle(cellPlot, {'marker.color':[DEFAULT_LOAD_COLORS]}, [LOADING_TRACE]);
  Plotly.restyle(genePlot, {'marker.color':[DEFAULT_LOAD_COLORS]}, [LOADING_TRACE]);
  const pl = document.getElementById('pole-legend');
  if (pl) pl.innerHTML = newPoleTop.map((g, p) =>
    '<span style="display:inline-block;width:10px;height:10px;background:'+POLE_COLORS_[p]+';margin-right:4px;border-radius:50%;"></span> '+POLE_NAMES_[p]+' ('+g+') &nbsp;&nbsp;').join('');

  const dt = ((performance.now() - t0) / 1000).toFixed(2);
  recomputeStatus.innerHTML = '<b>recomputed</b> on ' + m + ' / ' + n_cells_total
    + ' cells, ' + n_panel + ' ' + basisLabel + ' (rank ' + K_emb + ', ' + dt + 's)';
  // Recolour the cells in the new embedding and (if a focus gene is set) re-rank.
  if (focusGene >= 0) computeAssoc(); else { qcColors = null; setMode('subtype','c-subtype'); }
}
document.getElementById('recompute-btn').addEventListener('click', () => {
  const btn = document.getElementById('recompute-btn');
  btn.disabled = true; recomputeStatus.textContent = 'computing…';
  setTimeout(() => { try { recomputeSVD(panel_idx, 'panel HVG'); } finally { btn.disabled = false; } }, 30);
});
document.getElementById('recompute-genes-btn').addEventListener('click', () => {
  const visible = visibleGeneIdx();
  if (visible.length < 3) { recomputeStatus.innerHTML = '<span style="color:#c00">need &ge;3 considered genes (got '+visible.length+'). Loosen the filters.</span>'; return; }
  const btn = document.getElementById('recompute-genes-btn');
  btn.disabled = true; recomputeStatus.textContent = 'computing on ' + visible.length + ' shown genes…';
  setTimeout(() => { try { recomputeSVD(visible, 'shown genes'); } finally { btn.disabled = false; } }, 30);
});

// ---- "Number of similar cells": local density in the CURRENT gene space -----
function colorBySimilarCells() {
  syncCellActive();
  const cellSel = [];
  for (let i = 0; i < cell_active.length; i++) if (cell_active[i]) cellSel.push(i);
  const m = cellSel.length;
  if (m < 10) { status.innerHTML = '<span style="color:#c00">need &ge;10 cells</span>'; return; }
  let basis = visibleGeneIdx();
  if (!basis || basis.length < 3) basis = panel_idx.slice();
  if (basis.length > 600) basis = basis.slice().sort((a,b) => gene_std[b]-gene_std[a]).slice(0,600);
  const nb = basis.length;
  const K = Math.max(2, Math.min(25, nb-1, m-1));
  const Zp = new Array(m); for (let i=0;i<m;i++) Zp[i] = new Float64Array(nb);
  for (let k=0;k<nb;k++) { const j=basis[k]; let s=0, ss=0;
    for (let ii=0;ii<m;ii++) { const v=readVal(cellSel[ii], j); Zp[ii][k]=v; s+=v; ss+=v*v; }
    const mean=s/m, sd=Math.sqrt(Math.max(ss/m-mean*mean,1e-18));
    for (let ii=0;ii<m;ii++) Zp[ii][k]=(Zp[ii][k]-mean)/sd; }
  const C=new Array(nb);
  for (let a=0;a<nb;a++) { const Ca=new Float64Array(nb);
    for (let ii=0;ii<m;ii++) { const za=Zp[ii][a]; if (za===0) continue; const Zi=Zp[ii];
      for (let b=a;b<nb;b++) Ca[b]+=za*Zi[b]; } C[a]=Ca; }
  for (let a=0;a<nb;a++) for (let b=a+1;b<nb;b++) C[b][a]=C[a][b];
  const pk = powerIterTopK(C, K, 60); const Vc = pk.U;
  const sc=new Float64Array(m*K);
  for (let k=0;k<K;k++) { const vk=Vc[k];
    for (let ii=0;ii<m;ii++) { let acc=0; const Zi=Zp[ii];
      for (let a=0;a<nb;a++) acc+=Zi[a]*vk[a]; sc[ii*K+k]=acc; } }
  const A=Math.min(m,1200), step=m/A; const anc=new Int32Array(A);
  for (let a=0;a<A;a++) anc[a]=Math.floor(a*step);
  const Bn=Math.min(A,400); const dd=[];
  for (let a=0;a<Bn;a++) for (let b=a+1;b<Bn;b++) { const ia=anc[a]*K, ib=anc[b]*K; let d2=0;
    for (let k=0;k<K;k++) { const t=sc[ia+k]-sc[ib+k]; d2+=t*t; } dd.push(d2); }
  dd.sort((x,y)=>x-y);
  const h2=Math.max(dd.length?dd[dd.length>>1]:1, 1e-12), inv2h2=1/(2*h2), scale=m/A;
  const dens=new Float64Array(cell_subtype.length);
  for (let ii=0;ii<m;ii++) { const base=ii*K; let acc=0;
    for (let a=0;a<A;a++) { const ia=anc[a]*K; let d2=0;
      for (let k=0;k<K;k++) { const t=sc[base+k]-sc[ia+k]; d2+=t*t; }
      acc+=Math.exp(-d2*inv2h2); }
    dens[cellSel[ii]]=acc*scale; }
  colorByQC(dens, 'similar cells (PCA K='+K+', '+nb+' genes)', v => Math.round(v).toLocaleString(), 'qc-density');
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
  syncCellActive();
  const cellSel = [];
  for (let i = 0; i < cell_subtype.length; i++) if (cellActive(i)) cellSel.push(i);
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
  colorMode='qc'; activeGene=-1;
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
  applyGeneFilter(); if (focusGene >= 0) computeAssoc();
});

// ---- focus gene + metric controls ------------------------------------------
const focusInput=document.getElementById('focus-gene');
function setFocus(name){ const j=gene_name.indexOf(name); if(j<0){ status.innerHTML='Gene <b>'+name+'</b> not found.'; return; }
  focusGene=j; status.innerHTML='Focus gene: <b>'+name+'</b> — '+(metric==='corr'?'genes ranked by correlation.':'genes ranked by mutual information.');
  computeAssoc(); }
focusInput.addEventListener('change',()=>{ const v=focusInput.value.trim(); if(v) setFocus(v); });
document.getElementById('focus-clear').addEventListener('click',()=>{ focusInput.value=''; focusGene=-1; assoc=null;
  document.getElementById('ga-results').innerHTML=''; ringGene(-1);
  Plotly.restyle(genePlot,{'marker.color':[gene_default_colors]},[POINTS_TRACE]); setMode('subtype','c-subtype'); status.innerHTML='Pick a focus gene to begin.'; });
document.getElementById('metric-corr').addEventListener('click',()=>{ metric='corr';
  document.getElementById('metric-corr').classList.add('active'); document.getElementById('metric-mi').classList.remove('active'); if(focusGene>=0) computeAssoc(); });
document.getElementById('metric-mi').addEventListener('click',()=>{ metric='mi';
  document.getElementById('metric-mi').classList.add('active'); document.getElementById('metric-corr').classList.remove('active'); if(focusGene>=0) computeAssoc(); });
document.getElementById('ga-topn').addEventListener('change',()=>{ if(focusGene>=0&&assoc) renderAssocList(); });

// hovering a gene in the right plot sets it as the focus gene
genePlot.on('plotly_hover',function(data){ if(!data.points||!data.points.length)return;
  const pt=data.points[0]; if(pt.curveNumber!==POINTS_TRACE)return; const j=pt.pointNumber; if(!geneActive[j])return;
  if(j===focusGene)return; focusInput.value=gene_name[j]; setFocus(gene_name[j]); });

function boot(){
  syncCellActive();
  applyGeneFilter();
  const e=document.getElementById('ga-sub-count'); if(e) e.textContent=getSel().length.toLocaleString()+' cells included';
  renderCells();
  refreshTitles();
  setTimeout(function(){ try{ Plotly.Plots.resize(cellPlot); Plotly.Plots.resize(genePlot); renderCells(); }catch(e){} },250);
}
if (cellPlot.data) boot(); else cellPlot.on('plotly_afterplot', function once(){ boot(); cellPlot.removeListener('plotly_afterplot', once); });
"""


if __name__ == '__main__':
    main()
