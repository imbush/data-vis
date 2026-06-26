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
    cell_subclass = np.array(proj.get(
        'cell_subclass',
        [GROUP_NAME] * len(subs)
    ))
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
    cell_donor_arr = proj.get('cell_donor')
    cell_donor_list = ([str(x) for x in cell_donor_arr]
                       if cell_donor_arr is not None else None)
    # Per-cell developmental age (dev-VIS cohort only). When non-None we
    # enable the Age colour-by button.
    cell_age_arr = proj.get('cell_age')
    cell_age_list = ([str(x) for x in cell_age_arr]
                     if cell_age_arr is not None else None)
    cell_layer_arr = proj.get('cell_layer')
    if cell_layer_arr is None:
        cell_layer_list = None
    else:
        cell_layer_list = [str(x) for x in cell_layer_arr]
        if len(set(cell_layer_list)) <= 1:
            cell_layer_list = None
    n_cells   = X_keep.shape[0]
    n_genes   = len(gene_names)

    qc = base.compute_or_load_qc_full()
    assert np.array_equal(np.array(qc['subs']), subs), 'QC/proj cell-order mismatch'
    qc_total, qc_ngenes, qc_ribo = (np.asarray(qc['total_counts']),
                                    np.asarray(qc['n_genes']),
                                    np.asarray(qc['pct_ribo']))
    _r = qc_ribo.astype(np.float64) - qc_ribo.mean()
    _rs = float(np.sqrt((_r * _r).sum()) + 1e-12)
    _Xc = X_keep.astype(np.float32) - X_keep.mean(0)
    _denom = np.sqrt((_Xc * _Xc).sum(0)) * _rs + 1e-12
    gene_ribo_corr = np.asarray((_Xc * _r[:, None]).sum(0) / _denom, dtype=np.float32)
    # Per-gene OLS slope of expression ~ pct_ribo. Used by the "Regress out
    # %ribo" toggle to residualise each gene before the recompute.
    _var_ribo = float((_r * _r).sum()) + 1e-12
    gene_ribo_slope = np.asarray((_Xc * _r[:, None]).sum(0) / _var_ribo, dtype=np.float32)
    mean_pct_ribo = float(qc_ribo.mean())

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

    # Per-axis top genes for GO: rank panel genes by their centroid coord on
    # each DC. ±30 most extreme on each pole feed Enrichr.
    GO_TOP_PER_POLE = 30
    top_genes_per_axis = []
    for k in range(NDC):
        col = gene_xyz[panel_idx, k]
        order = np.argsort(col)
        neg = [panel_genes_list[i] for i in order[:GO_TOP_PER_POLE]]
        pos = [panel_genes_list[i] for i in order[-GO_TOP_PER_POLE:]][::-1]
        top_genes_per_axis.append({'name': f'DC{k+1}', 'pos': pos, 'neg': neg})
    go_axes = []  # GO bars removed from UI; placeholder for JS data
    cats = sorted(set(subs.tolist()))
    subtype_palette = base.build_subtype_palette(cats)
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

    import base64
    EXPR_SCALE = 16
    _expr_q = np.clip(np.round(X_keep * EXPR_SCALE), 0, 255).astype(np.uint8)
    expr_b64 = base64.b64encode(_expr_q.tobytes()).decode('ascii')
    n_cells_emit = int(_expr_q.shape[0])
    n_genes_emit = int(_expr_q.shape[1])
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
        pole_lab = [POLE_NAMES[p] for p in range(6)]
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
            marker=dict(size=0,  color=['#e0e0e0']*6, opacity=0.0,
                        line=dict(width=0)),
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
                xaxis=dict(visible=False, range=[-lim, lim], autorange=False),
                yaxis=dict(visible=False, range=[-lim, lim], autorange=False),
                zaxis=dict(visible=False, range=[-lim, lim], autorange=False),
                aspectmode='cube', dragmode='orbit',
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.4),
                            center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1))),
            margin=dict(l=0, r=0, t=(40 if title else 4), b=0),
            paper_bgcolor='white', plot_bgcolor='white')
        return fig

    def _age_tag(i):
        if cell_age_list is None: return ''
        return f' · age {cell_age_list[i]}'
    cell_hover_text = [
        f'#{i}<br>subtype: {subs[i]}{_age_tag(i)}<br>'
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

    # No in-plot titles — each plot sits under its own HTML title box (see body).
    fig_cells = build_fig(cell_xyz, cell_color_default, cell_hover_text, '')
    fig_genes = build_fig(gene_xyz, gene_color_default, gene_hover_text, '')

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

    # Subtype checkbox row, grouped by cell_subclass with per-group all/none.
    subtype_counts = {c: int(np.sum(subs == c)) for c in cats}
    subtype_to_subclass = {}
    for s, csc in zip(subs, cell_subclass):
        if s not in subtype_to_subclass:
            subtype_to_subclass[s] = csc
    subclass_order = sorted(set(cell_subclass.tolist()))
    by_subclass = {csc: [c for c in cats if subtype_to_subclass.get(c) == csc]
                   for csc in subclass_order}
    default_subset = set(base.GROUP.get('default_selected_subtypes') or ())
    def _chk(c):
        return 'checked' if (not default_subset or c in default_subset) else ''
    subtype_group_html_parts = []
    for csc in subclass_order:
        sub_subs = by_subclass[csc]
        if not sub_subs: continue
        total_in_grp = sum(subtype_counts[c] for c in sub_subs)
        subtype_group_html_parts.append(
            f'<div class="subt-group" data-subclass="{csc}">'
            f'<div class="subt-group-head">'
            f'<b>{csc}</b> <span class="ct">({total_in_grp} cells, {len(sub_subs)} subtypes)</span>'
            f'<button class="grp-toggle" data-grp="{csc}" data-action="all" title="Check all in {csc}">all</button>'
            f'<button class="grp-toggle" data-grp="{csc}" data-action="none" title="Uncheck all in {csc}">none</button>'
            f'</div>'
            f'<div class="subt-group-checkboxes">'
            + ''.join(
                f'<label class="subt-chk" data-grp-sub="{csc}">'
                f'<input type="checkbox" data-sub="{c}" data-grp="{csc}" {_chk(c)}> '
                f'<span style="color:{subtype_palette[c]}; font-weight:700;">●</span> '
                f'{c} <span class="ct">({subtype_counts[c]})</span></label>'
                for c in sub_subs)
            + '</div></div>'
        )
    subtype_checkbox_html = ''.join(subtype_group_html_parts)
    auto_recompute_on_load = bool(default_subset)

    # Build interned (cats + uint8 index) form of cell_subtype.
    _subs_list = subs.tolist()
    _cs_cats = list(dict.fromkeys(_subs_list))
    if len(_cs_cats) > 256:
        raise RuntimeError(f"cell_subtype has >256 unique categories; bump to uint16")
    _cs_lookup = {c: i for i, c in enumerate(_cs_cats)}
    _cs_idx_arr = np.array([_cs_lookup[v] for v in _subs_list], dtype=np.uint8)
    import base64 as _b64m
    _cs_idx_b64 = _b64m.b64encode(_cs_idx_arr.tobytes()).decode("ascii")
    js_data = (
        f"const EXPR_SCALE  = {EXPR_SCALE};\n"
        f"const N_CELLS = {n_cells_emit};\n"
        f"const N_GENES = {n_genes_emit};\n"
        f"const EXPR_B64 = {json.dumps(expr_b64)};\n"
        # Decode the base64 expression matrix into a flat Uint8Array once.
        # Index as expr_matrix[i * N_GENES + j] (divide by EXPR_SCALE for log-CPM).
        f"const expr_matrix = (function() {{\n"
        f"  const bin = atob(EXPR_B64);\n"
        f"  const u8 = new Uint8Array(bin.length);\n"
        f"  for (let k = 0; k < bin.length; k++) u8[k] = bin.charCodeAt(k);\n"
        f"  return u8;\n"
        f"}})();\n"
        f"const cell_default_colors = {json.dumps(cell_color_default)};\n"
        f"let gene_default_colors = {json.dumps(gene_color_default)};\n"
        f"const _cell_subtype_cats = {json.dumps(_cs_cats)};\n"
        f"const _cell_subtype_idx_b64 = {json.dumps(_cs_idx_b64)};\n"
        "const cell_subtype = (function() {\n"
        "  const bin = atob(_cell_subtype_idx_b64);\n"
        "  const out = new Array(bin.length);\n"
        "  for (let k = 0; k < bin.length; k++) out[k] = _cell_subtype_cats[bin.charCodeAt(k)];\n"
        "  return out;\n"
        "})();\n"
        f"const cell_region = {json.dumps(cell_region_list)};\n"
        f"const region_options = {json.dumps(region_options)};\n"
        f"const cell_age = {json.dumps(cell_age_list)};\n"
        f"const cell_layer = {json.dumps(cell_layer_list)};\n"
        f"const subtype_palette = {json.dumps(subtype_palette)};\n"
        f"const gene_name    = {json.dumps(gene_names)};\n"
        f"const gene_in_panel = {json.dumps(panel_mask_list)};\n"
        f"const gene_mean    = {json.dumps([round(float(v),3) for v in mean_expr])};\n"
        f"const gene_std     = {json.dumps([round(float(v),3) for v in std_expr])};\n"
        f"const gene_ribo_corr = {json.dumps([round(float(v),3) for v in gene_ribo_corr])};\n"
        f"const gene_ribo_slope = {json.dumps([round(float(v),4) for v in gene_ribo_slope])};\n"
        f"const mean_pct_ribo = {round(float(mean_pct_ribo), 3)};\n"
        f"const RIBO_CORR_THRESHOLD = 0.3;\n"
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

    if region_options:
        region_count = {r: cell_region_list.count(r) for r in region_options}
        region_toggle_html = (
            '<span class="region-toggle">'
            '<span class="label">Region:</span>'
            '<button class="rg-btn active" data-region="both" '
            f'title="Use all cells regardless of region">Both ({len(cell_region_list)})</button>'
            + ''.join(
                f'<button class="rg-btn" data-region="{r}" '
                f'title="Use only cells dissected from {r}">{r} ({region_count[r]})</button>'
                for r in region_options)
            + '</span>'
        )
    else:
        region_toggle_html = ''

    def _age_sort_key(s):
        s = str(s).strip()
        if s.startswith(('E','e')):
            try: return -float(s[1:])
            except ValueError: return 0
        if s.startswith(('P','p')):
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
                + '</span>'
            )
        else:
            age_toggle_html = ''
    else:
        age_toggle_html = ''


    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{GROUP_NAME} diffmap recompute explorer</title>
<style>
{base.UNIFIED_DESIGN_CSS}
html, body {{ height: 100%; margin: 0; padding: 0; }}
body {{ display: flex; flex-direction: column; padding: 6px 12px; box-sizing: border-box; }}
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
.region-toggle {{ display: inline-flex; align-items: center; gap: 4px;
                    padding-right: 8px; margin-right: 6px;
                    border-right: 1px dashed #d0a060; }}
.region-toggle .rg-btn {{ padding: 2px 8px; font-size: 12px;
                            border: 1px solid #ccc; background: #fff;
                            border-radius: 3px; cursor: pointer; font-weight: 600; }}
.region-toggle .rg-btn:hover {{ background: #f6f6f6; }}
.region-toggle .rg-btn.active {{ background: #6b46c1; color: white;
                                   border-color: #553c9a; cursor: default; }}
.age-toggle {{ display: inline-flex; align-items: center; gap: 3px;
               padding-right: 8px; margin-right: 6px;
               border-right: 1px dashed #5fa86a;
               flex-wrap: wrap; max-width: 100%; }}
.age-toggle .ag-btn, .age-toggle .ag-btn-all {{
  padding: 1px 6px; font-size: 11px;
  border: 1px solid #b0c4a0; background: #fff;
  border-radius: 3px; cursor: pointer; font-weight: 500; }}
.age-toggle .ag-btn-all {{ font-size: 10px; padding: 1px 5px;
                            background: #f3f7f0; border-color: #c0d0b0; }}
.age-toggle .ag-btn:hover {{ background: #eef5ea; }}
.age-toggle .ag-btn.active {{ background: #2e7d32; color: white;
                                border-color: #1b5e20; }}
.age-toggle .ag-btn:not(.active) {{ opacity: 0.5; text-decoration: line-through; }}

.subt-group {{ display: flex; flex-direction: column; gap: 2px;
                padding: 4px 6px; border: 1px solid #e0c79a; background: #fffaf0;
                border-radius: 4px; }}
.subt-group-head {{ display: flex; align-items: center; gap: 6px; font-size: 12px; }}
.subt-group-head .ct {{ color: #888; font-weight: 400; }}
.subt-group-head .grp-toggle {{ padding: 1px 7px; font-size: 11px;
                                 border: 1px solid #ccc; background: #f6f6f6;
                                 border-radius: 3px; cursor: pointer; }}
.subt-group-head .grp-toggle:hover {{ background: #eee; }}
.subt-group-checkboxes {{ display: flex; flex-wrap: wrap; gap: 4px;
                           padding-left: 4px; }}
.subt-chk {{ display:inline-flex; align-items:center; gap:3px; padding:1px 6px;
             border:1px solid #ddd; border-radius:3px; background:#fafafa;
             font-size:12px; cursor:pointer; user-select:none; }}
.subt-chk .ct {{ color:#888; }}
#recompute-btn, #recompute-all-btn {{ font-weight:600; background:#ff7f0e; color:white;
                  border:1px solid #cc6510; padding:4px 12px; border-radius:3px; cursor:pointer; }}
#recompute-all-btn {{ background:#d9690a; border-color:#a8500a; margin-left:6px; }}
#recompute-btn:hover {{ background:#ec6a00; }}
#recompute-all-btn:hover {{ background:#c05c08; }}
#recompute-btn:disabled, #recompute-all-btn:disabled {{ background:#aaa; border-color:#888; cursor:not-allowed; }}
#mean-slider, #std-slider {{ width: 180px; }}
.ribo-toggle {{ font-size: 12px; color: #555; display: inline-flex;
                align-items: center; gap: 4px; margin-left: 12px;
                padding: 2px 6px; border: 1px dashed #bbb; border-radius: 3px; cursor: pointer; }}
.ribo-toggle input {{ margin: 0; }}
.row {{ flex: 1 1 auto; display: flex; flex-direction: row; gap: 8px; min-height: 0; }}
.col {{ flex: 1 1 0; min-width: 0; display: flex; flex-direction: column; }}
.col > .plotly-graph-div {{ flex: 1 1 auto; min-height: 0; height: 100% !important; }}
#status, #recompute-status {{ color: #555; font-size: 12px; }}
.legend {{ flex: 0 0 auto; font-size: 11px; color: #444; margin-top: 4px; }}
button {{ font-size: 13px; padding: 4px 10px; }}
details summary {{ cursor: pointer; color: #666; font-size: 12px; }}
.go-row {{ display: flex; gap: 8px; padding: 4px 6px; background: #f9fafb;
            border: 1px solid #dadde2; border-radius: 3px; }}
.go-axis-card {{ flex: 1 1 0; min-width: 0; display: flex; flex-direction: column;
                  gap: 2px; font-size: 11px; }}
.go-axis-card > .ttl {{ font-weight: 700; color: #222; }}
.go-axis-card > .ttl .axis-stripe {{ display: inline-block; width: 10px; height: 10px;
                                       border-radius: 50%; margin-right: 4px;
                                       vertical-align: middle; }}
.go-bar {{ display: grid; grid-template-columns: 1fr auto; gap: 6px;
            align-items: center; cursor: help; padding: 1px 2px;
            border-bottom: 1px dashed transparent; }}
.go-bar:hover {{ background: #ffffff; border-bottom-color: #c0c4cc; }}
.go-bar .bar-text {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
                      font-size: 11px; color: #333; }}
.go-bar .bar-fill {{ position: relative; height: 6px; min-width: 4px;
                      background: linear-gradient(to right, #1f77b4, #1f77b4);
                      border-radius: 2px; }}
.go-bar .bar-pval {{ font-size: 10px; color: #888;
                      font-variant-numeric: tabular-nums; min-width: 50px;
                      text-align: right; }}
.go-bar.empty {{ color: #999; font-style: italic; }}
.go-bars-label {{ font-size: 11px; color: #555; font-weight: 600;
                   align-self: center; margin-right: 4px; }}
{base.LAYOUT_CSS}
{base.VIZ_NAV_CSS}
{base.BUTTON_CSS}
{base.COLOR_KEY_CSS}
{base.HOME_LINK_CSS}
</style>
</head>
<body>
{base.HOME_LINK_HTML}
<div class="wrap">
<h2>{base.cohort_title(GROUP_NAME)}</h2>
<div class="hint">
<b>Pick which subtypes you want to fit the embedding to</b>, then click <b>Replot</b>.
</div>

<div class="ctrl-box">
  <div class="ctrl-box-title">Plotting Method</div>
  <div class="controls-row">{base.viz_nav_html(SLUG, 'diffmap')}</div>
</div>

<div class="ctrl-box">
  <div class="ctrl-box-title">Color Scheme</div>
  <div class="controls-row">
    <button id="qc-counts" class="qc-btn">Counts</button>
    <button id="qc-genes" class="qc-btn">Genes</button>
    <button id="qc-ribo" class="qc-btn">% ribo</button>
    <button id="qc-region" class="qc-btn" title="Colour each cell by its dissected region (V1 = orange, ALM = purple). Greyed out for single-region cohorts.">Region</button>
    <button id="qc-age" class="qc-btn" title="Colour each cell by its developmental stage (E11.5 → P56). Greyed out for cohorts without age info.">Age</button>
    <button id="qc-layer" class="qc-btn" title="Colour each cell by its dissected cortical layer (L1 → L6b), depth-encoded on viridis. Cells from multi-layer or pan (e.g. L1-L6) dissections are not layer-specific and shown faint grey. Hidden for cohorts without layer info.">Layer of Microdissection</button>
    <button id="reset-btn">Reset colours (subtype)</button>
  </div>
</div>

<div class="ctrl-box">
  <div class="ctrl-box-title">Filter Which Cells are Shown</div>
  <div class="controls-row">
    {region_toggle_html}
    {age_toggle_html}
    <span class="label">Subtypes:</span>
    {subtype_checkbox_html}
    <button id="subt-all" title="Check every subtype">all</button>
    <button id="subt-none" title="Uncheck every subtype">none</button>
    <span class="lin-sep">|</span>
    <button class="lin-btn" data-lin="MGE" title="Select MGE-derived subclasses: Pvalb (incl chandelier) + Sst (incl Chodl)">+MGE</button>
    <button class="lin-btn" data-lin="CGE" title="Select CGE-derived subclasses: Vip + Lamp5 + Sncg + Serpinf1 (+ Lamp5 Lhx6)">+CGE</button>
    <button class="lin-btn" data-lin="LGE" title="Select LGE-derived subclasses (rare in cortex; mostly striatal)">+LGE</button>
    <button id="recompute-btn" title="Refit the diffusion embedding on the panel HVG, using only the checked-subtype cells.">Replot with gene panel</button>
    <button id="recompute-all-btn" title="Refit the diffusion embedding on every gene currently shown in the gene filter (the active gene set ∩ mean/std/ribo sliders), using only the checked-subtype cells. Slower than the panel fit.">Replot (shown genes)</button>
    <span id="recompute-status" style="margin-left:8px;"></span>
  </div>
</div>

<div class="ctrl-box">
  <div class="ctrl-box-title">Filter which Genes are Shown</div>
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
    <label class="ribo-toggle" title="Hide genes whose log-CPM correlates with %ribosomal above this threshold. 1.00 = no filtering (default). Lower values strip out more 'metabolic' / housekeeping genes that track per-cell ribosomal content rather than cell type.">
      Metabolism filter <span id="ribo-corr-label">|r|≤1.00</span>
      <input type="range" id="ribo-corr-slider" min="0.10" max="1.00" step="0.05" value="1.00" style="vertical-align:middle; width:120px;">
      <span id="ribo-corr-count" style="color:#888;"></span></label>
    <label class="ribo-toggle" title="Subtract each gene's linear fit on pct_ribo before the recompute. Equivalent to projecting expression onto the subspace orthogonal to %ribo, so SVD/UMAP/diffmap see only the residual (non-metabolic) variance. NMF clips negative residuals to 0.">
      <input type="checkbox" id="regress-ribo"> Regress out %ribo
    </label>
    <span class="label" style="margin-left:14px;">Visible:</span>
    <span id="visible-count">{n_panel_disp} / {n_genes}</span>
  </div>
</div>

<div class="controls-row" style="justify-content:center;"><span id="status">Hover a cell (left) or a gene (right) to colour by expression.</span></div>

<div class="plot-pair">
  <div class="plot-box">
    <div class="plot-box-title" id="cell-plot-title"></div>
    {cells_html}
  </div>
  <div class="plot-box">
    <div class="plot-box-title" id="gene-plot-title"></div>
    {genes_html}
  </div>
</div>

<span id="pole-legend" style="display:none">{pole_legend}</span>
<footer class="cite">{base.cohort_citation(GROUP_NAME)}</footer>
</div>
<script>
{js_data}
{base.COLOR_KEY_JS}

const cellPlot = document.getElementById('cell-plot');
const genePlot = document.getElementById('gene-plot');
const status   = document.getElementById('status');
const recomputeStatus = document.getElementById('recompute-status');
const POINTS_TRACE = 2, VERTEX_TRACE = 1, LOADING_TRACE = 3, HIGHLIGHT_TRACE = 4;
const DEFAULT_LOAD_COLORS = ['#e0e0e0','#e0e0e0','#e0e0e0','#e0e0e0','#e0e0e0','#e0e0e0'];
const KNN_K = 15;
let lastHoveredCell = null, lastHoveredGene = null;

// ---- dynamic plot-box titles --------------------------------------------
const VIZ_METHOD = 'Diffusion map';
let titleCellColor = 'subtype';
let titleGeneRef   = 'strongest DC';
let titleGeneN     = {n_panel_disp};
function activeCellCount() {{
  let n = 0; for (let i = 0; i < cell_active.length; i++) if (cell_active[i]) n++; return n;
}}
function refreshTitles() {{
  const ct = document.getElementById('cell-plot-title');
  const gt = document.getElementById('gene-plot-title');
  if (ct) ct.innerHTML = 'Cells plotted on <b>' + VIZ_METHOD + '</b> axes · coloured by <b>'
    + titleCellColor + '</b> · n=' + activeCellCount().toLocaleString();
  if (gt) gt.innerHTML = 'Genes plotted on <b>' + VIZ_METHOD + '</b> axes · coloured by <b>'
    + titleGeneRef + '</b> · n=' + titleGeneN.toLocaleString();
}}

function exprToMagma(values) {{
  let lo = Infinity, hi = -Infinity;
  for (const v of values) {{ if (v < lo) lo = v; if (v > hi) hi = v; }}
  const range = (hi > lo) ? (hi - lo) : 1;
  return Array.from(values, v => magma[Math.max(0, Math.min(255, Math.round(255*(v-lo)/range)))]);
}}
function loadingToMagma(loadings) {{
  return loadings.map(v => magma[Math.round(255 * Math.max(0, Math.min(1, v)))]);
}}

cellPlot.on('plotly_hover', function(data) {{
  const pt = data.points[0]; if (pt.curveNumber !== POINTS_TRACE) return;
  const i = pt.pointNumber; if (lastHoveredCell === i) return; lastHoveredCell = i;
  if (!cell_active[i]) return;
  const row = expr_matrix.subarray(i * N_GENES, (i + 1) * N_GENES);
  Plotly.restyle(genePlot, {{'marker.color': [exprToMagma(row)]}}, [POINTS_TRACE]);
  const lc = loadingToMagma(cell_load[i]);
  Plotly.restyle(cellPlot, {{'marker.color': [lc]}}, [LOADING_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color': [lc]}}, [LOADING_TRACE]);
  let lo = Infinity, hi = -Infinity; for (const v of row) {{ if (v<lo) lo=v; if (v>hi) hi=v; }}
  const d = cell_dc[i];
  status.innerHTML = '<b style="color:' + cell_dom_color[i] + '">Cell #' + i
    + '</b> <span style="color:#555">(' + cell_subtype[i] + (typeof cell_age !== 'undefined' && cell_age ? ' · age ' + cell_age[i] : '') + ')</span> &nbsp; '
    + 'DC1,2,3 = (' + d[0].toFixed(3) + ', ' + d[1].toFixed(3) + ', ' + d[2].toFixed(3) + ') &nbsp; '
    + 'genes recoloured by expression (range ' + (lo/EXPR_SCALE).toFixed(2) + '..' + (hi/EXPR_SCALE).toFixed(2) + ', magma)';
  titleGeneRef = 'expression in cell #' + i + ' (' + cell_subtype[i] + ')'; refreshTitles();
}});

genePlot.on('plotly_hover', function(data) {{
  const pt = data.points[0]; if (pt.curveNumber !== POINTS_TRACE) return;
  const j = pt.pointNumber; if (lastHoveredGene === j) return; lastHoveredGene = j;
  const n = N_CELLS; const col = new Array(n);
  for (let i = 0; i < n; i++) col[i] = cell_active[i] ? expr_matrix[i * N_GENES + j] : null;
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
  setColorKeyGradient(gene_name[j] + ' expression', 'magma', lo/EXPR_SCALE, hi/EXPR_SCALE, v => v.toFixed(2));
  titleCellColor = gene_name[j] + ' expression'; refreshTitles();
}});

document.getElementById('reset-btn').addEventListener('click', function() {{
  const cells = cell_default_colors.map((c, i) => cell_active[i] ? c : '#dddddd');
  Plotly.restyle(cellPlot, {{'marker.color': [cells]}}, [POINTS_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color': [gene_default_colors]}}, [POINTS_TRACE]);
  Plotly.restyle(cellPlot, {{'marker.color': [DEFAULT_LOAD_COLORS]}}, [LOADING_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color': [DEFAULT_LOAD_COLORS]}}, [LOADING_TRACE]);
  lastHoveredCell = null; lastHoveredGene = null;
  status.innerHTML = 'Reset. Hover a cell or gene to colour by expression.'; clearColorKey();
  titleCellColor = 'subtype'; titleGeneRef = 'strongest DC'; refreshTitles();
}});

let activeSet = 'panel';
const meanSlider = document.getElementById('mean-slider'), stdSlider = document.getElementById('std-slider');
const meanValueEl = document.getElementById('mean-value'), stdValueEl = document.getElementById('std-value');
const visibleCount = document.getElementById('visible-count');
const riboSlider    = document.getElementById('ribo-corr-slider');
const riboCorrCount = document.getElementById('ribo-corr-count');
const riboCorrLabel = document.getElementById('ribo-corr-label');
function riboThreshold() {{
  return riboSlider ? parseFloat(riboSlider.value) : 1.0;
}}
function isRiboCorr(j) {{
  // True when this gene's |corr(%ribo)| exceeds the current slider threshold.
  return Math.abs(gene_ribo_corr[j]) > riboThreshold();
}}
const regressRiboToggle = document.getElementById('regress-ribo');
function readVal(i, j) {{
  // Per-cell, per-gene expression (log-CPM). If "Regress out %ribo" is on,
  // subtracts gene j's linear fit on pct_ribo so the recompute sees only the
  // component orthogonal to the %ribo axis.
  let v = expr_matrix[i * N_GENES + j] / EXPR_SCALE;
  if (regressRiboToggle && regressRiboToggle.checked) {{
    v -= gene_ribo_slope[j] * (qc_ribo[i] - mean_pct_ribo);
  }}
  return v;
}}
function readValNN(i, j) {{ return Math.max(0, readVal(i, j)); }}
function refreshRiboCount() {{
  if (!riboSlider) return;
  const t = riboThreshold();
  let n = 0;
  for (let j = 0; j < gene_name.length; j++) if (Math.abs(gene_ribo_corr[j]) > t) n++;
  if (riboCorrCount) riboCorrCount.textContent = '(' + n + ' hidden)';
  if (riboCorrLabel) riboCorrLabel.textContent = '|r|≤' + t.toFixed(2);
}}
function applyGeneFilter() {{
  const meanThr = parseFloat(meanSlider.value), stdThr = parseFloat(stdSlider.value);
  const hideRibo = !!riboSlider && riboThreshold() < 1.0;
  const mask = gene_sets[activeSet], n = gene_name.length;
  const xs = new Array(n), ys = new Array(n), zs = new Array(n); let visible = 0;
  for (let j = 0; j < n; j++) {{
    if (mask[j] && gene_mean[j] >= meanThr && gene_std[j] >= stdThr
        && !(hideRibo && isRiboCorr(j))) {{
      xs[j]=gene_x[j]; ys[j]=gene_y[j]; zs[j]=gene_z[j]; visible++;
    }} else {{ xs[j]=null; ys[j]=null; zs[j]=null; }}
  }}
  Plotly.restyle(genePlot, {{x:[xs], y:[ys], z:[zs]}}, [POINTS_TRACE]);
  meanValueEl.textContent = meanThr.toFixed(2); stdValueEl.textContent = stdThr.toFixed(2);
  visibleCount.textContent = visible + ' / ' + n;
  titleGeneN = visible; refreshTitles();
}}
meanSlider.addEventListener('input', applyGeneFilter);
stdSlider.addEventListener('input', applyGeneFilter);
if (riboSlider) {{
  refreshRiboCount();
  riboSlider.addEventListener('input', () => {{ refreshRiboCount(); applyGeneFilter(); }});
}}
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
  const n = N_CELLS; const col = new Array(n);
  for (let i = 0; i < n; i++) col[i] = cell_active[i] ? expr_matrix[i * N_GENES + j] : null;
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
                   && gene_std[j] >= parseFloat(stdSlider.value)
                   && !(riboSlider && riboThreshold() < 1.0 && isRiboCorr(j)));
  status.innerHTML = '<b style="color:' + gene_dom_color[j] + '">' + gene_name[j] + '</b> '
    + (gene_in_panel[j] ? '(panel)' : '(broader)')
    + (hidden ? ' <span style="color:#c00">[hidden by current filter — ring still shows its position]</span>' : '')
    + ' — cells recoloured by its expression (magma).';
  titleCellColor = gene_name[j] + ' expression'; refreshTitles();
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
  status.innerHTML = '';
  setColorKeyGradient(label, 'viridis', lo, hi, fmt);
  titleCellColor = label; refreshTitles();
}}
const fmtInt = v => Math.round(v).toLocaleString();
const fmtPct = v => v.toFixed(1) + '%';
document.getElementById('qc-counts').addEventListener('click', () => colorByQC(qc_total, 'total counts', fmtInt));
document.getElementById('qc-genes').addEventListener('click', () => colorByQC(qc_ngenes, 'genes detected', fmtInt));
document.getElementById('qc-ribo').addEventListener('click', () => colorByQC(qc_ribo, '% ribosomal', fmtPct));

// Colour by region (V1 vs ALM). The Tasic and merged V1+ALM cohorts carry
// `dissected_region` per cell; we paint VISp (V1) orange and ALM purple,
// and fall back to grey for cells without a region label or for cells
// excluded by the subtype filter.
const REGION_COLORS = {{ 'VISp': '#ff7f0e', 'V1': '#ff7f0e', 'ALM': '#9467bd' }};
const regionBtn = document.getElementById('qc-region');
if (regionBtn) {{
  const _regions = (typeof cell_region !== 'undefined' && cell_region) ? cell_region : [];
  const uniqRegions = Array.from(new Set(_regions));
  if (uniqRegions.length < 2) {{
    regionBtn.remove();
  }} else {{
    regionBtn.addEventListener('click', () => {{
      const colors = cell_region.map((r, i) =>
        cell_active[i] ? (REGION_COLORS[r] || '#888888') : '#dddddd');
      Plotly.restyle(cellPlot, {{'marker.color': [colors]}}, [POINTS_TRACE]);
      const swatch = r => `<span style="display:inline-block;width:9px;height:9px;`
        + `background:${{REGION_COLORS[r] || '#888'}};margin:0 3px 0 8px;`
        + `border-radius:50%;vertical-align:middle;"></span>${{r}}`;
      status.innerHTML = '';
      setColorKeyCats('region', uniqRegions.sort().map(r => ({{color: REGION_COLORS[r] || '#888888', label: r}})));
    }});
  }}
}}

// Colour by developmental age. Greyed out unless `cell_age` is defined.
function ageToNumber(s) {{
  if (s == null) return NaN;
  s = String(s).trim();
  if (s[0] === 'E' || s[0] === 'e') return -(parseFloat(s.slice(1)) || 0);
  if (s[0] === 'P' || s[0] === 'p') return  (parseFloat(s.slice(1)) || 0);
  return parseFloat(s);
}}
const ageBtn = document.getElementById('qc-age');
if (ageBtn) {{
  const _ages = (typeof cell_age !== 'undefined' && cell_age) ? cell_age : [];
  const uniqAges = Array.from(new Set(_ages));
  if (uniqAges.length < 2) {{
    ageBtn.remove();   // no age info or single-age cohort → hide entirely
  }} else {{
    const ageNum = cell_age.map(ageToNumber);
    ageBtn.addEventListener('click', () => {{
      colorByQC(ageNum, 'age (E neg / P pos days)', v => v < 0 ? 'E' + (-v) : 'P' + v);
    }});
  }}
}}

// Colour by dissected cortical layer (Tasic V1/ALM). Each layer string
// (L1, L2/3, L4, L5, L6, L6b; compound like L2/3-L4) maps to a viridis-depth
// scalar (L1=1 → L6b=6.5; midpoint for compound). Hidden when None / single.
function layerToDepth(s) {{
  if (s == null) return NaN;
  s = String(s).trim().toUpperCase();
  function lone(tok) {{
    tok = tok.trim();
    if (tok === 'L1')   return 1.0;
    if (tok === 'L2/3') return 2.5;
    if (tok === 'L4')   return 4.0;
    if (tok === 'L5')   return 5.0;
    if (tok === 'L6')   return 6.0;
    if (tok === 'L6B')  return 6.5;
    return NaN;
  }}
  if (s.indexOf('-') >= 0) return NaN;   // multi-layer / pan dissection: not a specific microdissected layer
  return lone(s);
}}
function layerLabel(d) {{
  if (d <= 1.25) return 'L1';
  if (d <= 2.9)  return 'L2/3';
  if (d <= 4.5)  return 'L4';
  if (d <= 5.5)  return 'L5';
  if (d <= 6.25) return 'L6';
  return 'L6b';
}}
// Layer of Microdissection: always visible. Cells without dissection info
// render in low-alpha grey instead of being hidden or miscoloured.
const layerBtn = document.getElementById('qc-layer');
const GREY_NO_LAYER = 'rgba(170,170,170,0.12)';
if (layerBtn) {{
  const _layers = (typeof cell_layer !== 'undefined' && cell_layer) ? cell_layer : null;
  const hasAnyLayer = _layers && new Set(_layers.filter(v => v != null)).size >= 1;
  layerBtn.addEventListener('click', () => {{
    if (!hasAnyLayer) {{
      const colors = cell_active.map(a => a ? GREY_NO_LAYER : '#dddddd');
      Plotly.restyle(cellPlot, {{'marker.color': [colors]}}, [POINTS_TRACE]);
      status.innerHTML = '<i>This cohort has no microdissection layer info — all cells greyed.</i>';
      return;
    }}
    const depths = _layers.map(layerToDepth);
    const validVals = depths.filter((d, i) => cell_active[i] && !isNaN(d));
    const palette = valuesToViridis(validVals);
    let pi = 0;
    const colors = depths.map((d, i) => {{
      if (!cell_active[i]) return '#dddddd';
      if (!isNaN(d)) return palette[pi++];
      return GREY_NO_LAYER;
    }});
    Plotly.restyle(cellPlot, {{'marker.color': [colors]}}, [POINTS_TRACE]);
    let lo = Infinity, hi = -Infinity;
    for (const v of validVals) {{ if (v < lo) lo = v; if (v > hi) hi = v; }}
    const greyN = depths.filter((d, i) => cell_active[i] && isNaN(d)).length;
    status.innerHTML = '';
    if (validVals.length) setColorKeyGradient('layer (microdissection)', 'viridis', lo, hi, layerLabel); else clearColorKey();
  }});
}}

// ============================================================================
// Subtype-subset diffmap recompute (Laplacian eigenmap)
// ============================================================================
const subtypeCheckboxes = Array.from(document.querySelectorAll('.subt-chk input[type="checkbox"]'));
document.querySelectorAll('.grp-toggle').forEach(btn => {{
  btn.addEventListener('click', () => {{
    const grp = btn.dataset.grp;
    const want = btn.dataset.action === 'all';
    subtypeCheckboxes.forEach(cb => {{
      if (cb.dataset.grp === grp) cb.checked = want;
    }});
  }});
}});
document.getElementById('subt-all').addEventListener('click', () => {{
  subtypeCheckboxes.forEach(cb => cb.checked = true);
}});
document.getElementById('subt-none').addEventListener('click', () => {{
  subtypeCheckboxes.forEach(cb => cb.checked = false);
}});
const LINEAGE_SUBCLASSES = {{
  MGE: new Set(['Pvalb', 'Pvalb chandelier', 'Sst', 'Sst Chodl']),
  CGE: new Set(['Vip', 'Lamp5', 'Lamp5 Lhx6', 'Sncg', 'Serpinf1']),
  LGE: new Set(['LGE']),
}};
document.querySelectorAll('.lin-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    const wanted = LINEAGE_SUBCLASSES[btn.dataset.lin] || new Set();
    subtypeCheckboxes.forEach(cb => {{
      if (wanted.has(cb.dataset.grp)) cb.checked = true;
    }});
  }});
}});
function selectedSubtypes() {{
  const out = new Set();
  subtypeCheckboxes.forEach(cb => {{ if (cb.checked) out.add(cb.dataset.sub); }});
  return out;
}}

let activeRegion = 'both';
document.querySelectorAll('.rg-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    activeRegion = btn.dataset.region;
    document.querySelectorAll('.rg-btn').forEach(b =>
      b.classList.toggle('active', b === btn));
  }});
}});
function regionAllowed(i) {{
  if (activeRegion === 'both' || !cell_region) return true;
  return cell_region[i] === activeRegion;
}}

const activeAges = new Set(
  (typeof cell_age !== 'undefined' && cell_age) ? Array.from(new Set(cell_age)) : []
);
document.querySelectorAll('.ag-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    const a = btn.dataset.age;
    if (activeAges.has(a)) {{ activeAges.delete(a); btn.classList.remove('active'); }}
    else                   {{ activeAges.add(a);    btn.classList.add('active'); }}
  }});
}});
document.querySelectorAll('.ag-btn-all').forEach(btn => {{
  btn.addEventListener('click', () => {{
    const want = btn.dataset.act === 'all';
    document.querySelectorAll('.ag-btn').forEach(b => {{
      b.classList.toggle('active', want);
      if (want) activeAges.add(b.dataset.age); else activeAges.delete(b.dataset.age);
    }});
  }});
}});
function ageAllowed(i) {{
  if (typeof cell_age === 'undefined' || !cell_age) return true;
  return activeAges.has(cell_age[i]);
}}

// The gene indices currently SHOWN in the gene plot (active set ∩ mean/std/ribo
// sliders) — the same predicate applyGeneFilter() uses for visibility. Used by
// the "Replot (shown genes)" button so the embedding is fit on whatever genes
// the user is currently looking at, not just the panel HVG.
function shownGeneIdx() {{
  const meanThr = parseFloat(meanSlider.value), stdThr = parseFloat(stdSlider.value);
  const hideRibo = !!riboSlider && riboThreshold() < 1.0;
  const mask = gene_sets[activeSet];
  const out = [];
  for (let j = 0; j < gene_name.length; j++) {{
    if (mask[j] && gene_mean[j] >= meanThr && gene_std[j] >= stdThr
        && !(hideRibo && isRiboCorr(j))) out.push(j);
  }}
  return out;
}}

// Build z-scored (subset rows, selected gene columns) matrix as Array of
// Float64Array rows. geneIdx defaults to the panel HVG.
function buildSubsetZ(cellSel, geneIdx) {{
  const m = cellSel.length, n = geneIdx.length;
  // Per-column mean/std on the subset
  const mean = new Float64Array(n), std = new Float64Array(n);
  for (let k = 0; k < n; k++) {{
    const j = geneIdx[k];
    let s = 0, ss = 0;
    for (let ii = 0; ii < m; ii++) {{
      const v = readVal(cellSel[ii], j);
      s += v; ss += v*v;
    }}
    mean[k] = s / m;
    std[k]  = Math.sqrt(Math.max(ss / m - mean[k]*mean[k], 1e-18));
  }}
  const Z = new Array(m);
  for (let ii = 0; ii < m; ii++) {{
    const Zi = new Float64Array(n);
    for (let k = 0; k < n; k++) {{
      const v = readVal(cellSel[ii], geneIdx[k]);
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

function recomputeDiffmap(useShownGenes) {{
  const t0 = performance.now();
  const sel = selectedSubtypes();
  const cellSel = [];
  for (let i = 0; i < cell_subtype.length; i++) {{
    if (sel.has(cell_subtype[i]) && regionAllowed(i) && ageAllowed(i)) cellSel.push(i);
  }}
  const m = cellSel.length;
  if (m < KNN_K + 1) {{
    recomputeStatus.innerHTML = '<span style="color:#c00">need ≥' + (KNN_K + 1)
      + ' cells (got ' + m + ')</span>';
    return;
  }}

  // Genes to fit on: the panel HVG (default) or every gene currently shown.
  const geneIdx = useShownGenes ? shownGeneIdx() : panel_idx;
  if (geneIdx.length < 2) {{
    recomputeStatus.innerHTML = '<span style="color:#c00">need ≥2 shown genes (got '
      + geneIdx.length + ') — widen the gene filter</span>';
    return;
  }}

  recomputeStatus.textContent = 'building kNN graph on ' + geneIdx.length + ' genes…';
  const Z = buildSubsetZ(cellSel, geneIdx);
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
      const w = readValNN(cellSel[ii], j);
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
  const newPoleLab = POLE_NAMES_.slice();
  Plotly.restyle(cellPlot, {{text:[newPoleLab], hovertext:[newPoleLab]}}, [VERTEX_TRACE]);
  Plotly.restyle(genePlot, {{text:[newPoleLab], hovertext:[newPoleLab]}}, [VERTEX_TRACE]);
  Plotly.restyle(cellPlot, {{'marker.color':[DEFAULT_LOAD_COLORS]}}, [LOADING_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color':[DEFAULT_LOAD_COLORS]}}, [LOADING_TRACE]);
  Plotly.restyle(genePlot, {{x:[[null]], y:[[null]], z:[[null]]}}, [HIGHLIGHT_TRACE]);
  document.getElementById('pole-legend').innerHTML = newPoleTop.map((g, p) =>
    `<span style="display:inline-block;width:10px;height:10px;background:${{POLE_COLORS_[p]}};` +
    `margin-right:4px;border-radius:50%;"></span> ${{POLE_NAMES_[p]}} (${{g}}) &nbsp;&nbsp;`).join('');

  const dt = ((performance.now() - t0) / 1000).toFixed(2);
  recomputeStatus.innerHTML = '<b>recomputed</b> on ' + m + ' / ' + n_cells_total + ' cells × '
    + geneIdx.length + (useShownGenes ? ' shown' : ' panel') + ' genes (' + dt + 's) — '
    + 'poles: ' + newPoleTop.map((g, p) => POLE_NAMES_[p] + '=' + g).join(', ');
  lastHoveredCell = null; lastHoveredGene = null;
  // GO bars — rank panel-HVG centroids on each DC axis, ±30 per pole, refetch
  const goAxes = [];
  for (let k = 0; k < 3; k++) {{
    const indexed = [];
    for (let pi = 0; pi < panel_idx.length; pi++) {{
      const j = panel_idx[pi];
      indexed.push([newGeneCentroid[j][k], j]);
    }}
    indexed.sort((a, b) => a[0] - b[0]);
    const N = Math.min(30, indexed.length);
    const neg = indexed.slice(0, N).map(p => gene_name[p[1]]);
    const pos = indexed.slice(-N).reverse().map(p => gene_name[p[1]]);
    goAxes.push({{name: 'DC' + (k+1), genes: pos.concat(neg)}});
  }}
}}

function wireRecompute(btnId, useShownGenes) {{
  document.getElementById(btnId).addEventListener('click', () => {{
    const btns = [document.getElementById('recompute-btn'),
                  document.getElementById('recompute-all-btn')].filter(Boolean);
    btns.forEach(b => b.disabled = true);
    recomputeStatus.textContent = 'computing…';
    setTimeout(() => {{
      try {{ recomputeDiffmap(useShownGenes); }}
      finally {{ btns.forEach(b => b.disabled = false); }}
    }}, 30);
  }});
}}
wireRecompute('recompute-btn', false);
wireRecompute('recompute-all-btn', true);

// --- Per-axis GO enrichment bars ---------------------------------------------
const AXIS_STRIPE_COLORS = ['#d62728', '#1f77b4', '#2ca02c'];
function shortGo(term) {{
  return term.replace(/ \\(GO:\\d+\\)\\s*$/, '');
}}
function fmtPadj(p) {{
  if (p === 0) return '0';
  if (p < 1e-3) return p.toExponential(1);
  return p.toFixed(3);
}}
function resizePlots() {{ Plotly.Plots.resize(cellPlot); Plotly.Plots.resize(genePlot); }}
window.addEventListener('resize', resizePlots);
refreshTitles();
setTimeout(function() {{ resizePlots(); applyGeneFilter(); refreshTitles(); }}, 50);

const AUTO_RECOMPUTE_DEFAULT = {str(auto_recompute_on_load).lower()};
if (AUTO_RECOMPUTE_DEFAULT && !location.hash.includes('s=')) {{
  // Signal the copy-link overlay so users don't see the full-cohort flash
  // before the default-subset recompute kicks in.
  window.__autoRecomputeDefault = true;
  setTimeout(() => {{
    const btn = document.getElementById('recompute-btn');
    if (btn && !btn.disabled) btn.click();
  }}, 80);
}}
</script>
</body>
</html>"""

    if base.LOG_SCALE_X:
        page = page.replace('</body>', base.LOG_X_RIBO_DISABLER + '</body>')
    with open(OUT, 'w') as f: f.write(page)
    print(f'  done. {os.path.getsize(OUT)/1e6:.1f} MB self-contained HTML.')
    print(f'  open: file://{OUT}')


if __name__ == '__main__':
    main()
