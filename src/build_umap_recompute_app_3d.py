#!/usr/bin/env python
"""3D UMAP explorer with in-browser UMAP recompute on a selected subtype subset.

Same biplot UI as build_svd_recompute_app_3d.py but with UMAP as the embedding.
Cells live in a 3D cube spanned by UMAP1/2/3; genes are placed at the
expression-weighted centroid of cell UMAP coords. Two recompute buttons:
  - "Recompute UMAP on selected": refit UMAP using only the checked-subtype cells.
  - "Recompute on shown genes": refit UMAP using only currently-visible genes.

Server-side: initial 3D UMAP via scanpy (PCA → 15-NN → UMAP).
Client-side: umap-js library (CDN) for the recompute. Takes 3–10 s per refit.

The variance-explained bar row is hidden — UMAP has no analytic variance ratios.

Usage:  python build_umap_recompute_app_3d.py [GROUP]   # default: base GROUP_NAME
Output: notebooks/{group}_umap_recompute_explorer_3d.html
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
                           f'{SLUG}_umap_recompute_explorer_3d.html')
NPC         = 3
POLE_COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf']
POLE_NAMES  = ['UMAP1+', 'UMAP1-', 'UMAP2+', 'UMAP2-', 'UMAP3+', 'UMAP3-']

def prep_cols(M):
    """Per-gene z-score before UMAP."""
    M = np.asarray(M, dtype=np.float64)
    return (M - M.mean(0)) / (M.std(0) + 1e-9)


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

    # ---- initial UMAP: scanpy via PCA → 15-NN → UMAP n_components=3 ------------
    # Variance-explained bars don't apply to UMAP; we keep BARS_K = 0 so the
    # bar row stays empty (it's hidden via CSS too).
    BARS_K = 0
    Xp = X_keep[:, in_panel]
    Zp = prep_cols(Xp)
    adp = ad_mod.AnnData(Zp.astype(np.float32))
    sc.pp.pca(adp, n_comps=min(30, n_cells - 1), random_state=0)
    sc.pp.neighbors(adp, n_neighbors=15, random_state=0)
    sc.tl.umap(adp, n_components=NPC, random_state=0, min_dist=0.3)
    UC = np.asarray(adp.obsm['X_umap'], dtype=np.float64)
    UC = UC - UC.mean(0)                        # centre
    cell_scores = UC                            # 'cell_score' in JS — UMAP coords
    print(f'  initial UMAP: n_cells={n_cells}, panel HVG={int(in_panel.sum())}, '
          f'NPC={NPC}, min_dist=0.3, random_state=0')

    # Genes = expression-weighted centroid of cell UMAP coords (over the FULL
    # cohort; the JS recompute updates this on subsets).
    Wn = X_keep.copy()
    Wn[Wn < 0] = 0
    Wn = Wn / (Wn.sum(0, keepdims=True) + 1e-9)
    gene_load3 = Wn.T @ UC                      # 'gene_loading' in JS

    def fill_cube(M):
        m = np.max(np.abs(M), axis=0) + 1e-9
        return M / m
    cell_xyz = fill_cube(cell_scores)
    gene_xyz = fill_cube(gene_load3)

    # Pole labels: most extreme panel-gene centroid on each ±UMAP axis (no V matrix)
    panel_idx_full = np.where(in_panel)[0]
    panel_genes_list = [gene_names[i] for i in panel_idx_full]
    pole_top = []
    for k in range(NPC):
        col = gene_xyz[panel_idx_full, k]
        pole_top.append(panel_genes_list[int(np.argmax(col))])
        pole_top.append(panel_genes_list[int(np.argmin(col))])

    # Per-axis top genes for GO: rank panel genes by their centroid coord on
    # each UMAP axis. ±30 most extreme on each pole feed Enrichr.
    GO_TOP_PER_POLE = 30
    top_genes_per_axis = []
    for k in range(NPC):
        col = gene_xyz[panel_idx_full, k]
        order = np.argsort(col)
        neg = [panel_genes_list[i] for i in order[:GO_TOP_PER_POLE]]
        pos = [panel_genes_list[i] for i in order[-GO_TOP_PER_POLE:]][::-1]
        top_genes_per_axis.append({'name': f'UMAP{k+1}', 'pos': pos, 'neg': neg})
    go_axes = []  # GO bars removed from UI; placeholder for JS data
    cats = sorted(set(subs.tolist()))
    subtype_palette = base.build_subtype_palette(cats)
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

    import base64
    EXPR_SCALE = 16
    _expr_q = np.clip(np.round(X_keep * EXPR_SCALE), 0, 255).astype(np.uint8)
    expr_b64 = base64.b64encode(_expr_q.tobytes()).decode('ascii')
    n_cells_emit = int(_expr_q.shape[0])
    n_genes_emit = int(_expr_q.shape[1])
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
        f'UMAP1,2,3 = ({cell_scores[i,0]:.2f}, {cell_scores[i,1]:.2f}, {cell_scores[i,2]:.2f})'
        for i in range(n_cells)]
    gene_hover_text = [
        f'<b>{gene_names[j]}</b>'
        + (' (panel HVG)' if in_panel[j] else ' (projected)')
        + f'<br>strongest: {POLE_NAMES[gene_pole[j]]} ({pole_top[gene_pole[j]]})<br>'
        + f'mean={mean_expr[j]:.2f}, std={std_expr[j]:.2f}<br>'
        + f'loadings UMAP1,2,3 = ({gene_load3[j,0]:.2f}, {gene_load3[j,1]:.2f}, {gene_load3[j,2]:.2f})'
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


    vr = ['n/a', 'n/a', 'n/a']   # UMAP has no analytic variance ratios
    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{GROUP_NAME} UMAP recompute explorer</title>
<!-- umap-js: client-side UMAP for the in-browser recompute. ~150 KB minified. -->
<script src="https://cdn.jsdelivr.net/npm/umap-js@1.4.0/lib/umap-js.min.js"></script>
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
.genestruct-box {{ display: flex; align-items: center; flex-wrap: wrap; gap: 6px;
                   border: 1px solid #cdd6e0; background: #f4f8fc; border-radius: 6px;
                   padding: 6px 10px; margin: 4px 0; font-size: 12px; }}
.genestruct-box .gs-title {{ font-weight: 700; color: #1f5d8c; margin-right: 6px; }}
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
.ribo-toggle {{ font-size: 12px; color: #555; display: inline-flex;
                align-items: center; gap: 4px; margin-left: 12px;
                padding: 2px 6px; border: 1px dashed #bbb; border-radius: 3px; cursor: pointer; }}
.ribo-toggle input {{ margin: 0; }}
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
                    margin-bottom: 2px; display: flex; justify-content: space-between;
                    flex-wrap: wrap; gap: 4px; }}
.heatmap-controls {{ display: inline-flex; align-items: center; gap: 4px;
                      font-size: 11px; color: #555; }}
.heatmap-controls .label {{ font-weight: 600; }}
.heatmap-controls .order-btn {{ padding: 1px 7px; font-size: 11px;
                                 border: 1px solid #bbb; background: #f6f6f6;
                                 border-radius: 3px; cursor: pointer; }}
.heatmap-controls .order-btn:hover {{ background: #eee; }}
.heatmap-controls .order-btn.active {{ background: #1f77b4; color: white;
                                        border-color: #1f77b4; cursor: default; }}
.group-by-toggle {{ display: inline-flex; align-items: center; gap: 3px;
                     padding-left: 6px; cursor: pointer; }}
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
.celltype-strip-wrap {{ flex: 0 0 14px; position: relative; margin-top: 2px;
                        border-top: 1px solid #c0c0c0; background: #fff; }}
#celltype-strip-canvas {{ position: absolute; left: 0; top: 0;
                          width: 100%; height: 100%; image-rendering: pixelated; }}
.celltype-order-row {{ flex: 0 0 auto; display: flex; align-items: center; gap: 6px;
                       flex-wrap: wrap; padding: 4px 2px 0; font-size: 11px; color: #555; }}
.celltype-order-chips {{ display: inline-flex; gap: 4px; flex-wrap: wrap; }}
.ct-chip {{ display: inline-flex; align-items: center; gap: 4px; padding: 2px 6px;
            background: #fff; border: 1px solid #bbb; border-radius: 10px; cursor: grab;
            font-size: 11px; user-select: none; }}
.ct-chip.dragging {{ opacity: 0.4; cursor: grabbing; }}
.ct-chip.drop-target {{ outline: 2px dashed #1f77b4; outline-offset: 1px; }}
.ct-chip-dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; }}
.celltype-order-reset {{ font-size: 11px; padding: 1px 6px;
                         background: #f6f6f6; border: 1px solid #bbb; border-radius: 3px; cursor: pointer; }}
#status, #recompute-status {{ color: #555; font-size: 12px; }}
button {{ font-size: 13px; padding: 4px 10px; }}
details summary {{ cursor: pointer; color: #666; font-size: 12px; }}
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
  <div class="controls-row">{base.viz_nav_html(SLUG, 'umap')}</div>
</div>

<div class="ctrl-box">
  <div class="ctrl-box-title">Color Scheme</div>
  <div class="controls-row">
    <button id="qc-counts" class="qc-btn">Counts</button>
    <button id="qc-genes" class="qc-btn">Genes</button>
    <button id="qc-ribo" class="qc-btn">% ribo</button>
    <button id="qc-density" class="qc-btn" title="Number of similar cells: local density of each cell in the CURRENT gene space (the genes shown in the biplot). Genes are first reduced to a K-dim PCA latent so neighbour counts stay meaningful despite the curse of dimensionality, then each cell gets a Gaussian-kernel effective count of nearby cells (median-heuristic bandwidth, anchor-subsampled). Recompute-aware.">&asymp; similar cells</button>
    <button id="qc-pt" class="qc-btn">Pseudotime</button>
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
  </div>
  <div class="controls-row">
    <label class="rank-label" title="Number of UMAP components to fit (1–3). Lower rank collapses unused axes: rank=2 puts all points on the UMAP1×UMAP2 plane; rank=1 onto UMAP1.">rank
      <input id="rank-input" type="number" min="1" max="3" value="3" step="1"></label>
    <button id="recompute-btn" title="Refit UMAP on the panel HVG, using only the checked-subtype cells.">Replot with gene panel</button>
    <button id="recompute-genes-btn" title="Refit UMAP using only the genes currently shown in the right biplot (gene set ∩ mean/std/metabolism filters). With no gene filter active this is all genes.">Replot with shown genes</button>
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
  <div class="genestruct-box" title="Embed the selected cells into a K-dim PCA latent space, then score each gene by how much of its variance is recoverable from that embedding (reconstruction R^2). High = the gene's expression tracks the shared cell-state structure of the selected cells; low = private/independent variation. Uses the same expression the embedding sees (incl. 'Regress out %ribo' if checked).">
    <span class="gs-title">Gene structure (recoverability)</span>
    <button id="genestruct-btn" class="qc-btn">Score genes</button>
    <button id="genestruct-color" class="qc-btn" disabled>Colour by recoverability</button>
    <span class="label" style="margin-left:8px;">latent dims K:</span>
    <input id="genestruct-k" type="range" min="2" max="100" step="1" value="30" style="width:110px; vertical-align:middle;">
    <span id="genestruct-k-val" style="color:#1f77b4; font-weight:600;">30</span>
    <span class="label" style="margin-left:10px;">Min recoverability R²:</span>
    <input id="genestruct-slider" type="range" min="0" max="1" step="0.01" value="0" style="width:110px; vertical-align:middle;" disabled>
    <span id="genestruct-thr-val">off</span>
    <span id="genestruct-status" style="color:#555; margin-left:8px;"></span>
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

<div class="viz-stack">
  <div class="heatmap-row">
    <div class="heatmap-caption">
      <span>Panel HVG z-expression — genes by argmax of smoothed expression. <span id="heatmap-info" style="color:#888;"></span></span>
      <span class="heatmap-controls">
        <span class="label">order cells by</span>
        <button class="order-btn active" data-axis="0">UMAP1</button>
        <button class="order-btn" data-axis="1">UMAP2</button>
        <button class="order-btn" data-axis="2">UMAP3</button>
        <button class="order-btn" data-axis="-1" title="Order cells along the first principal embedding axis (= biological pseudotime).">Pseudotime</button>
        <label class="group-by-toggle"><input type="checkbox" id="group-by-celltype"> group by cell type first</label>
      </span>
    </div>
    <div class="line-strip-wrap"><canvas id="line-canvas"></canvas></div>
    <div class="heatmap-canvas-wrap">
      <canvas id="heatmap-canvas"></canvas>
      <canvas id="heatmap-overlay"></canvas>
    </div>
    <div class="celltype-strip-wrap"><canvas id="celltype-strip-canvas"></canvas></div>
    <div id="celltype-order-row" class="celltype-order-row" style="display:none;">
      <span class="label">Cell-type order:</span>
      <span id="celltype-order-chips" class="celltype-order-chips"></span>
      <button id="celltype-order-reset" class="celltype-order-reset"
              title="Restore the axis-mean order (drop manual rearrangement).">↺ axis-mean</button>
    </div>
  </div>
</div>
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
let lastHoveredCell = null, lastHoveredGene = null;

// ---- dynamic plot-box titles --------------------------------------------
const VIZ_METHOD = 'UMAP';
let titleCellColor = 'subtype';
let titleGeneRef   = 'strongest UMAP axis';
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
  const s = cell_score[i];
  status.innerHTML = '<b style="color:' + cell_dom_color[i] + '">Cell #' + i
    + '</b> <span style="color:#555">(' + cell_subtype[i] + (typeof cell_age !== 'undefined' && cell_age ? ' · age ' + cell_age[i] : '') + ')</span> &nbsp; '
    + 'UMAP1,2,3 = (' + s[0].toFixed(2) + ', ' + s[1].toFixed(2) + ', ' + s[2].toFixed(2) + ') &nbsp; '
    + 'genes recoloured by expression (range ' + (lo/EXPR_SCALE).toFixed(2) + '..' + (hi/EXPR_SCALE).toFixed(2) + ', magma)';
  titleGeneRef = 'expression in cell #' + i + ' (' + cell_subtype[i] + ')'; refreshTitles();
}});

genePlot.on('plotly_hover', function(data) {{
  const pt = data.points[0]; if (pt.curveNumber !== POINTS_TRACE) return;
  const j = pt.pointNumber; if (lastHoveredGene === j) return; lastHoveredGene = j;
  const n = N_CELLS; const col = new Array(n);
  for (let i = 0; i < n; i++) col[i] = cell_active[i] ? expr_matrix[i * N_GENES + j] : null;
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
    + 'loadings UMAP1,2,3 = (' + L[0].toFixed(2) + ', ' + L[1].toFixed(2) + ', ' + L[2].toFixed(2) + ') &nbsp; '
    + 'cells recoloured by expression (range ' + (lo/EXPR_SCALE).toFixed(2) + '..' + (hi/EXPR_SCALE).toFixed(2) + ', magma)';
  setColorKeyGradient(gene_name[j] + ' expression', 'magma', lo/EXPR_SCALE, hi/EXPR_SCALE, v => v.toFixed(2));
  titleCellColor = gene_name[j] + ' expression'; refreshTitles();
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
  status.innerHTML = 'Reset. Hover a cell or gene to colour by expression and reveal UMAP coords.'; clearColorKey();
  titleCellColor = 'subtype'; titleGeneRef = 'strongest UMAP axis'; refreshTitles();
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
// Per-gene recoverability R^2 from a K-dim PCA of the selected cells; null until
// the user clicks "Score genes by recoverability". Invalidated when the cell
// selection changes (recompute / subtype toggles) so the filter never uses stale values.
let gene_struct = null;

function applyGeneFilter() {{
  const meanThr = parseFloat(meanSlider.value), stdThr = parseFloat(stdSlider.value);
  const hideRibo = !!riboSlider && riboThreshold() < 1.0;
  const gsSlider = document.getElementById('genestruct-slider');
  const structThr = (gene_struct && gsSlider) ? parseFloat(gsSlider.value) : 0;
  const mask = gene_sets[activeSet], n = gene_name.length;
  const xs = new Array(n), ys = new Array(n), zs = new Array(n); let visible = 0;
  for (let j = 0; j < n; j++) {{
    if (mask[j] && gene_mean[j] >= meanThr && gene_std[j] >= stdThr
        && !(hideRibo && isRiboCorr(j))
        && (!gene_struct || gene_struct[j] >= structThr)) {{
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
    + (gene_in_panel[j] ? '(panel)' : '(projected)')
    + (hidden ? ' <span style="color:#c00">[hidden by current filter — ring still shows its position]</span>' : '')
    + ' — cells recoloured by its expression (magma).';
  titleCellColor = gene_name[j] + ' expression'; refreshTitles();
}}
geneSearch.addEventListener('change', runSearch);
geneSearch.addEventListener('keydown', e => {{ if (e.key === 'Enter') runSearch(); }});
document.getElementById('search-clear').addEventListener('click', function() {{
  geneSearch.value = ''; clearSearch();
}});

// ---- Gene-structure (recoverability R^2) ---------------------------------
// Embed the selected cells into a K-dim PCA latent space (eigendecomposition of
// the panel covariance, equivalent to SVD but cheaper for large K), then score
// each gene by the fraction of its variance recoverable from that embedding:
//   R^2_j = sum_k (u_k . z_j)^2 / ||z_j||^2   (u_k = orthonormal cell-space PCs,
//   z_j = gene j z-scored over the selected cells). Computed entirely client-side.
const gsBtn      = document.getElementById('genestruct-btn');
const gsColor    = document.getElementById('genestruct-color');
const gsK        = document.getElementById('genestruct-k');
const gsKval     = document.getElementById('genestruct-k-val');
const gsSlider2  = document.getElementById('genestruct-slider');
const gsThrVal   = document.getElementById('genestruct-thr-val');
const gsStatus   = document.getElementById('genestruct-status');
gsK.addEventListener('input', () => {{ gsKval.textContent = gsK.value; }});

function invalidateGeneStruct() {{
  if (!gene_struct) return;
  gene_struct = null;
  gsSlider2.disabled = true; gsColor.disabled = true;
  gsStatus.innerHTML = '<span style="color:#999">cell selection changed — rescore</span>';
  applyGeneFilter();   // drop the (now-inactive) recoverability filter from the view
}}

function computeGeneStructure() {{
  const sel = selectedSubtypes();
  const cellSel = [];
  for (let i = 0; i < cell_subtype.length; i++)
    if (sel.has(cell_subtype[i]) && regionAllowed(i)) cellSel.push(i);
  const m = cellSel.length;
  if (m < 5) {{ gsStatus.innerHTML = '<span style="color:#c00">need ≥5 cells</span>'; return; }}
  const np = panel_idx.length;
  let K = Math.max(2, Math.min(parseInt(gsK.value) || 30, np - 1, m - 1));

  // Zp: m × np, panel genes z-scored on the selected cells.
  const Zp = new Array(m); for (let i = 0; i < m; i++) Zp[i] = new Float64Array(np);
  for (let k = 0; k < np; k++) {{
    const j = panel_idx[k]; let s = 0, ss = 0;
    for (let ii = 0; ii < m; ii++) {{ const v = readVal(cellSel[ii], j); Zp[ii][k] = v; s += v; ss += v*v; }}
    const mean = s/m, sd = Math.sqrt(Math.max(ss/m - mean*mean, 1e-18));
    for (let ii = 0; ii < m; ii++) Zp[ii][k] = (Zp[ii][k] - mean) / sd;
  }}
  // Panel covariance C = Zp^T Zp  (np × np), eigendecompose top-K via power iteration.
  const C = new Array(np);
  for (let a = 0; a < np; a++) {{
    const Ca = new Float64Array(np);
    for (let ii = 0; ii < m; ii++) {{ const za = Zp[ii][a]; if (za === 0) continue;
      const Zi = Zp[ii]; for (let b = a; b < np; b++) Ca[b] += za * Zi[b]; }}
    C[a] = Ca;
  }}
  for (let a = 0; a < np; a++) for (let b = a+1; b < np; b++) C[b][a] = C[a][b];  // symmetrize
  const {{U: Vc, S: eig}} = powerIterTopK(C, K, 60);   // Vc[k]=eigvec(np), eig[k]=eigval=S_k^2

  // Cell-space orthonormal vectors u_k = Zp · Vc_k / S_k  (length m).
  const Uk = [];
  for (let k = 0; k < K; k++) {{
    const sk = Math.sqrt(Math.max(eig[k], 1e-18));
    const uk = new Float64Array(m); const vk = Vc[k];
    for (let ii = 0; ii < m; ii++) {{ let acc = 0; const Zi = Zp[ii];
      for (let a = 0; a < np; a++) acc += Zi[a] * vk[a]; uk[ii] = acc / sk; }}
    Uk.push(uk);
  }}
  // Per-gene R^2 over ALL genes (panel + broader).
  const n_all = gene_name.length;
  const r2 = new Float32Array(n_all);
  for (let j = 0; j < n_all; j++) {{
    let s = 0, ss = 0;
    for (let ii = 0; ii < m; ii++) {{ const v = readVal(cellSel[ii], j); s += v; ss += v*v; }}
    const mean = s/m, varj = Math.max(ss/m - mean*mean, 1e-18);
    let energy = 0;
    for (let k = 0; k < K; k++) {{ const uk = Uk[k]; let dot = 0;
      for (let ii = 0; ii < m; ii++) dot += (readVal(cellSel[ii], j) - mean) * uk[ii];
      energy += dot * dot; }}
    r2[j] = Math.max(0, Math.min(1, energy / (varj * m)));
  }}
  gene_struct = r2;
  // enable controls; KEEP the current threshold and apply it immediately so the
  // genes shown always match the recoverability filter, then colour by R².
  gsSlider2.disabled = false; gsColor.disabled = false;
  gsThrVal.textContent = parseFloat(gsSlider2.value) > 0 ? (+gsSlider2.value).toFixed(2) : 'off';
  colorGenesByStruct();
  applyGeneFilter();
  const order = Array.from(r2.keys()).sort((a,b) => r2[b] - r2[a]);
  const top = order.slice(0, 8).map(j => gene_name[j] + ' (' + r2[j].toFixed(2) + ')');
  const panelMean = panel_idx.reduce((a,j)=>a+r2[j],0)/np;
  gsStatus.innerHTML = '<b>K=' + K + '</b>, ' + m + ' cells · top: ' + top.join(', ')
    + ' · panel mean R²=' + panelMean.toFixed(2);
}}

function colorGenesByStruct() {{
  if (!gene_struct) return;
  const colors = Array.from(gene_struct, v => viridis[Math.max(0, Math.min(255, Math.round(255*v)))]);
  Plotly.restyle(genePlot, {{'marker.color': [colors]}}, [POINTS_TRACE]);
  setColorKeyGradient('gene recoverability R²', 'viridis', 0, 1, v => v.toFixed(2));
}}

gsBtn.addEventListener('click', () => {{
  gsBtn.disabled = true; gsStatus.textContent = 'scoring…';
  setTimeout(() => {{ try {{ computeGeneStructure(); }} finally {{ gsBtn.disabled = false; }} }}, 30);
}});
gsColor.addEventListener('click', colorGenesByStruct);
gsSlider2.addEventListener('input', () => {{
  gsThrVal.textContent = parseFloat(gsSlider2.value) > 0 ? (+gsSlider2.value).toFixed(2) : 'off';
  applyGeneFilter();
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

// ---- "Number of similar cells": local density in the CURRENT gene space -----
// Reduce the currently-shown genes to a K-dim PCA latent (curse-of-dimensionality
// fix: raw distances over 100s of genes concentrate and stop discriminating),
// then each cell's value = Gaussian-kernel effective neighbour count in that
// latent (median-heuristic bandwidth, anchor-subsampled). Recomputed on click so
// it always reflects the current gene filter / embedding.
function colorBySimilarCells() {{
  const cellSel = [];
  for (let i = 0; i < cell_active.length; i++) if (cell_active[i]) cellSel.push(i);
  const m = cellSel.length;
  if (m < 10) {{ status.innerHTML = '<span style="color:#c00">need &ge;10 cells</span>'; return; }}
  let basis = (typeof visibleGeneIdx === 'function') ? visibleGeneIdx()
            : (typeof shownGeneIdx === 'function') ? shownGeneIdx() : panel_idx.slice();
  if (!basis || basis.length < 3) basis = panel_idx.slice();
  if (basis.length > 600) basis = basis.slice().sort((a,b) => gene_std[b]-gene_std[a]).slice(0,600);
  const nb = basis.length;
  const K = Math.max(2, Math.min(25, nb-1, m-1));
  // Zp: m x nb, z-scored on the selected cells.
  const Zp = new Array(m); for (let i=0;i<m;i++) Zp[i] = new Float64Array(nb);
  for (let k=0;k<nb;k++) {{ const j=basis[k]; let s=0, ss=0;
    for (let ii=0;ii<m;ii++) {{ const v=readVal(cellSel[ii], j); Zp[ii][k]=v; s+=v; ss+=v*v; }}
    const mean=s/m, sd=Math.sqrt(Math.max(ss/m-mean*mean,1e-18));
    for (let ii=0;ii<m;ii++) Zp[ii][k]=(Zp[ii][k]-mean)/sd; }}
  // Covariance C = Zp^T Zp (nb x nb), top-K eigenvectors (cheap when nb small).
  const C=new Array(nb);
  for (let a=0;a<nb;a++) {{ const Ca=new Float64Array(nb);
    for (let ii=0;ii<m;ii++) {{ const za=Zp[ii][a]; if (za===0) continue; const Zi=Zp[ii];
      for (let b=a;b<nb;b++) Ca[b]+=za*Zi[b]; }} C[a]=Ca; }}
  for (let a=0;a<nb;a++) for (let b=a+1;b<nb;b++) C[b][a]=C[a][b];
  const pk = powerIterTopK(C, K, 60); const Vc = pk.U;
  // Cell PCA scores sc (m x K, row-major): sc_ik = Zp_i . Vc_k.
  const sc=new Float64Array(m*K);
  for (let k=0;k<K;k++) {{ const vk=Vc[k];
    for (let ii=0;ii<m;ii++) {{ let acc=0; const Zi=Zp[ii];
      for (let a=0;a<nb;a++) acc+=Zi[a]*vk[a]; sc[ii*K+k]=acc; }} }}
  // Anchor subsample (stride) -> O(m.A.K) density.
  const A=Math.min(m,1200), step=m/A; const anc=new Int32Array(A);
  for (let a=0;a<A;a++) anc[a]=Math.floor(a*step);
  // Bandwidth h^2 = median squared distance among up to 400 anchors (median heuristic).
  const Bn=Math.min(A,400); const dd=[];
  for (let a=0;a<Bn;a++) for (let b=a+1;b<Bn;b++) {{ const ia=anc[a]*K, ib=anc[b]*K; let d2=0;
    for (let k=0;k<K;k++) {{ const t=sc[ia+k]-sc[ib+k]; d2+=t*t; }} dd.push(d2); }}
  dd.sort((x,y)=>x-y);
  const h2=Math.max(dd.length?dd[dd.length>>1]:1, 1e-12), inv2h2=1/(2*h2), scale=m/A;
  const dens=new Float64Array(cell_subtype.length);
  for (let ii=0;ii<m;ii++) {{ const base=ii*K; let acc=0;
    for (let a=0;a<A;a++) {{ const ia=anc[a]*K; let d2=0;
      for (let k=0;k<K;k++) {{ const t=sc[base+k]-sc[ia+k]; d2+=t*t; }}
      acc+=Math.exp(-d2*inv2h2); }}
    dens[cellSel[ii]]=acc*scale; }}
  colorByQC(Array.from(dens), 'similar cells (PCA K='+K+', '+nb+' genes)', v => Math.round(v).toLocaleString());
}}
document.getElementById('qc-density').addEventListener('click', () => {{
  const b=document.getElementById('qc-density'); b.disabled=true; status.innerHTML='computing density&hellip;';
  setTimeout(() => {{ try {{ colorBySimilarCells(); }} finally {{ b.disabled=false; }} }}, 30);
}});
document.getElementById('qc-pt').addEventListener('click', () => {{
  // Pseudotime colour = UMAP1 (cell_score[:,0]) ascending → viridis on active cells.
  const arr = cell_score.map(s => s[0]);
  colorByQC(arr, 'pseudotime (UMAP1)', v => v.toFixed(2));
}});

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
const GREY_NO_LAYER = 'rgba(0,0,0,0)';
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
// Subtype-subset UMAP recompute
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

// Any change to the cell selection makes a previously-computed gene-structure
// score stale, so drop it (and disable its filter) when the selection changes.
subtypeCheckboxes.forEach(cb => cb.addEventListener('change', invalidateGeneStruct));
document.querySelectorAll('.grp-toggle, #subt-all, #subt-none, .lin-btn, .rg-btn, .ag-btn, .ag-btn-all').forEach(b => b.addEventListener('click', invalidateGeneStruct));

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
  // Cell index list (subtype-checked AND region-allowed)
  const cellSel = [];
  for (let i = 0; i < cell_subtype.length; i++) {{
    if (sel.has(cell_subtype[i]) && regionAllowed(i)) cellSel.push(i);
  }}
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
      const v = readVal(cellSel[ii], j);
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

  // ---- UMAP fit via umap-js. Synchronous; takes ~3-10 s for ~1k cells.
  // The user can pick a lower rank: rank=2 fits a 2D UMAP (z=0), rank=1 a 1D one.
  const ZpArrays = Zp.map(row => Array.from(row));
  const nNeighborsHere = Math.max(2, Math.min(15, m - 1));
  const rankReq = Math.max(1, Math.min(3, parseInt(document.getElementById('rank-input').value) || 3));
  const K_emb   = rankReq;
  if (typeof UMAP === 'undefined') {{
    recomputeStatus.innerHTML = '<span style="color:#c00">umap-js failed to load (offline?). '
      + 'Refresh while online, or open this HTML over HTTPS.</span>';
    return;
  }}
  const umap = new UMAP.UMAP({{
    nComponents: K_emb, nNeighbors: nNeighborsHere, minDist: 0.3,
    random: (function() {{ let s = 0x9E3779B9; return function() {{
      s |= 0; s = s + 0x6D2B79F5 | 0;
      let t = Math.imul(s ^ s >>> 15, 1 | s);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }}; }})(),
  }});
  const embeddingRaw = umap.fit(ZpArrays);   // array of m × K_emb plain arrays
  // Pad to 3D with zeros for the unused trailing axes
  const embedding = new Array(m);
  for (let ii = 0; ii < m; ii++) {{
    const row = embeddingRaw[ii];
    embedding[ii] = [row[0] || 0, (K_emb > 1 ? (row[1] || 0) : 0), (K_emb > 2 ? (row[2] || 0) : 0)];
  }}
  // Centre coords around 0
  let cx = 0, cy = 0, cz = 0;
  for (let ii = 0; ii < m; ii++) {{ cx += embedding[ii][0]; cy += embedding[ii][1]; cz += embedding[ii][2]; }}
  cx /= m; cy /= m; cz /= m;
  const scoreSel = new Array(m);
  for (let ii = 0; ii < m; ii++) scoreSel[ii] = [embedding[ii][0]-cx, embedding[ii][1]-cy, embedding[ii][2]-cz];

  // Gene centroid in the new UMAP space: weighted-average of selected-cell coords
  // by each gene's non-negative expression. This mirrors the static UMAP/diffmap
  // viewers' "expression-weighted centroid" gene-placement convention.
  const newGeneLoading = new Array(n_all);
  for (let j = 0; j < n_all; j++) {{
    let sw = 0, s0 = 0, s1 = 0, s2 = 0;
    for (let ii = 0; ii < m; ii++) {{
      const w = readValNN(cellSel[ii], j);
      sw += w;
      s0 += w * scoreSel[ii][0];
      s1 += w * scoreSel[ii][1];
      s2 += w * scoreSel[ii][2];
    }}
    newGeneLoading[j] = sw > 1e-12 ? [s0/sw, s1/sw, s2/sw] : [0, 0, 0];
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
    // UMAP has no V matrix — find pole-label genes as the *most extreme PANEL
    // gene centroids* on each axis k, computed from newGeneLoading (centroids).
    let bestPosIdx = 0, bestPos = -Infinity, bestNegIdx = 0, bestNeg = Infinity;
    for (let p = 0; p < panel_idx.length; p++) {{
      const j = panel_idx[p];
      const w = newGeneLoading[j][k];
      if (w > bestPos) {{ bestPos = w; bestPosIdx = j; }}
      if (w < bestNeg) {{ bestNeg = w; bestNegIdx = j; }}
    }}
    newPoleTop[2*k]   = gene_name[bestPosIdx];
    newPoleTop[2*k+1] = gene_name[bestNegIdx];
  }}
  // UMAP has no analytic var-explained ratios.
  const vr = [null, null, null];

  // ---- commit new state to globals ----
  gene_x = newGeneX; gene_y = newGeneY; gene_z = newGeneZ;
  cell_load = newCellLoad; gene_load = newGeneLoad;
  cell_score = newCellScore;
  gene_loading = newGeneLoading.map(L => [+L[0].toFixed(3), +L[1].toFixed(3), +L[2].toFixed(3)]);
  cell_dom_color = newCellDom; gene_dom_color = newGeneDom;
  cell_active = newCellActive;
  pole_top = newPoleTop;

  // New gene_default_colors (by dominant signed UMAP axis)
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
  const newPoleLab = POLE_NAMES_.slice();
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
    + ' cells, ' + n_panel + ' ' + basisLabel + ' (rank ' + K_emb + ', ' + dt + 's) &nbsp; '
    + 'poles: ' + newPoleTop.map((g, p) => POLE_NAMES_[p] + '=' + g).join(', ');
  lastHoveredCell = null; lastHoveredGene = null;
  // 8. heatmap reflects the new UMAP1 ordering on the new active cell set
  renderHeatmap();}}

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
  const hideRibo = !!riboSlider && riboThreshold() < 1.0;
  const gsSlider = document.getElementById('genestruct-slider');
  const structThr = (gene_struct && gsSlider) ? parseFloat(gsSlider.value) : 0;
  const out = [];
  for (let j = 0; j < gene_name.length; j++) {{
    if (mask[j] && gene_mean[j] >= meanThr && gene_std[j] >= stdThr
        && !(hideRibo && isRiboCorr(j))
        && (!gene_struct || gene_struct[j] >= structThr)) out.push(j);
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
let heatmapActiveIdx = null;            // length-m_active cell indices in UMAP1 order
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
// Cell-ordering state for the heatmap.
let heatmapOrderAxis = 0;
let heatmapGroupByType = false;

function renderHeatmap() {{
  sizeHeatmapCanvas();
  const ctx = heatCanvas.getContext('2d', {{alpha: false}});
  const W = heatCanvas.width, H = heatCanvas.height;
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, W, H);

  // 1. Active cells ordered by the chosen axis (with optional subtype grouping)
  let activeIdx = [];
  for (let i = 0; i < cell_active.length; i++) if (cell_active[i]) activeIdx.push(i);
  const axis = heatmapOrderAxis;
  const sortKey = (axis === -1)
    ? (i => cell_score[i][0])     // pseudotime = first embedding axis (UMAP1)
    : (i => cell_score[i][axis]);
  if (heatmapGroupByType) {{
    const byType = new Map();
    for (const i of activeIdx) {{
      const t = cell_subtype[i];
      if (!byType.has(t)) byType.set(t, []);
      byType.get(t).push(i);
    }}
    let typeOrder;
    if (celltypeOrder && celltypeOrder.length) {{
      typeOrder = celltypeOrder.filter(t => byType.has(t));
      for (const t of byType.keys()) if (!typeOrder.includes(t)) typeOrder.push(t);
    }} else {{
      const subtypeMean = [];
      for (const [t, list] of byType) {{
        let s = 0; for (const i of list) s += sortKey(i);
        subtypeMean.push([t, s / list.length]);
      }}
      subtypeMean.sort((a, b) => a[1] - b[1]);
      typeOrder = subtypeMean.map(p => p[0]);
    }}
    activeIdx = [];
    for (const t of typeOrder) {{
      const list = byType.get(t);
      list.sort((a, b) => sortKey(a) - sortKey(b));
      for (const i of list) activeIdx.push(i);
    }}
    renderHeatmap._typeOrder = typeOrder;
  }} else {{
    activeIdx.sort((a, b) => sortKey(a) - sortKey(b));
    renderHeatmap._typeOrder = null;
  }}
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
      const v = expr_matrix[activeIdx[ii] * N_GENES + j] / EXPR_SCALE;
      s += v; ss += v*v;
    }}
    const mean = s / m;
    const stdv = Math.sqrt(Math.max(ss/m - mean*mean, 1e-12));
    const row = new Float32Array(m);
    for (let ii = 0; ii < m; ii++) {{
      row[ii] = (expr_matrix[activeIdx[ii] * N_GENES + j] / EXPR_SCALE - mean) / stdv;
    }}
    Zg[p] = row;
  }}

  // 3. Smooth each gene row with rolling mean for argmax detection
  const win = Math.max(5, Math.min(m, Math.floor(m / 120) | 0));
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
  paintCellTypeStrip(activeIdx);
  updateCellTypeOrderUI();
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
// (cells in UMAP1 order) but it never overlaps the gene rows below.
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
    const v = expr_matrix[heatmapActiveIdx[ii] * N_GENES + j] / EXPR_SCALE;
    s += v; ss += v*v;
  }}
  const mean = s / m;
  const stdv = Math.sqrt(Math.max(ss/m - mean*mean, 1e-12));
  const raw = new Float32Array(m);
  for (let ii = 0; ii < m; ii++) raw[ii] = (expr_matrix[heatmapActiveIdx[ii] * N_GENES + j] / EXPR_SCALE - mean) / stdv;
  const win = heatmapSmoothWin > 0 ? heatmapSmoothWin : Math.max(5, Math.floor(m / 120) | 0);
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
let heatResizeTimer = null;
function scheduleHeatmapRedraw() {{
  if (heatResizeTimer) clearTimeout(heatResizeTimer);
  heatResizeTimer = setTimeout(renderHeatmap, 80);
}}
window.addEventListener('resize', () => {{ resizePlots(); scheduleHeatmapRedraw(); }});
document.querySelectorAll('.order-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    heatmapOrderAxis = parseInt(btn.dataset.axis) || 0;
    document.querySelectorAll('.order-btn').forEach(b =>
      b.classList.toggle('active', b === btn));
    renderHeatmap();
  }});
}});
document.getElementById('group-by-celltype').addEventListener('change', e => {{
  heatmapGroupByType = e.target.checked;
  if (!heatmapGroupByType) celltypeOrder = null;
  renderHeatmap();
}});

function paintCellTypeStrip(activeIdx) {{
  const cv = document.getElementById('celltype-strip-canvas');
  if (!cv) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.parentElement.clientWidth;
  const cssH = cv.parentElement.clientHeight || 14;
  cv.width  = Math.max(1, Math.floor(cssW * dpr));
  cv.height = Math.max(1, Math.floor(cssH * dpr));
  cv.style.width  = cssW + 'px';
  cv.style.height = cssH + 'px';
  const ctx = cv.getContext('2d', {{alpha: false}});
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, cv.width, cv.height);
  const W = cv.width, H = cv.height;
  const m = activeIdx.length;
  const cellW = W / m;
  for (let i = 0; i < m; i++) {{
    const t = cell_subtype[activeIdx[i]];
    ctx.fillStyle = subtype_palette[t] || '#cccccc';
    ctx.fillRect(i * cellW, 0, cellW + 1, H);
  }}
}}

function updateCellTypeOrderUI() {{
  const row = document.getElementById('celltype-order-row');
  if (!row) return;
  if (!heatmapGroupByType) {{ row.style.display = 'none'; return; }}
  row.style.display = '';
  const order = renderHeatmap._typeOrder || [];
  const chipsEl = document.getElementById('celltype-order-chips');
  chipsEl.innerHTML = '';
  order.forEach(t => {{
    const chip = document.createElement('span');
    chip.className = 'ct-chip';
    chip.draggable = true;
    chip.dataset.type = t;
    chip.innerHTML = `<span class="ct-chip-dot" style="background:${{subtype_palette[t] || '#ccc'}};"></span>${{t}}`;
    chipsEl.appendChild(chip);
  }});
  let dragSrc = null;
  chipsEl.querySelectorAll('.ct-chip').forEach(chip => {{
    chip.addEventListener('dragstart', e => {{
      dragSrc = chip; chip.classList.add('dragging');
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', chip.dataset.type);
    }});
    chip.addEventListener('dragend', () => {{
      chip.classList.remove('dragging');
      chipsEl.querySelectorAll('.ct-chip').forEach(c => c.classList.remove('drop-target'));
    }});
    chip.addEventListener('dragover', e => {{
      e.preventDefault();
      if (chip !== dragSrc) chip.classList.add('drop-target');
    }});
    chip.addEventListener('dragleave', () => chip.classList.remove('drop-target'));
    chip.addEventListener('drop', e => {{
      e.preventDefault();
      chip.classList.remove('drop-target');
      if (!dragSrc || dragSrc === chip) return;
      chipsEl.insertBefore(dragSrc, chip);
      celltypeOrder = Array.from(chipsEl.querySelectorAll('.ct-chip')).map(c => c.dataset.type);
      renderHeatmap();
    }});
  }});
}}
document.getElementById('celltype-order-reset').addEventListener('click', () => {{
  celltypeOrder = null;
  renderHeatmap();
}});

refreshTitles();
setTimeout(function() {{ resizePlots(); applyGeneFilter(); renderHeatmap(); clearLineGraph(); refreshTitles(); }}, 80);

// Cohort-level default subset: when this cohort ships with only a curated
// subset of subtypes checked, auto-fire the panel-HVG recompute so the
// initial view shows the subset's own basis instead of the full-cohort fit.
// Skipped when a Copy-Link URL hash is present (that snippet triggers its
// own recompute).
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
