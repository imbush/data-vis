#!/usr/bin/env python
"""PCA biplot explorer with in-browser PCA recompute on a selected subtype subset.

Same biplot UI as build_lamp5_svd_app_3d.py plus:
  - A row of subtype checkboxes (default: all checked).
  - A "Recompute PCA on selected" button. Pressing it filters cells to the
    checked subtypes, re-z-scores the panel matrix on that subset, runs
    power-iteration PCA (top-3) in the browser, projects every gene onto the
    new basis, and updates both biplots in place (positions, vertex labels,
    default colours).

The initial page is exactly the full-cohort PCA; recompute is a one-click
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
    # Per-cell subclass — used to group subtype checkboxes by subclass.
    # Older proj_full caches may not have this; fall back to the group name.
    cell_subclass = np.array(proj.get(
        'cell_subclass',
        [GROUP_NAME] * len(subs)
    ))
    # Optional dual-resolution marker-named subtype levels (Gao dev-VIS cohort;
    # see build_devvis_marker_names.py). When both are present the recompute UI
    # exposes a cluster/subcluster granularity toggle and DEFAULTS to the finer
    # subcluster level. Otherwise it uses the single `subs` level unchanged.
    _named_subcluster = proj.get('subs_named_subcluster')
    _named_cluster    = proj.get('subs_named_cluster')
    DUAL_LEVEL = (_named_subcluster is not None and _named_cluster is not None)
    if DUAL_LEVEL:
        subs     = np.array([str(x) for x in _named_subcluster])  # default = subcluster
        subs_alt = np.array([str(x) for x in _named_cluster])     # coarser = cluster
        DEFAULT_LEVEL, ALT_LEVEL = 'subcluster', 'cluster'
    else:
        subs_alt = None
        DEFAULT_LEVEL, ALT_LEVEL = 'subtype', None
    # Optional per-cell region (only set when the cohort spans multiple
    # cortical areas — e.g. AllInhib_V1ALM). When present we expose a
    # Region: toggle in the recompute UI.
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
    # Per-cell donor for the "color by sample" button. None when proj predates
    # the field — UI hides the button gracefully in that case.
    cell_donor_arr = proj.get('cell_donor')
    cell_donor_list = ([str(x) for x in cell_donor_arr]
                       if cell_donor_arr is not None else None)
    # Per-cell developmental age (dev-VIS cohort only). When non-None we
    # enable the Age colour-by button + Age age filter.
    cell_age_arr = proj.get('cell_age')
    cell_age_list = ([str(x) for x in cell_age_arr]
                     if cell_age_arr is not None else None)
    # Per-cell dissected cortical layer (Tasic 2018 V1/ALM). The Layer QC
    # button reads this. Hidden when None or single-valued.
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
    # Compare against the ORIGINAL proj subs (cell_cluster); `subs` may have been
    # switched to a marker-named display level for dual-resolution cohorts, but
    # cell order is identical (same filtered anndata), which is what this checks.
    assert np.array_equal(np.array(qc['subs']), np.array(proj['subs'])), 'QC/proj cell-order mismatch'
    qc_total, qc_ngenes, qc_ribo = (np.asarray(qc['total_counts']),
                                    np.asarray(qc['n_genes']),
                                    np.asarray(qc['pct_ribo']))
    # Per-gene Pearson correlation with %ribo. Genes with |r| above the toggle
    # threshold are filtered out of the visible set when "Hide ribo-corr" is on.
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

    # ---- initial PCA: all cells. Top-NPC=3 used for embedding.
    Xp = X_keep[:, in_panel]
    Zp = prep_cols(Xp)
    U, S, Vt = np.linalg.svd(Zp, full_matrices=False)
    U, S, Vt = U[:, :NPC], S[:NPC], Vt[:NPC]
    cell_scores = U * S
    Zall = prep_cols(X_keep)
    gene_load3 = (Zall.T @ U) / S
    print(f'  initial PCA on {Xp.shape[0]} cells × {int(in_panel.sum())} panel genes')

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

    # Top genes per axis for GO enrichment: 30 + and 30 - panel genes per PC
    # (pooled into one query of ~60 genes per axis at the Enrichr endpoint).
    GO_TOP_PER_POLE = 30
    top_genes_per_axis = []
    for k in range(NPC):
        order = np.argsort(Vpanel[:, k])
        neg = [panel_genes_list[i] for i in order[:GO_TOP_PER_POLE]]
        pos = [panel_genes_list[i] for i in order[-GO_TOP_PER_POLE:]][::-1]
        top_genes_per_axis.append({'name': f'PC{k+1}', 'pos': pos, 'neg': neg})
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

    # Embed the expression matrix as base64-encoded uint8 (EXPR_SCALE=16, so
    # log-CPM ∈ [0, 16] covers the full dynamic range with ~0.06 unit
    # resolution). 3–4× smaller than the previous JSON-of-nested-arrays AND
    # the JS decode is a single typed-array allocation (no JSON.parse over
    # millions of values).
    import base64
    EXPR_SCALE = 16
    _expr_q = np.clip(np.round(X_keep * EXPR_SCALE), 0, 255).astype(np.uint8)
    expr_b64 = base64.b64encode(_expr_q.tobytes()).decode('ascii')
    n_cells_emit = int(_expr_q.shape[0])
    n_genes_emit = int(_expr_q.shape[1])
    panel_idx = [j for j, p in enumerate(in_panel.tolist()) if p]

    # ---- figure construction (same as PCA script) ----------------------------
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

    # gene-set masks (same as the PCA script)
    panel_mask_list = in_panel.tolist()
    set_masks  = {'all': [True]*n_genes, 'panel': panel_mask_list}
    set_counts = {'all': n_genes,        'panel': n_panel_disp}
    for name, gene_list in base.GENE_SETS.items():
        gset = set(gene_list)
        mask = [g in gset for g in gene_names]
        set_masks[name] = mask; set_counts[name] = sum(mask)

    mean_min, mean_max = float(np.min(mean_expr)), float(np.max(mean_expr))
    std_min,  std_max  = float(np.min(std_expr)),  float(np.max(std_expr))

    # Subtype checkbox row, grouped by cell_subclass. Each subclass gets a header
    # with select-all / deselect-all shortcuts, then its constituent subtypes
    # below as individual checkboxes. Single-subclass cohorts collapse to one
    # group (still useful: the header buttons act as bulk all/none for the group).
    # Cohort-level default subset: only these subtypes start checked. Empty
    # tuple (default) means every checkbox starts checked.
    default_subset = set(base.GROUP.get('default_selected_subtypes') or ())
    subclass_order = sorted(set(cell_subclass.tolist()))
    import base64 as _b64m

    def _build_level(subs_x, cats_x, palette_x, level_id, active):
        """Return (panel_html, cs_cats, cs_idx_b64, color_default) for one
        subtype resolution. The panel is a `.subt-panel` wrapper (hidden unless
        `active`) holding subclass-grouped checkboxes tagged with data-level so
        the JS can scope selection to the active level."""
        counts = {c: int(np.sum(subs_x == c)) for c in cats_x}
        s2sc = {}
        for s, csc in zip(subs_x, cell_subclass):
            s2sc.setdefault(s, csc)
        by_sc = {csc: [c for c in cats_x if s2sc.get(c) == csc]
                 for csc in subclass_order}
        def _chk(c):
            return 'checked' if (not default_subset or c in default_subset) else ''
        parts = [f'<div class="subt-panel{" active" if active else ""}" data-level="{level_id}">']
        for csc in subclass_order:
            grp = by_sc[csc]
            if not grp: continue
            tot = sum(counts[c] for c in grp)
            parts.append(
                f'<div class="subt-group" data-subclass="{csc}">'
                f'<div class="subt-group-head">'
                f'<b>{csc}</b> <span class="ct">({tot} cells, {len(grp)} subtypes)</span>'
                f'<button class="grp-toggle" data-grp="{csc}" data-action="all" title="Check all in {csc}">all</button>'
                f'<button class="grp-toggle" data-grp="{csc}" data-action="none" title="Uncheck all in {csc}">none</button>'
                f'</div><div class="subt-group-checkboxes">'
                + ''.join(
                    f'<label class="subt-chk" data-grp-sub="{csc}">'
                    f'<input type="checkbox" data-sub="{c}" data-grp="{csc}" {_chk(c)}> '
                    f'<span style="color:{palette_x[c]}; font-weight:700;">●</span> '
                    f'{c} <span class="ct">({counts[c]})</span></label>'
                    for c in grp)
                + '</div></div>')
        parts.append('</div>')
        cs_cats = list(dict.fromkeys(subs_x.tolist()))
        if len(cs_cats) > 256:
            raise RuntimeError("cell_subtype has >256 unique categories; bump to uint16")
        look = {c: i for i, c in enumerate(cs_cats)}
        idx = np.array([look[v] for v in subs_x.tolist()], dtype=np.uint8)
        color_default = [palette_x[s] for s in subs_x]
        return (''.join(parts), cs_cats, _b64m.b64encode(idx.tobytes()).decode("ascii"),
                color_default)

    # Default (finer, subcluster for dual cohorts) level.
    subtype_checkbox_html, _cs_cats, _cs_idx_b64, cell_color_default = _build_level(
        subs, cats, subtype_palette, DEFAULT_LEVEL, active=True)
    # Coarser (cluster) level — only for dual-resolution cohorts.
    if DUAL_LEVEL:
        cats_alt = sorted(set(subs_alt.tolist()))
        palette_alt = base.build_subtype_palette(cats_alt)
        _panel_alt, _cs_cats_alt, _cs_idx_alt_b64, _color_default_alt = _build_level(
            subs_alt, cats_alt, palette_alt, ALT_LEVEL, active=False)
        subtype_checkbox_html += _panel_alt
    auto_recompute_on_load = bool(default_subset)
    # Granularity toggle button (dual-resolution cohorts only). Its label always
    # names the level you'll switch TO; JS keeps it in sync.
    level_toggle_html = (
        '<span class="lin-sep">|</span>'
        '<button id="level-toggle" class="level-btn" '
        'data-default="subcluster" data-alt="cluster" '
        'title="Switch the subtype menu (and cell colours) between the finer '
        'subcluster level and the coarser cluster level.">▸ show cluster level</button>'
    ) if DUAL_LEVEL else ''
    js_data = (
        f"const EXPR_SCALE  = {EXPR_SCALE};\n"
        # Expression units, verified from the source h5ad: Tasic X = ln(CPM+1)
        # (natural log; exp(X)-1 reproduces the cpm layer exactly), dev-VIS/ABC
        # cohorts are log2(x+1). Used to label the saved-figure colour bar.
        f"const EXPR_UNIT = {json.dumps('log2(x+1)' if base.LOG_SCALE_X else ('ln(CP10K+1)' if base.GROUP_NAME.startswith('Fezf2') else 'ln(CPM+1)'))};\n"
        f"const N_CELLS = {n_cells_emit};\n"
        f"const N_GENES = {n_genes_emit};\n"
        f"const EXPR_B64 = {json.dumps(expr_b64)};\n"
        # Decode the base64 expression matrix into a flat Uint8Array once.
        # Use expr_matrix[i * N_GENES + j] to read cell i, gene j (uint8 in
        # [0, 255]; divide by EXPR_SCALE to recover log-CPM).
        f"const expr_matrix = (function() {{\n"
        f"  const bin = atob(EXPR_B64);\n"
        f"  const u8 = new Uint8Array(bin.length);\n"
        f"  for (let k = 0; k < bin.length; k++) u8[k] = bin.charCodeAt(k);\n"
        f"  return u8;\n"
        f"}})();\n"
        f"let gene_default_colors = {json.dumps(gene_color_default)};\n"
        f"const _cell_subtype_cats = {json.dumps(_cs_cats)};\n"
        f"const _cell_subtype_idx_b64 = {json.dumps(_cs_idx_b64)};\n"
        "function _decodeSubtype(cats, b64) {\n"
        "  const bin = atob(b64); const out = new Array(bin.length);\n"
        "  for (let k = 0; k < bin.length; k++) out[k] = cats[bin.charCodeAt(k)];\n"
        "  return out;\n"
        "}\n"
        # cell_subtype / subtype_palette / cell_default_colors are mutable so the
        # granularity toggle (dual-resolution cohorts) can swap the active level.
        "let cell_subtype = _decodeSubtype(_cell_subtype_cats, _cell_subtype_idx_b64);\n"
        f"let subtype_palette = {json.dumps(subtype_palette)};\n"
        "let cell_default_colors = cell_subtype.map(s => subtype_palette[s] || '#888888');\n"
        # Dual-resolution level table (empty {} for single-level cohorts).
        + (f"const SUBTYPE_LEVELS = {{"
           f"{json.dumps(DEFAULT_LEVEL)}: {{cats: {json.dumps(_cs_cats)}, "
           f"idx: {json.dumps(_cs_idx_b64)}, palette: {json.dumps(subtype_palette)}}}, "
           f"{json.dumps(ALT_LEVEL)}: {{cats: {json.dumps(_cs_cats_alt)}, "
           f"idx: {json.dumps(_cs_idx_alt_b64)}, palette: {json.dumps(palette_alt)}}}}};\n"
           f"let activeLevel = {json.dumps(DEFAULT_LEVEL)};\n"
           if DUAL_LEVEL else
           "const SUBTYPE_LEVELS = {}; let activeLevel = null;\n") +
        f"const cell_region = {json.dumps(cell_region_list)};\n"
        f"const region_options = {json.dumps(region_options)};\n"
        f"const cell_age = {json.dumps(cell_age_list)};\n"
        f"const cell_layer = {json.dumps(cell_layer_list)};\n"
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

    # Region toggle — rendered only when the cohort spans multiple regions.
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

    # Age toggle — rendered when the cohort spans multiple developmental ages
    # (dev-VIS). Each chip toggles individual ages on/off, ordered E → P.
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
                '<button class="ag-btn-all" data-act="all" '
                'title="Enable all ages">all</button>'
                '<button class="ag-btn-all" data-act="none" '
                'title="Disable all ages">none</button>'
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
<title>{GROUP_NAME} PCA recompute explorer</title>
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
/* Granularity panels: only the active resolution is laid out; `display:contents`
   keeps the group divs as direct flex children of the controls row. */
.subt-panel {{ display: none; }}
.subt-panel.active {{ display: contents; }}
#level-toggle {{ font-weight: 700; }}
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
  <div class="controls-row">{base.viz_nav_html(SLUG, 'svd')}</div>
</div>

<div class="ctrl-box">
  <div class="ctrl-box-title">Color Scheme</div>
  <div class="controls-row">
    <button id="qc-counts" class="qc-btn">Counts</button>
    <button id="qc-genes" class="qc-btn">Genes</button>
    <button id="qc-ribo" class="qc-btn">% ribo</button>
    <button id="qc-density" class="qc-btn" title="Colour BOTH plots by local density: cells by number of similar cells in the CURRENT gene space, and genes by number of similar genes in the gene-embedding space.">Density</button>
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
    {level_toggle_html}
    <span class="lin-sep">|</span>
    <button class="lin-btn" data-lin="MGE" title="Select MGE-derived subclasses: Pvalb (incl chandelier) + Sst (incl Chodl)">+MGE</button>
    <button class="lin-btn" data-lin="CGE" title="Select CGE-derived subclasses: Vip + Lamp5 + Sncg + Serpinf1 (+ Lamp5 Lhx6)">+CGE</button>
    <button class="lin-btn" data-lin="LGE" title="Select LGE-derived subclasses (rare in cortex; mostly striatal)">+LGE</button>
  </div>
  <div class="controls-row">
    <label class="rank-label" title="Number of singular components to keep in the recompute (1–3). Lower rank collapses axes: rank=2 puts all points on the PC1×PC2 plane (z=0); rank=1 puts them on the PC1 axis.">rank
      <input id="rank-input" type="number" min="1" max="3" value="3" step="1"></label>
    <button id="recompute-btn" title="Refit PCA on the panel HVG, using only the checked-subtype cells.">Replot with gene panel</button>
    <button id="recompute-genes-btn" title="Refit PCA using only the genes currently shown in the right biplot (gene set ∩ mean/std/metabolism ∩ recoverability filters), on the checked-subtype cells.">Replot with shown genes</button>
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
    <label class="ribo-toggle" title="Subtract each gene's linear fit on pct_ribo before the recompute. Equivalent to projecting expression onto the subspace orthogonal to %ribo, so PCA/UMAP/diffmap see only the residual (non-metabolic) variance. NMF clips negative residuals to 0.">
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
        <button class="order-btn active" data-axis="0">PC1</button>
        <button class="order-btn" data-axis="1">PC2</button>
        <button class="order-btn" data-axis="2">PC3</button>
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
const VIZ_METHOD = 'PCA';            // axis label shown in both plot titles
let titleCellColor = 'subtype';      // what the cells are coloured by
let titleGeneRef   = 'strongest PC'; // what the genes are coloured by
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

// ---- Save figure: PNG of the current cell plot, camera view + colouring ----
// dark=true themes backgrounds/text/axes/legend for a dark background (data
// colours are kept — viridis/magma/subtype palettes read well on both).
function saveFigure(dark) {{
  if (typeof cellPlot === 'undefined' || !cellPlot || !cellPlot.data || !cellPlot.data[POINTS_TRACE]) {{ alert('Figure export not available.'); return; }}
  const rankEl = document.getElementById('rank-input');
  const rank = Math.max(1, Math.min(3, parseInt(rankEl ? rankEl.value : 3) || 3));
  const src = cellPlot.data[POINTS_TRACE];
  const title = 'Cells plotted on ' + VIZ_METHOD + ' axes · coloured by ' + titleCellColor
    + ' · n=' + activeCellCount().toLocaleString();
  const col = src.marker && src.marker.color;
  const colors = Array.isArray(col) ? col : src.x.map(() => col);
  const is3d = rank >= 3;                        // rank 1/2 -> drop the 3rd axis
  const PL = {{magma:'Magma', viridis:'Viridis'}};
  const BG = dark ? '#0e1116' : '#ffffff', FG = dark ? '#eaeaea' : '#1a1a1a', GRID = dark ? '#39404d' : '#e6e6e6';
  const traces = [];
  const layout = {{ title:{{text:title, font:{{size:15, color:FG}}, x:0.5, xanchor:'center', y:0.98}},
    margin:{{l:8,r:8,t:48,b:78}}, paper_bgcolor:BG, plot_bgcolor:BG, font:{{color:FG}}, showlegend:false }};
  const fc = window.figColor;
  if (fc && fc.kind === 'grad') {{
    const pts = {{mode:'markers', x:src.x, y:src.y, marker:{{size:is3d?3:5, color:colors}}, hoverinfo:'skip', showlegend:false}};
    if (is3d) {{ pts.type='scatter3d'; pts.z=src.z; }} else pts.type='scattergl';
    traces.push(pts);
    // expression uses the verified per-cohort unit; QC titles (total counts,
    // % ribosomal, genes detected, …) are already self-describing.
    const unit = /express/i.test(fc.title) ? ' (' + EXPR_UNIT + ')' : '';
    const cb = {{mode:'markers', x:[src.x[0]], y:[src.y[0]], opacity:0, hoverinfo:'skip', showlegend:false,
      marker:{{color:[fc.lo], colorscale:PL[fc.palette]||'Viridis', cmin:fc.lo, cmax:fc.hi, showscale:true,
        colorbar:{{orientation:'h', x:0.5, xanchor:'center', y:-0.06, yanchor:'top', len:0.55, thickness:16,
          outlinecolor:GRID, tickfont:{{color:FG}}, title:{{text:fc.title + unit, side:'bottom', font:{{color:FG}}}}}}}}}};
    if (is3d) {{ cb.type='scatter3d'; cb.z=[src.z[0]]; }} else cb.type='scattergl';
    traces.push(cb);
  }} else {{
    let items;
    if (fc && fc.kind === 'cats') items = fc.items;
    else items = Object.keys(subtype_palette).map(c => ({{color:subtype_palette[c], label:c}}));
    const groups = {{}};
    for (let i = 0; i < colors.length; i++) {{ const c = colors[i]; (groups[c] = groups[c] || []).push(i); }}
    const used = {{}};
    const mkTrace = (idx, displayColor, name) => {{
      const tr = {{mode:'markers', name: name || '', x: idx.map(i=>src.x[i]), y: idx.map(i=>src.y[i]),
        marker:{{size:is3d?3:5, color:displayColor}}, showlegend: !!name, hoverinfo:'skip'}};
      if (is3d) tr.z = idx.map(i=>src.z[i]);
      tr.type = is3d ? 'scatter3d' : 'scattergl'; traces.push(tr);
    }};
    // labelled categories first, in the key's own order (layers go superficial→deep)
    items.forEach(it => {{ if (groups[it.color]) {{ mkTrace(groups[it.color], it.color, it.label); used[it.color] = 1; }} }});
    // leftover groups: a fully-transparent colour (alpha 0, e.g. no-layer cells) can
    // blank scatter3d during toImage on some GPUs — render it grey (theme-aware) and label it.
    const isLayer = /layer|microdiss/i.test((fc && fc.title) || '');
    const NLGREY = dark ? '#6b7280' : '#c9c9c9';
    Object.keys(groups).forEach(c => {{
      if (used[c]) return;
      const transp = /,\s*0\s*\)\s*$/.test(String(c));
      mkTrace(groups[c], transp ? NLGREY : c, transp ? (isLayer ? 'Not microdissected' : 'unassigned') : '');
    }});
    layout.showlegend = true;
    layout.legend = {{orientation:'h', x:0.5, xanchor:'center', y:-0.02, yanchor:'top', font:{{size:10, color:FG}}, itemsizing:'constant'}};
  }}
  if (is3d) {{
    let cam; try {{ cam = cellPlot._fullLayout.scene._scene.getCamera(); }} catch(e) {{ cam = (cellPlot.layout.scene||{{}}).camera; }}
    const ax = t => ({{title:{{text:t, font:{{color:FG}}}}, tickfont:{{color:FG}}, gridcolor:GRID, zerolinecolor:GRID, backgroundcolor:BG, showbackground:true, color:FG}});
    layout.scene = {{ camera:cam, aspectmode:'data', xaxis:ax(VIZ_METHOD+' 1'), yaxis:ax(VIZ_METHOD+' 2'), zaxis:ax(VIZ_METHOD+' 3') }};
  }} else {{
    const ax2 = t => ({{title:{{text:t, font:{{color:FG}}}}, tickfont:{{color:FG}}, gridcolor:GRID, zerolinecolor:GRID, zeroline:false, linecolor:GRID}});
    layout.xaxis = ax2(VIZ_METHOD+' 1'); layout.yaxis = ax2(VIZ_METHOD+' 2');
  }}
  Plotly.toImage({{data:traces, layout:layout}}, {{format:'png', width:1000, height:820, scale:2}}).then(url => {{
    const a = document.createElement('a'); a.href = url; a.download = figFilename(rank, dark); a.click();
  }}).catch(e => {{ alert('Export failed: ' + e); }});
}}
// filename encodes the current settings (colouring, subset, gene restrictions) in plaintext
function figFilename(rank, dark) {{
  const parts = [VIZ_METHOD, titleCellColor, 'rank' + rank, dark ? 'dark' : 'light'];
  if (typeof activeSet !== 'undefined') parts.push('genes-' + activeSet);
  try {{
    const sel = Array.from(selectedSubtypes()), total = subtypeCheckboxes ? subtypeCheckboxes.length : sel.length;
    if (sel.length === 0) parts.push('no-subtypes');
    else if (sel.length >= total) parts.push('all-subtypes');
    else if (sel.length <= 8) parts.push('subtypes-' + sel.join('+'));
    else parts.push(sel.length + 'of' + total + '-subtypes');
  }} catch (e) {{}}
  if (typeof activeRegion !== 'undefined' && activeRegion && activeRegion !== 'both') parts.push('region-' + activeRegion);
  try {{
    const ab = document.querySelectorAll('.ag-btn');
    if (typeof activeAges !== 'undefined' && activeAges && ab.length && activeAges.size > 0 && activeAges.size < ab.length)
      parts.push('ages-' + Array.from(activeAges).join('+'));
  }} catch (e) {{}}
  const mn = meanSlider, sd = stdSlider, rb = riboSlider, rg = regressRiboToggle;
  if (mn && +mn.value > +mn.min + 1e-9) parts.push('minmean' + (+mn.value).toFixed(2));
  if (sd && +sd.value > +sd.min + 1e-9) parts.push('minstd' + (+sd.value).toFixed(2));
  if (rb && +rb.value < 1) parts.push('metabfilt' + (+rb.value).toFixed(2));
  if (rg && rg.checked) parts.push('regress-ribo');
  return parts.join('_').replace(/[^A-Za-z0-9+._-]+/g, '-').replace(/-+/g, '-').replace(/^[-_]+|[-_]+$/g, '').slice(0, 200) + '.png';
}}
(function() {{  // Save-figure buttons (light / dark), sit just left of Copy-link
  const mk = (label, tip, dark, rightPx) => {{
    const b = document.createElement('button'); b.className = 'save-fig-btn'; b.textContent = label; b.title = tip;
    b.style.cssText = 'position:fixed;top:10px;right:calc(max(10px, 50vw - 845px) + ' + rightPx + 'px);z-index:1000;'
      + 'font-size:12px;padding:5px 10px;border:1px solid #cdbf9f;border-radius:7px;background:rgba(255,255,255,.9);'
      + 'cursor:pointer;font-weight:600;color:#5a4326;box-shadow:0 1px 3px rgba(0,0,0,.08)';
    b.onclick = () => saveFigure(dark); document.body.appendChild(b);
  }};
  mk('⤓ Save (dark)',  'Download a PNG (dark background) of the current cell plot, view and colouring',  true,  104);
  mk('⤓ Save (light)', 'Download a PNG (light background) of the current cell plot, view and colouring', false, 210);
}})();

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
  const _ageTag = (typeof cell_age !== 'undefined' && cell_age) ? ' · age ' + cell_age[i] : '';
  status.innerHTML = '<b style="color:' + cell_dom_color[i] + '">Cell #' + i
    + '</b> <span style="color:#555">(' + cell_subtype[i] + _ageTag + ')</span> &nbsp; '
    + 'PC1,2,3 = (' + s[0].toFixed(2) + ', ' + s[1].toFixed(2) + ', ' + s[2].toFixed(2) + ') &nbsp; '
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
    + 'loadings PC1,2,3 = (' + L[0].toFixed(2) + ', ' + L[1].toFixed(2) + ', ' + L[2].toFixed(2) + ') &nbsp; '
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
  status.innerHTML = 'Reset. Hover a cell or gene to colour by expression and reveal PC projection.'; clearColorKey();
  titleCellColor = 'subtype'; titleGeneRef = 'strongest PC'; refreshTitles();
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
  // update the colour key + saved-figure state to THIS gene (hover-colouring does
  // this; search previously did not, so the saved colour bar was stale/missing).
  let _lo = Infinity, _hi = -Infinity;
  for (const v of visible) {{ if (v < _lo) _lo = v; if (v > _hi) _hi = v; }}
  if (visible.length) setColorKeyGradient(gene_name[j] + ' expression', 'magma', _lo/EXPR_SCALE, _hi/EXPR_SCALE, v => v.toFixed(2));
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
// the panel covariance, equivalent to PCA but cheaper for large K), then score
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
    if (sel.has(cell_subtype[i]) && regionAllowed(i) && ageAllowed(i)) cellSel.push(i);
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

// ---- "Number of similar genes": local density in the gene-embedding space ----
// Gene analogue of "similar cells". Each SHOWN gene gets a Gaussian-kernel
// effective neighbour count among the OTHER shown genes, using the current 3D
// gene biplot coords (gene_x/gene_y/gene_z). Bandwidth h^2 = median squared
// pairwise distance among up to ~400 shown genes (median heuristic). Colours are
// built as a PLAIN Array (typed arrays coerce colour strings to NaN).
function colorBySimilarGenes() {{
  let shown = (typeof visibleGeneIdx === 'function') ? visibleGeneIdx()
            : (typeof shownGeneIdx === 'function') ? shownGeneIdx() : null;
  if (!shown || shown.length < 2) shown = Array.from(gene_name.keys());
  const G = shown.length;
  const inShown = new Uint8Array(gene_name.length);
  for (let i = 0; i < G; i++) inShown[shown[i]] = 1;
  // Anchors: subsample shown genes when many (median-heuristic + density sums).
  const A = Math.min(G, 1200), step = G / A; const anc = new Int32Array(A);
  for (let a = 0; a < A; a++) anc[a] = shown[Math.floor(a * step)];
  // Bandwidth h^2 = median squared distance among up to 400 anchors.
  const Bn = Math.min(A, 400); const dd = [];
  for (let a = 0; a < Bn; a++) {{ const ga = anc[a];
    for (let b = a + 1; b < Bn; b++) {{ const gb = anc[b];
      const dx = gene_x[ga] - gene_x[gb], dy = gene_y[ga] - gene_y[gb], dz = gene_z[ga] - gene_z[gb];
      dd.push(dx*dx + dy*dy + dz*dz); }} }}
  dd.sort((x, y) => x - y);
  const h2 = Math.max(dd.length ? dd[dd.length >> 1] : 1, 1e-12), inv2h2 = 1 / (2 * h2), scale = G / A;
  // density[g] = scale * sum over anchors exp(-d^2 / (2 h^2)) for shown genes.
  const dens = new Float64Array(gene_name.length);
  for (let i = 0; i < G; i++) {{ const g = shown[i]; let acc = 0;
    const gx = gene_x[g], gy = gene_y[g], gz = gene_z[g];
    for (let a = 0; a < A; a++) {{ const ga = anc[a];
      const dx = gx - gene_x[ga], dy = gy - gene_y[ga], dz = gz - gene_z[ga];
      acc += Math.exp(-(dx*dx + dy*dy + dz*dz) * inv2h2); }}
    dens[g] = acc * scale; }}
  // Map shown-gene densities to viridis (plain Array), grey the rest.
  const vals = []; for (let i = 0; i < G; i++) vals.push(dens[shown[i]]);
  const palette = valuesToViridis(vals);
  let vi = 0;
  const colors = [];
  for (let j = 0; j < gene_name.length; j++) colors.push(inShown[j] ? palette[vi++] : '#e6e6e6');
  Plotly.restyle(genePlot, {{'marker.color': [colors]}}, [POINTS_TRACE]);
  let lo = Infinity, hi = -Infinity;
  for (const v of vals) {{ if (v < lo) lo = v; if (v > hi) hi = v; }}
  setColorKeyGradient('similar genes (' + G + ' shown)', 'viridis', lo, hi, v => v.toFixed(1));
}}
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
// Principal-curve pseudotime (Hastie-Stuetzle arc length), ported from
// bayesian-continua/src/ordering.py. Given a list of cell indices and an
// optional root cell (global index, or -1/null for unrooted), returns a plain
// Array of lambda in [0,1] indexed 0..cellIdxs.length-1 (aligned to cellIdxs).
// Embedding coords come from cell_score[i] (length-3 [PC1,PC2,PC3]; missing
// dims treated as 0). Uses NC=min(n,200) moving-average curve anchors for speed.
// Returns a PLAIN Array (not Float64Array) so the viridis colour path is safe.
function principalCurvePT(cellIdxs, root) {{
  const D = 3;
  const n = cellIdxs.length;
  const coord = (idx, k) => {{ const s = cell_score[idx]; return (s && s.length > k) ? s[k] : 0; }};
  // X: n x D embedding for these cells.
  const X = new Array(n);
  for (let a = 0; a < n; a++) {{
    const idx = cellIdxs[a]; const row = new Array(D);
    for (let k = 0; k < D; k++) row[k] = coord(idx, k);
    X[a] = row;
  }}
  const rootValid = (root !== null && root !== undefined && root >= 0);
  // Position of the root within cellIdxs (for orientation), -1 if not in set.
  let rootPos = -1;
  if (rootValid) {{ for (let a = 0; a < n; a++) if (cellIdxs[a] === root) {{ rootPos = a; break; }} }}
  // init lambda.
  let lam = new Array(n);
  if (rootValid) {{
    // Euclidean distance in cell_score from the root cell.
    const r = new Array(D); for (let k = 0; k < D; k++) r[k] = coord(root, k);
    for (let a = 0; a < n; a++) {{ let s = 0; for (let k = 0; k < D; k++) {{ const d = X[a][k] - r[k]; s += d * d; }} lam[a] = Math.sqrt(s); }}
  }} else {{
    for (let a = 0; a < n; a++) lam[a] = X[a][0];   // PC1
  }}
  const normalize = arr => {{
    let lo = Infinity, hi = -Infinity;
    for (const v of arr) {{ if (v < lo) lo = v; if (v > hi) hi = v; }}
    const ptp = (hi - lo) + 1e-9;
    return arr.map(v => (v - lo) / ptp);
  }};
  if (n < 15) return normalize(lam);
  const w = Math.max(5, Math.floor(0.08 * n));
  const NC = Math.min(n, 200);
  const nIter = 8;
  for (let it = 0; it < nIter; it++) {{
    lam = normalize(lam);
    // order cells by lambda.
    const o = Array.from({{length: n}}, (_, a) => a);
    o.sort((a, b) => lam[a] - lam[b]);
    // anchors: NC moving-average points along the order.
    const anchors = new Array(NC);
    for (let j = 0; j < NC; j++) {{
      const p = (NC === 1) ? 0 : Math.round(j * (n - 1) / (NC - 1));
      const lo = Math.max(0, p - w), hi = Math.min(n - 1, p + w);
      const c = new Array(D).fill(0); let cnt = 0;
      for (let q = lo; q <= hi; q++) {{ const row = X[o[q]]; for (let k = 0; k < D; k++) c[k] += row[k]; cnt++; }}
      for (let k = 0; k < D; k++) c[k] /= cnt;
      anchors[j] = c;
    }}
    // arc length s[j] along the anchors.
    const s = new Array(NC); s[0] = 0;
    for (let j = 1; j < NC; j++) {{
      let d2 = 0; for (let k = 0; k < D; k++) {{ const d = anchors[j][k] - anchors[j - 1][k]; d2 += d * d; }}
      s[j] = s[j - 1] + Math.sqrt(d2);
    }}
    // project each cell to nearest anchor -> lambda = s[nearest].
    for (let a = 0; a < n; a++) {{
      const row = X[a]; let best = 0, bestD = Infinity;
      for (let j = 0; j < NC; j++) {{
        const ac = anchors[j]; let d2 = 0;
        for (let k = 0; k < D; k++) {{ const d = row[k] - ac[k]; d2 += d * d; }}
        if (d2 < bestD) {{ bestD = d2; best = j; }}
      }}
      lam[a] = s[best];
    }}
  }}
  lam = normalize(lam);
  // Orient so the root sits at t=0.
  if (rootPos >= 0 && lam[rootPos] > 0.5) {{ for (let a = 0; a < n; a++) lam[a] = 1 - lam[a]; }}
  return lam;
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
  setTimeout(() => {{ try {{ colorBySimilarCells(); colorBySimilarGenes(); }} finally {{ b.disabled=false; }} }}, 30);
}});
document.getElementById('qc-pt').addEventListener('click', () => {{
  // Pseudotime colour = principal-curve arc length (Hastie-Stuetzle) over the
  // active cells → viridis. Unrooted here (PC1-seeded); no root picker in this app.
  const active = [];
  for (let i = 0; i < cell_active.length; i++) if (cell_active[i]) active.push(i);
  const lam = principalCurvePT(active, -1);
  const arr = new Array(cell_score.length).fill(0);
  for (let a = 0; a < active.length; a++) arr[active[a]] = lam[a];
  colorByQC(arr, 'pseudotime (principal curve)', v => v.toFixed(2));
}});

// Colour by region (V1 vs ALM). Carries per-cell `dissected_region` —
// VISp/V1 cells get orange, ALM cells purple, others grey. Greyed out for
// single-region cohorts.
const REGION_COLORS = {{ 'VISp': '#ff7f0e', 'V1': '#ff7f0e', 'ALM': '#9467bd' }};
const regionBtn = document.getElementById('qc-region');
if (regionBtn) {{
  const _regions = (typeof cell_region !== 'undefined' && cell_region) ? cell_region : [];
  const uniqRegions = Array.from(new Set(_regions));
  if (uniqRegions.length < 2) {{
    regionBtn.remove();   // no region info or single-region cohort → hide entirely
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
// Ages are E11.5/E12/.../P56 — parse into a numeric ordering and map to a
// viridis gradient (purple → blue → cyan → green → yellow) for monotone
// developmental time.
function ageToNumber(s) {{
  // E11.5 → -7.5 (E = embryonic, days before birth set to negative).
  // P3 → 3, P56 → 56.
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
      colorByQC(ageNum, 'age (E neg / P pos days)', v => {{
        if (v < 0) return 'E' + (-v);
        return 'P' + v;
      }});
    }});
  }}
}}

// Colour by dissected cortical layer. Tasic 2018 metadata carries values
// like 'L1', 'L2/3', 'L4', 'L5', 'L6', 'L6b' plus compound dissections like
// 'L2/3-L4'. We encode each as a viridis-depth scalar (L1=1 → L6b=6.5; mean
// of endpoints for compound), then re-use colorByQC. Hidden when None or
// single-valued.
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
    // depth-proportional viridis, but coloured per DISCRETE layer (one colour
    // per layer) with a categorical key rather than a continuous gradient.
    let lo = Infinity, hi = -Infinity;
    depths.forEach((d, i) => {{ if (cell_active[i] && !isNaN(d)) {{ if (d < lo) lo = d; if (d > hi) hi = d; }} }});
    const range = (hi > lo) ? (hi - lo) : 1;
    const depthColor = d => viridis[Math.max(0, Math.min(255, Math.round(255 * (d - lo) / range)))];
    const colors = depths.map((d, i) => {{
      if (!cell_active[i]) return '#dddddd';
      if (!isNaN(d)) return depthColor(d);
      return GREY_NO_LAYER;                 // multi-layer / pan dissection: faint
    }});
    Plotly.restyle(cellPlot, {{'marker.color': [colors]}}, [POINTS_TRACE]);
    // discrete per-layer key: distinct layers present, ordered superficial→deep.
    // Drives both the on-page colour key and the saved-figure legend.
    const seen = new Map();
    depths.forEach((d, i) => {{ if (cell_active[i] && !isNaN(d) && !seen.has(d)) seen.set(d, layerLabel(d)); }});
    const items = Array.from(seen.keys()).sort((a, b) => a - b).map(d => ({{color: depthColor(d), label: seen.get(d)}}));
    const greyN = depths.filter((d, i) => cell_active[i] && isNaN(d)).length;
    titleCellColor = 'layer of microdissection'; refreshTitles();
    status.innerHTML = greyN ? ('<i>' + greyN.toLocaleString() + ' cells from multi-layer / pan dissections shown faint.</i>') : '';
    if (items.length) setColorKeyCats('layer (microdissection)', items); else clearColorKey();
  }});
}}

// ============================================================================
// Subtype-subset PCA recompute
// ============================================================================
const subtypeCheckboxes = Array.from(document.querySelectorAll('.subt-chk input[type="checkbox"]'));
// Checkboxes belonging to the currently-active granularity panel. For
// single-level cohorts the one panel is always active, so this == all boxes.
function activeCbs() {{
  return subtypeCheckboxes.filter(cb => {{
    const p = cb.closest('.subt-panel');
    return !p || p.classList.contains('active');
  }});
}}
// Granularity toggle (dual-resolution cohorts): switch the subtype menu AND the
// per-cell subtype labels/colours between the finer and coarser level.
const levelToggleBtn = document.getElementById('level-toggle');
if (levelToggleBtn) {{
  const def = levelToggleBtn.dataset.default, alt = levelToggleBtn.dataset.alt;
  levelToggleBtn.addEventListener('click', () => {{
    activeLevel = (activeLevel === def) ? alt : def;
    document.querySelectorAll('.subt-panel').forEach(p =>
      p.classList.toggle('active', p.dataset.level === activeLevel));
    const lvl = SUBTYPE_LEVELS[activeLevel];
    cell_subtype = _decodeSubtype(lvl.cats, lvl.idx);
    subtype_palette = lvl.palette;
    cell_default_colors = cell_subtype.map(s => subtype_palette[s] || '#888888');
    activeCbs().forEach(cb => cb.checked = true);          // show all in new level
    levelToggleBtn.textContent = '▸ show ' + (activeLevel === def ? alt : def) + ' level';
    invalidateGeneStruct();
    document.getElementById('reset-btn').click();          // repaint by new colours
  }});
}}
// Per-subclass group-level all/none buttons (in the AllInhib cohort + future
// multi-subclass cohorts; single-subclass cohorts still get a group header
// that conveniently toggles its whole subclass).
document.querySelectorAll('.grp-toggle').forEach(btn => {{
  btn.addEventListener('click', () => {{
    const grp = btn.dataset.grp;
    const want = btn.dataset.action === 'all';
    activeCbs().forEach(cb => {{
      if (cb.dataset.grp === grp) cb.checked = want;
    }});
  }});
}});
document.getElementById('subt-all').addEventListener('click', () => {{
  activeCbs().forEach(cb => cb.checked = true);
}});
document.getElementById('subt-none').addEventListener('click', () => {{
  activeCbs().forEach(cb => cb.checked = false);
}});

// Developmental-lineage shortcuts. MGE → Pvalb (incl chandelier) + Sst (incl
// Chodl). CGE → Vip + Lamp5 (incl Lhx6) + Sncg + Serpinf1. LGE rarely makes it
// to cortex; matches anything tagged as such. Each button checks every
// subtype belonging to one of the named subclasses (subtype's data-grp
// attribute holds the cell_subclass).
const LINEAGE_SUBCLASSES = {{
  MGE: new Set(['Pvalb', 'Pvalb chandelier', 'Sst', 'Sst Chodl']),
  CGE: new Set(['Vip', 'Lamp5', 'Lamp5 Lhx6', 'Sncg', 'Serpinf1']),
  LGE: new Set(['LGE']),  // placeholder — matches any 'LGE'-prefixed subclass
}};
document.querySelectorAll('.lin-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    const wanted = LINEAGE_SUBCLASSES[btn.dataset.lin] || new Set();
    activeCbs().forEach(cb => {{
      if (wanted.has(cb.dataset.grp)) cb.checked = true;
    }});
  }});
}});

function selectedSubtypes() {{
  const out = new Set();
  activeCbs().forEach(cb => {{ if (cb.checked) out.add(cb.dataset.sub); }});
  return out;
}}

// Any change to the cell selection makes a previously-computed gene-structure
// score stale, so drop it (and disable its filter) when the selection changes.
subtypeCheckboxes.forEach(cb => cb.addEventListener('change', invalidateGeneStruct));
document.querySelectorAll('.grp-toggle, #subt-all, #subt-none, .lin-btn, .rg-btn, .ag-btn, .ag-btn-all')
  .forEach(b => b.addEventListener('click', invalidateGeneStruct));

// Region toggle ("both" → no filter, else only cells with that dissected_region)
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

// Age toggle — independent per-age chips, used as an AND filter on top of
// subtype + region. Default: every chip on; user clicks to drop ages.
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
  // A: m x n (array of Float64Array rows). Returns {{U, S, V}} for top-K PCA via
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

function recomputePCA(basisIdx, basisLabel) {{
  // basisIdx is the list of gene indices used as the PCA's basis ("panel" of
  // genes the PCA is *fit on*). Default = the panel HVG. Pass the currently-
  // visible gene indices to refit on whatever the gene biplot is showing.
  basisIdx = basisIdx || panel_idx;
  basisLabel = basisLabel || 'panel HVG';
  const t0 = performance.now();
  const sel = selectedSubtypes();
  // Cell index list of selected cells (subtype-checked AND region-allowed).
  const cellSel = [];
  for (let i = 0; i < cell_subtype.length; i++) {{
    if (sel.has(cell_subtype[i]) && regionAllowed(i) && ageAllowed(i)) cellSel.push(i);
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

  // Power-iteration PCA top-3 on Zp (just the embedding rank; variance bars
  // were removed from the UI so we no longer compute the top-10 spectrum).
  const K_compute = Math.min(3, n_panel);
  const {{U, S, V}} = powerIterTopK(Zp, K_compute);

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
      const v = readVal(cellSel[ii], j);
      s += v; ss += v*v;
    }}
    const mean = s / m;
    const stdv = Math.sqrt(Math.max(ss / m - mean*mean, 1e-18));
    const accs = [0, 0, 0];
    for (let ii = 0; ii < m; ii++) {{
      const z = (readVal(cellSel[ii], j) - mean) / stdv;
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
    + ' cells, ' + n_panel + ' ' + basisLabel + ' (rank ' + K_emb + ', ' + dt + 's) — '
    + 'poles: ' + newPoleTop.map((g, p) => POLE_NAMES_[p] + '=' + g).join(', ');
  lastHoveredCell = null; lastHoveredGene = null;
  // 8. heatmap reflects the new PC1 ordering on the new active cell set
  renderHeatmap();
  // 9. expose the fresh basis so an overlay (e.g. patch-seq MET cells) can
  //    re-project its own cells onto the new PCA and move with the recompute.
  window.__svdBasis = {{ basisIdx: basisIdx, panelMean: panelMean, panelStd: panelStd,
                         V: V, kEmb: K_emb, cmax: cmax }};
  document.dispatchEvent(new CustomEvent('svd-recomputed'));}}

document.getElementById('recompute-btn').addEventListener('click', () => {{
  const btn = document.getElementById('recompute-btn');
  btn.disabled = true; recomputeStatus.textContent = 'computing…';
  // Defer to next frame so the disabled state actually renders
  setTimeout(() => {{ try {{ recomputePCA(panel_idx, 'panel HVG'); }} finally {{ btn.disabled = false; }} }}, 30);
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
    try {{ recomputePCA(visible, 'shown genes'); }}
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
// Cell-ordering state for the heatmap (read by renderHeatmap).
let heatmapOrderAxis = 0;       // 0=PC1, 1=PC2, 2=PC3
let heatmapGroupByType = false; // if true, stratify by cell_subtype before sorting
let celltypeOrder = null;       // when grouping by cell-type: explicit user-set order
                                 // (null = use axis-mean of subtype). Array of subtype strings.

function renderHeatmap() {{
  sizeHeatmapCanvas();
  const ctx = heatCanvas.getContext('2d', {{alpha: false}});
  const W = heatCanvas.width, H = heatCanvas.height;
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, W, H);

  // 1. Active cells ordered by the chosen axis (optionally grouped by subtype first)
  let activeIdx = [];
  for (let i = 0; i < cell_active.length; i++) if (cell_active[i]) activeIdx.push(i);
  const axis = heatmapOrderAxis;
  // Pseudotime axis (-1): order along the first principal embedding axis.
  // For PCA that's PC1; we just project onto cell_score[i][0].
  const sortKey = (axis === -1)
    ? (i => cell_score[i][0])
    : (i => cell_score[i][axis]);
  if (heatmapGroupByType) {{
    // Stratify by cell_subtype; sort subtype groups by mean axis value
    // (or by celltypeOrder if the user has manually re-arranged them).
    const byType = new Map();
    for (const i of activeIdx) {{
      const t = cell_subtype[i];
      if (!byType.has(t)) byType.set(t, []);
      byType.get(t).push(i);
    }}
    let typeOrder;
    if (celltypeOrder && celltypeOrder.length) {{
      typeOrder = celltypeOrder.filter(t => byType.has(t));
      // Append any active types not yet in celltypeOrder (new from recompute)
      for (const t of byType.keys()) {{
        if (!typeOrder.includes(t)) typeOrder.push(t);
      }}
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
    // Stash so the chip row + strip painter use the same order
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
  // Paint the cell-type colour strip below the heatmap (one column per cell,
  // coloured by that cell's subtype). Same column width as the heatmap above.
  paintCellTypeStrip(activeIdx);
  // Show/hide the reorder chip row + sync chips when grouping is on.
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
  // strip the trailing (GO:0001234) ID for the displayed label.
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
// Wire heatmap-control buttons.
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
  // Drop any prior manual reorder when toggling off so the next "group by"
  // starts from axis-mean order again.
  if (!heatmapGroupByType) celltypeOrder = null;
  renderHeatmap();
}});

// ---- Cell-type colour strip + reorder chips ------------------------------
function paintCellTypeStrip(activeIdx) {{
  const cv = document.getElementById('celltype-strip-canvas');
  if (!cv) return;
  // Match strip width to heatmap canvas
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
  // HTML5 drag-and-drop reorder
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
      // Insert dragSrc before this chip
      chipsEl.insertBefore(dragSrc, chip);
      // Commit new order to global state
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
// initial view shows the subset's own PCA basis instead of the full-cohort
// fit. Skipped when a Copy-Link URL hash is present (that snippet handles
// its own recompute trigger).
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
