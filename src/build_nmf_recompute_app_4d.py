#!/usr/bin/env python
"""4-archetype NMF explorer with in-browser NMF recompute on a subset of cells/genes.

Same controls layout as build_svd_recompute_app_3d.py (subtype checkboxes,
gene filter sliders, recompute buttons, sample/region toggles, heatmap, GO
bars, copy-link) but the embedding is NMF: cells live inside a 3D
tetrahedron whose vertices are 4 archetypes (W rows are normalised to sum
to 1 then barycentric-projected to cartesian), and genes live inside a
second tetrahedron (H column barycentrics).

Recompute uses multiplicative-update NMF (Lee & Seung) in pure JS — same
one-click mutation pattern as the SVD recompute, no server / Pyodide.

Usage:  python build_nmf_recompute_app_4d.py [GROUP]   # default: base GROUP_NAME
Output: notebooks/{group}_nmf_recompute_explorer_4d.html
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import plotly.graph_objects as go
from plotly.io import to_html
from bokeh.palettes import Magma256, Viridis256, Category20, Set3, Set1, Category10
from sklearn.decomposition import non_negative_factorization

import build_lamp5_archetype_app_4d as base

GROUP_NAME  = base.GROUP_NAME
SLUG        = base.SLUG
OUT         = os.path.join(base.ROOT, 'notebooks',
                           f'{SLUG}_nmf_recompute_explorer_4d.html')
K_ARCH      = 4
ARCH_NAMES  = [f'A{k+1}' for k in range(K_ARCH)]
ARCH_COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']
# Regular tetrahedron vertices (centered, unit-radius). Cells / genes are
# placed inside this tetrahedron via barycentric (W normalised) → cartesian.
TET_V = np.array([
    [ 1,  1,  1],
    [ 1, -1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
], dtype=float) / np.sqrt(3)


def nmf_fit(X, K, seed=42, max_iter=400, tol=1e-4):
    """Sklearn-CD NMF, non-negative inputs only. Returns (W, H) such that X≈W@H."""
    X_nn = np.clip(X, 0, None).astype(np.float32)
    W, H, _ = non_negative_factorization(
        X_nn, n_components=K, init='nndsvd', beta_loss='frobenius',
        solver='cd', max_iter=max_iter, tol=tol, random_state=seed)
    return W, H


def barycentric_to_xyz(W):
    """W: n×K non-negative. Returns n×3 cartesian inside the tetrahedron."""
    W_norm = W / (W.sum(1, keepdims=True) + 1e-12)
    return W_norm @ TET_V


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

    # ---- initial NMF on the panel HVG matrix (non-negative log-CPM).
    # K=4 archetypes; W is n_cells × K (cell archetype loadings), H is K × n_panel
    # (archetype gene profiles). For the full broader-gene matrix we then NNLS-
    # project: H_all is K × n_genes (each gene's archetype profile).
    Xp = X_keep[:, in_panel]                      # n × n_panel, non-negative
    W, H_panel = nmf_fit(Xp, K_ARCH)
    # Project broader gene matrix onto archetype space: H_all = (W^+ X_all)
    # via NNLS for non-negativity (one solve per gene).
    from scipy.optimize import nnls
    H_all = np.zeros((K_ARCH, n_genes), dtype=np.float32)
    for j in range(n_genes):
        H_all[:, j], _ = nnls(W, X_keep[:, j])
    print(f'  initial NMF on {Xp.shape[0]} cells × {Xp.shape[1]} panel genes (K={K_ARCH})')

    cell_xyz = barycentric_to_xyz(W)            # n × 3 inside tetrahedron
    # Gene barycentrics: H column normalised — each gene has K archetype weights.
    gene_xyz = barycentric_to_xyz(H_all.T)      # n_genes × 3
    # Dominant-archetype index per cell / gene (for default colouring)
    cell_arch = W.argmax(1)
    gene_arch = H_all.argmax(0)

    # Vertex labels: top panel gene per archetype (peak gene)
    panel_genes_list = [g for g, k in zip(gene_names, in_panel) if k]
    vertex_top = [panel_genes_list[int(np.argmax(H_panel[k]))] for k in range(K_ARCH)]

    # Top genes per archetype for GO enrichment: top-N panel genes by H_panel weight.
    GO_TOP_PER_AXIS = 60
    top_genes_per_axis = []
    for k in range(K_ARCH):
        order = np.argsort(H_panel[k])[::-1][:GO_TOP_PER_AXIS]
        top_genes_per_axis.append({
            'name': ARCH_NAMES[k],
            'pos': [panel_genes_list[i] for i in order],
            'neg': [],  # NMF has no negative pole
        })
    go_axes = []  # GO bars removed from UI; placeholder for JS data
    cats = sorted(set(subs.tolist()))
    subtype_palette = base.build_subtype_palette(cats)
    cell_color_default = [subtype_palette[s] for s in subs]
    gene_color_default = [ARCH_COLORS[int(k)] for k in gene_arch]
    cell_dom_color     = [ARCH_COLORS[int(k)] for k in cell_arch]
    gene_dom_color     = gene_color_default
    # Per-cell / per-gene archetype loadings (K values each, for hover/heatmap order).
    cell_load = W.round(4).tolist()             # n × K
    gene_load = H_all.T.round(4).tolist()       # n_genes × K
    # cell_score / gene_loading reuse the SVD-recompute JS names; they hold the
    # full K-vector now (K=4 instead of 3).
    cell_scores = W                             # alias for consistency below
    gene_load3 = H_all.T                        # alias for consistency below

    import base64
    EXPR_SCALE = 16
    _expr_q = np.clip(np.round(X_keep * EXPR_SCALE), 0, 255).astype(np.uint8)
    expr_b64 = base64.b64encode(_expr_q.tobytes()).decode('ascii')
    n_cells_emit = int(_expr_q.shape[0])
    n_genes_emit = int(_expr_q.shape[1])
    panel_idx = [j for j, p in enumerate(in_panel.tolist()) if p]

    # ---- figure construction: tetrahedron with 4 vertex markers, no cube axes.
    POLE = 1.22                                # vertex label radius
    # axis_ends still has 6 slots so JS code that indexes them stays compatible:
    # positions 0–3 are the 4 archetype vertices, positions 4–5 hidden.
    axis_ends = np.zeros((6, 3))
    axis_ends[:K_ARCH] = TET_V * POLE

    # Edges of the regular tetrahedron (6 lines).
    tet_edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

    def build_fig(xyz, colors, hover_text, title):
        ax_x, ax_y, ax_z = [], [], []
        for (i, j) in tet_edges:
            a, b = TET_V[i], TET_V[j]
            ax_x += [float(a[0]), float(b[0]), None]
            ax_y += [float(a[1]), float(b[1]), None]
            ax_z += [float(a[2]), float(b[2]), None]
        edge_trace = go.Scatter3d(x=ax_x, y=ax_y, z=ax_z, mode='lines',
                                  line=dict(color='lightgray', width=2),
                                  hoverinfo='skip', showlegend=False)
        vlab_active = [ARCH_NAMES[k] for k in range(K_ARCH)]
        vlab        = vlab_active + ['', '']    # padding to 6 entries (compat)
        vcolors     = list(ARCH_COLORS) + ['rgba(0,0,0,0)', 'rgba(0,0,0,0)']
        vertex_trace = go.Scatter3d(
            x=axis_ends[:, 0], y=axis_ends[:, 1], z=axis_ends[:, 2],
            mode='markers+text', marker=dict(size=4, color=vcolors),
            text=vlab, textposition='top center',
            textfont=dict(size=11, color='black'),
            hoverinfo='text', hovertext=vlab, showlegend=False)
        points_trace = go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers',
            marker=dict(size=4, color=colors, opacity=0.85, line=dict(width=0)),
            text=hover_text, hoverinfo='text', showlegend=False)
        loading_colors = ['#e0e0e0'] * K_ARCH + ['rgba(0,0,0,0)', 'rgba(0,0,0,0)']
        loading_trace = go.Scatter3d(
            x=axis_ends[:, 0], y=axis_ends[:, 1], z=axis_ends[:, 2], mode='markers',
            marker=dict(size=0,  color=loading_colors, opacity=0.0,
                        line=dict(width=0)),
            hoverinfo='text',
            hovertext=ARCH_NAMES + ['', ''], showlegend=False)
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
        f'W = ({cell_scores[i,0]:.2f}, {cell_scores[i,1]:.2f}, '
        f'{cell_scores[i,2]:.2f}, {cell_scores[i,3]:.2f}); '
        f'dom: {ARCH_NAMES[cell_arch[i]]}'
        for i in range(n_cells)]
    gene_hover_text = [
        f'<b>{gene_names[j]}</b>'
        + (' (panel HVG)' if in_panel[j] else ' (projected)')
        + f'<br>strongest: {ARCH_NAMES[gene_arch[j]]} ({vertex_top[gene_arch[j]]})<br>'
        + f'mean={mean_expr[j]:.2f}, std={std_expr[j]:.2f}<br>'
        + f'H = ({gene_load3[j,0]:.2f}, {gene_load3[j,1]:.2f}, '
        + f'{gene_load3[j,2]:.2f}, {gene_load3[j,3]:.2f})'
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
        f'<span style="display:inline-block;width:10px;height:10px;background:{ARCH_COLORS[k]};'
        f'margin-right:4px;border-radius:50%;"></span> {ARCH_NAMES[k]} ({vertex_top[k]}) &nbsp;&nbsp;'
        for k in range(K_ARCH))

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

    # Subtype checkbox row, grouped by cell_subclass. Each subclass gets a header
    # with select-all / deselect-all shortcuts, then its constituent subtypes
    # below as individual checkboxes. Single-subclass cohorts collapse to one
    # group (still useful: the header buttons act as bulk all/none for the group).
    subtype_counts = {c: int(np.sum(subs == c)) for c in cats}
    # Build subclass → ordered list of subtypes mapping
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
        # NMF tetrahedron padding: emit 6-slot arrays so legacy JS that indexes
        # 0..5 (signed-poles from the SVD template) still works. Slots 4-5 are
        # always invisible / empty for NMF.
        f"let pole_top = {json.dumps(vertex_top + ['', ''])};\n"
        f"const panel_idx = {json.dumps(panel_idx)};\n"
        f"const POLE_NAMES_  = {json.dumps(ARCH_NAMES + ['', ''])};\n"
        f"const POLE_COLORS_ = {json.dumps(list(ARCH_COLORS) + ['rgba(0,0,0,0)', 'rgba(0,0,0,0)'])};\n"
        f"const TET_V = {json.dumps(TET_V.tolist())};\n"
        f"const K_ARCH = {K_ARCH};\n"
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
<title>{GROUP_NAME} NMF recompute explorer</title>
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
.legend {{ flex: 0 0 auto; font-size: 11px; color: #444; margin-top: 4px; }}
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
<b>Pick which subtypes you want to fit the archetypes to</b>, then click <b>Replot</b>.
</div>

<div class="ctrl-box">
  <div class="ctrl-box-title">Plotting Method</div>
  <div class="controls-row">{base.viz_nav_html(SLUG, 'archetype')}</div>
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
    <span class="lin-sep">|</span>
    <button class="lin-btn" data-lin="MGE" title="Select MGE-derived subclasses: Pvalb (incl chandelier) + Sst (incl Chodl)">+MGE</button>
    <button class="lin-btn" data-lin="CGE" title="Select CGE-derived subclasses: Vip + Lamp5 + Sncg + Serpinf1 (+ Lamp5 Lhx6)">+CGE</button>
    <button class="lin-btn" data-lin="LGE" title="Select LGE-derived subclasses (rare in cortex; mostly striatal)">+LGE</button>
    <button id="recompute-btn" title="Refit NMF on the panel HVG, using only the checked-subtype cells.">Replot with gene panel</button>
    <button id="recompute-genes-btn" title="Refit NMF using only the genes currently shown in the right biplot (gene set ∩ mean/std/metabolism filters). With no gene filter active this is all genes.">Replot with shown genes</button>
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
        <button class="order-btn active" data-axis="0">A1</button>
        <button class="order-btn" data-axis="1">A2</button>
        <button class="order-btn" data-axis="2">A3</button>
        <button class="order-btn" data-axis="3">A4</button>
        <button class="order-btn" data-axis="-1" title="Order cells by dominant archetype then by max-archetype weight — a 'trajectory through archetypes' pseudotime.">Pseudotime</button>
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
const VIZ_METHOD = 'NMF archetype';
let titleCellColor = 'subtype';
let titleGeneRef   = 'dominant archetype';
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
    + 'W = (' + s[0].toFixed(2) + ', ' + s[1].toFixed(2) + ', ' + s[2].toFixed(2) + ', ' + s[3].toFixed(2) + ') &nbsp; '
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
    + 'H = (' + L[0].toFixed(2) + ', ' + L[1].toFixed(2) + ', ' + L[2].toFixed(2) + ') &nbsp; '
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
  status.innerHTML = 'Reset. Hover a cell or gene to colour by expression and reveal archetype weights.'; clearColorKey();
  titleCellColor = 'subtype'; titleGeneRef = 'dominant archetype'; refreshTitles();
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
  // Pseudotime colour = A1 weight (cell_score[:,0]) ascending → viridis on active cells.
  const arr = cell_score.map(s => s[0]);
  colorByQC(arr, 'A1 weight', v => v.toFixed(2));
}});

// Colour by region (V1 vs ALM). VISp orange, ALM purple, others grey.
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
// Subtype-subset SVD recompute
// ============================================================================
const subtypeCheckboxes = Array.from(document.querySelectorAll('.subt-chk input[type="checkbox"]'));
// Per-subclass group-level all/none buttons (in the AllInhib cohort + future
// multi-subclass cohorts; single-subclass cohorts still get a group header
// that conveniently toggles its whole subclass).
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

// Multiplicative-update NMF (Lee & Seung 2001, Frobenius loss).
// X: m × n non-negative matrix (array of Float32Array rows).
// Returns {{W, H}} with W (m × K) and H (K × n), both as flat Float32Arrays.
// W[i*K + k], H[k*n + j].
function multiplicativeUpdateNMF(X, K, maxIter) {{
  maxIter = maxIter || 80;
  const m = X.length, n = X[0].length;
  // Deterministic-ish init: random positive values seeded by index
  // (browser Math.random would change between recomputes; deterministic
  // helps the bars + legend feel stable across runs of the same inputs.)
  const W = new Float32Array(m * K);
  const H = new Float32Array(K * n);
  for (let i = 0; i < m; i++) {{
    for (let k = 0; k < K; k++) {{
      W[i*K + k] = 0.5 + 0.5 * Math.abs(Math.sin((i + 1) * (k + 1) * 0.731));
    }}
  }}
  for (let k = 0; k < K; k++) {{
    for (let j = 0; j < n; j++) {{
      H[k*n + j] = 0.5 + 0.5 * Math.abs(Math.sin((j + 1) * (k + 7) * 0.617));
    }}
  }}
  const EPS = 1e-9;
  const WtX = new Float32Array(K * n);
  const WtW = new Float32Array(K * K);
  const WtWH = new Float32Array(K * n);
  const XHt = new Float32Array(m * K);
  const HHt = new Float32Array(K * K);
  const WHHt = new Float32Array(m * K);
  for (let iter = 0; iter < maxIter; iter++) {{
    // ---- H update: H ← H ⊙ (W^T X) / (W^T W H) ------------------------
    // WtX[k, j] = sum_i W[i, k] * X[i, j]
    WtX.fill(0);
    for (let i = 0; i < m; i++) {{
      const Xi = X[i];
      for (let k = 0; k < K; k++) {{
        const w_ik = W[i*K + k];
        if (w_ik === 0) continue;
        for (let j = 0; j < n; j++) WtX[k*n + j] += w_ik * Xi[j];
      }}
    }}
    // WtW[k, l] = sum_i W[i, k] * W[i, l]
    WtW.fill(0);
    for (let i = 0; i < m; i++) {{
      for (let k = 0; k < K; k++) {{
        const w_ik = W[i*K + k];
        if (w_ik === 0) continue;
        for (let l = 0; l < K; l++) WtW[k*K + l] += w_ik * W[i*K + l];
      }}
    }}
    // WtWH[k, j] = sum_l WtW[k, l] * H[l, j]
    WtWH.fill(0);
    for (let k = 0; k < K; k++) {{
      for (let l = 0; l < K; l++) {{
        const v = WtW[k*K + l];
        if (v === 0) continue;
        for (let j = 0; j < n; j++) WtWH[k*n + j] += v * H[l*n + j];
      }}
    }}
    for (let k = 0; k < K; k++) {{
      for (let j = 0; j < n; j++) {{
        const idx = k*n + j;
        H[idx] = H[idx] * WtX[idx] / (WtWH[idx] + EPS);
      }}
    }}
    // ---- W update: W ← W ⊙ (X H^T) / (W H H^T) -------------------------
    // HHt[k, l] = sum_j H[k, j] * H[l, j]
    HHt.fill(0);
    for (let k = 0; k < K; k++) {{
      for (let l = 0; l < K; l++) {{
        let s = 0;
        for (let j = 0; j < n; j++) s += H[k*n + j] * H[l*n + j];
        HHt[k*K + l] = s;
      }}
    }}
    // XHt[i, k] = sum_j X[i, j] * H[k, j]
    XHt.fill(0);
    for (let i = 0; i < m; i++) {{
      const Xi = X[i];
      for (let k = 0; k < K; k++) {{
        let s = 0;
        for (let j = 0; j < n; j++) s += Xi[j] * H[k*n + j];
        XHt[i*K + k] = s;
      }}
    }}
    // WHHt[i, k] = sum_l W[i, l] * HHt[l, k]
    WHHt.fill(0);
    for (let i = 0; i < m; i++) {{
      for (let l = 0; l < K; l++) {{
        const v = W[i*K + l];
        if (v === 0) continue;
        for (let k = 0; k < K; k++) WHHt[i*K + k] += v * HHt[l*K + k];
      }}
    }}
    for (let i = 0; i < m; i++) {{
      for (let k = 0; k < K; k++) {{
        const idx = i*K + k;
        W[idx] = W[idx] * XHt[idx] / (WHHt[idx] + EPS);
      }}
    }}
  }}
  return {{W, H}};
}}

function recomputeNMF(basisIdx, basisLabel) {{
  // basisIdx is the list of gene indices the NMF is *fit on* (panel of genes).
  // Default = the panel HVG. Pass currently-visible gene indices to refit on
  // whatever the gene biplot is showing.
  basisIdx = basisIdx || panel_idx;
  basisLabel = basisLabel || 'panel HVG';
  const t0 = performance.now();
  const sel = selectedSubtypes();
  // Cell index list of selected cells (subtype-checked AND region-allowed AND age-allowed).
  const cellSel = [];
  for (let i = 0; i < cell_subtype.length; i++) {{
    if (sel.has(cell_subtype[i]) && regionAllowed(i) && ageAllowed(i)) cellSel.push(i);
  }}
  const m = cellSel.length;
  if (m < K_ARCH + 1) {{
    recomputeStatus.innerHTML = '<span style="color:#c00">need ≥' + (K_ARCH+1)
      + ' cells (got ' + m + ')</span>';
    return;
  }}
  const n_panel = basisIdx.length;
  const n_all   = gene_name.length;
  if (n_panel < K_ARCH) {{
    recomputeStatus.innerHTML = '<span style="color:#c00">need ≥' + K_ARCH
      + ' basis genes (got ' + n_panel + ')</span>';
    return;
  }}

  // Build non-negative basis matrix on selected cells (m × n_panel).
  // Values come straight from the log-CPM expression matrix; readValNN
  // applies the optional %ribo regress-out and clips to 0.
  const Xp = new Array(m);
  for (let i = 0; i < m; i++) Xp[i] = new Float32Array(n_panel);
  for (let ii = 0; ii < m; ii++) {{
    const i_global = cellSel[ii];
    const Xi = Xp[ii];
    for (let k = 0; k < n_panel; k++) {{
      Xi[k] = readValNN(i_global, basisIdx[k]);
    }}
  }}
  let frob2 = 0;
  for (let i = 0; i < m; i++) {{
    const Xi = Xp[i];
    for (let k = 0; k < n_panel; k++) frob2 += Xi[k] * Xi[k];
  }}

  // Run multiplicative-update NMF. W: m×K, H: K×n_panel (flat).
  const ITER = 60;
  const {{W: Wflat, H: Hflat}} = multiplicativeUpdateNMF(Xp, K_ARCH, ITER);

  // ---- Project broader genes onto W via least-squares (one solve per gene).
  // We use one step of multiplicative update on H_all (with W frozen): this
  // is a fast non-negative projection and converges in 20-30 iters on most
  // genes. For browser perf we just do 1 pass with a closed-form K=4 NNLS
  // approximation: H_all = max(0, (W^T W)^{{-1}} W^T X_all). Sign-flip happens
  // rarely; we clip negatives to keep semantic consistency.
  //
  // For K=4 the K×K matrix inversion is cheap; do it once.
  const WtW = new Float32Array(K_ARCH * K_ARCH);
  for (let k = 0; k < K_ARCH; k++) {{
    for (let l = 0; l < K_ARCH; l++) {{
      let s = 0;
      for (let i = 0; i < m; i++) s += Wflat[i*K_ARCH + k] * Wflat[i*K_ARCH + l];
      WtW[k*K_ARCH + l] = s;
    }}
  }}
  // Invert WtW (K×K = 4×4) via cofactors. Light & cheap.
  function invertSmall(M, K) {{
    // Gauss-Jordan on a copy
    const A = new Float64Array(K*K), I = new Float64Array(K*K);
    for (let k = 0; k < K*K; k++) A[k] = M[k];
    for (let k = 0; k < K; k++) I[k*K + k] = 1;
    for (let p = 0; p < K; p++) {{
      let piv = A[p*K + p];
      if (Math.abs(piv) < 1e-12) {{
        // Try to find a row swap
        for (let r = p+1; r < K; r++) {{
          if (Math.abs(A[r*K + p]) > 1e-12) {{
            for (let c = 0; c < K; c++) {{
              [A[p*K + c], A[r*K + c]] = [A[r*K + c], A[p*K + c]];
              [I[p*K + c], I[r*K + c]] = [I[r*K + c], I[p*K + c]];
            }}
            piv = A[p*K + p]; break;
          }}
        }}
        if (Math.abs(piv) < 1e-12) return null;
      }}
      const invp = 1 / piv;
      for (let c = 0; c < K; c++) {{ A[p*K + c] *= invp; I[p*K + c] *= invp; }}
      for (let r = 0; r < K; r++) {{
        if (r === p) continue;
        const f = A[r*K + p];
        if (f === 0) continue;
        for (let c = 0; c < K; c++) {{
          A[r*K + c] -= f * A[p*K + c];
          I[r*K + c] -= f * I[p*K + c];
        }}
      }}
    }}
    return I;
  }}
  const WtW_inv = invertSmall(WtW, K_ARCH);

  // For every gene j: WtX[k] = sum_i W[i,k] * x_ij; H_proj[k,j] = max(0, WtW_inv @ WtX)
  const newGeneLoading = new Array(n_all);
  const WtX_buf = new Float32Array(K_ARCH);
  for (let j = 0; j < n_all; j++) {{
    WtX_buf.fill(0);
    for (let ii = 0; ii < m; ii++) {{
      const v = readValNN(cellSel[ii], j);
      for (let k = 0; k < K_ARCH; k++) WtX_buf[k] += Wflat[ii*K_ARCH + k] * v;
    }}
    const out = new Array(K_ARCH).fill(0);
    if (WtW_inv) {{
      for (let k = 0; k < K_ARCH; k++) {{
        let s = 0;
        for (let l = 0; l < K_ARCH; l++) s += WtW_inv[k*K_ARCH + l] * WtX_buf[l];
        out[k] = s > 0 ? s : 0;
      }}
    }} else {{
      // Degenerate W (rank-deficient): fall back to mean-weighted projection
      for (let k = 0; k < K_ARCH; k++) out[k] = WtX_buf[k];
    }}
    newGeneLoading[j] = out;
  }}

  // Project to tetrahedron (barycentric → cartesian). One pass for cells, one for genes.
  function bary2xyz(weights) {{
    const sum = weights.reduce((a, b) => a + b, 0) + 1e-12;
    const x = (weights[0]*TET_V[0][0] + weights[1]*TET_V[1][0]
             + weights[2]*TET_V[2][0] + weights[3]*TET_V[3][0]) / sum;
    const y = (weights[0]*TET_V[0][1] + weights[1]*TET_V[1][1]
             + weights[2]*TET_V[2][1] + weights[3]*TET_V[3][1]) / sum;
    const z = (weights[0]*TET_V[0][2] + weights[1]*TET_V[1][2]
             + weights[2]*TET_V[2][2] + weights[3]*TET_V[3][2]) / sum;
    return [x, y, z];
  }}

  const n_cells_total = cell_subtype.length;
  const newCellX = new Array(n_cells_total).fill(null);
  const newCellY = new Array(n_cells_total).fill(null);
  const newCellZ = new Array(n_cells_total).fill(null);
  const newCellScore = cell_score.slice();
  const newCellLoad = new Array(n_cells_total);
  for (let i = 0; i < n_cells_total; i++) newCellLoad[i] = cell_load[i];
  const newCellActive = new Array(n_cells_total).fill(false);
  cellSel.forEach((i, ii) => {{
    newCellActive[i] = true;
    const w = [Wflat[ii*K_ARCH], Wflat[ii*K_ARCH+1], Wflat[ii*K_ARCH+2], Wflat[ii*K_ARCH+3]];
    const xyz = bary2xyz(w);
    newCellX[i] = +xyz[0].toFixed(4);
    newCellY[i] = +xyz[1].toFixed(4);
    newCellZ[i] = +xyz[2].toFixed(4);
    newCellScore[i] = [+w[0].toFixed(3), +w[1].toFixed(3),
                       +w[2].toFixed(3), +w[3].toFixed(3)];
    newCellLoad[i]  = [+w[0].toFixed(4), +w[1].toFixed(4),
                       +w[2].toFixed(4), +w[3].toFixed(4)];
  }});

  const newGeneX = new Array(n_all), newGeneY = new Array(n_all), newGeneZ = new Array(n_all);
  const newGeneLoad = new Array(n_all);
  for (let j = 0; j < n_all; j++) {{
    const h = newGeneLoading[j];
    const xyz = bary2xyz(h);
    newGeneX[j] = +xyz[0].toFixed(4);
    newGeneY[j] = +xyz[1].toFixed(4);
    newGeneZ[j] = +xyz[2].toFixed(4);
    newGeneLoad[j] = [+h[0].toFixed(4), +h[1].toFixed(4),
                      +h[2].toFixed(4), +h[3].toFixed(4)];
  }}

  // Dominant-archetype index per cell / gene
  function domArch(w) {{
    let k = 0, mx = w[0];
    for (let i = 1; i < K_ARCH; i++) if (w[i] > mx) {{ mx = w[i]; k = i; }}
    return k;
  }}
  const newCellDom = new Array(n_cells_total);
  for (let i = 0; i < n_cells_total; i++) {{
    if (newCellActive[i]) newCellDom[i] = POLE_COLORS_[domArch(newCellLoad[i])];
    else newCellDom[i] = '#cccccc';
  }}
  const newGeneDom = new Array(n_all);
  for (let j = 0; j < n_all; j++) {{
    newGeneDom[j] = POLE_COLORS_[domArch(newGeneLoad[j])];
  }}

  // Update vertex-label "top gene per archetype": from H[k,:] argmax over basis.
  const newPoleTop = ['', '', '', '', '', ''];
  for (let k = 0; k < K_ARCH; k++) {{
    let bestIdx = 0, best = -Infinity;
    for (let i = 0; i < n_panel; i++) {{
      const w = Hflat[k*n_panel + i];
      if (w > best) {{ best = w; bestIdx = i; }}
    }}
    newPoleTop[k] = gene_name[basisIdx[bestIdx]];
  }}
  // ---- commit new state to globals ----
  gene_x = newGeneX; gene_y = newGeneY; gene_z = newGeneZ;
  cell_load = newCellLoad; gene_load = newGeneLoad;
  cell_score = newCellScore;
  gene_loading = newGeneLoading.map(L => [+L[0].toFixed(3), +L[1].toFixed(3), +L[2].toFixed(3)]);
  cell_dom_color = newCellDom; gene_dom_color = newGeneDom;
  cell_active = newCellActive;
  pole_top = newPoleTop;

  // New gene_default_colors (by dominant archetype)
  gene_default_colors = newGeneDom.slice();

  // ---- update both plots ----
  // 1. cell points: positions + colors (subtype-coloured, dimmed for hidden)
  const cellColors = cell_default_colors.map((c, i) => newCellActive[i] ? c : '#dddddd');
  Plotly.restyle(cellPlot, {{x:[newCellX], y:[newCellY], z:[newCellZ],
                              'marker.color':[cellColors]}}, [POINTS_TRACE]);
  // 2. gene points: positions + colors
  Plotly.restyle(genePlot, {{'marker.color':[gene_default_colors]}}, [POINTS_TRACE]);
  applyGeneFilter();   // applies the current mean/std sliders → positions for visible genes
  // 3. vertex labels (archetypes): update text on both plots
  const newPoleLab = [];
  for (let p = 0; p < 6; p++) newPoleLab.push(POLE_NAMES_[p] || '');
  Plotly.restyle(cellPlot, {{text:[newPoleLab], hovertext:[newPoleLab]}}, [VERTEX_TRACE]);
  Plotly.restyle(genePlot, {{text:[newPoleLab], hovertext:[newPoleLab]}}, [VERTEX_TRACE]);
  // 4. loading dots: reset to gray (hovered state is stale)
  Plotly.restyle(cellPlot, {{'marker.color':[DEFAULT_LOAD_COLORS]}}, [LOADING_TRACE]);
  Plotly.restyle(genePlot, {{'marker.color':[DEFAULT_LOAD_COLORS]}}, [LOADING_TRACE]);
  // 5. clear gene search highlight
  Plotly.restyle(genePlot, {{x:[[null]], y:[[null]], z:[[null]]}}, [HIGHLIGHT_TRACE]);
  // 6. refresh the archetype legend in the page footer
  document.getElementById('pole-legend').innerHTML = newPoleTop.slice(0, K_ARCH).map((g, k) =>
    `<span style="display:inline-block;width:10px;height:10px;background:${{POLE_COLORS_[k]}};` +
    `margin-right:4px;border-radius:50%;"></span> ${{POLE_NAMES_[k]}} (${{g}}) &nbsp;&nbsp;`).join('');

  // 7. status
  const dt = ((performance.now() - t0) / 1000).toFixed(2);
  recomputeStatus.innerHTML = '<b>recomputed</b> NMF on ' + m + ' / ' + n_cells_total
    + ' cells, ' + n_panel + ' ' + basisLabel + ' (' + dt + 's) — '
    + 'peaks: ' + newPoleTop.slice(0, K_ARCH).map((g, k) =>
        POLE_NAMES_[k] + '=' + g).join(', ');
  lastHoveredCell = null; lastHoveredGene = null;
  // 8. heatmap reflects the new archetype ordering on the new active cell set
  renderHeatmap();}}

document.getElementById('recompute-btn').addEventListener('click', () => {{
  const btn = document.getElementById('recompute-btn');
  btn.disabled = true; recomputeStatus.textContent = 'computing…';
  setTimeout(() => {{ try {{ recomputeNMF(panel_idx, 'panel HVG'); }} finally {{ btn.disabled = false; }} }}, 30);
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
    try {{ recomputeNMF(visible, 'shown genes'); }}
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
let heatmapActiveIdx = null;            // length-m_active cell indices in A1-weight order
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
let heatmapOrderAxis = 0;       // 0=A1, 1=A2, 2=A3, 3=A4
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
  // Pseudotime for NMF: cells ordered by dominant-archetype index then by
  // max-archetype weight within that archetype — i.e. a "trajectory through
  // archetypes A1 → A2 → A3 → A4". Encoded into a single sort key as
  // (domIdx + 1 - W_max), so dom=0 cells with high W_max come first.
  const sortKey = (axis === -1)
    ? (i => {{ let m = 0, k = 0; const s = cell_score[i];
              for (let j = 0; j < K_ARCH; j++) if (s[j] > m) {{ m = s[j]; k = j; }}
              return k + 1 - m; }})
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
// (cells in A1-weight order) but it never overlaps the gene rows below.
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
