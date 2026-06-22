#!/usr/bin/env python
"""Standalone spatial explorer for Zhuang ABCA-1 + ABCA-3 V1/ALM-containing sections.

Each section's cortical GABA cells are plotted in their native cutting-plane
coordinates, coloured by Tasic subclass (from the marker-overlap crosswalk).
Per-substructure convex hulls outline cortical regions. Users pick a section
from a dropdown, filter Tasic subtypes with grouped checkboxes, and search a
gene to recolour cells by Zhuang panel expression (top-200 most variable
genes are embedded as uint8 to keep the HTML compact).

Run:
    .venv/bin/python scripts/build_spatial_v1alm_explorer.py
"""
import os, json, base64, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial import ConvexHull, Delaunay
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

ROOT = '/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish'
ANNDATA = os.path.join(ROOT, 'data', 'zhuang_cortical_GABA_with_tasic.h5ad')
CCF_POLYS = os.path.join(ROOT, 'data', 'ccfv3', 'section_polygons.pkl')
OUT = os.path.join(ROOT, 'notebooks', 'spatial_v1alm_explorer.html')

PLANE_OF = {'Zhuang-ABCA-1': ('z', 'y'), 'Zhuang-ABCA-3': ('x', 'y')}
DONORS = ['Zhuang-ABCA-1', 'Zhuang-ABCA-3']

SUBCLASS_COLORS = {
    'Lamp5':    '#3F6F76',
    'Vip':      '#A11D21',
    'Sst':      '#69B3A6',
    'Pvalb':    '#F4A300',
    'Sncg':     '#7E1E9C',
    'Serpinf1': '#1F77B4',
    'Meis2':    '#2CA02C',
    'Other GABA': '#888888',
}
SUBCLASS_ORDER = ['Lamp5', 'Vip', 'Sst', 'Pvalb', 'Sncg', 'Serpinf1', 'Meis2', 'Other GABA']

# Tasic interneuron subtype lookup so we can group checkboxes by Tasic subclass.
# Mirrors GROUPS['AllInhib'] from build_lamp5_archetype_app_4d.py.
SUBTYPES_BY_SUBCLASS = {
    'Lamp5':    ['Lamp5'],   # placeholder; rebuilt from data below if needed
    'Vip':      ['Vip'],
    'Sst':      ['Sst'],
    'Pvalb':    ['Pvalb'],
    'Sncg':     ['Sncg'],
    'Serpinf1': ['Serpinf1'],
    'Meis2':    ['Meis2'],
    'Other GABA': ['Other GABA'],
}


def _alpha_shape_rings(pts, alpha):
    """Concave hull (alpha shape) via Delaunay edge filter.

    Returns a list of (xs, ys) polygon rings — possibly multiple disjoint
    pieces. `alpha` controls concavity: smaller alpha gives tighter hulls
    that may break into multiple components. With CCFv3 coordinates in mm,
    alpha ≈ 5–15 (i.e. circumradius cutoff ≈ 1/5–1/15 mm) gives nice
    region-shaped outlines for Zhuang ABCA cell densities (~few cells per
    100 µm² in cortex).
    """
    if len(pts) < 4:
        return []
    try:
        tri = Delaunay(pts)
    except Exception:
        return []
    # Triangle filter: keep simplices whose circumradius is below 1/alpha.
    triangles = pts[tri.simplices]
    a = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
    b = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
    c = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    s = (a + b + c) / 2.0
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 1e-12))
    circ_r = (a * b * c) / (4.0 * area)
    keep = circ_r < (1.0 / alpha)
    polys = [Polygon(triangles[i]) for i in np.where(keep)[0]]
    if not polys:
        return []
    union = unary_union(polys)
    rings = []
    geoms = union.geoms if hasattr(union, 'geoms') else [union]
    for g in geoms:
        if g.is_empty or g.geom_type not in ('Polygon', 'MultiPolygon'):
            continue
        if g.geom_type == 'MultiPolygon':
            for gg in g.geoms:
                rings.append(list(gg.exterior.coords))
        else:
            rings.append(list(g.exterior.coords))
    return rings


def hulls_for_slice(obs_slice, plane, alpha=3.5):
    """Concave-hull outline per substructure on a single section.

    Uses alpha shapes (concave hulls) instead of convex hulls — much closer
    to the actual region shapes when CCFv3 substructures curve, branch, or
    have concave borders. Returns one dict per substructure (or per
    disconnected component within a substructure).
    """
    out = []
    for sub, g in obs_slice.groupby('substructure', observed=True):
        if len(g) < 6:
            continue
        pts = np.column_stack([g[plane[0]].values, g[plane[1]].values])
        if len(np.unique(pts[:, 0])) < 3 or len(np.unique(pts[:, 1])) < 3:
            continue
        rings = _alpha_shape_rings(pts, alpha)
        if not rings:
            # Fall back to convex hull on sparse substructures.
            try:
                h = ConvexHull(pts)
                rings = [list(zip(pts[h.vertices, 0].tolist(),
                                  pts[h.vertices, 1].tolist()))
                         + [(pts[h.vertices[0], 0], pts[h.vertices[0], 1])]]
            except Exception:
                continue
        for coords in rings:
            if len(coords) < 4:
                continue
            xs = [round(float(x), 3) for x, _ in coords]
            ys = [round(float(y), 3) for _, y in coords]
            out.append({'sub': str(sub), 'x': xs, 'y': ys})
    return out


def main():
    print('loading ...')
    a = ad.read_h5ad(ANNDATA)
    print(f'  loaded: {a.shape}')

    obs = a.obs.copy()
    obs['donor'] = obs['brain_section_label'].astype(str).str.split('.').str[0]
    keep_donor = obs['donor'].isin(DONORS).values
    a = a[keep_donor].copy()
    obs = a.obs.copy()
    obs['donor'] = obs['brain_section_label'].astype(str).str.split('.').str[0]
    print(f'  {a.shape[0]} cells from {", ".join(DONORS)}')

    has_visp = (obs['structure'].astype(str) == 'VISp').astype(int)
    has_mo = obs['structure'].astype(str).isin(['MOs', 'MOp']).astype(int)
    section_counts = (
        obs.assign(visp=has_visp.values, mo=has_mo.values)
        .groupby('brain_section_label', observed=True)
        .agg(visp=('visp', 'sum'), mo=('mo', 'sum'))
    )
    keep_labels = set(
        section_counts.index[
            (section_counts['visp'] >= 20) | (section_counts['mo'] >= 50)
        ].astype(str)
    )
    mask = obs['brain_section_label'].astype(str).isin(keep_labels).values
    a = a[mask].copy()
    obs = a.obs.copy()
    obs['donor'] = obs['brain_section_label'].astype(str).str.split('.').str[0]
    print(f'  {a.shape[0]} cells across {len(keep_labels)} sections')

    if 'tasic_subclass' in obs.columns:
        sc_col = obs['tasic_subclass'].astype(str).replace({'nan': 'Other GABA'})
        sc_col = sc_col.fillna('Other GABA')
        sc_col = sc_col.where(sc_col.isin(SUBCLASS_ORDER), 'Other GABA')
    else:
        sc_col = pd.Series(['Other GABA'] * len(obs), index=obs.index)
    obs['subclass'] = sc_col.values

    if 'tasic_cluster_match' in obs.columns:
        st_col = obs['tasic_cluster_match'].astype(str).replace({'nan': ''}).fillna('')
        obs['subtype'] = st_col.values
    else:
        obs['subtype'] = ''

    subtypes_present = sorted(set(s for s in obs['subtype'] if s))
    subt_to_sub = {}
    for st in subtypes_present:
        rows = obs[obs['subtype'] == st]
        if len(rows):
            subt_to_sub[st] = rows['subclass'].mode().iloc[0]
    grouped_subtypes = {}
    for sc, st in [(subt_to_sub[s], s) for s in subtypes_present]:
        grouped_subtypes.setdefault(sc, []).append(st)
    for sc in grouped_subtypes:
        grouped_subtypes[sc] = sorted(grouped_subtypes[sc])
    # Cells without a Tasic-subtype label go into a special "Unmapped" group per subclass
    for sc in SUBCLASS_ORDER:
        if (obs['subclass'] == sc).any():
            unmapped_count = ((obs['subclass'] == sc) & (obs['subtype'] == '')).sum()
            if unmapped_count > 0:
                grouped_subtypes.setdefault(sc, []).append(f'{sc} (unmapped)')

    print(f'  {len(subtypes_present)} distinct Tasic subtypes mapped (Jaccard crosswalk)')

    # ---- Yao taxonomy alternative subset (subclass + supertype). The Zhuang
    # ABCA release labels every cell with the Yao 2023 hierarchy directly, so
    # the user can subset by Yao subclass / supertype instead of Tasic subtype.
    obs['yao_class']     = obs['class'].astype(str)     if 'class'     in obs.columns else ''
    obs['yao_subclass']  = obs['subclass_y'] if 'subclass_y' in obs.columns else obs['subclass'].astype(str) if False else (obs['class'].astype(str) if False else '')
    # The Zhuang anndata uses 'subclass' and 'supertype' for Yao labels; we keep
    # them under fresh names so the existing Tasic 'subclass' column above
    # (mapped from tasic_subclass via SUBCLASS_ORDER) keeps working.
    src_subclass  = a.obs['subclass'].astype(str)  if 'subclass'  in a.obs.columns else pd.Series([''] * len(obs), index=obs.index)
    src_supertype = a.obs['supertype'].astype(str) if 'supertype' in a.obs.columns else pd.Series([''] * len(obs), index=obs.index)
    obs['yao_subclass']  = src_subclass.values
    obs['yao_supertype'] = src_supertype.values
    yao_subclasses  = sorted(set(s for s in obs['yao_subclass'] if s))
    yao_supertypes  = sorted(set(s for s in obs['yao_supertype'] if s))
    # Subclass → supertype grouping (use the supertype prefix that matches subclass).
    yao_groups = {}
    if 'yao_subclass' in obs.columns and 'yao_supertype' in obs.columns:
        pairs = obs[['yao_subclass', 'yao_supertype']].dropna().drop_duplicates()
        for ysc, yst in pairs.values:
            yao_groups.setdefault(str(ysc), []).append(str(yst))
        for k in yao_groups:
            yao_groups[k] = sorted(set(yao_groups[k]))
    print(f'  Yao: {len(yao_subclasses)} subclasses, {len(yao_supertypes)} supertypes')

    print('selecting top-variable Zhuang panel genes ...')
    X = a.X.toarray() if hasattr(a.X, 'toarray') else np.asarray(a.X)
    X = X.astype(np.float32)
    means = X.mean(axis=0)
    vars_ = X.var(axis=0)
    cv = vars_ / (means + 1e-3)
    top_idx = np.argsort(-cv)[:200]
    genes = [str(g) for g in a.var_names[top_idx]]
    X_sub = X[:, top_idx]
    gene_max = np.clip(X_sub.max(axis=0), 1e-6, None)
    X_q = np.clip(X_sub / gene_max * 255, 0, 255).astype(np.uint8)
    print(f'  {len(genes)} genes embedded as uint8 ({X_q.nbytes/1e6:.1f} MB raw)')

    # Load pre-computed CCFv3 region polygons (from build_ccf_polygons.py).
    # Each section gets a list of {'rid', 'name', 'x', 'y'} polygons in mm —
    # accurate region boundaries from the 25-µm Allen annotation volume,
    # already converted to the same coordinate frame as the cells.
    ccf_polys = {}
    rid_to_name = {}
    if os.path.exists(CCF_POLYS):
        with open(CCF_POLYS, 'rb') as f:
            ccf_polys = pickle.load(f)
        rid_to_name = ccf_polys.pop('_rid_name', {})
        print(f'  loaded {len(ccf_polys)} CCFv3 section-polygon entries')
    else:
        print(f'  WARN: {CCF_POLYS} missing — falling back to cell-derived alpha hulls')

    sections = sorted(set(obs['brain_section_label'].astype(str)))
    section_data = {}
    for sec in sections:
        m = (obs['brain_section_label'].astype(str) == sec).values
        donor = obs.loc[m, 'donor'].iloc[0]
        plane = PLANE_OF[donor]
        sub_obs = obs.loc[m]
        x = sub_obs[plane[0]].values.astype(np.float32)
        y = sub_obs[plane[1]].values.astype(np.float32)
        idx = np.where(m)[0].astype(np.int32)
        subt = sub_obs['subtype'].astype(str).fillna('').values
        sub = sub_obs['subclass'].astype(str).values
        subt_for_grouping = [(s if s else f'{sc} (unmapped)')
                             for s, sc in zip(subt, sub)]

        if sec in ccf_polys:
            hulls = []
            for p in ccf_polys[sec]['polys']:
                info = rid_to_name.get(p['rid'])
                if isinstance(info, dict):
                    acr = info.get('acronym') or f'rid{p["rid"]}'
                    nm  = info.get('name') or ''
                elif isinstance(info, str):  # legacy single-string entry
                    acr, nm = info, ''
                else:
                    acr, nm = f'rid{p["rid"]}', ''
                hulls.append({'sub': acr, 'name': nm, 'rid': p['rid'],
                              'x': p['x'], 'y': p['y']})
        else:
            hulls = hulls_for_slice(sub_obs, plane)

        section_data[sec] = {
            'donor': donor,
            'plane': list(plane),
            'idx': idx.tolist(),
            'x': [round(float(v), 3) for v in x],
            'y': [round(float(v), 3) for v in y],
            'subclass': sub.tolist(),
            'subtype': subt_for_grouping,
            'yao_subclass':  sub_obs['yao_subclass'].astype(str).tolist(),
            'yao_supertype': sub_obs['yao_supertype'].astype(str).tolist(),
            'structure': sub_obs['structure'].astype(str).tolist(),
            'cluster': sub_obs['cluster_name'].astype(str).tolist() if 'cluster_name' in sub_obs.columns else [''] * int(m.sum()),
            'hulls': hulls,
        }

    expr_b64 = base64.b64encode(X_q.tobytes()).decode('ascii')

    # Optional Zhuang full-brain background — stratified-sampled positions
    # for every cell type in each section, rendered as light-grey dots so
    # the section silhouette is visible even outside the cortical-GABA
    # foreground. Skipped silently if the pickle isn't built yet.
    bg_path = os.path.join(ROOT, 'data', 'zhuang_full_background.pkl')
    bg_sections = {}
    if os.path.exists(bg_path):
        with open(bg_path, 'rb') as f:
            bg = pickle.load(f)
        # Keep only sections that also exist in our foreground set
        bg_sections = {s: v for s, v in bg.get('sections', {}).items()
                       if s in section_data}
        print(f'  loaded {len(bg_sections)} sections from full Zhuang background '
              f'(target {len(section_data)})')
    else:
        print(f'  (no Zhuang background pickle at {bg_path})')

    # Build Yao subclass color palette (8 colours) and supertype palette
    # (deterministic golden-angle hue per index).
    def hash_palette(items):
        out = {}
        for i, x in enumerate(sorted(items)):
            hue = (i * 137.508) % 360
            out[x] = f'hsl({int(hue)}, 65%, 52%)'
        return out
    yao_subclass_colors  = hash_palette(yao_subclasses)
    yao_supertype_colors = hash_palette(yao_supertypes)

    js_data = (
        f"const SECTIONS = {json.dumps(sections)};\n"
        f"const SECTION_DATA = {json.dumps(section_data, separators=(',', ':'))};\n"
        f"const BG_SECTIONS = {json.dumps(bg_sections, separators=(',', ':'))};\n"
        f"const GENES = {json.dumps(genes)};\n"
        f"const GENE_MAX = {json.dumps([float(round(v, 4)) for v in gene_max.tolist()])};\n"
        f"const N_CELLS = {int(a.shape[0])};\n"
        f"const N_GENES = {len(genes)};\n"
        f"const TAXONOMIES = {{\n"
        f"  'tasic': {{ field: 'subtype', groups: {json.dumps(grouped_subtypes)},\n"
        f"             classes: {json.dumps(SUBCLASS_ORDER)},\n"
        f"             colors: {json.dumps(SUBCLASS_COLORS)} }},\n"
        f"  'yao_subclass': {{ field: 'yao_subclass',\n"
        f"             groups: {json.dumps({k: [k] for k in yao_subclasses})},\n"
        f"             classes: {json.dumps(yao_subclasses)},\n"
        f"             colors: {json.dumps(yao_subclass_colors)} }},\n"
        f"  'yao_supertype': {{ field: 'yao_supertype',\n"
        f"             groups: {json.dumps(yao_groups)},\n"
        f"             classes: {json.dumps(sorted(yao_groups.keys()))},\n"
        f"             colors: {json.dumps(yao_subclass_colors)} }},\n"
        f"}};\n"
        f"const SUBCLASSES = {json.dumps(SUBCLASS_ORDER)};\n"
        f"const SUBCLASS_COLORS = {json.dumps(SUBCLASS_COLORS)};\n"
        f"const GROUPED_SUBTYPES = {json.dumps(grouped_subtypes)};\n"
        f"const EXPR_B64 = {json.dumps(expr_b64)};\n"
    )

    # Build subtype-checkbox HTML grouped by subclass (like AllInhib explorer)
    def chip(label, css_class='subt-chip'):
        return f'<label class="{css_class}"><input type="checkbox" data-sub="{label}" checked> {label}</label>'

    group_html_parts = []
    for sc in SUBCLASS_ORDER:
        if sc not in grouped_subtypes: continue
        sts = grouped_subtypes[sc]
        color = SUBCLASS_COLORS.get(sc, '#888')
        chips = ''.join(chip(s) for s in sts)
        group_html_parts.append(
            f'<div class="subt-group" style="border-left:4px solid {color};">'
            f'<div class="subt-group-head">{sc} '
            f'<button class="subt-mini" data-act="all" data-grp="{sc}">all</button>'
            f'<button class="subt-mini" data-act="none" data-grp="{sc}">none</button>'
            f'</div>'
            f'<div class="subt-chips" data-grp="{sc}">{chips}</div>'
            f'</div>'
        )
    subtype_html = '\n'.join(group_html_parts)

    section_options = '\n'.join(
        f'<option value="{s}">{s}</option>' for s in sections
    )

    html = """<!doctype html>
<html><head>
<meta charset="utf-8"/>
<title>Spatial V1+ALM explorer (Zhuang ABCA-1 / ABCA-3)</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 12px 18px; color: #222; }
  h2 { margin: 6px 0; font-size: 18px; }
  .sub { color: #555; font-size: 13px; margin-bottom: 8px; }
  .controls { display: flex; gap: 14px; align-items: flex-start; flex-wrap: wrap; padding-bottom: 10px; border-bottom: 1px solid #ddd; margin-bottom: 10px; }
  .ctl { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: #444; }
  .ctl select, .ctl input[type="text"] { font-size: 13px; padding: 4px 6px; border: 1px solid #bbb; border-radius: 4px; }
  .ctl select { min-width: 200px; }
  .ctl input[type="text"] { width: 180px; }
  .clear-gene { font-size: 11px; padding: 2px 8px; background: #eee; border: 1px solid #bbb; border-radius: 3px; cursor: pointer; margin-top: 2px; }
  .subt-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 6px; margin: 8px 0 14px; }
  .subt-group { padding: 4px 8px 6px; background: #fafafa; border-radius: 4px; }
  .subt-group-head { font-size: 12px; font-weight: 600; margin-bottom: 4px; display: flex; align-items: center; gap: 4px; }
  .subt-mini { font-size: 10px; padding: 1px 5px; background: #fff; border: 1px solid #ccc; border-radius: 3px; cursor: pointer; }
  .subt-chips { display: flex; flex-wrap: wrap; gap: 4px; }
  .subt-chip { font-size: 11px; background: #fff; border: 1px solid #ccc; border-radius: 3px; padding: 1px 5px; cursor: pointer; display: inline-flex; align-items: center; gap: 3px; }
  .subt-chip input { margin: 0; }
  .all-controls { display: flex; gap: 6px; margin-bottom: 4px; }
  .all-controls button { font-size: 11px; padding: 2px 8px; background: #fff; border: 1px solid #ccc; border-radius: 3px; cursor: pointer; }
  #plot { width: 100%; height: 720px; }
  .legend { display: flex; flex-wrap: wrap; gap: 8px; font-size: 12px; margin-top: 6px; }
  .legend-item { display: flex; align-items: center; gap: 3px; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .stat { font-size: 12px; color: #555; margin-left: 8px; }
  #gene-status, #area-status { font-size: 11px; color: #888; margin-top: 2px; }
  .area-results { margin: 4px 0 10px; padding: 6px 8px; background: #f3f7fa;
                  border-left: 3px solid #1f77b4; border-radius: 3px; font-size: 12px; }
  .area-results-head { font-weight: 600; color: #1f77b4; margin-bottom: 3px; display: block; }
  .area-results-chips { display: flex; flex-wrap: wrap; gap: 4px; }
  .area-results-chip { padding: 2px 6px; font-size: 11px; cursor: pointer;
                        background: #fff; border: 1px solid #b0c4d6; border-radius: 3px; }
  .area-results-chip:hover { background: #e7f0fa; }
  .area-results-chip.current { background: #1f77b4; color: white; border-color: #1f77b4; }
  .home-link { position: fixed; top: 10px; left: max(12px, calc(50vw - 845px)); z-index: 1000; font-size: 13px;
               text-decoration: none; color: #5a4326; background: rgba(255,255,255,.9);
               border: 1px solid #cdbf9f; border-radius: 7px; padding: 5px 11px;
               font-weight: 600; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  .home-link:hover { background: #f3ecdc; }
</style></head>
<body>
<a class="home-link" href="../index.html">← All explorers</a>
<h2>Spatial V1+ALM explorer — Zhuang ABCA-1 / ABCA-3</h2>
<div class="sub">Cortical GABA cells with Tasic-subclass labels (marker-overlap crosswalk). Convex-hull outlines per substructure. Search a gene to recolour cells by Zhuang panel expression (top-200 variable genes embedded).</div>

<div class="controls">
  <div class="ctl">
    <label>Section</label>
    <select id="section-select">__SECTION_OPTIONS__</select>
    <div class="stat" id="section-stat"></div>
  </div>
  <div class="ctl">
    <label>Gene (recolour by expression)</label>
    <input type="text" id="gene-input" list="gene-list" placeholder="e.g. Pvalb, Sst, Vip"/>
    <datalist id="gene-list"></datalist>
    <button class="clear-gene" id="clear-gene">↺ subclass colour</button>
    <div id="gene-status"></div>
  </div>
  <div class="ctl">
    <label>Taxonomy (subset by)</label>
    <select id="taxonomy-select">
      <option value="tasic" selected>Tasic subtype (crosswalk)</option>
      <option value="yao_subclass">Yao subclass (8 — Pvalb/Sst/Vip/…)</option>
      <option value="yao_supertype">Yao supertype (~60 fine groups)</option>
    </select>
    <div class="stat" id="taxonomy-stat"></div>
  </div>
  <div class="ctl">
    <label>Find area (across slices)</label>
    <input type="text" id="area-input" list="area-list"
           placeholder="acronym or name, e.g. VISp"/>
    <datalist id="area-list"></datalist>
    <button class="clear-gene" id="find-area-btn">→ list slices</button>
    <div id="area-status"></div>
  </div>
  <div class="ctl">
    <label>Outlines</label>
    <label style="font-size:11px;"><input type="checkbox" id="show-hulls" checked> show region outlines</label>
    <label style="font-size:11px;"><input type="checkbox" id="label-hulls" checked> label substructure</label>
  </div>
</div>

<div id="area-results" class="area-results" style="display:none;">
  <span class="area-results-head"></span>
  <div class="area-results-chips"></div>
</div>

<div class="all-controls">
  <button id="all-on">all subtypes on</button>
  <button id="all-off">all subtypes off</button>
</div>
<div id="subt-grid" class="subt-grid">__SUBTYPE_HTML__</div>

<div id="plot"></div>
<div class="legend" id="legend"></div>

<script>
__JS_DATA__

// decode expression payload
const EXPR_BYTES = (function() {
  const bin = atob(EXPR_B64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
})();
function geneExprFor(idx, geneCol) {
  // Returns float CPM-like value for cell idx and gene index geneCol
  return EXPR_BYTES[idx * N_GENES + geneCol] / 255 * GENE_MAX[geneCol];
}
const GENE_INDEX = (function() {
  const m = {};
  for (let i = 0; i < GENES.length; i++) m[GENES[i].toLowerCase()] = i;
  return m;
})();

// populate datalist
const dl = document.getElementById('gene-list');
GENES.forEach(g => {
  const o = document.createElement('option');
  o.value = g;
  dl.appendChild(o);
});

// Build a global registry of every CCF region acronym + full name across all
// sections, so the "Find area" datalist offers them and the lookup can run
// case-insensitively against acronym OR name fragment.
const AREA_BY_ACRONYM = (function() {
  const acc = {};   // acronym (lowercased) -> { acronym, name, sections: [secLabel,...] }
  Object.entries(SECTION_DATA).forEach(([sec, d]) => {
    (d.hulls || []).forEach(h => {
      const key = (h.sub || '').toLowerCase();
      if (!key) return;
      const r = acc[key] || (acc[key] = { acronym: h.sub, name: h.name || '', sections: new Set() });
      r.sections.add(sec);
    });
  });
  Object.values(acc).forEach(v => v.sections = Array.from(v.sections).sort());
  return acc;
})();
const areaDl = document.getElementById('area-list');
Object.values(AREA_BY_ACRONYM).sort((a, b) => a.acronym.localeCompare(b.acronym)).forEach(r => {
  const o = document.createElement('option');
  o.value = r.acronym;
  if (r.name) o.label = r.name;
  areaDl.appendChild(o);
});

// magma colorscale (Plotly-compatible array)
const MAGMA = [
  [0.0, '#000004'], [0.1, '#160b39'], [0.2, '#420a68'], [0.3, '#6a176e'],
  [0.4, '#932667'], [0.5, '#bb3754'], [0.6, '#dd513a'], [0.7, '#f37819'],
  [0.8, '#fba40a'], [0.9, '#f6d746'], [1.0, '#fcfdbf']
];
function magmaColor(t) {
  // simple sample from MAGMA for legend dots
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < MAGMA.length - 1; i++) {
    if (t <= MAGMA[i+1][0]) {
      const a = MAGMA[i], b = MAGMA[i+1];
      const f = (t - a[0]) / (b[0] - a[0]);
      const ah = a[1].slice(1), bh = b[1].slice(1);
      const ar = parseInt(ah.slice(0,2),16), ag = parseInt(ah.slice(2,4),16), ab = parseInt(ah.slice(4,6),16);
      const br = parseInt(bh.slice(0,2),16), bg = parseInt(bh.slice(2,4),16), bb = parseInt(bh.slice(4,6),16);
      return `rgb(${Math.round(ar + (br-ar)*f)},${Math.round(ag + (bg-ag)*f)},${Math.round(ab + (bb-ab)*f)})`;
    }
  }
  return MAGMA[MAGMA.length-1][1];
}

// state
let currentGene = null;
let currentSection = SECTIONS[0];
let currentTaxonomy = 'tasic';

function activeTax() { return TAXONOMIES[currentTaxonomy]; }
function cellLabel(d, i) { return d[activeTax().field][i] || '(none)'; }
function cellSubclassFor(d, i) {
  // For Tasic: use the explicit subclass field. For Yao: derive from the
  // active label itself (since groups[label] = [label] for subclass) — or
  // map supertype → subclass via the GROUPED structure (reverse lookup).
  if (currentTaxonomy === 'tasic') return d.subclass[i] || 'Other GABA';
  if (currentTaxonomy === 'yao_subclass') return d.yao_subclass[i] || '';
  // yao_supertype: find which subclass key in groups contains this supertype.
  const lab = d.yao_supertype[i];
  const groups = activeTax().groups;
  for (const sc in groups) if (groups[sc].includes(lab)) return sc;
  return '';
}
function colorFor(d, i) {
  const tax = activeTax();
  if (currentTaxonomy === 'tasic') {
    return SUBCLASS_COLORS[d.subclass[i]] || '#888';
  }
  return tax.colors[cellSubclassFor(d, i)] || '#888';
}

const subtypeCheckboxes = () =>
  Array.from(document.querySelectorAll('#subt-grid .subt-chip input[type="checkbox"]'));

function selectedSubtypes() {
  const out = new Set();
  subtypeCheckboxes().forEach(cb => { if (cb.checked) out.add(cb.dataset.sub); });
  return out;
}

function rebuildSubtypeGrid() {
  const tax = activeTax();
  const grid = document.getElementById('subt-grid');
  grid.innerHTML = '';
  // tax.classes is the ordered subclass list; tax.groups[sc] is the list of
  // labels to expose as individual chips. For Yao subclass each group has a
  // single label = itself; for Yao supertype each subclass groups its
  // supertypes; for Tasic each subclass groups its Tasic subtypes.
  for (const sc of tax.classes) {
    const labels = (tax.groups && tax.groups[sc]) || [];
    if (!labels.length) continue;
    const color = tax.colors[sc] || '#888';
    const grp = document.createElement('div');
    grp.className = 'subt-group';
    grp.style.borderLeft = `4px solid ${color}`;
    grp.innerHTML =
      `<div class="subt-group-head">${sc} ` +
      `<button class="subt-mini" data-act="all" data-grp="${sc}">all</button>` +
      `<button class="subt-mini" data-act="none" data-grp="${sc}">none</button></div>` +
      `<div class="subt-chips" data-grp="${sc}">` +
      labels.map(lab =>
        `<label class="subt-chip"><input type="checkbox" data-sub="${lab}" checked> ${lab}</label>`
      ).join('') + `</div>`;
    grid.appendChild(grp);
  }
  // re-bind per-group all/none and per-chip change handlers
  grid.querySelectorAll('.subt-mini').forEach(btn => {
    btn.addEventListener('click', () => {
      const want = btn.dataset.act === 'all';
      grid.querySelectorAll(`.subt-chips[data-grp="${btn.dataset.grp}"] input`).forEach(cb => {
        cb.checked = want;
      });
      renderPlot();
    });
  });
  subtypeCheckboxes().forEach(cb => cb.addEventListener('change', renderPlot));
  const totalChips = grid.querySelectorAll('.subt-chip').length;
  document.getElementById('taxonomy-stat').textContent =
    `${tax.classes.length} groups, ${totalChips} labels`;
}

function renderPlot() {
  const sec = currentSection;
  const d = SECTION_DATA[sec];
  const sel = selectedSubtypes();
  const showHulls = document.getElementById('show-hulls').checked;
  const labelHulls = document.getElementById('label-hulls').checked;

  // partition into background (deselected) and foreground (selected) cells
  const bgX = [], bgY = [], bgText = [];
  const fgX = [], fgY = [], fgText = [], fgColor = [], fgCustom = [];
  const geneCol = currentGene == null ? -1 : GENE_INDEX[currentGene.toLowerCase()];
  const useExpr = currentGene && geneCol != null && geneCol >= 0;
  let exprMax = 0;
  if (useExpr) {
    for (let i = 0; i < d.idx.length; i++) {
      if (sel.has(cellLabel(d, i))) {
        const v = geneExprFor(d.idx[i], geneCol);
        if (v > exprMax) exprMax = v;
      }
    }
    if (exprMax === 0) exprMax = 1;
  }

  for (let i = 0; i < d.idx.length; i++) {
    const lab = cellLabel(d, i);
    if (sel.has(lab)) {
      fgX.push(d.x[i]);
      fgY.push(d.y[i]);
      let txt = `${lab}<br>${d.structure[i]}`;
      if (d.cluster[i]) txt += `<br>${d.cluster[i]}`;
      if (useExpr) {
        const v = geneExprFor(d.idx[i], geneCol);
        txt += `<br>${currentGene}: ${v.toFixed(2)}`;
        fgColor.push(v);
      } else {
        fgColor.push(colorFor(d, i));
      }
      fgText.push(txt);
      fgCustom.push([d.idx[i]]);
    } else {
      bgX.push(d.x[i]);
      bgY.push(d.y[i]);
      bgText.push(`${lab} · ${d.structure[i]}`);
    }
  }

  const traces = [];

  // Full-Zhuang background layer (~500 cells/section/donor) — every cell in
  // the section regardless of type, drawn as tiny light-grey dots BEHIND the
  // cortical-GABA foreground so the brain silhouette is visible.
  const bg = BG_SECTIONS[sec];
  if (bg) {
    traces.push({
      type: 'scattergl',
      mode: 'markers',
      x: bg.x, y: bg.y,
      marker: { color: 'rgba(190,190,190,0.45)', size: 2.2 },
      hoverinfo: 'skip',
      showlegend: false,
      name: 'background',
    });
  }


  // hull outlines (one trace each so each gets its own label). Hover shows
  // the acronym plus the full Allen name; the centroid label only shows the
  // acronym (kept short to avoid overlap with neighbouring regions).
  if (showHulls) {
    for (const h of d.hulls) {
      const fullLabel = h.name ? `<b>${h.sub}</b><br>${h.name}` : `<b>${h.sub}</b>`;
      traces.push({
        type: 'scattergl',
        mode: 'lines',
        x: h.x, y: h.y,
        line: { color: 'rgba(80,80,80,0.55)', width: 1 },
        hoverinfo: 'text',
        text: fullLabel,
        showlegend: false,
      });
      if (labelHulls) {
        let cx = 0, cy = 0;
        for (let i = 0; i < h.x.length - 1; i++) { cx += h.x[i]; cy += h.y[i]; }
        cx /= (h.x.length - 1); cy /= (h.y.length - 1);
        traces.push({
          type: 'scatter',
          mode: 'text',
          x: [cx], y: [cy],
          text: [h.sub],
          textfont: { size: 9, color: '#444' },
          hoverinfo: 'skip',
          showlegend: false,
        });
      }
    }
  }

  // background (deselected) cells
  if (bgX.length) {
    traces.push({
      type: 'scattergl',
      mode: 'markers',
      x: bgX, y: bgY,
      marker: { color: 'rgba(180,180,180,0.35)', size: 3 },
      text: bgText,
      hoverinfo: 'skip',
      showlegend: false,
      name: 'background',
    });
  }

  // foreground cells (selected subtypes)
  if (fgX.length) {
    const markerCfg = useExpr
      ? {
          color: fgColor,
          colorscale: MAGMA,
          cmin: 0, cmax: exprMax,
          size: 5,
          showscale: true,
          colorbar: { title: { text: currentGene, font: { size: 11 } }, len: 0.5, thickness: 12 },
        }
      : { color: fgColor, size: 5 };
    traces.push({
      type: 'scattergl',
      mode: 'markers',
      x: fgX, y: fgY,
      marker: markerCfg,
      text: fgText,
      hoverinfo: 'text',
      customdata: fgCustom,
      showlegend: false,
      name: 'cells',
    });
  }

  const plane = d.plane;
  const layout = {
    margin: { l: 50, r: 30, t: 30, b: 50 },
    xaxis: { title: `${plane[0]} (mm, CCFv3)`, scaleanchor: 'y', constrain: 'domain' },
    yaxis: { title: `${plane[1]} (mm, CCFv3)`, autorange: 'reversed' },
    title: { text: `${sec} (${d.donor})`, font: { size: 14 } },
    hovermode: 'closest',
  };

  Plotly.react('plot', traces, layout, { responsive: true, displaylogo: false });

  // legend
  const legend = document.getElementById('legend');
  legend.innerHTML = '';
  if (useExpr) {
    const grad = 'linear-gradient(to right,#000004,#3b0f70,#8c2981,#de4968,#fe9f6d,#fcfdbf)';
    legend.innerHTML = `<span class="legend-item" style="gap:8px;font-weight:600;">${currentGene} (CPM-like)</span>`
      + `<span class="legend-item" style="gap:6px;"><span style="color:#666">0</span>`
      + `<span style="display:inline-block;width:220px;height:12px;border-radius:3px;border:1px solid #c4c4c4;background:${grad}"></span>`
      + `<span style="color:#666">${exprMax.toFixed(2)}</span></span>`;
  } else {
    // Build from groups (taxonomy "classes") actually present in selection
    const tax = activeTax();
    const present = new Set();
    for (let i = 0; i < d.idx.length; i++)
      if (sel.has(cellLabel(d, i))) present.add(cellSubclassFor(d, i));
    tax.classes.forEach(sc => {
      if (!present.has(sc)) return;
      const div = document.createElement('span');
      div.className = 'legend-item';
      div.innerHTML = `<span class="legend-dot" style="background:${tax.colors[sc] || '#888'}"></span>${sc}`;
      legend.appendChild(div);
    });
  }

  // stat text
  document.getElementById('section-stat').textContent =
    `${d.idx.length} cells (${fgX.length} selected)`;
  document.getElementById('gene-status').textContent = useExpr
    ? `expr max in selection: ${exprMax.toFixed(2)}`
    : (currentGene
        ? `gene "${currentGene}" not in Zhuang top-200 panel`
        : '');
}

// section selector
document.getElementById('section-select').addEventListener('change', e => {
  currentSection = e.target.value;
  renderPlot();
});
document.getElementById('section-select').value = currentSection;

// taxonomy selector
document.getElementById('taxonomy-select').addEventListener('change', e => {
  currentTaxonomy = e.target.value;
  rebuildSubtypeGrid();
  renderPlot();
});

// --- Find area across slices ---------------------------------------------
function findAreaMatches(query) {
  // Returns the sorted list of {acronym, name, sections} entries that match
  // the query. Matches against acronym (case-insensitive exact or prefix) AND
  // against the full Allen name (substring).
  query = (query || '').trim();
  if (!query) return [];
  const q = query.toLowerCase();
  // 1) exact acronym match (case-insensitive)
  if (AREA_BY_ACRONYM[q]) return [AREA_BY_ACRONYM[q]];
  // 2) acronym prefix
  const pref = Object.values(AREA_BY_ACRONYM).filter(r =>
    r.acronym.toLowerCase().startsWith(q));
  if (pref.length) return pref.slice(0, 20);
  // 3) substring in name
  return Object.values(AREA_BY_ACRONYM).filter(r =>
    r.name && r.name.toLowerCase().includes(q)).slice(0, 20);
}

function runFindArea() {
  const q = document.getElementById('area-input').value;
  const results = findAreaMatches(q);
  const box = document.getElementById('area-results');
  const head = box.querySelector('.area-results-head');
  const chipsEl = box.querySelector('.area-results-chips');
  const status = document.getElementById('area-status');
  chipsEl.innerHTML = '';
  if (!q.trim()) { box.style.display = 'none'; status.textContent = ''; return; }
  if (!results.length) {
    box.style.display = '';
    head.textContent = `No CCF area matches "${q}"`;
    status.textContent = `No matches in ${Object.keys(AREA_BY_ACRONYM).length} known CCF regions.`;
    return;
  }
  // If we matched a single region, list its sections. If multiple regions
  // matched, show one row per region with their section counts (click an
  // acronym to drill in).
  if (results.length === 1) {
    const r = results[0];
    box.style.display = '';
    head.innerHTML = `<b>${r.acronym}</b>${r.name ? ' — ' + r.name : ''} · `
      + r.sections.length + ' slice(s)';
    r.sections.forEach(sec => {
      const chip = document.createElement('span');
      chip.className = 'area-results-chip' + (sec === currentSection ? ' current' : '');
      chip.textContent = sec.replace('Zhuang-ABCA-', '');
      chip.title = sec;
      chip.addEventListener('click', () => {
        currentSection = sec;
        document.getElementById('section-select').value = sec;
        renderPlot();
        chipsEl.querySelectorAll('.area-results-chip').forEach(c =>
          c.classList.toggle('current', c.textContent === sec.replace('Zhuang-ABCA-', '')));
      });
      chipsEl.appendChild(chip);
    });
    status.textContent = `${r.sections.length} sections contain ${r.acronym}.`;
  } else {
    box.style.display = '';
    head.innerHTML = `${results.length} CCF area(s) match "${q}" — click an acronym to list its slices:`;
    results.forEach(r => {
      const chip = document.createElement('span');
      chip.className = 'area-results-chip';
      chip.textContent = `${r.acronym} (${r.sections.length})`;
      chip.title = r.name || r.acronym;
      chip.addEventListener('click', () => {
        document.getElementById('area-input').value = r.acronym;
        runFindArea();
      });
      chipsEl.appendChild(chip);
    });
    status.textContent = `${results.length} regions match. Click one to expand.`;
  }
}
document.getElementById('find-area-btn').addEventListener('click', runFindArea);
document.getElementById('area-input').addEventListener('keyup', e => {
  if (e.key === 'Enter') runFindArea();
});

// gene input
const geneInput = document.getElementById('gene-input');
geneInput.addEventListener('change', () => {
  currentGene = geneInput.value.trim() || null;
  renderPlot();
});
geneInput.addEventListener('keyup', e => {
  if (e.key === 'Enter') {
    currentGene = geneInput.value.trim() || null;
    renderPlot();
  }
});
document.getElementById('clear-gene').addEventListener('click', () => {
  geneInput.value = '';
  currentGene = null;
  renderPlot();
});

document.getElementById('show-hulls').addEventListener('change', renderPlot);
document.getElementById('label-hulls').addEventListener('change', renderPlot);

document.getElementById('all-on').addEventListener('click', () => {
  subtypeCheckboxes().forEach(cb => cb.checked = true);
  renderPlot();
});
document.getElementById('all-off').addEventListener('click', () => {
  subtypeCheckboxes().forEach(cb => cb.checked = false);
  renderPlot();
});

document.querySelectorAll('.subt-mini').forEach(btn => {
  btn.addEventListener('click', () => {
    const grp = btn.dataset.grp;
    const want = btn.dataset.act === 'all';
    document.querySelectorAll(`.subt-chips[data-grp="${grp}"] input`).forEach(cb => {
      cb.checked = want;
    });
    renderPlot();
  });
});

subtypeCheckboxes().forEach(cb => cb.addEventListener('change', renderPlot));

renderPlot();
</script>
</body></html>
"""
    html = (html
            .replace('__SECTION_OPTIONS__', section_options)
            .replace('__SUBTYPE_HTML__', subtype_html)
            .replace('__JS_DATA__', js_data))

    print(f'writing {OUT} ...')
    with open(OUT, 'w') as f:
        f.write(html)
    print(f'  done. {os.path.getsize(OUT)/1e6:.1f} MB self-contained HTML.')
    print(f'  open: file://{OUT}')


if __name__ == '__main__':
    main()
