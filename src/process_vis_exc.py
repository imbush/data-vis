#!/usr/bin/env python
"""Process Gouwens et al. 2026 (exc_vis_manuscript) into a cache for the visualizer.

Modalities per Patch-seq cell (n=1528): T-type + MET-type labels, electrophysiology
(E-UMAP provided by the paper + interpretable features), morphology (50 features,
n=389). We reproduce the paper's DR *substrates*:
  - E: their t-seeded UMAP of 62 sparse-PCA components (used as-is).
  - M: UMAP of the 50 normalized morphology features (their feature set).
  - T: a global UMAP built from their per-subclass transcriptomic PCA (their tx-DR
       substrate; gene counts are fastq-only on NeMO so a from-counts global T-UMAP
       is not reproducible here — documented limitation).
Plus the WNM projectome: 341 whole-neuron morphologies with axonal projection
strengths to 220 CCF targets, predicted MET-type, soma location, azimuth/altitude.
"""
import os, json, pickle, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import umap
R = '/private/tmp/claude-501/-Users-inlebush-cs-lab-green-data-vis/95ef1b74-d552-4696-b236-9181bf853664/scratchpad/exc_vis_manuscript'
OUT = '/private/tmp/claude-501/-Users-inlebush-cs-lab-green-data-vis/95ef1b74-d552-4696-b236-9181bf853664/scratchpad/vis_exc_cache.pkl'

def met_to_subclass(m):
    if not isinstance(m, str) or not m: return None
    if m.startswith('L2/3 IT'): return 'L23-IT'
    if m in ('L4 IT','L4/L5 IT') or m.startswith('L5 IT'): return 'L4-L5-IT'
    if m.startswith('L6 IT'): return 'L6-IT'
    if m.startswith('L5/L6 IT Car3'): return 'L5L6-IT-Car3'
    if m.startswith('L5 ET'): return 'L5-ET'
    if m == 'L5 NP': return 'L5-NP'
    if m.startswith('L6 CT'): return 'L6-CT'
    if m.startswith('L6b'): return 'L6b'
    return None

SUBCLASS_ORDER = ['L2/3 IT','L4 IT','L4/L5 IT','L5 IT','L6 IT','L5/L6 IT Car3','L5 ET','L5 NP','L6 CT','L6b']
def met_to_broad(m):
    sc = met_to_subclass(m)
    return {'L23-IT':'L2/3 IT','L4-L5-IT':'L4/L5 IT','L6-IT':'L6 IT','L5L6-IT-Car3':'L5/L6 IT Car3',
            'L5-ET':'L5 ET','L5-NP':'L5 NP','L6-CT':'L6 CT','L6b':'L6b'}.get(sc, 'other')

# ---------- Patch-seq labels ----------
met = pd.read_csv(f'{R}/derived_data/inferred_met_types.csv', index_col=0)
met.index = met.index.astype(str)
met['MET'] = met['inferred_met_type'].fillna('').where(met['inferred_met_type'].notna(), met['met_type'])
met['MET'] = met['MET'].fillna('').replace('', 'unassigned')
met['subclass'] = met['MET'].map(met_to_broad)
ids = met.index.tolist()
n = len(ids)
print(f'Patch-seq cells: {n}  | MET-types: {met.MET.nunique()}  | T-types: {met.t_type.nunique()}')

# ---------- E-UMAP (paper-provided) ----------
eu = pd.read_csv(f'{R}/derived_data/ephys_umap_coordinates_t_seeded.csv', index_col=0)
eu.index = eu.index.astype(str)
e_umap = eu.reindex(ids)[['x','y']].values.astype(np.float32)
print(f'E-UMAP: {np.isfinite(e_umap[:,0]).sum()} cells')

# ---------- interpretable ephys features ----------
eph = pd.read_csv(f'{R}/data/patchseq/exc_mMET_ephys_features.csv', index_col=0)
eph.index = eph.index.astype(str)
EPHYS_KEEP = {
 'rheobase_i':'rheobase (pA)','input_resistance':'input resistance (MΩ)','tau':'membrane tau (ms)',
 'v_baseline':'resting Vm (mV)','sag_nearest_minus_100':'sag ratio','fi_linear_fit_slope':'f-I slope',
 'avg_rate_0_long_square':'firing rate (Hz)','ap_1_width_short_square':'AP width (ms)',
 'ap_1_threshold_v_short_square':'AP threshold (mV)','ap_1_upstroke_downstroke_ratio_short_square':'up:down ratio',
 'first_isi_0_long_square':'first ISI (ms)','isi_cv_0_long_square':'ISI CV'}
ephys = {}
for col,lab in EPHYS_KEEP.items():
    if col in eph.columns: ephys[lab] = eph.reindex(ids)[col].values.astype(np.float32)
print(f'ephys features: {list(ephys)}')

# ---------- morphology: 50 features + M-UMAP (n=389) ----------
mf = pd.read_csv(f'{R}/data/patchseq/morph_features_mMET_exc_wide_normalized.csv', index_col=0)
mf.index = mf.index.astype(str)
morph_cols = [c for c in mf.columns]
mf_ids = [i for i in ids if i in mf.index]
Xm = mf.reindex(mf_ids)[morph_cols].values.astype(np.float64)
Xm = np.nan_to_num(Xm, nan=0.0)
m_reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0)
m_xy_sub = m_reducer.fit_transform(Xm)
m_umap = np.full((n,2), np.nan, np.float32)
pos = {i:k for k,i in enumerate(ids)}
for k,i in enumerate(mf_ids): m_umap[pos[i]] = m_xy_sub[k]
# per-cell morph features (full length, NaN where absent)
morph = {}
for c in morph_cols:
    arr = np.full(n, np.nan, np.float32); s = mf.reindex(mf_ids)[c].values.astype(np.float32)
    for k,i in enumerate(mf_ids): arr[pos[i]] = s[k]
    morph[c] = arr
print(f'M-UMAP: {len(mf_ids)} cells; {len(morph_cols)} morph features')

# soma depth from pia (morph) for the 389
soma_depth = morph.get('soma_aligned_dist_from_pia', np.full(n, np.nan, np.float32))

# ---------- global T-UMAP from per-subclass transcriptomic PCA (block-diagonal) ----------
sc_files = {'L23-IT':'L23-IT','L4-L5-IT':'L4-L5-IT','L6-IT':'L6-IT','L5L6-IT-Car3':'L5L6-IT-Car3',
            'L5-ET':'L5-ET','L5-NP':'L5-NP','L6-CT':'L6-CT','L6b':'L6b'}
tx = {}   # subclass -> DataFrame(specimen -> PCs)
for sc,f in sc_files.items():
    p = f'{R}/derived_data/ps_tx_pca_results/{f}_ps_transformed_pcs.csv'
    if os.path.exists(p):
        d = pd.read_csv(p, index_col=0); d.index = d.index.astype(str); tx[sc] = d
# block layout
blocks = {sc:d.shape[1] for sc,d in tx.items()}
order = list(blocks); offsets = {}; tot=0
for sc in order: offsets[sc]=tot; tot+=blocks[sc]
tx_ids = []; rows = []
for i in ids:
    sc = met_to_subclass(met.loc[i,'MET'])
    if sc in tx and i in tx[sc].index:
        v = np.zeros(tot, np.float64)
        pcs = tx[sc].loc[i].values.astype(np.float64)
        # standardize each subclass block to unit-ish scale so no block dominates
        v[offsets[sc]:offsets[sc]+len(pcs)] = pcs / (np.std(tx[sc].values)+1e-9)
        tx_ids.append(i); rows.append(v)
Xt = np.array(rows)
t_reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0, metric='euclidean')
t_xy_sub = t_reducer.fit_transform(Xt)
t_umap = np.full((n,2), np.nan, np.float32)
for k,i in enumerate(tx_ids): t_umap[pos[i]] = t_xy_sub[k]
print(f'T-UMAP: {len(tx_ids)} cells (block-diag over {len(tx)} subclasses, {tot} dims)')

# also keep per-cell within-subclass tx PC1/PC2 (continuous transcriptomic axis)
tx_pc1 = np.full(n, np.nan, np.float32); tx_pc2 = np.full(n, np.nan, np.float32)
for i in ids:
    sc = met_to_subclass(met.loc[i,'MET'])
    if sc in tx and i in tx[sc].index:
        r = tx[sc].loc[i].values.astype(np.float32)
        tx_pc1[pos[i]] = r[0]; tx_pc2[pos[i]] = r[1] if len(r)>1 else np.nan

# ---------- colors ----------
hexd = json.load(open(f'{R}/data/hex_color_dict.json'))
def color_of(name):
    if name in hexd and isinstance(hexd[name], str): return hexd[name]
    # fallbacks for naming variants
    for k,v in hexd.items():
        if isinstance(v,str) and (name.startswith(k) or k.startswith(name)): return v
    return None
met_types = sorted([m for m in met.MET.unique() if m!='unassigned'])
import matplotlib; from matplotlib import cm
fallback = [matplotlib.colors.to_hex(c) for c in cm.tab20(np.linspace(0,1,20))]
met_colors = {}
for k,mt in enumerate(met_types):
    met_colors[mt] = color_of(mt) or fallback[k%20]
met_colors['unassigned'] = '#cfcfcf'

# ---------- WNM projectome ----------
wm = pd.read_csv(f'{R}/data/wnm/FullMorphMetaData_Master.csv', index_col=0)
pm = pd.read_csv(f'{R}/data/wnm/ProjectionMatrix_tip_and_branch_roll_up.csv', index_col=0)
# align by swc id (strip .swc)
def strip(s): return str(s).replace('.swc','')
wm.index = [strip(x) for x in wm.index]; pm.index = [strip(x) for x in pm.index]
wnm_ids = [i for i in pm.index if i in wm.index]
targets = list(pm.columns)
P = pm.reindex(wnm_ids)[targets].values.astype(np.float32)   # axon length per target
wnm_met = wm.reindex(wnm_ids)['predicted_met_type'].fillna('unassigned').values
wnm = dict(ids=wnm_ids, targets=targets, P=P, met=wnm_met.tolist(),
           soma_region=wm.reindex(wnm_ids)['ccf_soma_location'].fillna('').values.tolist(),
           soma_x=wm.reindex(wnm_ids)['ccf_soma_x'].values.astype(np.float32),
           soma_y=wm.reindex(wnm_ids)['ccf_soma_y'].values.astype(np.float32),
           soma_z=wm.reindex(wnm_ids)['ccf_soma_z'].values.astype(np.float32),
           azimuth=pd.to_numeric(wm.reindex(wnm_ids)['azimuth'],errors='coerce').values.astype(np.float32),
           altitude=pd.to_numeric(wm.reindex(wnm_ids)['altitude'],errors='coerce').values.astype(np.float32))
print(f'WNM projectome: {len(wnm_ids)} cells × {len(targets)} targets')

cache = dict(
  ids=ids, met=met.MET.values.tolist(), t_type=met.t_type.fillna('unassigned').values.tolist(),
  subclass=met.subclass.values.tolist(),
  e_umap=e_umap, m_umap=m_umap, t_umap=t_umap, tx_pc1=tx_pc1, tx_pc2=tx_pc2,
  ephys=ephys, morph=morph, morph_cols=morph_cols, soma_depth=np.asarray(soma_depth,np.float32),
  met_types=met_types, met_colors=met_colors, subclass_order=[s for s in SUBCLASS_ORDER],
  wnm=wnm)
pickle.dump(cache, open(OUT,'wb'), protocol=4)
print(f'\nwrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)')
print('MET distribution:', {k:met.MET.tolist().count(k) for k in met_types[:6]}, '...')
