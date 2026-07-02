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

# ---------- morphology: 50 features + M-UMAP (paper's own t-seeded coords, n=389) ----------
mf = pd.read_csv(f'{R}/data/patchseq/morph_features_mMET_exc_wide_normalized.csv', index_col=0)
mf.index = mf.index.astype(str)
morph_cols = [c for c in mf.columns]
mf_ids = [i for i in ids if i in mf.index]
pos = {i:k for k,i in enumerate(ids)}
# Use the paper's published morphology UMAP (derived_data/morpho_umap_coordinates_t_seeded.csv)
mu = pd.read_csv(f'{R}/derived_data/morpho_umap_coordinates_t_seeded.csv', index_col=0)
mu.index = mu.index.astype(str)
m_umap = np.full((n,2), np.nan, np.float32)
for i in mu.index:
    if i in pos: m_umap[pos[i]] = mu.loc[i,['x','y']].values.astype(np.float32)
print(f"M-UMAP: {int(np.isfinite(m_umap[:,0]).sum())} cells (paper's t-seeded coords)")
# per-cell morph features (full length, NaN where absent)
morph = {}
for c in morph_cols:
    arr = np.full(n, np.nan, np.float32); s = mf.reindex(mf_ids)[c].values.astype(np.float32)
    for k,i in enumerate(mf_ids): arr[pos[i]] = s[k]
    morph[c] = arr
print(f'M-UMAP: {len(mf_ids)} cells; {len(morph_cols)} morph features')

# soma depth from pia (morph) for the 389
soma_depth = morph.get('soma_aligned_dist_from_pia', np.full(n, np.nan, np.float32))

# ---------- T-UMAP: the paper's REAL co-embedded UMAP (FACS reference + Patch-seq) ----------
# tx_umap_coordinates_patchseq_facs_co-embedded.csv is keyed by RNA sample_id; there is no
# sample_id<->specimen_id crosswalk in the repo (it lives in an internal Allen anno file), so
# the T panel is its own population: FACS reference cells (grey backdrop) + Patch-seq cells
# coloured by subclass. This is the genuine Fig. 2a transcriptomic landscape.
co = pd.read_csv(f'{R}/derived_data/tx_umap_coordinates_patchseq_facs_co-embedded.csv', index_col=0)
co.index = co.index.astype(str)
ref_lab = pd.read_csv(f'{R}/derived_data/ref_dataset_subclass_labels.csv', index_col=0).iloc[:,0].astype(str)
ps_lab  = pd.read_csv(f'{R}/derived_data/ps_dataset_subclass_labels.csv', index_col=0).iloc[:,0].astype(str)
def norm_sc(s):
    return str(s).replace('patch-seq-','').replace('facs-','').strip().replace('L4 & L5 IT','L4/L5 IT')
ref_sids = [s for s in co.index if s in ref_lab.index]
ps_sids  = [s for s in co.index if s in ps_lab.index]
t_ref = dict(x=co.reindex(ref_sids)['x'].values.astype(np.float32),
             y=co.reindex(ref_sids)['y'].values.astype(np.float32))
t_ps  = dict(x=co.reindex(ps_sids)['x'].values.astype(np.float32),
             y=co.reindex(ps_sids)['y'].values.astype(np.float32),
             subclass=[norm_sc(ps_lab[s]) for s in ps_sids])
print(f'T-UMAP (paper co-embedded): {len(ref_sids)} FACS reference + {len(ps_sids)} Patch-seq')

# per-cell within-subclass transcriptomic PC1/PC2 (specimen-keyed; for E/M feature colouring)
sc_files = {'L23-IT':'L23-IT','L4-L5-IT':'L4-L5-IT','L6-IT':'L6-IT','L5L6-IT-Car3':'L5L6-IT-Car3',
            'L5-ET':'L5-ET','L5-NP':'L5-NP','L6-CT':'L6-CT','L6b':'L6b'}
tx = {}
for sc,f in sc_files.items():
    p = f'{R}/derived_data/ps_tx_pca_results/{f}_ps_transformed_pcs.csv'
    if os.path.exists(p):
        d = pd.read_csv(p, index_col=0); d.index = d.index.astype(str); tx[sc] = d
tx_pc1 = np.full(n, np.nan, np.float32); tx_pc2 = np.full(n, np.nan, np.float32)
for i in ids:
    sc = met_to_subclass(met.loc[i,'MET'])
    if sc in tx and i in tx[sc].index:
        r = tx[sc].loc[i].values.astype(np.float32)
        tx_pc1[pos[i]] = r[0]; tx_pc2[pos[i]] = r[1] if len(r)>1 else np.nan

# ---------- reconstructed gene expression (log2 CPM+1) from the paper's tx-PCA ----------
# X_hat = scores · loadings^T + center, per transcriptomic subclass. This recovers per-cell
# expression of the per-subclass highly-variable genes the paper's PCA was fit on (a low-rank,
# HVG-restricted estimate — the true full count matrix is fastq-only on NeMO). A cell gets a
# value for a gene only if that gene is in its subclass's HVG set.
broad_of_sc = {'L23-IT':'L2/3 IT','L4-L5-IT':'L4/L5 IT','L6-IT':'L6 IT','L5L6-IT-Car3':'L5/L6 IT Car3',
               'L5-ET':'L5 ET','L5-NP':'L5 NP','L6-CT':'L6 CT','L6b':'L6b'}
gene_expr = {}
gene_present_n = {}
for sc, f in sc_files.items():
    wp = f'{R}/derived_data/ref_tx_pca_results/{f}_tx_pca_weights.csv'
    cp = f'{R}/derived_data/ref_tx_pca_results/{f}_tx_pca_centers.csv'
    if not (os.path.exists(wp) and os.path.exists(cp) and f in [s for s in ['L23-IT','L4-L5-IT','L6-IT','L5L6-IT-Car3','L5-ET','L5-NP','L6-CT','L6b']]):
        continue
    W = pd.read_csv(wp, index_col=0); Cc = pd.read_csv(cp, index_col=0).iloc[:,0]
    S = tx.get(sc)
    if S is None: continue
    k = min(W.shape[1], S.shape[1])
    Xhat = S.values[:, :k] @ W.values[:, :k].T + Cc.values[None, :]     # cells × genes
    genes = list(W.index)
    rows = [(pos[i], r) for r, i in enumerate(S.index) if i in pos and met_to_broad(met.loc[i,'MET'])==broad_of_sc[f]]
    if not rows: continue
    ridx = np.array([p for p,_ in rows]); sidx = np.array([r for _,r in rows])
    for gj, g in enumerate(genes):
        if g not in gene_expr: gene_expr[g] = np.full(n, np.nan, np.float32); gene_present_n[g]=0
        gene_expr[g][ridx] = Xhat[sidx, gj].astype(np.float32)
        gene_present_n[g] += len(ridx)
genes_all = sorted(gene_expr)
print(f'reconstructed expression: {len(genes_all)} unique HVG/DEGs across {len(sc_files)} subclasses')
# top genes by variance (for the pushed HTML); full set kept for the local build
def gvar(g):
    v = gene_expr[g]; v = v[np.isfinite(v)]; return float(np.var(v)) if v.size>5 else 0.0
genes_ranked = sorted(genes_all, key=gvar, reverse=True)

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
# subclass colours (consistent across T/E/M panels), from a representative MET per subclass
_rep = {'L2/3 IT':'L2/3 IT','L4/L5 IT':'L4 IT','L6 IT':'L6 IT-1','L5/L6 IT Car3':'L5/L6 IT Car3',
        'L5 ET':'L5 ET-2','L5 NP':'L5 NP','L6 CT':'L6 CT-1','L6b':'L6b','other':'unassigned'}
subclass_colors = {sc: met_colors.get(rep, '#bbbbbb') for sc,rep in _rep.items()}

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
  e_umap=e_umap, m_umap=m_umap, tx_pc1=tx_pc1, tx_pc2=tx_pc2,
  t_ref=t_ref, t_ps=t_ps,
  ephys=ephys, morph=morph, morph_cols=morph_cols, soma_depth=np.asarray(soma_depth,np.float32),
  met_types=met_types, met_colors=met_colors, subclass_colors=subclass_colors,
  subclass_order=[s for s in SUBCLASS_ORDER], wnm=wnm,
  genes_ranked=genes_ranked, gene_expr={g:np.asarray(gene_expr[g],np.float32) for g in genes_all})
pickle.dump(cache, open(OUT,'wb'), protocol=4)
print(f'\nwrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)')
print('MET distribution:', {k:met.MET.tolist().count(k) for k in met_types[:6]}, '...')
