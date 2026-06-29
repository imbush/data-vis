#!/usr/bin/env python
"""Add a 'number of similar cells' colour-by to all 4 recompute builders.

Density is computed IN-BROWSER on the current gene space (it changes as the user
filters genes / recomputes): the currently-shown genes are reduced to a K-dim PCA
latent (the curse-of-dimensionality fix — raw distances over hundreds of genes
concentrate and stop discriminating), then each cell's value is a Gaussian-kernel
effective neighbour count in that latent (median-heuristic bandwidth, anchor-
subsampled for speed). Reuses the existing powerIterTopK / readVal / colorByQC.

Inserts a button after #qc-ribo and the handler after its click registration.
JS goes into an f-string region, so braces are doubled. Idempotent."""
import sys

BUILDERS = [
    'scripts/build_svd_recompute_app_3d.py',
    'scripts/build_umap_recompute_app_3d.py',
    'scripts/build_diffmap_recompute_app_3d.py',
    'scripts/build_nmf_recompute_app_4d.py',
]
ROOT = '/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish/'

BTN_ANCHOR = '    <button id="qc-ribo" class="qc-btn">% ribo</button>\n'
BTN_NEW = ('    <button id="qc-density" class="qc-btn" title="Number of similar cells: '
           'local density of each cell in the CURRENT gene space (the genes shown in the '
           'biplot). Genes are first reduced to a K-dim PCA latent so neighbour counts stay '
           'meaningful despite the curse of dimensionality, then each cell gets a '
           'Gaussian-kernel effective count of nearby cells (median-heuristic bandwidth, '
           'anchor-subsampled). Recompute-aware.">&asymp; similar cells</button>\n')

JS_ANCHOR = ("document.getElementById('qc-ribo').addEventListener('click', "
             "() => colorByQC(qc_ribo, '% ribosomal', fmtPct));\n")

# JS written with SINGLE braces; doubled below for the f-string region.
JS_NEW = r"""
// ---- "Number of similar cells": local density in the CURRENT gene space -----
// Reduce the currently-shown genes to a K-dim PCA latent (curse-of-dimensionality
// fix: raw distances over 100s of genes concentrate and stop discriminating),
// then each cell's value = Gaussian-kernel effective neighbour count in that
// latent (median-heuristic bandwidth, anchor-subsampled). Recomputed on click so
// it always reflects the current gene filter / embedding.
function colorBySimilarCells() {
  const cellSel = [];
  for (let i = 0; i < cell_active.length; i++) if (cell_active[i]) cellSel.push(i);
  const m = cellSel.length;
  if (m < 10) { status.innerHTML = '<span style="color:#c00">need &ge;10 cells</span>'; return; }
  let basis = (typeof visibleGeneIdx === 'function') ? visibleGeneIdx()
            : (typeof shownGeneIdx === 'function') ? shownGeneIdx() : panel_idx.slice();
  if (!basis || basis.length < 3) basis = panel_idx.slice();
  if (basis.length > 600) basis = basis.slice().sort((a,b) => gene_std[b]-gene_std[a]).slice(0,600);
  const nb = basis.length;
  const K = Math.max(2, Math.min(25, nb-1, m-1));
  // Zp: m x nb, z-scored on the selected cells.
  const Zp = new Array(m); for (let i=0;i<m;i++) Zp[i] = new Float64Array(nb);
  for (let k=0;k<nb;k++) { const j=basis[k]; let s=0, ss=0;
    for (let ii=0;ii<m;ii++) { const v=readVal(cellSel[ii], j); Zp[ii][k]=v; s+=v; ss+=v*v; }
    const mean=s/m, sd=Math.sqrt(Math.max(ss/m-mean*mean,1e-18));
    for (let ii=0;ii<m;ii++) Zp[ii][k]=(Zp[ii][k]-mean)/sd; }
  // Covariance C = Zp^T Zp (nb x nb), top-K eigenvectors (cheap when nb small).
  const C=new Array(nb);
  for (let a=0;a<nb;a++) { const Ca=new Float64Array(nb);
    for (let ii=0;ii<m;ii++) { const za=Zp[ii][a]; if (za===0) continue; const Zi=Zp[ii];
      for (let b=a;b<nb;b++) Ca[b]+=za*Zi[b]; } C[a]=Ca; }
  for (let a=0;a<nb;a++) for (let b=a+1;b<nb;b++) C[b][a]=C[a][b];
  const pk = powerIterTopK(C, K, 60); const Vc = pk.U;
  // Cell PCA scores sc (m x K, row-major): sc_ik = Zp_i . Vc_k.
  const sc=new Float64Array(m*K);
  for (let k=0;k<K;k++) { const vk=Vc[k];
    for (let ii=0;ii<m;ii++) { let acc=0; const Zi=Zp[ii];
      for (let a=0;a<nb;a++) acc+=Zi[a]*vk[a]; sc[ii*K+k]=acc; } }
  // Anchor subsample (stride) -> O(m.A.K) density.
  const A=Math.min(m,1200), step=m/A; const anc=new Int32Array(A);
  for (let a=0;a<A;a++) anc[a]=Math.floor(a*step);
  // Bandwidth h^2 = median squared distance among up to 400 anchors (median heuristic).
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
  colorByQC(dens, 'similar cells (PCA K='+K+', '+nb+' genes)', v => Math.round(v).toLocaleString());
}
document.getElementById('qc-density').addEventListener('click', () => {
  const b=document.getElementById('qc-density'); b.disabled=true; status.innerHTML='computing density&hellip;';
  setTimeout(() => { try { colorBySimilarCells(); } finally { b.disabled=false; } }, 30);
});
"""

def patch(path):
    s = open(path).read()
    if 'qc-density' in s:
        print(f'  {path}: already patched'); return
    assert BTN_ANCHOR in s, f'button anchor missing in {path}'
    assert JS_ANCHOR in s, f'js anchor missing in {path}'
    s = s.replace(BTN_ANCHOR, BTN_ANCHOR + BTN_NEW, 1)
    js_doubled = JS_NEW.replace('{', '{{').replace('}', '}}')
    s = s.replace(JS_ANCHOR, JS_ANCHOR + js_doubled, 1)
    open(path, 'w').write(s)
    print(f'  {path}: patched (+button +handler)')

if __name__ == '__main__':
    for b in BUILDERS:
        patch(ROOT + b)
