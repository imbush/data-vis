# data-vis

Self-contained interactive visualizations of lab data, served via GitHub Pages.

Live: <https://imbush.github.io/data-vis/>

## What's here

```
data-vis/
  index.html                              Landing page (imbush.github.io/data-vis/)
  .nojekyll                               Tell Pages to skip Jekyll
  lamp5/
    index.html                            → archetype explorer (same as below)
    lamp5_archetype_explorer_4d.html      K=4 NMF tetrahedron
    lamp5_svd_recompute_explorer_3d.html  Top-3 SVD biplot, recompute-on-subset
    lamp5_umap_recompute_explorer_3d.html 3D UMAP, recompute-on-subset (umap-js)
    lamp5_diffmap_recompute_explorer_3d.html Diffmap (Laplacian eigenmap recompute)
  sst/   …same 4 files for Sst…
  pvalb/ …same 4 files for Pvalb…
  vip/   …same 4 files for Vip…
  sncg/  …same 4 files for Sncg…
  src/
    build_lamp5_archetype_app_4d.py       Base config + NMF archetype builder
    build_svd_recompute_app_3d.py         SVD biplot with in-browser SVD recompute
    build_umap_recompute_app_3d.py        UMAP biplot with in-browser UMAP recompute
    build_diffmap_recompute_app_3d.py     Diffmap with Laplacian-eigenmap recompute
    build_umap_app_3d.py                  Static UMAP (no recompute)
```

Each explorer is a single self-contained HTML (Plotly via CDN; data embedded
inline as JSON). They're built from the scripts in `src/` and copied into the
per-group folders before `git push`.

## Variants

Each cell-type folder has four explorers sharing a common viz-nav row:

| Variant | Embedding | Recompute on subset? |
|---|---|---|
| Archetype | K=4 NMF tetrahedron (consensus over 20 inits) | No |
| SVD biplot | Top-3 SVD of the z-scored panel HVG | Yes (power iteration) |
| UMAP | 3D scanpy UMAP (PCA → 15-NN → UMAP) | Yes (umap-js client-side) |
| Diffmap | Top-3 diffusion components (scanpy) | Yes (Laplacian eigenmap) |

The recompute viewers let you pick a subset of subtypes via checkboxes, then
click **Recompute on panel HVG** or **Recompute on shown genes** to refit the
embedding to that subset (and a chosen lower rank, for SVD and UMAP). The SVD
recompute also has a live variance-explained bar chart, a heatmap of cells ×
panel HVG ordered by PC1 pseudotime, and a smoothed-expression line graph that
appears when you hover a gene.

## What's NOT here (excluded by `.gitignore`)

The underlying scRNA-seq data is **not** in this repo. To rebuild from scratch
you need access to the lab machine and:

- `data/v1_neurons_proc.h5ad` — Tasic 2018 V1 anndata (12,863 GABAergic cells ×
  27,505 genes), log-CPM normalized. Source: GEO GSE115746.
- `notebooks/cache/{Group}.pkl` — per-subclass cleaned-HVG caches built by
  `cleaned_cascade_subtype_order.ipynb` (HVG selection on Seurat-v3 variance
  followed by KMeans-based noise cluster removal).
- `notebooks/cache/{Group}_archetype_proj.pkl` (and `..._full.pkl`) — projection
  caches produced by the build scripts on first run (consensus NMF + NNLS
  projection).

## Updating the deployed visualizations

```sh
# On the lab machine, with caches in place:
cd green/sequencing/tasic2018_v1_merfish
for g in Lamp5 Sst Pvalb Vip Sncg; do
  .venv/bin/python scripts/build_lamp5_archetype_app_4d.py     $g
  .venv/bin/python scripts/build_svd_recompute_app_3d.py       $g
  .venv/bin/python scripts/build_umap_recompute_app_3d.py      $g
  .venv/bin/python scripts/build_diffmap_recompute_app_3d.py   $g
done

# Copy outputs into the data-vis folders + bump index.html copies:
cd ../../data-vis
for g in lamp5 sst pvalb vip sncg; do
  for k in archetype_explorer_4d svd_recompute_explorer_3d \
           umap_recompute_explorer_3d diffmap_recompute_explorer_3d; do
    cp ../sequencing/tasic2018_v1_merfish/notebooks/${g}_${k}.html $g/${g}_${k}.html
  done
  cp $g/${g}_archetype_explorer_4d.html $g/index.html
done

git add -A && git commit -m "Refresh explorers" && git push
```

## How each explorer is built (one paragraph)

For each cell-type group (one subclass or a small union, e.g. `Sncg +
Serpinf1`): filter the Tasic 2018 V1 GABAergic anndata to that group, drop the
published outlier subtypes (`Sst Chodl`, `Pvalb Vipr2`, `Pvalb Gabrg1`,
`Lamp5 Lhx6`, plus `Lamp5 Krt73` at runtime) and per-cell QC drops
(`F2S4_171122_010_H01` for Lamp5 and `F2S4_170406_018_D01` for Pvalb — both
high-ribo low-detection outliers), select ~150 high-variance "cleaned" genes
via a KMeans-on-pseudotime noise filter, run a 20-init consensus NMF with K=4,
then NNLS-project the top ~3,000 by-mean genes from the broader transcriptome
onto the fixed W to get imputed barycentric coordinates. The 4 archetype
loadings (`W` for cells, column-normalized `H` for genes) give barycentric
positions in a regular tetrahedron. The SVD / UMAP / diffmap recompute variants
re-use the same per-cell cohort and broader gene pool but fit a different
embedding to the panel matrix — and on click, refit that embedding in the
browser to whichever subtype subset (or shown-gene subset) you select.
