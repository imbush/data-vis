# data-vis

Self-contained interactive visualizations of lab data, served via GitHub Pages.

Live: <https://imbush.github.io/data-vis/>

## What's here

```
data-vis/
  index.html              Landing page (imbush.github.io/data-vis/)
  .nojekyll               Tell Pages to skip Jekyll
  lamp5/index.html        Lamp5 K=4 archetype explorer
  sst/index.html          Sst   K=4 archetype explorer
  pvalb/index.html        Pvalb K=4 archetype explorer
  vip/index.html          Vip   K=4 archetype explorer
  sncg/index.html         Sncg  K=4 archetype explorer
  src/
    build_archetype_app_4d.py    Source for the explorers above
```

Each explorer is a single self-contained HTML (Plotly via CDN; data embedded
inline as JSON). They're built from `src/build_archetype_app_4d.py` and copied
into the per-group folders before `git push`.

## What's NOT here (excluded by `.gitignore`)

The underlying scRNA-seq data is **not** in this repo. To rebuild from scratch
you need access to the lab machine and:

- `data/v1_neurons_proc.h5ad` — Tasic 2018 V1 anndata (12,863 GABAergic cells ×
  27,505 genes), log-CPM normalized. Source: GEO GSE115746.
- `notebooks/cache/{Group}.pkl` — per-subclass cleaned-HVG caches built by
  `cleaned_cascade_subtype_order.ipynb` (HVG selection on Seurat-v3 variance
  followed by KMeans-based noise cluster removal).
- `notebooks/cache/{Group}_archetype_proj.pkl` — projection caches produced by
  this script on first run (consensus NMF + NNLS projection).

## Updating the deployed visualizations

```sh
# On the lab machine, with caches in place:
cd green/sequencing/tasic2018_v1_merfish
for g in Lamp5 Sst Pvalb Vip Sncg; do
  .venv/bin/python scripts/build_lamp5_archetype_app_4d.py $g
done

# Copy each output into its data-vis folder (rename to index.html):
cd ../../data-vis
for g in lamp5 sst pvalb vip sncg; do
  cp ../sequencing/tasic2018_v1_merfish/notebooks/${g}_archetype_explorer_4d.html $g/index.html
done

git add -A && git commit -m "Refresh explorers" && git push
```

## How each explorer is built (one paragraph)

For each cell-type group (one subclass or a small union, e.g. `Sncg + Serpinf1`):
filter the Tasic 2018 V1 GABAergic anndata to that group, drop the published
outlier subtypes (`Sst Chodl`, `Pvalb Vipr2`, `Pvalb Gabrg1`, `Lamp5 Lhx6`, plus
`Lamp5 Krt73` at runtime), select ~150 high-variance "cleaned" genes via a
KMeans-on-pseudotime noise filter, run a 20-init consensus NMF with K=4, then
NNLS-project the top ~3,000 by-mean genes from the broader transcriptome onto
the fixed W to get imputed barycentric coordinates. The 4 archetype loadings
(`W` for cells, column-normalized `H` for genes) give barycentric positions in
a regular tetrahedron. Plotly's `Scatter3d` renders both tetrahedrons side by
side; the embedded JSON expression matrix (rounded to int×10 to keep file size
manageable) drives the cross-hover recoloring.
