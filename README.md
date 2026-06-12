# data-vis

Self-contained interactive visualizations of lab data, served via GitHub Pages.

Live: <https://imbush.github.io/data-vis/>

## Layout

```
data-vis/
  index.html              Landing page (linked from imbush.github.io/data-vis/)
  .nojekyll               Tell Pages to skip Jekyll
  lamp5/index.html        Lamp5 K=4 archetype explorer
  sst/index.html          Sst   K=4 archetype explorer
  pvalb/index.html        Pvalb K=4 archetype explorer
  vip/index.html          Vip   K=4 archetype explorer
  sncg/index.html         Sncg  K=4 archetype explorer
```

Each explorer is a single self-contained HTML (Plotly via CDN; data embedded
inline as JSON). They're rebuilt from
[`green/sequencing/tasic2018_v1_merfish/scripts/build_lamp5_archetype_app_4d.py`](https://github.com/imbush)
on the local machine and copied here before `git push`.

## Updating

```sh
cd green/sequencing/tasic2018_v1_merfish
.venv/bin/python scripts/build_lamp5_archetype_app_4d.py Lamp5
# (repeat for Sst, Pvalb, Vip, Sncg)
cp notebooks/{lamp5,sst,pvalb,vip,sncg}_archetype_explorer_4d.html \
   ../../data-vis/{lamp5,sst,pvalb,vip,sncg}/index.html  # rename per folder
cd ../../data-vis && git add -A && git commit -m "Refresh explorers" && git push
```
