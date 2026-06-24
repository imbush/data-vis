# Zhuang ABCA whole-brain MERFISH — lazy-loading spatial explorer

`index.html` fetches `manifest.json` + per-section `data/<section>.bin` files on demand.

- Built by `src/build_zhuang_sections.py` (one uint8 binary per brain section;
  per-gene min–max → 0..255; oversized sections drop their lowest-detection genes
  to stay under the per-file limit).
- The `data/` binaries total ~8.2 GB for all 4 donors (~1.5 GB for ABCA-3 alone),
  so they are **git-ignored** and must be hosted off-repo (GitHub Pages caps a site
  at 1 GB). Point `BASE` in index.html at the data host if not co-located.
