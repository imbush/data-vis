# Zhuang ABCA whole-brain MERFISH — lazy-loading spatial explorer

`index.html` fetches `manifest.json` + per-section `data/<section>.bin` on demand.

- Built by `src/build_zhuang_sections.py`: one **uint4** binary per brain section
  (per-gene min–max → 16 levels, 2 nibbles/byte; coords float32; class index uint8).
  uint4 halves storage, so the full 1,122-gene panel fits in every section.
- Gene names are Allen WMB symbols (var `gene_symbol`); cells coloured by WMB class
  or by a searched gene (magma). Deep-link with `?donor=&section=&gene=`.
- Currently deployed: **Zhuang-ABCA-3** (23 sections, ~0.9 GB). Re-run
  `build_zhuang_sections.py <DONOR>` to add more donors.
