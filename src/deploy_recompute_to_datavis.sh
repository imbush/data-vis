#!/bin/bash
# Copy rebuilt cohort HTMLs from notebooks/ to data-vis/, then re-apply the
# copy-link snippet patcher. Covers all 10 cohorts:
#   per-subclass (lamp5/vip/sst/pvalb/sncg): archetype + svd + umap + diffmap
#   pooled       (allinhib/allexc/devinhib/allinhib_v1alm/allexc_v1alm): nmf + svd + umap + diffmap
#
# Files >95 MB are skipped (GitHub's per-file hard limit is 100 MB; soft warn
# at 50 MB). When a file is skipped its prior copy in data-vis/ is removed so
# the deploy doesn't ship a stale version under the live URL.
set -e

SRC=/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish/notebooks
DST=/Users/inlebush/cs/lab/green/data-vis
MAX_BYTES=$((99 * 1000 * 1000))   # GitHub hard limit is 100 MB; this is the safety margin

# copy_one_or_skip: enforces the 95 MB GitHub guard. Removes any stale prior
# copy under data-vis/ when a file is skipped so the live URL doesn't keep
# serving an outdated explorer.
copy_one_or_skip() {
  local src=$1 dst=$2 g=$3 f=$4
  [ -f "$src" ] || { echo "  MISSING: $src"; return; }
  local bytes=$(stat -f%z "$src")
  if [ "$bytes" -gt "$MAX_BYTES" ]; then
    if [ -f "$dst" ]; then rm "$dst"; fi
    printf '  SKIP (%s MB > %s MB cap): %s/%s — kept local only\n' \
      "$(( bytes / 1000000 ))" "$(( MAX_BYTES / 1000000 ))" "$g" "$f"
    return
  fi
  cp "$src" "$dst"
  printf '  copied: %s/%s (%s MB)\n' "$g" "$f" "$(( bytes / 1000000 ))"
}

# NOTE: the per-subclass cohorts (lamp5/vip/sst/pvalb/sncg) were retired — they're
# fully covered by the pooled AllInhib + MGE/CGE lineage cohorts. The per-subclass
# .pkl HVG caches are kept because the pooled cohorts derive their HVG from them.

# Standalone explorers (no NMF/SVD/UMAP/diffmap quartet)
for f in patchseq_overlay_explorer.html; do
  if [ -f "$SRC/$f" ]; then
    DST_DIR=$DST/patchseq
    mkdir -p "$DST_DIR"
    copy_one_or_skip "$SRC/$f" "$DST_DIR/$f" patchseq "$f"
  fi
done

# pooled cohorts: nmf is the 4th view
for g in allinhib allinhib_mge allinhib_cge allexc devinhib devinhib_mge devinhib_cge allinhib_v1alm allexc_v1alm; do
  for f in "${g}_nmf_recompute_explorer_4d.html" \
           "${g}_svd_recompute_explorer_3d.html" \
           "${g}_umap_recompute_explorer_3d.html" \
           "${g}_diffmap_recompute_explorer_3d.html" \
           "${g}_marker_explorer_3d.html" \
           "${g}_geneassoc_explorer_3d.html"; do
    copy_one_or_skip "$SRC/$f" "$DST/$g/$f" "$g" "$f"
  done
done

# Re-apply the copy-link snippet (cp wipes it from the files)
cd /Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish
.venv/bin/python scripts/patch_copy_link.py | tail -25

echo
echo "Spot-checking one file..."
P=$DST/allinhib/allinhib_svd_recompute_explorer_3d.html
.venv/bin/python -c "
import os
h = open('$P').read()
print(f'  copy-link snippet:  {h.count(\"copy-link snippet v4\")} (1 OK)')
print(f'  Region button:      {h.count(\"qc-region\")} (>=1 OK)')
print(f'  Regress-ribo btn:   {h.count(\"regress-ribo\")} (>=1 OK)')
print(f'  recompute btn:      {h.count(\"recompute-btn\")} (>=1 OK)')
print(f'  GO-removed marker:  {0 if \"GO biological process\" in h else 1} (1 OK = GO terms absent)')
"
