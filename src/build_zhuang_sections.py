#!/usr/bin/env python
"""Shard the Zhuang ABCA whole-brain MERFISH atlas into one binary file per
brain section, for the lazy-loading spatial transcriptomics explorer.

For each section we write a compact little-endian .bin holding:
    gene_min  : float32[ngene]          per-gene min (for de-quantising)
    gene_max  : float32[ngene]          per-gene max
    x         : float32[ncell]          in-plane coord 1
    y         : float32[ncell]          in-plane coord 2
    class_idx : uint8[ncell]            index into manifest.classes
    expr      : uint8[ncell*ngene]      per-gene min-max -> 0..255, row-major
                                        (cell i, local gene g at i*ngene+g)

Expression is uint8 (full 0.0625-equivalent precision via per-gene min-max).
A section that would exceed MAX_BYTES at the full 1122-gene panel drops its
lowest-detection genes until it fits (the MERFISH panel is curated, so this
only trims a tail off the few largest sections).

A single manifest.json indexes every section + the global gene list and the
class colour table. The explorer fetches manifest.json once, then a section's
.bin on demand.

Usage:  python build_zhuang_sections.py [DONOR ...]   (default: all 4)
"""
import os, sys, json, glob, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as ss

ABC  = '/Users/Shared/lab-data/zhuang-abca-whole-brain/_abc_cache'
META = ABC + '/metadata'
EXPR = ABC + '/expression_matrices'
OUT_DIR = '/Users/inlebush/cs/lab/green/data-vis/spatial_zhuang'
DATA_DIR = os.path.join(OUT_DIR, 'data')

ALL_DONORS = ['Zhuang-ABCA-1', 'Zhuang-ABCA-2', 'Zhuang-ABCA-3', 'Zhuang-ABCA-4']
MAX_BYTES  = 95 * 1024 * 1024        # keep each .bin safely under GitHub's 100 MB/file


def annotation_view(donor):
    p = glob.glob(f'{META}/{donor}/*/views/cell_metadata_with_cluster_annotation.csv')
    return p[0] if p else None


def expr_h5ad(donor):
    p = glob.glob(f'{EXPR}/{donor}/*/{donor}-log2.h5ad')
    return p[0] if p else None


def detect_plane(meta):
    """Cutting axis = the coordinate with the least within-section spread;
    plot the other two. Returns (axis_a, axis_b)."""
    var = {}
    for ax in ('x', 'y', 'z'):
        var[ax] = meta.groupby('brain_section_label')[ax].std().mean()
    cut = min(var, key=var.get)            # near-constant within a section
    plane = [ax for ax in ('x', 'y', 'z') if ax != cut]
    return plane


def main():
    donors = [d for d in sys.argv[1:] if d in ALL_DONORS] or ALL_DONORS
    os.makedirs(DATA_DIR, exist_ok=True)

    # Build a GLOBAL class colour table + gene list incrementally.
    class_color = {}                       # class name -> hex
    global_genes = None
    sections_manifest = []

    for donor in donors:
        av = annotation_view(donor); hp = expr_h5ad(donor)
        if not av or not hp:
            print(f'  {donor}: missing metadata/expression, skip'); continue
        print(f'\n=== {donor} ===')
        meta = pd.read_csv(av, usecols=['cell_label', 'brain_section_label', 'x', 'y', 'z',
                                        'class', 'subclass', 'class_color'])
        meta = meta.dropna(subset=['x', 'y', 'z'])
        plane = detect_plane(meta)
        print(f'  cutting plane -> plot axes {plane}; {len(meta):,} cells, '
              f'{meta["brain_section_label"].nunique()} sections')
        for c, col in zip(meta['class'], meta['class_color']):
            if c not in class_color and isinstance(col, str):
                class_color[c] = col

        print('  loading expression matrix ...')
        A = ad.read_h5ad(hp)
        genes = list(A.var_names)
        if global_genes is None:
            global_genes = genes
        assert genes == global_genes, 'gene panel differs between donors!'
        Xall = A.X
        # map cell_label -> row in expression matrix
        row_of = {cl: i for i, cl in enumerate(A.obs_names)}

        meta = meta[meta['cell_label'].isin(row_of)]
        for sec, g in meta.groupby('brain_section_label', sort=True):
            rows = np.array([row_of[c] for c in g['cell_label']], dtype=np.int64)
            X = Xall[rows]
            X = X.toarray() if ss.issparse(X) else np.asarray(X)
            X = np.asarray(X, dtype=np.float32)
            ncell = X.shape[0]
            det = (X > 0).mean(0)                          # per-gene detection

            # How many genes fit? keep highest-detection genes first.
            # bytes = 8*ngene (minmax) + 9*ncell (x,y f32 + class u8) + ncell*ngene
            def fits(ng): return 8 * ng + 9 * ncell + ncell * ng <= MAX_BYTES
            ng_max = (MAX_BYTES - 9 * ncell) // (ncell + 8)
            ng_keep = int(min(len(genes), max(2, ng_max)))
            if ng_keep < len(genes):
                keep = np.argsort(-det)[:ng_keep]
                keep.sort()
            else:
                keep = np.arange(len(genes))
            Xk = X[:, keep]
            lo = Xk.min(0); hi = Xk.max(0); rng = np.maximum(hi - lo, 1e-9)
            q = np.clip(np.round((Xk - lo) / rng * 255.0), 0, 255).astype(np.uint8)

            xy = g[plane].values.astype(np.float32)
            cls_idx = g['class'].map(lambda c: list(class_color).index(c)
                                     if c in class_color else 0).values.astype(np.uint8)

            # write .bin
            fn = f'{sec}.bin'
            with open(os.path.join(DATA_DIR, fn), 'wb') as fh:
                fh.write(lo.astype('<f4').tobytes())
                fh.write(hi.astype('<f4').tobytes())
                fh.write(np.ascontiguousarray(xy[:, 0]).astype('<f4').tobytes())
                fh.write(np.ascontiguousarray(xy[:, 1]).astype('<f4').tobytes())
                fh.write(cls_idx.tobytes())
                fh.write(np.ascontiguousarray(q).tobytes())
            sz = os.path.getsize(os.path.join(DATA_DIR, fn))
            sections_manifest.append({
                'id': str(sec), 'donor': donor, 'plane': plane,
                'ncell': int(ncell), 'ngene': int(len(keep)),
                'genes': 'all' if ng_keep == len(genes) else keep.tolist(),
                'bbox': [float(xy[:, 0].min()), float(xy[:, 0].max()),
                         float(xy[:, 1].min()), float(xy[:, 1].max())],
                'file': 'data/' + fn, 'bytes': sz,
            })
        del A, Xall
        print(f'  wrote {sum(1 for s in sections_manifest if s["donor"]==donor)} sections')

    classes = [{'name': c, 'color': class_color[c]} for c in class_color]
    manifest = {
        'genes': global_genes,
        'classes': classes,
        'expr_levels': 256,
        'sections': sorted(sections_manifest, key=lambda s: (s['donor'], s['id'])),
    }
    with open(os.path.join(OUT_DIR, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh)
    tot = sum(s['bytes'] for s in sections_manifest)
    over = [s for s in sections_manifest if s['bytes'] > 100 * 1024 * 1024]
    trimmed = [s for s in sections_manifest if s['genes'] != 'all']
    print(f'\nmanifest: {len(sections_manifest)} sections, {len(global_genes)} genes, '
          f'{len(classes)} classes')
    print(f'total data: {tot/1e9:.2f} GB | sections over 100MB: {len(over)} | '
          f'gene-trimmed sections: {len(trimmed)}')
    if trimmed:
        mn = min(s['ngene'] for s in trimmed)
        print(f'  smallest trimmed gene panel: {mn} genes')


if __name__ == '__main__':
    main()
