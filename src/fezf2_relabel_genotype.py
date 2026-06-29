#!/usr/bin/env python
"""Post-process the Fezf2 recompute HTMLs: the shared template hardcodes the
`dissected_region` dimension as "Region" with V1/ALM colours. For Fezf2 that
dimension is GENOTYPE, so relabel the toggle + colour button and give
Control/Fezf2-KO distinct colours."""
import os, glob
NB = '/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish/notebooks'
REPL = [
    ('<span class="label">Region:</span>', '<span class="label">Genotype:</span>'),
    ('title="Colour each cell by its dissected region (V1 = orange, ALM = purple). '
     'Greyed out for single-region cohorts.">Region</button>',
     'title="Colour each cell by genotype (Control = slate, Fezf2-KO = red).">'
     'Genotype</button>'),
    ('title="Use all cells regardless of region">Both',
     'title="Use cells of both genotypes">Both'),
    ("'ALM': '#9467bd' }",
     "'ALM': '#9467bd', 'Control': '#5b6770', 'Fezf2-KO': '#d62728' }"),
    ("'ALM': '#9467bd'}",
     "'ALM': '#9467bd', 'Control': '#5b6770', 'Fezf2-KO': '#d62728'}"),
]
for f in sorted(glob.glob(os.path.join(NB, 'fezf2_*recompute_explorer_*.html'))):
    s = open(f).read(); n = 0
    for a, b in REPL:
        if a in s: s = s.replace(a, b); n += 1
    open(f, 'w').write(s)
    print(f'  relabelled {os.path.basename(f)}: {n}/{len(REPL)} substitutions')
