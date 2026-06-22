#!/usr/bin/env python
"""Inject the copy-link snippet into every explorer HTML in data-vis/ + the
local notebooks/ directory. If an older snippet version is already present,
strip it out before injecting the new one.

The snippet is inserted right before `</body>` so it runs after all existing
script blocks have defined their handlers.
"""
import os, sys, glob, re

ROOT = '/Users/inlebush/cs/lab/green'
DEPLOY = os.path.join(ROOT, 'data-vis')
LOCAL  = os.path.join(ROOT, 'sequencing', 'tasic2018_v1_merfish', 'notebooks')
SNIPPET_PATH = os.path.join(
    ROOT, 'sequencing', 'tasic2018_v1_merfish',
    'scripts', 'copy_link_snippet.js',
)
CURRENT_VERSION = 'v5'
CURRENT_MARKER  = f'copy-link snippet {CURRENT_VERSION}'
# Regex matches a <script>…</script> block whose comment header starts with
# `copy-link snippet v` (any version), so we can strip the old block before
# re-injecting the latest one.
OLD_BLOCK_RE = re.compile(
    r'\n?<script>\s*\n?/\* copy-link snippet v\d+.*?</script>\n?',
    re.DOTALL,
)


def main():
    only_deploy = '--deploy-only' in sys.argv
    with open(SNIPPET_PATH) as f:
        snippet = f.read()

    block = '\n<script>\n' + snippet + '\n</script>\n'

    files = sorted(glob.glob(os.path.join(DEPLOY, '*', '*.html')))
    if not only_deploy:
        files += sorted(glob.glob(os.path.join(LOCAL, '*.html')))
    if not files:
        print('no HTMLs found')
        return

    patched = 0; refreshed = 0; skipped = 0
    for f in files:
        with open(f) as fh:
            html = fh.read()
        if CURRENT_MARKER in html:
            skipped += 1
            continue
        if '</body>' not in html:
            print(f'  WARN: no </body> in {f}, skipping')
            skipped += 1
            continue
        # If an OLDER snippet is already in there, strip it first.
        if 'copy-link snippet v' in html:
            stripped = OLD_BLOCK_RE.sub('', html, count=1)
            if stripped == html:
                # The regex failed to match an old block we know is in there —
                # don't risk double-injecting; skip with a warning.
                print(f'  WARN: old snippet present but regex did not strip in {f}')
                skipped += 1
                continue
            new = stripped.replace('</body>', block + '</body>', 1)
            refreshed += 1
            label = 'refreshed'
        else:
            new = html.replace('</body>', block + '</body>', 1)
            patched += 1
            label = 'patched'
        with open(f, 'w') as fh:
            fh.write(new)
        size_mb = os.path.getsize(f) / 1e6
        print(f'  {label}: {os.path.relpath(f, ROOT)} ({size_mb:.1f} MB)')

    print(f'\nDone: {patched} new-patched, {refreshed} refreshed from older version, '
          f'{skipped} already up-to-date.')


if __name__ == '__main__':
    main()
