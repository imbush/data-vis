#!/usr/bin/env python
"""Shared patch-seq MET overlay injected into a (redesigned) recompute explorer.

Both the Tolias (MOp / Scala 2020) and Gouwens (VISp / 2020) explorers use this
one implementation so they stay at feature parity. Each caller builds the common
JS-const block (see `js_consts`) and calls `assemble`.

The overlay:
  - Adds a diamond-marker trace of the patch-seq (MET) cells on the cell plot;
    the underlying Tasic cells are dimmed (low alpha) and made non-interactive.
  - Renders a "Patch-seq Info" box BELOW the plots (morphology image + metadata
    + ephys/morph feature grids). Before any hover it reads "Hover on a cell, to
    see info."
  - Syncs the page's Color Scheme controls to the patch-seq cells, colouring them
    from THEIR OWN transcriptome (gene-hover expression; Counts / Genes / %ribo
    computed from MET_EXPR) and their dissection layer (Layer button). Pseudotime
    falls back to the imputed-type colour.
  - Subtype-checkbox subsetting hides the matching patch-seq cells too.
  - Retitles the page and swaps in the patch-seq data citation footer.
"""
import os, re

ROOT = '/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish'
SITE = '/Users/inlebush/cs/lab/green/data-vis'

# Unified overlay logic. Plain string (single braces) — NOT an f-string.
OVERLAY_JS = r"""
(function(){
  const _bin = atob(MET_EXPR_B64);
  const met_expr = new Uint8Array(_bin.length);
  for (let k=0; k<_bin.length; k++) met_expr[k] = _bin.charCodeAt(k);
  const MET_GENES = MET_EXPR_SHAPE[1];

  function metColor(i) {
    const c = MET_IMPUTED_CLUSTER[i];
    return (typeof subtype_palette !== 'undefined' && subtype_palette[c])
            ? subtype_palette[c] : '#888';
  }
  const met_default_colors = Array.from({length: MET_N}, (_, i) => metColor(i));
  let metCurrentColors = met_default_colors.slice();

  // ---- per-cell metrics derived from the patch-seq transcriptome ----------
  const metCounts = new Float64Array(MET_N);
  const metGenes  = new Float64Array(MET_N);
  const metRibo   = new Float64Array(MET_N);
  let riboMask = null;
  if (typeof gene_name !== 'undefined' && gene_name.length === MET_GENES) {
    riboMask = gene_name.map(g => /^rp[sl]/i.test(g));
  }
  for (let i=0;i<MET_N;i++){
    let tot=0,nz=0,rib=0;
    const base=i*MET_GENES;
    for (let j=0;j<MET_GENES;j++){
      const v=met_expr[base+j];
      if (v>0){ tot+=v; nz++; if (riboMask && riboMask[j]) rib+=v; }
    }
    metCounts[i]=tot; metGenes[i]=nz; metRibo[i]= tot>0 ? 100*rib/tot : 0;
  }
  function layerToNum(s){
    if (!s) return NaN;
    const m=String(s).match(/(\d)\s*\/\s*(\d)/);   // e.g. 2/3 -> 2.5
    if (m) return (parseInt(m[1])+parseInt(m[2]))/2;
    const b=String(s).match(/6\s*b/i); if (b) return 6.5;
    const d=String(s).match(/(\d)/); return d ? parseInt(d[1]) : NaN;
  }
  const metLayerNum = (typeof MET_LAYER !== 'undefined' && MET_LAYER)
        ? MET_LAYER.map(layerToNum) : null;

  function metActive() {
    const allow = new Set();
    document.querySelectorAll('input[type=checkbox][data-sub]').forEach(cb => {
      if (cb.checked) allow.add(cb.dataset.sub);
    });
    // If the page exposes no subtype checkboxes, show all.
    if (allow.size === 0) return MET_IMPUTED_CLUSTER.map(() => 1);
    return MET_IMPUTED_CLUSTER.map(c => allow.has(c) ? 1 : 0);
  }

  function renderMet() {
    if (typeof window.MET_TRACE === 'undefined') return;
    const act = metActive();
    const xs=new Array(MET_N), ys=new Array(MET_N), zs=new Array(MET_N), cols=new Array(MET_N);
    for (let i=0;i<MET_N;i++){
      if (act[i]) { xs[i]=MET_XYZ[i][0]; ys[i]=MET_XYZ[i][1]; zs[i]=MET_XYZ[i][2];
                    cols[i]=metCurrentColors[i]; }
      else        { xs[i]=null; ys[i]=null; zs[i]=null; cols[i]='rgba(0,0,0,0)'; }
    }
    Plotly.restyle(cellPlot, {x:[xs], y:[ys], z:[zs], 'marker.color':[cols]}, [window.MET_TRACE]);
  }

  // ---- colour-scheme sync: patch-seq cells follow the page's colour mode ----
  function metViridisByValues(vals) {
    const act = metActive();
    const valid = vals.filter((v,i)=> act[i] && !isNaN(v) && v !== null);
    const pal = (typeof valuesToViridis !== 'undefined') ? valuesToViridis(valid)
                : valid.map(()=> '#888');
    let p=0;
    metCurrentColors = Array.from(vals, (v,i)=> (act[i] && !isNaN(v) && v!==null) ? pal[p++] : '#cccccc');
    renderMet();
  }
  function metByGene(j) {
    const col = new Array(MET_N);
    for (let i=0;i<MET_N;i++) col[i] = met_expr[i*MET_GENES + j];
    metCurrentColors = exprToMagma(col);
    renderMet();
  }
  function metReset() { metCurrentColors = met_default_colors.slice(); renderMet(); }

  function wireColorSync() {
    const on = (id, fn) => { const b=document.getElementById(id); if (b) b.addEventListener('click', fn); };
    on('reset-btn',   metReset);
    on('qc-counts', () => metViridisByValues(metCounts));
    on('qc-genes',  () => metViridisByValues(metGenes));
    on('qc-ribo',   () => metViridisByValues(metRibo));
    on('qc-pt',     metReset);   // no patch-seq pseudotime → keep imputed-type colour
    on('qc-layer',  () => { if (metLayerNum) metViridisByValues(metLayerNum); else metReset(); });
    // gene-hover on the gene plot recolours the patch-seq cells by that gene's
    // expression in their own transcriptome (same magma path as the base cells).
    if (typeof genePlot !== 'undefined' && genePlot) {
      genePlot.on('plotly_hover', (data) => {
        if (!data.points || !data.points.length) return;
        const pt = data.points[0];
        if (typeof POINTS_TRACE !== 'undefined' && pt.curveNumber !== POINTS_TRACE) return;
        metByGene(pt.pointNumber);
      });
    }
  }

  function addMetTrace() {
    const trace = {
      x: MET_XYZ.map(p=>p[0]), y: MET_XYZ.map(p=>p[1]), z: MET_XYZ.map(p=>p[2]),
      mode: 'markers', type: 'scatter3d', name: DATASET_LABEL,
      // No marker outline — when zoomed out the outline is all you'd see.
      marker: { size: 6, color: met_default_colors, symbol: 'diamond', line: {width: 0} },
      hoverinfo: 'text', hovertext: MET_HOVER, showlegend: false,
    };
    Plotly.addTraces(cellPlot, trace);
    window.MET_TRACE = cellPlot.data.length - 1;
    // Dim the underlying transcriptomic cells + take them out of hover so the
    // patch-seq diamonds read clearly on top.
    Plotly.restyle(cellPlot, {hoverinfo: 'skip', 'marker.opacity': 0.28}, [POINTS_TRACE]);
    renderMet();
  }

  // Re-project the patch-seq cells onto a freshly-recomputed SVD basis so they
  // move WITH the reference cells (otherwise they'd stay frozen at build-time
  // coords after a recompute). Exact for SVD: score_k = z·V_k, scaled by cmax.
  function reprojectMet() {
    const b = window.__svdBasis;
    if (!b || !b.V) return;
    const idx = b.basisIdx, np = idx.length, kEmb = b.kEmb;
    for (let i = 0; i < MET_N; i++) {
      const sc = [0, 0, 0];
      const base = i * MET_GENES;
      for (let kk = 0; kk < np; kk++) {
        const j = idx[kk];
        const z = (met_expr[base + j] / MET_EXPR_SCALE - b.panelMean[kk]) / b.panelStd[kk];
        for (let k = 0; k < kEmb; k++) sc[k] += z * b.V[k][kk];
      }
      MET_XYZ[i] = [ kEmb > 0 ? sc[0] / b.cmax[0] : 0,
                     kEmb > 1 ? sc[1] / b.cmax[1] : 0,
                     kEmb > 2 ? sc[2] / b.cmax[2] : 0 ];
    }
    renderMet();
  }

  function wireSubtypeSync() {
    document.querySelectorAll('input[type=checkbox][data-sub]').forEach(cb => {
      cb.addEventListener('change', renderMet);
    });
    document.querySelectorAll('.grp-toggle, #subt-all, #subt-none, .lin-btn').forEach(b => {
      b.addEventListener('click', () => setTimeout(renderMet, 60));
    });
  }

  // ---- "Patch-seq Info" box, below the plots --------------------------------
  const box = document.createElement('div');
  box.className = 'ctrl-box';
  box.id = 'patchseq-info-box';
  box.innerHTML =
      '<div class="ctrl-box-title">Patch-seq Info</div>'
    + '<div id="psq-empty" style="text-align:center; color:#777; padding:14px 0;">'
    + 'Hover on a cell, to see info.</div>'
    + '<div id="psq-body" style="display:none; gap:16px;">'
    + '  <div class="psq-imgwrap"><img id="psq-img" alt=""></div>'
    + '  <div class="psq-info">'
    + '    <div class="psq-head"><span class="swatch" id="psq-swatch"></span>'
    + '      <span id="psq-title">—</span></div>'
    + '    <div class="psq-meta" id="psq-meta"></div>'
    + '    <div class="psq-grids">'
    + '      <div class="psq-grid-col" id="psq-ephys-col" style="display:none;">'
    + '        <div class="psq-grid-ttl">Ephys</div><div class="psq-kv" id="psq-ephys"></div></div>'
    + '      <div class="psq-grid-col" id="psq-morph-col" style="display:none;">'
    + '        <div class="psq-grid-ttl">Morphology</div><div class="psq-kv" id="psq-morph"></div></div>'
    + '    </div>'
    + '  </div>'
    + '</div>';
  const boxCss = document.createElement('style');
  boxCss.textContent =
      '#patchseq-info-box #psq-body { display: flex; align-items: flex-start; flex-wrap: wrap; }'
    + '#patchseq-info-box .psq-imgwrap { flex: 0 0 300px; max-width: 300px; aspect-ratio: 4/3;'
    + '  background: #f7f5f1; border: 1px solid var(--line); border-radius: 6px;'
    + '  display: flex; align-items: center; justify-content: center; overflow: hidden; }'
    + '#patchseq-info-box .psq-imgwrap img { max-width: 100%; max-height: 100%; display: none; }'
    + '#patchseq-info-box .psq-info { flex: 1 1 auto; min-width: 0; }'
    + '#patchseq-info-box .psq-head { display: flex; align-items: center; gap: 7px;'
    + '  font-size: 15px; font-weight: 600; color: var(--fg); margin-bottom: 6px; }'
    + '#patchseq-info-box .psq-head .swatch { width: 13px; height: 13px; border-radius: 50%;'
    + '  flex: 0 0 13px; }'
    + '#patchseq-info-box .psq-meta { display: grid; grid-template-columns: max-content 1fr;'
    + '  gap: 2px 14px; font-size: 12.5px; color: #333; margin-bottom: 8px; }'
    + '#patchseq-info-box .psq-meta .k { color: #777; }'
    + '#patchseq-info-box .psq-meta .v { color: #222; }'
    + '#patchseq-info-box .psq-grids { display: flex; gap: 22px; flex-wrap: wrap; }'
    + '#patchseq-info-box .psq-grid-ttl { font-size: 11px; font-weight: 700; color: #555;'
    + '  text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 3px; }'
    + '#patchseq-info-box .psq-kv { display: grid; grid-template-columns: max-content auto;'
    + '  gap: 1px 10px; font-size: 11.5px; }'
    + '#patchseq-info-box .psq-kv .k { color: #888; }'
    + '#patchseq-info-box .psq-kv .v { color: #222; text-align: right;'
    + '  font-variant-numeric: tabular-nums; }';
  document.head.appendChild(boxCss);

  function placeBox() {
    const pair = document.querySelector('.plot-pair');
    if (pair && pair.parentNode) { pair.insertAdjacentElement('afterend', box); return true; }
    return false;
  }

  function fmtNum(v, dp) {
    if (v == null || isNaN(v)) return '—';
    if (Math.abs(v) >= 1000) return Number(v).toFixed(0);
    return Number(v).toFixed(dp == null ? 2 : dp);
  }
  function setPanel(i) {
    document.getElementById('psq-empty').style.display = 'none';
    document.getElementById('psq-body').style.display = 'flex';
    document.getElementById('psq-swatch').style.background = metColor(i);
    document.getElementById('psq-title').textContent = (MET_CELL_ID[i] || ('cell #' + i));
    // metadata rows from MET_META (ordered label -> per-cell value array)
    let rows = '';
    for (const lbl of MET_META_ORDER) {
      const arr = MET_META[lbl]; if (!arr) continue;
      const val = arr[i]; if (val == null || val === '') continue;
      rows += '<span class="k">' + lbl + '</span><span class="v">' + val + '</span>';
    }
    document.getElementById('psq-meta').innerHTML = rows;
    // ephys / morph feature grids (only if provided)
    const grid = (obj, el, col) => {
      const keys = obj ? Object.keys(obj) : [];
      if (!keys.length) { document.getElementById(col).style.display = 'none'; return; }
      document.getElementById(col).style.display = 'block';
      document.getElementById(el).innerHTML = keys.map(k =>
        '<span class="k">' + k + '</span><span class="v">' + fmtNum(obj[k] ? obj[k][i] : null) + '</span>'
      ).join('');
    };
    grid(typeof MET_EPHYS !== 'undefined' ? MET_EPHYS : null, 'psq-ephys', 'psq-ephys-col');
    grid(typeof MET_MORPH !== 'undefined' ? MET_MORPH : null, 'psq-morph', 'psq-morph-col');
    const img = document.getElementById('psq-img');
    const cid = String(MET_CELL_ID[i]);
    if (MET_MORPHO_IDS.has(cid)) { img.src = MORPHO_DIR + '/' + cid + '.png'; img.style.display = 'block'; }
    else { img.removeAttribute('src'); img.style.display = 'none'; }
  }

  function metHover(data) {
    if (!data.points || !data.points.length) return;
    const pt = data.points[0];
    if (pt.curveNumber !== window.MET_TRACE) return;
    const i = pt.pointNumber;
    const row = met_expr.subarray(i * MET_GENES, (i + 1) * MET_GENES);
    Plotly.restyle(genePlot, {'marker.color': [exprToMagma(row)]}, [POINTS_TRACE]);
    if (typeof status !== 'undefined' && status) {
      status.innerHTML = '<b style="color:#8a6038">' + (MET_CELL_ID[i] || ('cell #'+i)) + '</b>'
        + ' · imputed <span style="color:' + metColor(i) + '">' + MET_IMPUTED_CLUSTER[i] + '</span>'
        + ' · genes recoloured by this cell’s patch-seq expression';
    }
    setPanel(i);
  }

  function centerAndResize() {
    if (typeof cellPlot === 'undefined' || !cellPlot.data) return;
    const fix = {'scene.aspectmode':'cube','scene.aspectratio':{x:1,y:1,z:1},
                 'scene.camera.center':{x:0,y:0,z:0}};
    try { Plotly.relayout(cellPlot, fix); } catch(e) {}
    try { Plotly.relayout(genePlot, fix); } catch(e) {}
    try { Plotly.Plots.resize(cellPlot); } catch(e) {}
    try { Plotly.Plots.resize(genePlot); } catch(e) {}
  }

  // ---- Patch-seq Variables colour-by -------------------------------------
  // A dedicated control box that colours ONLY the patch-seq diamonds by their
  // own measured / annotated variables (ephys, morphology, Cre line, RNA family,
  // imputed type, soma depth, KNN distance, …). Base reference cells go grey so
  // the patch-seq variable reads clearly.
  function _catPalette(cats) {
    const pal = {};
    cats.forEach((c, i) => { pal[c] = 'hsl(' + ((i * 137.508) % 360).toFixed(1) + ',62%,52%)'; });
    return pal;
  }
  function _greyBaseCells() {
    const n = cellPlot.data[POINTS_TRACE].x.length;
    Plotly.restyle(cellPlot, {'marker.color': [new Array(n).fill('#e6e6e6')]}, [POINTS_TRACE]);
  }
  function metColorByVar(label, values) {
    const act = metActive();
    let nact = 0; for (let i = 0; i < MET_N; i++) if (act[i]) nact++; if (!nact) nact = 1;
    const nums = values.map(v => (v === '' || v == null) ? NaN : (typeof v === 'number' ? v : parseFloat(v)));
    let numOK = 0; for (let i = 0; i < MET_N; i++) if (act[i] && !isNaN(nums[i])) numOK++;
    if (numOK / nact > 0.8) {
      const valid = nums.filter((v, i) => act[i] && !isNaN(v));
      const pal = (typeof valuesToViridis !== 'undefined') ? valuesToViridis(valid) : valid.map(() => '#888');
      let p = 0;
      metCurrentColors = nums.map((v, i) => (act[i] && !isNaN(v)) ? pal[p++] : '#cccccc');
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < MET_N; i++) if (act[i] && !isNaN(nums[i])) { if (nums[i] < lo) lo = nums[i]; if (nums[i] > hi) hi = nums[i]; }
      if (typeof setColorKeyGradient !== 'undefined')
        setColorKeyGradient(label + ' · patch-seq', 'viridis', lo, hi,
          v => (Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : (Math.round(v * 100) / 100)));
    } else {
      const cats = Array.from(new Set(values.filter((v, i) => act[i] && v !== '' && v != null))).sort();
      const pal = _catPalette(cats);
      metCurrentColors = values.map((v, i) => (act[i] && pal[v]) ? pal[v] : '#cccccc');
      if (typeof setColorKeyCats !== 'undefined')
        setColorKeyCats(label + ' · patch-seq', cats.map(c => ({color: pal[c], label: c})));
    }
    _greyBaseCells();
    renderMet();
  }
  function buildPatchseqVarBox() {
    const wrap = document.querySelector('.wrap');
    if (!wrap || document.getElementById('patchseq-var-box')) return;
    const vars = [];
    if (typeof MET_META_ORDER !== 'undefined' && typeof MET_META !== 'undefined')
      MET_META_ORDER.forEach(l => { if (MET_META[l]) vars.push([l, MET_META[l]]); });
    if (typeof MET_EPHYS !== 'undefined' && MET_EPHYS)
      Object.keys(MET_EPHYS).forEach(l => vars.push([l, MET_EPHYS[l]]));
    if (typeof MET_MORPH !== 'undefined' && MET_MORPH)
      Object.keys(MET_MORPH).forEach(l => vars.push([l, MET_MORPH[l]]));
    if (!vars.length) return;
    const box = document.createElement('div');
    box.className = 'ctrl-box'; box.id = 'patchseq-var-box';
    let html = '<div class="ctrl-box-title">Patch-seq Variables — colour the ' + DATASET_LABEL
             + ' cells by their own measurements</div><div class="controls-row">';
    vars.forEach(([l], k) => { html += '<button class="qc-btn psq-var-btn" data-vi="' + k + '">' + l + '</button>'; });
    html += '</div>';
    box.innerHTML = html;
    wrap.appendChild(box);
    box.querySelectorAll('.psq-var-btn').forEach(b => b.addEventListener('click', () => {
      box.querySelectorAll('.psq-var-btn').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      const v = vars[parseInt(b.dataset.vi, 10)];
      metColorByVar(v[0], v[1]);
    }));
    // clicking any base colour-scheme button clears the patch-seq-var highlight
    document.querySelectorAll('.qc-btn:not(.psq-var-btn), #reset-btn').forEach(b =>
      b.addEventListener('click', () => box.querySelectorAll('.psq-var-btn').forEach(x => x.classList.remove('active'))));
  }

  function boot() {
    if (typeof cellPlot === 'undefined' || !cellPlot.data || !placeBox()) {
      setTimeout(boot, 200); return;
    }
    addMetTrace();
    wireSubtypeSync();
    wireColorSync();
    buildPatchseqVarBox();
    document.addEventListener('svd-recomputed', reprojectMet);
    cellPlot.on('plotly_hover', metHover);
    centerAndResize();
    window.addEventListener('resize', () => setTimeout(centerAndResize, 100));
    console.log('patch-seq overlay armed: ' + MET_N + ' cells, ' + MET_MORPHO_IDS.size + ' with morphology');
  }
  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', () => setTimeout(boot, 400));
  else setTimeout(boot, 400);
})();
"""


def assemble(base_html_path, out_html_path, js_consts, h2_title, citation_html):
    """Inject the overlay into a base recompute explorer HTML and write the result."""
    with open(base_html_path) as f:
        html = f.read()

    # 1. Page <title> (covers the redesigned-base form).
    html = re.sub(r'<title>.*?</title>', f'<title>{h2_title}</title>', html, count=1)
    # 2. Visible <h2> heading.
    html = re.sub(r'<h2>.*?</h2>', f'<h2>{h2_title}</h2>', html, count=1)
    # 3. Footer citation.
    html = re.sub(r'<footer class="cite">.*?</footer>',
                  f'<footer class="cite">{citation_html}</footer>', html, count=1, flags=re.S)

    block = ('\n<script>\n/* === patch-seq MET overlay === */\n'
             + js_consts + OVERLAY_JS + '\n</script>\n')
    html = html.replace('</body>', block + '</body>', 1)

    os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
    with open(out_html_path, 'w') as f:
        f.write(html)
    return os.path.getsize(out_html_path)
