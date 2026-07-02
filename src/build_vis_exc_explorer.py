#!/usr/bin/env python
"""Build the VIS excitatory MET + projectome explorer (Gouwens et al. 2026).

Self-contained HTML, unified with the rest of the data-vis site. Three linked UMAP
panels (Transcriptomic / Electrophysiology / Morphology) over 1,528 Patch-seq
neurons, colourable by MET-type, subclass, T-type, cortical depth, or any ephys /
morphology feature; a clickable MET-type legend cross-highlights the same cells in
all three modalities. A projectome panel shows, for the selected MET-type(s), the
mean axonal projection strength to 220 CCF targets (ipsi vs contra) from the 341
whole-neuron morphologies whose MET-type was predicted from dendrites.
"""
import os, sys, json, pickle, numpy as np
sys.path.insert(0, '/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish/scripts')
import build_lamp5_archetype_app_4d as base   # reuse unified design tokens

C = pickle.load(open('/private/tmp/claude-501/-Users-inlebush-cs-lab-green-data-vis/95ef1b74-d552-4696-b236-9181bf853664/scratchpad/vis_exc_cache.pkl','rb'))
OUT = '/Users/inlebush/cs/lab/green/data-vis/vis_exc/vis_exc_projectome_explorer.html'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

n = len(C['ids'])
def f2(a, d=3):
    out=[]
    for v in np.asarray(a, float):
        out.append(None if not np.isfinite(v) else round(float(v), d))
    return out

MET_ORDER = ['L2/3 IT','L4 IT','L4/L5 IT','L5 IT-1','L5 IT-2','L5 IT-3 Pld5','L6 IT-1','L6 IT-2','L6 IT-3',
             'L5/L6 IT Car3','L5 ET-1 Chrna6','L5 ET-2','L5 ET-3','L5 NP','L6 CT-1','L6 CT-2','L6b']
met_types = [m for m in MET_ORDER if m in C['met_types']] + [m for m in C['met_types'] if m not in MET_ORDER]
met_colors = C['met_colors']
morph_labels = {c: c.replace('_',' ') for c in C['morph_cols']}
ephys = C['ephys']; morph = C['morph']

wnm = C['wnm']; targets = wnm['targets']; P = np.asarray(wnm['P'], float)
tgt_hemi = ['contra' if t.startswith('contra_') else 'ipsi' for t in targets]
tgt_name = [t.split('_',1)[1] if '_' in t else t for t in targets]
wnm_met = wnm['met']
Plog = np.log1p(P)
met_proj, met_ncell = {}, {}
for m in set(wnm_met):
    idx = [i for i,x in enumerate(wnm_met) if x==m]
    if idx:
        met_proj[m] = [round(float(v),3) for v in Plog[idx].mean(0)]
        met_ncell[m] = len(idx)
all_proj = [round(float(v),3) for v in Plog.mean(0)]

data = dict(
  N=n, ids=[str(x) for x in C['ids']],
  met=C['met'], t_type=C['t_type'], subclass=C['subclass'],
  e_umap=[f2(C['e_umap'][:,0]), f2(C['e_umap'][:,1])],
  m_umap=[f2(C['m_umap'][:,0]), f2(C['m_umap'][:,1])],
  t_umap=[f2(C['t_umap'][:,0]), f2(C['t_umap'][:,1])],
  soma_depth=f2(C['soma_depth'],1), tx_pc1=f2(C['tx_pc1']), tx_pc2=f2(C['tx_pc2']),
  ephys={k:f2(v) for k,v in ephys.items()},
  morph={morph_labels[k]:f2(v) for k,v in morph.items()},
  met_types=met_types, met_colors={m:met_colors.get(m,'#cccccc') for m in met_types+['unassigned']},
  targets=tgt_name, tgt_hemi=tgt_hemi, met_proj=met_proj, met_ncell=met_ncell, all_proj=all_proj,
  wnm_met=wnm_met)
js_data = "const D = " + json.dumps(data) + ";\n"

title = "Mouse VISp excitatory neurons — MET-types & projectome"
cite = ("Gouwens, Sorensen, Wang et al. <i>Connecting single-cell transcriptomes to projectomes "
        "in the mouse visual cortex.</i> Nature (2026). "
        "<a href='https://doi.org/10.1038/s41586-026-10424-8' target='_blank'>doi:10.1038/s41586-026-10424-8</a> "
        "· data: Allen Institute exc_vis_manuscript (NeMO / DANDI / BIL).")

JS_LOGIC = r"""
const N = D.N;
const T_PLOT=document.getElementById('t-plot'), E_PLOT=document.getElementById('e-plot'), M_PLOT=document.getElementById('m-plot');
const viridis=['#440154','#482878','#3e4a89','#31688e','#26828e','#1f9e89','#35b779','#6ece58','#b5de2b','#fde725'];
function vircol(t){t=Math.max(0,Math.min(1,t));const i=Math.min(viridis.length-1,Math.floor(t*(viridis.length-1)));return viridis[i];}
const SUBC=[...new Set(D.subclass)].sort();
const TT=[...new Set(D.t_type)].sort();
const tab20=['#1f77b4','#aec7e8','#ff7f0e','#ffbb78','#2ca02c','#98df8a','#d62728','#ff9896','#9467bd','#c5b0d5','#8c564b','#c49c94','#e377c2','#f7b6d2','#7f7f7f','#c7c7c7','#bcbd22','#dbdb8d','#17becf','#9edae5'];
function catcolors(cats){const m={};cats.forEach((c,i)=>m[c]=tab20[i%tab20.length]);return m;}
const subcCol=catcolors(SUBC), ttCol=catcolors(TT);
let colorMode='met', isolate=null;
function numArr(mode){
  if(mode==='depth') return D.soma_depth;
  if(mode==='tx_pc1') return D.tx_pc1;
  if(mode==='tx_pc2') return D.tx_pc2;
  if(mode.startsWith('ephys:')) return D.ephys[mode.slice(6)];
  if(mode.startsWith('morph:')) return D.morph[mode.slice(6)];
  return null;
}
function cellColors(){
  const cols=new Array(N);
  if(colorMode==='met'){for(let i=0;i<N;i++)cols[i]=D.met_colors[D.met[i]]||'#ccc';document.getElementById('colorkey').textContent='';}
  else if(colorMode==='subclass'){for(let i=0;i<N;i++)cols[i]=subcCol[D.subclass[i]]||'#ccc';document.getElementById('colorkey').textContent='';}
  else if(colorMode==='ttype'){for(let i=0;i<N;i++)cols[i]=ttCol[D.t_type[i]]||'#ccc';document.getElementById('colorkey').textContent='';}
  else{
    const a=numArr(colorMode); let lo=Infinity,hi=-Infinity;
    for(let i=0;i<N;i++){const v=a[i];if(v!=null){if(v<lo)lo=v;if(v>hi)hi=v;}}
    const rng=(hi>lo)?(hi-lo):1;
    for(let i=0;i<N;i++){const v=a[i];cols[i]=(v==null)?'#e6e6e6':vircol((v-lo)/rng);}
    document.getElementById('colorkey').textContent=isFinite(lo)?`${lo.toFixed(1)} – ${hi.toFixed(1)}`:'no data';
  }
  if(isolate){for(let i=0;i<N;i++) if(D.met[i]!==isolate) cols[i]='rgba(210,210,210,0.22)';}
  return cols;
}
const HOVER=Array.from({length:N},(_,i)=>`${D.ids[i]}<br>MET: <b>${D.met[i]}</b><br>T-type: ${D.t_type[i]}<br>subclass: ${D.subclass[i]}`);
function trace(xy){return {x:xy[0],y:xy[1],mode:'markers',type:'scatter',
  marker:{size:5,color:cellColors(),line:{width:0}},text:HOVER,hoverinfo:'text',showlegend:false};}
const LAYOUT={margin:{l:6,r:6,t:6,b:6},xaxis:{visible:false},yaxis:{visible:false},
  paper_bgcolor:'white',plot_bgcolor:'white',hovermode:'closest',dragmode:'pan'};
const CFG={displayModeBar:false,responsive:true};
function drawAll(){Plotly.react(T_PLOT,[trace(D.t_umap)],LAYOUT,CFG);Plotly.react(E_PLOT,[trace(D.e_umap)],LAYOUT,CFG);Plotly.react(M_PLOT,[trace(D.m_umap)],LAYOUT,CFG);}
function recolor(){const c=cellColors();Plotly.restyle(T_PLOT,{'marker.color':[c]},[0]);Plotly.restyle(E_PLOT,{'marker.color':[c]},[0]);Plotly.restyle(M_PLOT,{'marker.color':[c]},[0]);}
document.querySelectorAll('.cb').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('.cb').forEach(x=>x.classList.remove('active'));
  b.classList.add('active'); colorMode=b.dataset.mode;
  document.getElementById('ephys-sel').value='';document.getElementById('morph-sel').value='';document.getElementById('tx-sel').value='';
  recolor();}));
function fillSel(id,obj,pfx){const s=document.getElementById(id);Object.keys(obj).sort().forEach(k=>{const o=document.createElement('option');o.value=pfx+k;o.textContent=k;s.appendChild(o);});}
fillSel('ephys-sel',D.ephys,'ephys:'); fillSel('morph-sel',D.morph,'morph:');
['ephys-sel','morph-sel','tx-sel'].forEach(id=>document.getElementById(id).addEventListener('change',e=>{
  if(!e.target.value)return; document.querySelectorAll('.cb').forEach(x=>x.classList.remove('active'));
  ['ephys-sel','morph-sel','tx-sel'].forEach(o=>{if(o!==id)document.getElementById(o).value='';});
  colorMode=e.target.value; recolor();}));
const lg=document.getElementById('legend');
D.met_types.forEach(m=>{
  const el=document.createElement('span');el.className='lg';el.dataset.met=m;
  el.innerHTML=`<span class="dot" style="background:${D.met_colors[m]}"></span>${m}`;
  el.addEventListener('click',()=>{isolate=(isolate===m)?null:m;document.querySelectorAll('.lg').forEach(e=>e.classList.toggle('off',isolate&&e.dataset.met!==isolate));recolor();drawProj();});
  lg.appendChild(el);});
let projView='bars';
function topTargets(prof,topn){return prof.map((v,i)=>[v,i]).filter(p=>p[0]>0).sort((a,b)=>b[0]-a[0]).slice(0,topn).map(p=>p[1]);}
function drawBars(){
  const topn=parseInt(document.getElementById('proj-topn').value)||30;
  const prof=(isolate&&D.met_proj[isolate])?D.met_proj[isolate]:D.all_proj;
  const ncell=isolate?(D.met_ncell[isolate]||0):341;
  document.getElementById('proj-title').textContent=isolate?`${isolate}  (n=${ncell} WNM)`:'all MET-types';
  const idx=topTargets(prof,topn);
  const y=idx.map(i=>D.targets[i]+(D.tgt_hemi[i]==='contra'?' (contra)':''));
  const x=idx.map(i=>prof[i]);
  const col=idx.map(i=>D.tgt_hemi[i]==='contra'?'#d1495b':'#3d6098');
  Plotly.react('proj-plot',[{x:x,y:y,type:'bar',orientation:'h',marker:{color:col},hovertemplate:'%{y}: %{x:.2f}<extra></extra>'}],
    {margin:{l:170,r:20,t:8,b:34},height:Math.max(320,22*idx.length+60),
     xaxis:{title:'mean log(1+axon length µm)'},yaxis:{autorange:'reversed',tickfont:{size:10}},
     paper_bgcolor:'white',plot_bgcolor:'white'},{displayModeBar:false,responsive:true});
}
function drawHeat(){
  const topn=parseInt(document.getElementById('proj-topn').value)||30;
  document.getElementById('proj-title').textContent='all MET-types (heatmap)';
  const idx=topTargets(D.all_proj,topn);
  const rows=D.met_types.filter(m=>D.met_proj[m]);            // MET-types with WNM cells
  const xlab=idx.map(i=>D.targets[i]+(D.tgt_hemi[i]==='contra'?'*':''));
  const z=rows.map(m=>idx.map(i=>D.met_proj[m][i]));
  const yl=rows.map(m=>`${m} (${D.met_ncell[m]})`);
  Plotly.react('proj-plot',[{z:z,x:xlab,y:yl,type:'heatmap',colorscale:'Viridis',
     hovertemplate:'%{y} → %{x}: %{z:.2f}<extra></extra>',colorbar:{title:'log µm',thickness:12}}],
    {margin:{l:150,r:20,t:20,b:110},height:Math.max(360,26*rows.length+150),
     xaxis:{tickangle:-60,tickfont:{size:9},side:'top'},yaxis:{autorange:'reversed',tickfont:{size:10}},
     paper_bgcolor:'white',plot_bgcolor:'white',
     annotations:[{x:0,y:-0.12,xref:'paper',yref:'paper',showarrow:false,font:{size:10,color:'#888'},text:'* = contralateral target',xanchor:'left'}]},
    {displayModeBar:false,responsive:true});
}
function drawProj(){ projView==='heat'?drawHeat():drawBars(); }
document.querySelectorAll('.pv').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('.pv').forEach(x=>x.classList.remove('active'));b.classList.add('active');projView=b.dataset.pv;drawProj();}));
document.getElementById('proj-topn').addEventListener('change',drawProj);
drawAll(); drawProj();
"""

page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
{base.UNIFIED_DESIGN_CSS}
html,body{{margin:0;padding:0;background:var(--bg);color:var(--fg);}}
.wrap{{max-width:1500px;margin:0 auto;padding:10px 16px 40px;}}
h2{{margin:6px 0 2px;}}
.sub{{color:var(--muted);font-size:13px;margin-bottom:10px;line-height:1.4;}}
.ctrl-box{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin:8px 0;}}
.ctrl-box-title{{font-size:12px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px;}}
.row{{display:flex;flex-wrap:wrap;gap:6px;align-items:center;}}
button.cb{{font-size:12px;padding:3px 10px;border:1px solid var(--line);background:#f6f2ea;border-radius:6px;cursor:pointer;}}
button.cb.active{{background:var(--earth,#a9794f);color:#fff;border-color:var(--earth-dark,#875e38);}}
select{{font-size:12px;padding:3px 6px;border:1px solid var(--line);border-radius:6px;background:#fff;}}
.triptych{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;}}
.panel{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:4px;}}
.panel .ttl{{text-align:center;font-weight:600;font-size:13px;padding:2px;}}
.panel .plt{{width:100%;height:360px;}}
.legend{{display:flex;flex-wrap:wrap;gap:3px;}}
.lg{{display:inline-flex;align-items:center;gap:4px;font-size:11px;padding:2px 7px;border:1px solid var(--line);border-radius:12px;background:#fff;cursor:pointer;user-select:none;}}
.lg .dot{{width:10px;height:10px;border-radius:50%;}}
.lg.off{{opacity:.35;}}
#proj-plot{{width:100%;}}
footer.cite{{color:var(--muted);font-size:11px;margin-top:16px;border-top:1px solid var(--line);padding-top:8px;}}
.hint{{font-size:12px;color:var(--muted);}}
a.home{{font-size:12px;color:var(--earth-dark,#875e38);text-decoration:none;}}
@media(max-width:1000px){{.triptych{{grid-template-columns:1fr;}}}}
</style></head>
<body><div class="wrap">
<a class="home" href="../index.html">← all visualizers</a>
<h2>{title}</h2>
<div class="sub">{cite}<br>
1,528 Patch-seq neurons with transcriptomic (T), electrophysiological (E) and (n=389) morphological (M) data, plus
341 whole-neuron reconstructions with long-range axonal projections. Click a MET-type to highlight it across all
three modalities and see where it projects.</div>

<div class="ctrl-box">
  <div class="ctrl-box-title">Colour cells by</div>
  <div class="row">
    <button class="cb active" data-mode="met">MET-type</button>
    <button class="cb" data-mode="subclass">Subclass</button>
    <button class="cb" data-mode="ttype">T-type</button>
    <button class="cb" data-mode="depth">Cortical depth</button>
    <span style="margin-left:8px" class="hint">ephys:</span><select id="ephys-sel"><option value="">—</option></select>
    <span class="hint">morphology:</span><select id="morph-sel"><option value="">—</option></select>
    <span class="hint">transcriptomic:</span><select id="tx-sel"><option value="">—</option><option value="tx_pc1">within-subclass PC1</option><option value="tx_pc2">within-subclass PC2</option></select>
    <span id="colorkey" class="hint" style="margin-left:10px"></span>
  </div>
</div>

<div class="ctrl-box">
  <div class="ctrl-box-title">MET-types <span class="hint">(click to isolate · click again to reset)</span></div>
  <div class="legend" id="legend"></div>
</div>

<div class="triptych">
  <div class="panel"><div class="ttl">Transcriptomic <span class="hint">(per-subclass PCA → UMAP)</span></div><div class="plt" id="t-plot"></div></div>
  <div class="panel"><div class="ttl">Electrophysiology <span class="hint">(62 sparse-PC UMAP)</span></div><div class="plt" id="e-plot"></div></div>
  <div class="panel"><div class="ttl">Morphology <span class="hint">(50-feature UMAP, n=389)</span></div><div class="plt" id="m-plot"></div></div>
</div>

<div class="ctrl-box" style="margin-top:12px;">
  <div class="ctrl-box-title">Projectome — where does this cell type send its axon?</div>
  <div class="row">
    <button class="pv cb active" data-pv="bars">Selected type (bars)</button>
    <button class="pv cb" data-pv="heat">All types (heatmap)</button>
    <span style="margin-left:12px" class="hint">showing:</span><b id="proj-title">all MET-types</b>
    <span style="margin-left:12px" class="hint">top targets:</span>
    <select id="proj-topn"><option>20</option><option selected>30</option><option>50</option></select>
    <span style="margin-left:12px" class="hint">341 whole-neuron morphologies · mean log(1+axon length µm) per CCF region · ipsi (blue) / contra (red)</span>
  </div>
  <div id="proj-plot"></div>
</div>

<div class="ctrl-box" style="background:#fbf7ef;">
  <div class="ctrl-box-title">Methods &amp; caveats</div>
  <div class="hint" style="line-height:1.5">
   <b>E</b> UMAP is the paper's own t-seeded UMAP of 62 sparse-PCA electrophysiology components.
   <b>M</b> UMAP is computed here from the paper's 50 normalized dendritic-morphology features (UMAP, seed 0; n=389).
   <b>T</b> is a global UMAP built from the paper's <i>per-subclass</i> transcriptomic PCA (their continuous-variation
   substrate) arranged block-diagonally — so within-island structure is the real transcriptomic gradient, but
   between-island distances are not meaningful; the raw counts are fastq-only on NeMO, so a from-counts global
   T-UMAP (and gene-level colouring) is not reproducible here. MET-types for the 1,528 Patch-seq cells are the
   paper's (manually curated where available, else T-type-inferred). Projectome MET-types are <i>predicted</i> from
   dendritic morphology by the paper's classifier, not measured, so the projection profiles are model-assigned.
  </div>
</div>

<footer class="cite">{cite}</footer>
</div>
<script>
{js_data}
{JS_LOGIC}
</script>
</body></html>"""

open(OUT,'w').write(page)
print(f'wrote {OUT} ({os.path.getsize(OUT)/1e6:.2f} MB)')
