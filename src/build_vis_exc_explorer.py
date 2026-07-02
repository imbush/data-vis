#!/usr/bin/env python
"""Build the VISp excitatory MET + projectome explorer (Gouwens et al. 2026).

Uses the paper's GENUINE embeddings: E-UMAP (their t-seeded 62-sPC UMAP), M-UMAP
(their t-seeded 50-feature UMAP), and the real co-embedded transcriptomic UMAP
(FACS reference + Patch-seq). Gene colouring uses per-cell log2(CPM+1) reconstructed
from the paper's per-subclass transcriptomic PCA (loadings·scores+centre) — the
true full count matrix is fastq-only on NeMO.

VIS_EXC_FULL=1 embeds all reconstructed genes (large, local only); otherwise the
top-N by variance are embedded (GitHub-sized).
"""
import os, sys, json, pickle, numpy as np
sys.path.insert(0, '/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish/scripts')
import build_lamp5_archetype_app_4d as base

FULL = os.environ.get('VIS_EXC_FULL') == '1'
TOPN_GENES = 100000 if FULL else 300
C = pickle.load(open('/private/tmp/claude-501/-Users-inlebush-cs-lab-green-data-vis/95ef1b74-d552-4696-b236-9181bf853664/scratchpad/vis_exc_cache.pkl','rb'))
OUT = ('/Users/inlebush/cs/lab/green/data-vis/vis_exc/vis_exc_projectome_explorer.html' if not FULL
       else '/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish/notebooks/vis_exc_projectome_explorer_FULL.html')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

n = len(C['ids'])
def f2(a, d=3):
    return [None if not np.isfinite(v) else round(float(v), d) for v in np.asarray(a, float)]

MET_ORDER = ['L2/3 IT','L4 IT','L4/L5 IT','L5 IT-1','L5 IT-2','L5 IT-3 Pld5','L6 IT-1','L6 IT-2','L6 IT-3',
             'L5/L6 IT Car3','L5 ET-1 Chrna6','L5 ET-2','L5 ET-3','L5 NP','L6 CT-1','L6 CT-2','L6b']
met_types = [m for m in MET_ORDER if m in C['met_types']] + [m for m in C['met_types'] if m not in MET_ORDER]
morph_labels = {c: c.replace('_',' ') for c in C['morph_cols']}
ephys = C['ephys']; morph = C['morph']

# gene subset for the embedded HTML
genes = C['genes_ranked'][:TOPN_GENES]
gene_expr = {g: f2(C['gene_expr'][g], 2) for g in genes}

# projectome
wnm = C['wnm']; targets = wnm['targets']; P = np.asarray(wnm['P'], float)
tgt_hemi = ['contra' if t.startswith('contra_') else 'ipsi' for t in targets]
tgt_name = [t.split('_',1)[1] if '_' in t else t for t in targets]
wnm_met = wnm['met']; Plog = np.log1p(P)
met_proj, met_ncell = {}, {}
for m in set(wnm_met):
    idx=[i for i,x in enumerate(wnm_met) if x==m]
    if idx: met_proj[m]=[round(float(v),3) for v in Plog[idx].mean(0)]; met_ncell[m]=len(idx)
all_proj=[round(float(v),3) for v in Plog.mean(0)]

data = dict(
  N=n, ids=[str(x) for x in C['ids']],
  met=C['met'], t_type=C['t_type'], subclass=C['subclass'],
  e_umap=[f2(C['e_umap'][:,0]), f2(C['e_umap'][:,1])],
  m_umap=[f2(C['m_umap'][:,0]), f2(C['m_umap'][:,1])],
  t_ref=[f2(C['t_ref']['x']), f2(C['t_ref']['y'])],
  t_ps=[f2(C['t_ps']['x']), f2(C['t_ps']['y']), C['t_ps']['subclass']],
  soma_depth=f2(C['soma_depth'],1), tx_pc1=f2(C['tx_pc1']), tx_pc2=f2(C['tx_pc2']),
  ephys={k:f2(v) for k,v in ephys.items()},
  morph={morph_labels[k]:f2(v) for k,v in morph.items()},
  genes=genes, gene_expr=gene_expr,
  met_types=met_types, met_colors={m:C['met_colors'].get(m,'#cccccc') for m in met_types+['unassigned']},
  subclass_colors=C['subclass_colors'],
  targets=tgt_name, tgt_hemi=tgt_hemi, met_proj=met_proj, met_ncell=met_ncell, all_proj=all_proj)
js_data = "const D = " + json.dumps(data) + ";\nconst FULL=" + ("true" if FULL else "false") + ";\n"

title = "Mouse VISp excitatory neurons — MET-types & projectome"
cite = ("Gouwens, Sorensen, Wang et al. <i>Connecting single-cell transcriptomes to projectomes "
        "in the mouse visual cortex.</i> Nature (2026). "
        "<a href='https://doi.org/10.1038/s41586-026-10424-8' target='_blank'>doi:10.1038/s41586-026-10424-8</a> "
        "· data: Allen Institute exc_vis_manuscript (NeMO / DANDI / BIL).")

JS_LOGIC = r"""
const N=D.N;
const T_PLOT=document.getElementById('t-plot'),E_PLOT=document.getElementById('e-plot'),M_PLOT=document.getElementById('m-plot');
const viridis=['#440154','#482878','#3e4a89','#31688e','#26828e','#1f9e89','#35b779','#6ece58','#b5de2b','#fde725'];
function vircol(t){t=Math.max(0,Math.min(1,t));return viridis[Math.min(9,Math.floor(t*9))];}
const TT=[...new Set(D.t_type)].sort();
function catcolors(cats){const m={},nn=cats.length;const tb=['#1f77b4','#aec7e8','#ff7f0e','#ffbb78','#2ca02c','#98df8a','#d62728','#ff9896','#9467bd','#c5b0d5','#8c564b','#c49c94','#e377c2','#f7b6d2','#7f7f7f','#c7c7c7','#bcbd22','#dbdb8d','#17becf','#9edae5'];cats.forEach((c,i)=>{m[c]=(nn<=tb.length)?tb[i%tb.length]:`hsl(${Math.round(360*i/nn)},62%,48%)`;});return m;}
const ttCol=catcolors(TT);
// MET -> subclass map (for linking the T panel, which is subclass-keyed)
const metSub={}; for(let i=0;i<N;i++) metSub[D.met[i]]=D.subclass[i];
let colorMode='met', isolate=null;
function numArr(m){if(m==='depth')return D.soma_depth;if(m==='tx_pc1')return D.tx_pc1;if(m==='tx_pc2')return D.tx_pc2;
  if(m.startsWith('ephys:'))return D.ephys[m.slice(6)];if(m.startsWith('morph:'))return D.morph[m.slice(6)];
  if(m.startsWith('gene:'))return D.gene_expr[m.slice(5)];return null;}
function keyText(t){document.getElementById('colorkey').textContent=t;}
function specimenColors(){ // colours for E/M panels (specimen-keyed cells)
  const cols=new Array(N);
  if(colorMode==='met'){for(let i=0;i<N;i++)cols[i]=D.met_colors[D.met[i]]||'#ccc';keyText('');}
  else if(colorMode==='subclass'){for(let i=0;i<N;i++)cols[i]=D.subclass_colors[D.subclass[i]]||'#ccc';keyText('');}
  else if(colorMode==='ttype'){for(let i=0;i<N;i++)cols[i]=ttCol[D.t_type[i]]||'#ccc';keyText('');}
  else{const a=numArr(colorMode);let lo=Infinity,hi=-Infinity;for(let i=0;i<N;i++){const v=a?a[i]:null;if(v!=null){if(v<lo)lo=v;if(v>hi)hi=v;}}
    const rng=(hi>lo)?(hi-lo):1;for(let i=0;i<N;i++){const v=a?a[i]:null;cols[i]=(v==null)?'#e6e6e6':vircol((v-lo)/rng);}
    keyText(isFinite(lo)?`${lo.toFixed(1)} – ${hi.toFixed(1)}`+(colorMode.startsWith('gene:')?' log2(CPM+1)':''):'no data');}
  if(isolate)for(let i=0;i<N;i++)if(D.met[i]!==isolate)cols[i]='rgba(210,210,210,0.20)';
  return cols;
}
function tpsColors(){ // T-panel patch-seq cells (subclass-keyed)
  const sub=D.t_ps[2],m=new Array(sub.length);
  for(let i=0;i<sub.length;i++)m[i]=D.subclass_colors[sub[i]]||'#999';
  if(isolate){const iso=metSub[isolate];for(let i=0;i<sub.length;i++)if(sub[i]!==iso)m[i]='rgba(210,210,210,0.15)';}
  return m;
}
const HOVER=Array.from({length:N},(_,i)=>`${D.ids[i]}<br>MET: <b>${D.met[i]}</b><br>T-type: ${D.t_type[i]}<br>subclass: ${D.subclass[i]}`);
function sc(xy,cols,txt){return{x:xy[0],y:xy[1],mode:'markers',type:'scatter',marker:{size:5,color:cols,line:{width:0}},text:txt,hoverinfo:txt?'text':'skip',showlegend:false};}
const LAYOUT={margin:{l:6,r:6,t:6,b:6},xaxis:{visible:false},yaxis:{visible:false},paper_bgcolor:'white',plot_bgcolor:'white',hovermode:'closest',dragmode:'pan'};
const CFG={displayModeBar:false,responsive:true};
function drawAll(){
  // T: reference backdrop (grey) + patch-seq (coloured)
  const refTr={x:D.t_ref[0],y:D.t_ref[1],mode:'markers',type:'scatter',marker:{size:3,color:'rgba(200,200,200,0.5)',line:{width:0}},hoverinfo:'skip',showlegend:false};
  const psTr=sc([D.t_ps[0],D.t_ps[1]],tpsColors(),null);
  Plotly.react(T_PLOT,[refTr,psTr],LAYOUT,CFG);
  Plotly.react(E_PLOT,[sc(D.e_umap,specimenColors(),HOVER)],LAYOUT,CFG);
  Plotly.react(M_PLOT,[sc(D.m_umap,specimenColors(),HOVER)],LAYOUT,CFG);
}
function recolor(){const c=specimenColors();Plotly.restyle(E_PLOT,{'marker.color':[c]},[0]);Plotly.restyle(M_PLOT,{'marker.color':[c]},[0]);
  Plotly.restyle(T_PLOT,{'marker.color':[tpsColors()]},[1]);}
document.querySelectorAll('.cb').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('.cb').forEach(x=>x.classList.remove('active'));b.classList.add('active');colorMode=b.dataset.mode;
  ['ephys-sel','morph-sel','tx-sel','gene-sel'].forEach(id=>{const e=document.getElementById(id);if(e)e.value='';});recolor();}));
function fillSel(id,keys,pfx){const s=document.getElementById(id);keys.forEach(k=>{const o=document.createElement('option');o.value=pfx+k;o.textContent=k;s.appendChild(o);});}
fillSel('ephys-sel',Object.keys(D.ephys).sort(),'ephys:');fillSel('morph-sel',Object.keys(D.morph).sort(),'morph:');fillSel('gene-sel',D.genes,'gene:');
['ephys-sel','morph-sel','tx-sel','gene-sel'].forEach(id=>{const el=document.getElementById(id);if(!el)return;el.addEventListener('change',e=>{
  if(!e.target.value)return;document.querySelectorAll('.cb').forEach(x=>x.classList.remove('active'));
  ['ephys-sel','morph-sel','tx-sel','gene-sel'].forEach(o=>{if(o!==id)document.getElementById(o).value='';});
  colorMode=e.target.value;recolor();});});
const lg=document.getElementById('legend');
D.met_types.forEach(m=>{const el=document.createElement('span');el.className='lg';el.dataset.met=m;
  el.innerHTML=`<span class="dot" style="background:${D.met_colors[m]}"></span>${m}`;
  el.addEventListener('click',()=>{isolate=(isolate===m)?null:m;document.querySelectorAll('.lg').forEach(e=>e.classList.toggle('off',isolate&&e.dataset.met!==isolate));recolor();drawProj();});
  lg.appendChild(el);});
// projectome
let projView='bars';
function topTargets(prof,k){return prof.map((v,i)=>[v,i]).filter(p=>p[0]>0).sort((a,b)=>b[0]-a[0]).slice(0,k).map(p=>p[1]);}
function drawBars(){const topn=parseInt(document.getElementById('proj-topn').value)||30;
  if(isolate&&!D.met_proj[isolate]){document.getElementById('proj-title').textContent=`${isolate} — no whole-neuron reconstructions`;
    Plotly.react('proj-plot',[{x:[],y:[],type:'bar'}],{height:140,margin:{t:20,b:20},xaxis:{visible:false},yaxis:{visible:false},paper_bgcolor:'white',plot_bgcolor:'white',annotations:[{x:.5,y:.5,xref:'paper',yref:'paper',showarrow:false,font:{size:13,color:'#999'},text:`No whole-neuron (projection) morphologies for <b>${isolate}</b>.`}]},{displayModeBar:false,responsive:true});return;}
  const prof=(isolate&&D.met_proj[isolate])?D.met_proj[isolate]:D.all_proj;const nc=isolate?(D.met_ncell[isolate]||0):341;
  document.getElementById('proj-title').textContent=isolate?`${isolate} (n=${nc} WNM)`:'all MET-types';
  const idx=topTargets(prof,topn);const y=idx.map(i=>D.targets[i]+(D.tgt_hemi[i]==='contra'?' (contra)':''));
  Plotly.react('proj-plot',[{x:idx.map(i=>prof[i]),y:y,type:'bar',orientation:'h',marker:{color:idx.map(i=>D.tgt_hemi[i]==='contra'?'#d1495b':'#3d6098')},hovertemplate:'%{y}: %{x:.2f}<extra></extra>'}],
    {margin:{l:170,r:20,t:8,b:34},height:Math.max(320,22*idx.length+60),xaxis:{title:'mean log(1+axon length µm)'},yaxis:{autorange:'reversed',tickfont:{size:10}},paper_bgcolor:'white',plot_bgcolor:'white'},{displayModeBar:false,responsive:true});}
function drawHeat(){const topn=parseInt(document.getElementById('proj-topn').value)||30;document.getElementById('proj-title').textContent='all MET-types (heatmap)';
  const idx=topTargets(D.all_proj,topn);const rows=D.met_types.filter(m=>D.met_proj[m]);
  Plotly.react('proj-plot',[{z:rows.map(m=>idx.map(i=>D.met_proj[m][i])),x:idx.map(i=>D.targets[i]+(D.tgt_hemi[i]==='contra'?'*':'')),y:rows.map(m=>`${m} (${D.met_ncell[m]})`),type:'heatmap',colorscale:'Viridis',hovertemplate:'%{y} → %{x}: %{z:.2f}<extra></extra>',colorbar:{title:'log µm',thickness:12}}],
    {margin:{l:150,r:20,t:20,b:110},height:Math.max(360,26*rows.length+150),xaxis:{tickangle:-60,tickfont:{size:9},side:'top'},yaxis:{autorange:'reversed',tickfont:{size:10}},paper_bgcolor:'white',plot_bgcolor:'white',annotations:[{x:0,y:-.12,xref:'paper',yref:'paper',showarrow:false,font:{size:10,color:'#888'},text:'* = contralateral target',xanchor:'left'}]},{displayModeBar:false,responsive:true});}
function drawProj(){projView==='heat'?drawHeat():drawBars();}
document.querySelectorAll('.pv').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.pv').forEach(x=>x.classList.remove('active'));b.classList.add('active');projView=b.dataset.pv;drawProj();}));
document.getElementById('proj-topn').addEventListener('change',drawProj);
drawAll();drawProj();
"""

page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
{base.UNIFIED_DESIGN_CSS}
html,body{{margin:0;padding:0;background:var(--bg);color:var(--fg);}}
.wrap{{max-width:1500px;margin:0 auto;padding:10px 16px 40px;}}
h2{{margin:6px 0 2px;}} .sub{{color:var(--muted);font-size:13px;margin-bottom:10px;line-height:1.4;}}
.ctrl-box{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin:8px 0;}}
.ctrl-box-title{{font-size:12px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px;}}
.row{{display:flex;flex-wrap:wrap;gap:6px;align-items:center;}}
button.cb,button.pv{{font-size:12px;padding:3px 10px;border:1px solid var(--line);background:#f6f2ea;border-radius:6px;cursor:pointer;}}
button.cb.active,button.pv.active{{background:var(--earth,#a9794f);color:#fff;border-color:var(--earth-dark,#875e38);}}
select,input[list]{{font-size:12px;padding:3px 6px;border:1px solid var(--line);border-radius:6px;background:#fff;}}
.triptych{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;}}
.panel{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:4px;}}
.panel .ttl{{text-align:center;font-weight:600;font-size:13px;padding:2px;}} .panel .plt{{width:100%;height:360px;}}
.legend{{display:flex;flex-wrap:wrap;gap:3px;}}
.lg{{display:inline-flex;align-items:center;gap:4px;font-size:11px;padding:2px 7px;border:1px solid var(--line);border-radius:12px;background:#fff;cursor:pointer;user-select:none;}}
.lg .dot{{width:10px;height:10px;border-radius:50%;}} .lg.off{{opacity:.35;}}
#proj-plot{{width:100%;}} .hint{{font-size:12px;color:var(--muted);}}
footer.cite{{color:var(--muted);font-size:11px;margin-top:16px;border-top:1px solid var(--line);padding-top:8px;}}
a.home{{font-size:12px;color:var(--earth-dark,#875e38);text-decoration:none;}}
@media(max-width:1000px){{.triptych{{grid-template-columns:1fr;}}}}
</style></head><body><div class="wrap">
<a class="home" href="../index.html">← all visualizers</a>
<h2>{title}</h2>
<div class="sub">{cite}<br>
1,528 Patch-seq neurons (transcriptomic + electrophysiology + morphology) and 341 whole-neuron reconstructions with
long-range projections. Colour by MET-type, subclass, T-type, a feature, or a <b>gene</b>; click a MET-type to
highlight it across modalities and see its projectome.</div>

<div class="ctrl-box"><div class="ctrl-box-title">Colour cells by</div>
<div class="row">
 <button class="cb active" data-mode="met">MET-type</button>
 <button class="cb" data-mode="subclass">Subclass</button>
 <button class="cb" data-mode="ttype">T-type</button>
 <button class="cb" data-mode="depth">Soma depth <span style="opacity:.7">(n=389)</span></button>
 <span style="margin-left:8px" class="hint">gene:</span><input id="gene-sel" list="gene-dl" placeholder="e.g. Ctgf"><datalist id="gene-dl"></datalist>
 <span class="hint">ephys:</span><select id="ephys-sel"><option value="">—</option></select>
 <span class="hint">morph:</span><select id="morph-sel"><option value="">—</option></select>
 <span class="hint">tx:</span><select id="tx-sel"><option value="">—</option><option value="tx_pc1">within-subclass PC1</option><option value="tx_pc2">within-subclass PC2</option></select>
 <span id="colorkey" class="hint" style="margin-left:8px"></span>
</div></div>

<div class="ctrl-box"><div class="ctrl-box-title">MET-types <span class="hint">(click to isolate)</span></div><div class="legend" id="legend"></div></div>

<div class="triptych">
 <div class="panel"><div class="ttl">Transcriptomic <span class="hint">(co-embedded: FACS ref ∙ grey + Patch-seq)</span></div><div class="plt" id="t-plot"></div></div>
 <div class="panel"><div class="ttl">Electrophysiology <span class="hint">(62 sparse-PC UMAP)</span></div><div class="plt" id="e-plot"></div></div>
 <div class="panel"><div class="ttl">Morphology <span class="hint">(50-feature UMAP, n=389)</span></div><div class="plt" id="m-plot"></div></div>
</div>

<div class="ctrl-box" style="margin-top:12px;"><div class="ctrl-box-title">Projectome — where does this cell type send its axon?</div>
<div class="row">
 <button class="pv active" data-pv="bars">Selected type (bars)</button><button class="pv" data-pv="heat">All types (heatmap)</button>
 <span style="margin-left:12px" class="hint">showing:</span><b id="proj-title">all MET-types</b>
 <span style="margin-left:12px" class="hint">top targets:</span><select id="proj-topn"><option>20</option><option selected>30</option><option>50</option></select>
 <span style="margin-left:12px" class="hint">341 whole-neuron morphologies · mean log(1+axon length µm) · ipsi (blue) / contra (red)</span>
</div><div id="proj-plot"></div></div>

<div class="ctrl-box" style="background:#fbf7ef;"><div class="ctrl-box-title">Methods &amp; caveats</div>
<div class="hint" style="line-height:1.5">
 <b>Transcriptomic</b> panel is the paper's real co-embedded UMAP (Patch-seq + FACS reference, grey); it is keyed by
 RNA sample-id (no sample↔specimen crosswalk is public), so Patch-seq points are coloured by subclass and a MET-type
 selection highlights that subclass. <b>E</b> and <b>M</b> panels are the paper's own t-seeded UMAPs (62 sparse-PC
 ephys; 50 morphology features, n=389), keyed by specimen so MET-type selection and feature/gene colouring apply
 per cell. <b>Gene</b> expression is log2(CPM+1) <i>reconstructed</i> from the paper's per-subclass transcriptomic
 PCA (scores·loadingsᵀ+centre) for {len(genes)} highly-variable genes — the true full count matrix is released only
 as raw fastqs on NeMO. Projectome MET-types are predicted from dendritic morphology (a model), not measured.
</div></div>
<footer class="cite">{cite}</footer>
</div>
<script>
{js_data}
{JS_LOGIC}
</script></body></html>"""

open(OUT,'w').write(page)
print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.2f} MB) | genes={len(genes)} FULL={FULL}")
