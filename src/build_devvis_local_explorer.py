"""Build self-contained local HTML cell-explorers (Plotly 3D UMAP, embedded base64 data).
Style follows github.com/imbush/data-vis: self-contained HTML, Plotly via CDN, expression
embedded inline as base64 uint8 (value = u8/EXPR_SCALE, log1p-CPM). LOCAL ONLY — not published.
Features: color by metadata / age / any gene; subset by subtype + age range; client-side
UMAP recompute (replot) on the current subset via umap-js.
"""
from __future__ import annotations
import base64, json, re, warnings, sys
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import scanpy as sc, anndata as ad
import matplotlib.pyplot as plt

OUTDIR = Path("/home/inb539/data-vis-local"); OUTDIR.mkdir(exist_ok=True)
CACHE = Path("/n/scratch/users/i/inb539/vis_cache"); CACHE.mkdir(parents=True, exist_ok=True)
RNG = 0; EXPR_SCALE = 16; N_HVG = 500; sc.settings.verbosity = 0  # embed curated panel + top-500 HVGs

PALETTE = [c for c in (
    plt.matplotlib.colors.to_hex(x) for x in
    list(plt.cm.tab20.colors)+list(plt.cm.tab20b.colors)+list(plt.cm.tab20c.colors))]

def b64(arr) -> str: return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()
def age_num(s):
    m = re.match(r"([EP])([0-9.]+)", str(s)); return (float(m.group(2))-19 if m.group(1)=="E" else float(m.group(2))) if m else np.nan

def build(adata, out_name, title, subtitle, cat_fields, panel, age_field=None, age_from=None):
    a = adata
    if a.X.max() > 50:
        sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat")
    panel = [g for g in dict.fromkeys(panel) if g in a.var_names]
    hvg = [g for g in a.var_names[a.var.highly_variable] if g not in panel]
    hv_rank = a.var.loc[hvg, "dispersions_norm"].sort_values(ascending=False).index.tolist()
    genes = list(dict.fromkeys(panel + hv_rank[:N_HVG])); panel_set = set(panel)
    aa = a[:, a.var_names[a.var.highly_variable]].copy(); sc.pp.scale(aa, max_value=10)
    sc.tl.pca(aa, n_comps=30, random_state=RNG); a.obsm["X_pca"] = aa.obsm["X_pca"]
    sc.pp.neighbors(a, use_rep="X_pca", n_neighbors=15, random_state=RNG)
    sc.tl.umap(a, n_components=3, min_dist=0.3, random_state=RNG)
    coord = a.obsm["X_umap"].astype(np.float32)
    coord = (coord - coord.mean(0)) / (coord.std(0).mean())
    pca = np.ascontiguousarray(a.obsm["X_pca"].astype(np.float32))
    E = a[:, genes].X; E = np.asarray(E.todense()) if hasattr(E, "todense") else np.asarray(E)
    u8 = np.clip(np.round(E * EXPR_SCALE), 0, 255).astype(np.uint8)
    gene_mean = E.mean(0)
    cats = {}
    for f in cat_fields:
        if f not in a.obs: continue
        s = a.obs[f].astype(str).values
        order = sorted(pd.unique(s), key=lambda x: -(s==x).sum())
        code = {c: i for i, c in enumerate(order)}
        cats[f] = {"categories": order, "codes": b64(np.array([code[x] for x in s], dtype=np.int16))}
    age = None
    if age_from:
        av = np.array([age_num(x) for x in a.obs[age_from].astype(str)], dtype=np.float32)
        if np.isfinite(av).any(): age = {"name": age_field or "age", "vals": b64(av),
                                         "min": float(np.nanmin(av)), "max": float(np.nanmax(av))}
    meta = {"title": title, "subtitle": subtitle, "n_cells": int(a.n_obs), "n_genes": len(genes),
            "genes": genes, "panel": [g for g in genes if g in panel_set], "expr_scale": EXPR_SCALE,
            "gene_mean": [round(float(x), 3) for x in gene_mean], "cats": cats,
            "cat_fields": [f for f in cat_fields if f in cats], "age": age, "npc": int(pca.shape[1]),
            "default_cat": next((f for f in cat_fields if f in cats), None)}
    html = TEMPLATE.replace("__META__", json.dumps(meta)) \
                   .replace("__EXPR__", b64(u8)).replace("__COORD__", b64(coord)) \
                   .replace("__PCA__", b64(pca)).replace("__PALETTE__", json.dumps(PALETTE))
    p = OUTDIR / out_name; p.write_text(html)
    print(f"  wrote {p}  ({p.stat().st_size/1e6:.1f} MB, {a.n_obs} cells, {len(genes)} genes)")
    return meta

TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Cell Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://unpkg.com/umap-js@1.4.0/lib/umap-js.min.js"></script>
<style>
 body{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#1a1a1a}
 #wrap{display:flex;height:100vh}
 #side{width:300px;padding:14px 16px;overflow-y:auto;border-right:1px solid #e3e3e3;background:#fafafa}
 #plot{flex:1}
 h1{font-size:16px;margin:0 0 2px} .sub{font-size:11px;color:#666;margin:0 0 12px}
 .sec{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;color:#888;margin:15px 0 6px;border-top:1px solid #e8e8e8;padding-top:12px}
 select,input,button{font-size:13px;padding:5px 7px;border:1px solid #ccc;border-radius:5px;background:#fff}
 select,input.full{width:100%;box-sizing:border-box} button{cursor:pointer;margin:3px 3px 0 0}
 button:hover{background:#eee} button.primary{background:#2166ac;color:#fff;border-color:#2166ac} button.primary:hover{background:#17518c}
 .gbtn{font-size:11px;padding:3px 6px} .muted{color:#999;font-size:11px}
 #subs{max-height:26vh;overflow-y:auto;border:1px solid #e6e6e6;border-radius:6px;padding:6px;background:#fff}
 #subs label{display:flex;align-items:center;gap:6px;font-size:12px;margin:1px 0}
 #legend{font-size:11px;margin-top:8px;max-height:30vh;overflow-y:auto}
 .lg{display:flex;align-items:center;gap:6px;margin:1px 0;cursor:pointer} .sw{width:11px;height:11px;border-radius:2px;flex:0 0 auto}
 #status{font-size:11px;color:#555;margin-top:8px;min-height:14px}
 .row{display:flex;gap:6px;align-items:center;font-size:12px}
</style></head><body>
<div id="wrap">
 <div id="side">
  <h1 id="ttl"></h1><p class="sub" id="stl"></p>
  <div class="sec">Color by</div>
  <select id="catsel"></select>
  <div style="margin-top:8px"><input class="full" id="genein" list="glist" placeholder="…or type a gene (Enter)"><datalist id="glist"></datalist></div>
  <div id="qpanel" style="margin-top:5px"></div>
  <div class="sec">Correlated genes</div>
  <div id="corr"><span class="muted">color by a gene to see correlated genes</span></div>
  <div class="sec">NNMF components</div>
  <div class="row"><label>k <input id="nmfk" type="number" min="2" max="12" value="8" style="width:52px"></label>
   <button id="nmfgo">Compute NNMF</button></div>
  <div id="nmfstat" class="muted"></div>
  <div id="nmfcomps" style="margin-top:5px"></div>
  <div id="nmftop"></div>
  <div class="sec">Subset</div>
  <label class="muted" style="display:block;margin-bottom:3px">subset by</label>
  <select id="subfield" class="full" style="margin-bottom:8px"></select>
  <div class="row" id="agerow" style="margin-bottom:6px"></div>
  <div class="muted" style="margin:2px 0 3px">subtypes <a href="#" id="allon">all</a> · <a href="#" id="alloff">none</a></div>
  <div id="subs"></div>
  <button class="primary" id="apply" style="margin-top:8px">Apply subset</button>
  <button id="recompute">Recompute UMAP ↻</button>
  <button id="reset">Reset all</button>
  <div id="status"></div>
  <div class="sec">Display</div>
  <label style="font-size:12px">point size <input type="range" id="psz" min="1" max="6" step="0.5" value="2.4" style="width:110px;vertical-align:middle"></label>
  <div class="sec">Legend</div><div id="legend"></div>
 </div>
 <div id="plot"></div>
</div>
<script>
const META=__META__, PALETTE=__PALETTE__;
function b64b(s){const bin=atob(s),n=bin.length,u=new Uint8Array(n);for(let i=0;i<n;i++)u[i]=bin.charCodeAt(i);return u;}
const EXPR=b64b("__EXPR__");
const _cb=b64b("__COORD__");const COORD=new Float32Array(_cb.buffer,_cb.byteOffset,_cb.byteLength/4);
const _pb=b64b("__PCA__");const PCA=new Float32Array(_pb.buffer,_pb.byteOffset,_pb.byteLength/4);
const N=META.n_cells,G=META.n_genes,NPC=META.npc,RECOMPUTE_CAP=15000;
// current coordinates (mutable; recompute overwrites)
const CX=new Float32Array(N),CY=new Float32Array(N),CZ=new Float32Array(N);
function resetCoords(){for(let i=0;i<N;i++){CX[i]=COORD[i*3];CY[i]=COORD[i*3+1];CZ[i]=COORD[i*3+2];}}
resetCoords();
const catCache={};
function codes(f){if(catCache[f])return catCache[f];const c=META.cats[f];const b=b64b(c.codes);const arr=new Int16Array(b.buffer,b.byteOffset,b.byteLength/2);catCache[f]={cats:c.categories,codes:arr};return catCache[f];}
let AGE=null;if(META.age){const b=b64b(META.age.vals);AGE=new Float32Array(b.buffer,b.byteOffset,b.byteLength/4);}
let PSZ=2.4, INCL=new Uint8Array(N).fill(1), curColor={kind:'cat',key:META.default_cat};
let SUBSET_FIELD=META.cat_fields[0];
const plotDiv=document.getElementById('plot');
const baseLayout={margin:{l:0,r:0,t:0,b:0},scene:{xaxis:{visible:false},yaxis:{visible:false},zaxis:{visible:false},aspectmode:'data',bgcolor:'#fff'},showlegend:false,paper_bgcolor:'#fff',uirevision:'keep'};
function setStatus(t){document.getElementById('status').innerText=t;}
function activeIdx(){const a=[];for(let i=0;i<N;i++)if(INCL[i])a.push(i);return a;}

function render(){ if(curColor.kind==='gene')renderGene(curColor.key);
  else if(curColor.kind==='age')renderCont(AGE,META.age.name,'Spectral');
  else if(curColor.kind==='nmf'){if(NMF)renderNMF(curColor.key);else renderCat(META.default_cat);}
  else renderCat(curColor.key); }
function renderCat(field){curColor={kind:'cat',key:field};clearCorr();
 const {cats,codes:cd}=codes(field);const idx=cats.map(()=>[]);
 for(let i=0;i<N;i++)if(INCL[i])idx[cd[i]].push(i);
 const traces=[];const legend=[];
 cats.forEach((name,k)=>{const ii=idx[k];if(!ii.length)return;const col=PALETTE[k%PALETTE.length];
   traces.push({type:'scatter3d',mode:'markers',name:name+' ('+ii.length+')',
     x:ii.map(i=>CX[i]),y:ii.map(i=>CY[i]),z:ii.map(i=>CZ[i]),
     marker:{size:PSZ,color:col},hovertext:ii.map(()=>name),hoverinfo:'text'});
   legend.push({name:name+' ('+ii.length+')',color:col});});
 Plotly.react(plotDiv,traces,baseLayout,{responsive:true,displaylogo:false});
 buildLegend(legend); setStatus('Colored by '+field+' — '+activeIdx().length.toLocaleString()+' cells');
}
function renderCont(vals,label,cmap){const ii=activeIdx();
 const v=ii.map(i=>vals[i]);const finite=v.filter(x=>isFinite(x));const lo=Math.min(...finite),hi=Math.max(...finite);
 const tr={type:'scatter3d',mode:'markers',x:ii.map(i=>CX[i]),y:ii.map(i=>CY[i]),z:ii.map(i=>CZ[i]),
   marker:{size:PSZ,color:v,colorscale:cmap||'Viridis',cmin:lo,cmax:hi,colorbar:{title:label,thickness:12,len:.55}},
   hovertext:v.map(x=>label+': '+(isFinite(x)?x.toFixed(2):'NA')),hoverinfo:'text'};
 Plotly.react(plotDiv,[tr],baseLayout,{responsive:true,displaylogo:false});
 document.getElementById('legend').innerHTML='<span class="muted">continuous: '+label+' ['+lo.toFixed(2)+' – '+hi.toFixed(2)+']</span>';
 setStatus('Colored by '+label+' — '+ii.length.toLocaleString()+' cells');
}
function renderGene(g){const j=META.genes.indexOf(g);if(j<0){setStatus('gene not found: '+g);return;}
 curColor={kind:'gene',key:g};const v=new Float32Array(N);for(let i=0;i<N;i++)v[i]=EXPR[i*G+j]/META.expr_scale;
 renderCont(v,g+' (log1p CPM)','Magma'); showCorr(g,j);
}
function clearCorr(){document.getElementById('corr').innerHTML='<span class="muted">color by a gene to see correlated genes</span>';}
function correlate(gj,idx){const n=idx.length;if(n<10)return[];
 let mx=0;for(let t=0;t<n;t++)mx+=EXPR[idx[t]*G+gj];mx/=n;
 const xs=new Float32Array(n);let vx=0;for(let t=0;t<n;t++){const d=EXPR[idx[t]*G+gj]-mx;xs[t]=d;vx+=d*d;}
 const sx=Math.sqrt(vx)||1;const out=[];
 for(let j=0;j<G;j++){if(j===gj)continue;let my=0;for(let t=0;t<n;t++)my+=EXPR[idx[t]*G+j];my/=n;
  let cov=0,vy=0;for(let t=0;t<n;t++){const dy=EXPR[idx[t]*G+j]-my;cov+=xs[t]*dy;vy+=dy*dy;}
  out.push([META.genes[j],cov/(sx*(Math.sqrt(vy)||1))]);}
 out.sort((a,b)=>b[1]-a[1]);return out;}
function corrRow(gn,rv){const c=rv>=0?'#b2182b':'#2166ac';return '<div class="crow" data-g="'+gn+'" style="cursor:pointer;display:flex;justify-content:space-between;padding:1px 2px;border-radius:3px"><span>'+gn+'</span><span style="color:'+c+'">'+(rv>=0?'+':'')+rv.toFixed(2)+'</span></div>';}
function showCorr(g,gj){const idx=activeIdx();const el=document.getElementById('corr');
 if(gj<0||idx.length<10){el.innerHTML='<span class="muted">too few cells in subset</span>';return;}
 el.innerHTML='<span class="muted">computing…</span>';
 setTimeout(()=>{const r=correlate(gj,idx);const pos=r.slice(0,15),neg=r.slice(-15).reverse();
  let html='<div class="muted" style="margin-bottom:3px">'+g+' vs '+(G-1)+' panel genes · '+idx.length.toLocaleString()+' cells (Pearson)</div>';
  html+='<div style="font-weight:700;color:#b2182b">positively correlated</div>';pos.forEach(x=>{html+=corrRow(x[0],x[1]);});
  html+='<div style="margin-top:5px;font-weight:700;color:#2166ac">negatively correlated</div>';neg.forEach(x=>{html+=corrRow(x[0],x[1]);});
  el.innerHTML=html;
  el.querySelectorAll('.crow').forEach(d=>{d.onclick=()=>{document.getElementById('genein').value=d.dataset.g;renderGene(d.dataset.g);};});},15);}
function buildLegend(items){const el=document.getElementById('legend');el.innerHTML='';
 items.forEach((it,k)=>{const d=document.createElement('div');d.className='lg';
  d.innerHTML='<span class="sw" style="background:'+it.color+'"></span><span>'+it.name+'</span>';
  d.onclick=()=>{const vis=plotDiv.data[k].visible;Plotly.restyle(plotDiv,{visible:vis===false?true:'legendonly'},[k]);};
  el.appendChild(d);});}

// ---- NNMF (client-side, main thread, chunked per iteration) ----
let NMF=null, NMF_RUN=false; const NMF_CAP=5000, NMF_ITERS=60;
function clearNMF(){document.getElementById('nmfcomps').innerHTML='';document.getElementById('nmftop').innerHTML='';document.getElementById('nmfstat').innerText='';}
function computeNMF(){
 if(NMF_RUN)return; let idx=activeIdx(); if(idx.length<20){setStatus('subset too small for NNMF');return;}
 if(idx.length>NMF_CAP){const s=[];const st=idx.length/NMF_CAP;for(let t=0;t<NMF_CAP;t++)s.push(idx[Math.floor(t*st)]);INCL.fill(0);s.forEach(i=>INCL[i]=1);idx=s;render();}
 const k=Math.max(2,Math.min(12,Math.round(+document.getElementById('nmfk').value)||8));
 const n=idx.length,V=new Float32Array(n*G),gmax=new Float32Array(G);
 for(let t=0;t<n;t++){const co=idx[t]*G,vo=t*G;for(let j=0;j<G;j++){const v=EXPR[co+j]/META.expr_scale;V[vo+j]=v;if(v>gmax[j])gmax[j]=v;}}
 for(let j=0;j<G;j++){const m=gmax[j]||1;for(let t=0;t<n;t++)V[t*G+j]/=m;}
 let s=12345;const rnd=()=>{s=(s*1664525+1013904223)>>>0;return s/4294967296;};
 const W=new Float32Array(n*k),H=new Float32Array(k*G);for(let i=0;i<n*k;i++)W[i]=rnd()+0.01;for(let i=0;i<k*G;i++)H[i]=rnd()+0.01;
 const eps=1e-9,WtV=new Float32Array(k*G),WtW=new Float32Array(k*k),WtWH=new Float32Array(k*G),HHt=new Float32Array(k*k),VHt=new Float32Array(n*k),WHHt=new Float32Array(n*k);
 const stat=document.getElementById('nmfstat');document.getElementById('nmfgo').disabled=true;NMF_RUN=true;let it=0;
 function stepNMF(){
  WtV.fill(0);for(let i=0;i<n;i++){const vo=i*G,wo=i*k;for(let a=0;a<k;a++){const w=W[wo+a];if(w===0)continue;const to=a*G;for(let j=0;j<G;j++)WtV[to+j]+=w*V[vo+j];}}
  WtW.fill(0);for(let i=0;i<n;i++){const wo=i*k;for(let a=0;a<k;a++){const wa=W[wo+a];for(let b=0;b<k;b++)WtW[a*k+b]+=wa*W[wo+b];}}
  WtWH.fill(0);for(let a=0;a<k;a++)for(let b=0;b<k;b++){const ww=WtW[a*k+b],ho=b*G,to=a*G;for(let j=0;j<G;j++)WtWH[to+j]+=ww*H[ho+j];}
  for(let i=0;i<k*G;i++)H[i]*=WtV[i]/(WtWH[i]+eps);
  HHt.fill(0);for(let a=0;a<k;a++)for(let b=0;b<k;b++){let sm=0;const ao=a*G,bo=b*G;for(let j=0;j<G;j++)sm+=H[ao+j]*H[bo+j];HHt[a*k+b]=sm;}
  VHt.fill(0);for(let i=0;i<n;i++){const vo=i*G,to=i*k;for(let a=0;a<k;a++){let sm=0;const ho=a*G;for(let j=0;j<G;j++)sm+=V[vo+j]*H[ho+j];VHt[to+a]=sm;}}
  WHHt.fill(0);for(let i=0;i<n;i++){const wo=i*k;for(let a=0;a<k;a++){let sm=0;for(let b=0;b<k;b++)sm+=W[wo+b]*HHt[b*k+a];WHHt[wo+a]=sm;}}
  for(let i=0;i<n*k;i++)W[i]*=VHt[i]/(WHHt[i]+eps);
  it++;stat.innerText='NNMF iter '+it+'/'+NMF_ITERS+' (k='+k+', '+n.toLocaleString()+' cells)';
  if(it<NMF_ITERS){setTimeout(stepNMF,0);}else{NMF={W:W,H:H,idx:idx,k:k};NMF_RUN=false;document.getElementById('nmfgo').disabled=false;stat.innerText='NNMF done (k='+k+') — click a component below';buildNMFcomps();renderNMF(0);}
 }
 setTimeout(stepNMF,10);
}
function buildNMFcomps(){const el=document.getElementById('nmfcomps');el.innerHTML='';for(let c=0;c<NMF.k;c++){const b=document.createElement('button');b.className='gbtn';b.id='nmfc'+c;b.innerText='C'+(c+1);b.onclick=()=>renderNMF(c);el.appendChild(b);}}
function renderNMF(c){if(!NMF)return;curColor={kind:'nmf',key:c};clearCorr();
 const g=new Float32Array(N);g.fill(NaN);for(let t=0;t<NMF.idx.length;t++)g[NMF.idx[t]]=NMF.W[t*NMF.k+c];
 renderCont(g,'NNMF C'+(c+1)+' loading','Viridis');
 for(let i=0;i<NMF.k;i++){const b=document.getElementById('nmfc'+i);if(b){b.style.background=i===c?'#2166ac':'';b.style.color=i===c?'#fff':'';}}
 showNMFtop(c);}
function showNMFtop(c){const H=NMF.H,arr=[];for(let j=0;j<G;j++)arr.push([META.genes[j],H[c*G+j]]);arr.sort((a,b)=>b[1]-a[1]);const top=arr.slice(0,20);
 let html='<div class="muted" style="margin:4px 0 2px">top 20 genes · NNMF C'+(c+1)+'</div>';
 top.forEach(x=>{html+='<div class="crow" data-g="'+x[0]+'" style="cursor:pointer;display:flex;justify-content:space-between;padding:1px 2px"><span>'+x[0]+'</span><span style="color:#555">'+x[1].toFixed(2)+'</span></div>';});
 const el=document.getElementById('nmftop');el.innerHTML=html;
 el.querySelectorAll('.crow').forEach(d=>{d.onclick=()=>{document.getElementById('genein').value=d.dataset.g;renderGene(d.dataset.g);};});}
document.getElementById('nmfgo').onclick=computeNMF;

// ---- controls: color by ----
document.getElementById('ttl').innerText=META.title;
document.getElementById('stl').innerText=META.subtitle+' · '+N.toLocaleString()+' cells · '+G+' genes';
const catsel=document.getElementById('catsel');
META.cat_fields.forEach(f=>{const o=document.createElement('option');o.value=f;o.text=f;catsel.appendChild(o);});
if(META.age){const o=document.createElement('option');o.value='__age__';o.text=META.age.name+' (numeric)';catsel.appendChild(o);}
catsel.value=META.default_cat;
catsel.onchange=()=>{document.getElementById('genein').value='';const v=catsel.value; if(v==='__age__'){curColor={kind:'age'};clearCorr();renderCont(AGE,META.age.name,'Spectral');}else renderCat(v);};
const gl=document.getElementById('glist');META.genes.forEach(g=>{const o=document.createElement('option');o.value=g;gl.appendChild(o);});
function doGene(){const g=document.getElementById('genein').value.trim();if(g)renderGene(g);}
document.getElementById('genein').addEventListener('keydown',e=>{if(e.key==='Enter')doGene();});
const qp=document.getElementById('qpanel');META.panel.slice(0,20).forEach(g=>{const b=document.createElement('button');b.className='gbtn';b.innerText=g;b.onclick=()=>{document.getElementById('genein').value=g;renderGene(g);};qp.appendChild(b);});

// ---- controls: subset ----
const subsEl=document.getElementById('subs');
const subfield=document.getElementById('subfield');
META.cat_fields.forEach(f=>{const o=document.createElement('option');o.value=f;o.text=f;subfield.appendChild(o);});
subfield.value=SUBSET_FIELD;
function buildSubs(){SUBSET_FIELD=subfield.value;const cats=codes(SUBSET_FIELD).cats;subsEl.innerHTML='';
  cats.forEach((c,k)=>{subsEl.insertAdjacentHTML('beforeend','<label><input type="checkbox" checked value="'+k+'"><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:'+PALETTE[k%PALETTE.length]+'"></span>'+c+'</label>');});}
subfield.onchange=buildSubs; buildSubs();
document.getElementById('allon').onclick=e=>{e.preventDefault();subsEl.querySelectorAll('input').forEach(c=>c.checked=true);};
document.getElementById('alloff').onclick=e=>{e.preventDefault();subsEl.querySelectorAll('input').forEach(c=>c.checked=false);};
let AGELO,AGEHI;
if(META.age){document.getElementById('agerow').innerHTML=
  META.age.name+' <input id="alo" type="number" step="1" value="'+Math.floor(META.age.min)+'" style="width:58px"> – <input id="ahi" type="number" step="1" value="'+Math.ceil(META.age.max)+'" style="width:58px">';}
function applySubset(){const keep=new Set();subsEl.querySelectorAll('input:checked').forEach(c=>keep.add(+c.value));
 const cd=codes(SUBSET_FIELD).codes; let lo=-1e9,hi=1e9;
 if(META.age){lo=+document.getElementById('alo').value;hi=+document.getElementById('ahi').value;}
 let n=0;for(let i=0;i<N;i++){const ok=keep.has(cd[i])&&(!META.age||!isFinite(AGE[i])||(AGE[i]>=lo&&AGE[i]<=hi));INCL[i]=ok?1:0;if(ok)n++;}
 NMF=null;clearNMF();if(curColor.kind==='nmf')curColor={kind:'cat',key:META.default_cat};
 render(); setStatus('Subset: '+n.toLocaleString()+' cells');
}
document.getElementById('apply').onclick=applySubset;

// ---- client-side UMAP recompute on current subset ----
document.getElementById('recompute').onclick=async ()=>{
 let idx=activeIdx();
 if(idx.length<10){setStatus('subset too small to recompute');return;}
 if(idx.length>RECOMPUTE_CAP){ // downsample for tractable in-browser UMAP
   const s=[];const step=idx.length/RECOMPUTE_CAP;for(let t=0;t<RECOMPUTE_CAP;t++)s.push(idx[Math.floor(t*step)]);
   INCL.fill(0);s.forEach(i=>INCL[i]=1);idx=s;
   setStatus('subset > '+RECOMPUTE_CAP+', downsampled to '+RECOMPUTE_CAP+' for recompute');
 }
 const data=idx.map(i=>{const r=new Array(NPC);for(let k=0;k<NPC;k++)r[k]=PCA[i*NPC+k];return r;});
 const U=(window.UMAP&&window.UMAP.UMAP)?window.UMAP.UMAP:window.UMAP;
 const umap=new U({nComponents:3,nNeighbors:Math.min(15,idx.length-1),minDist:0.3,random:mulberry32(0)});
 setStatus('recomputing UMAP on '+idx.length.toLocaleString()+' cells…');
 const nE=umap.initializeFit(data);
 await umap.fitAsync(data,ep=>{if(ep%20===0)setStatus('recomputing… epoch '+ep+'/'+nE);});
 const emb=umap.getEmbedding();
 let mx=[0,0,0];emb.forEach(p=>{mx[0]+=p[0];mx[1]+=p[1];mx[2]+=p[2];});mx=mx.map(v=>v/emb.length);
 let sd=0;emb.forEach(p=>{sd+=Math.hypot(p[0]-mx[0],p[1]-mx[1],p[2]-mx[2]);});sd/=emb.length;sd=sd||1;
 idx.forEach((i,t)=>{CX[i]=(emb[t][0]-mx[0])/sd;CY[i]=(emb[t][1]-mx[1])/sd;CZ[i]=(emb[t][2]-mx[2])/sd;});
 render(); setStatus('Recomputed UMAP on '+idx.length.toLocaleString()+' cells (subset re-embedded)');
};
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}

document.getElementById('reset').onclick=()=>{INCL.fill(1);resetCoords();
 subfield.value=META.cat_fields[0];buildSubs();
 if(META.age){document.getElementById('alo').value=Math.floor(META.age.min);document.getElementById('ahi').value=Math.ceil(META.age.max);}
 catsel.value=META.default_cat;renderCat(META.default_cat);setStatus('Reset to full data + original UMAP');};
document.getElementById('psz').oninput=e=>{PSZ=+e.target.value;Plotly.restyle(plotDiv,{'marker.size':PSZ});};
renderCat(META.default_cat);
</script></body></html>"""

# ------------------------------------------------------------------ panels
MGE=["Gad1","Gad2","Slc32a1","Lhx6","Nkx2-1","Maf","Mafb","Pvalb","Sst","Chodl","Nos1","Calb1","Calb2","Reln","Npy","Cck","Vip","Chrna2","Crhbp","Crh","Tac1","Pdyn","Th","Etv1","Nr2f2","Elfn1","Cbln4","Hpse","Myh8","Sema3c","Grin3a","Cort","Tacr1","Mef2c","Satb1","Lamp5","Id2","Adarb2","Pthlh","Dcx","Sox11","Sox4","Neurod2","Mki67","Sox2","Ascl1","Dlx1","Dlx2","Dlx5","Snap25","Syt1","Rbfox3","Syp","Fos","Egr1","Arc","Bdnf",
     # Sst/Pvalb cluster-discriminating markers (Tasic-mapping analysis)
     "Rbp4","Lypd6","Lypd6b","Mgat4c","Necab1","Pdlim5","Nts","Rxfp1","Esm1","Mme","Tac2","Myh4",
     "Gabrg1","Akr1c18","Ntf3","Sema3e","Kank4","Gpr149","Islr","Itm2a","Tpbg","Vipr2","Unc5b","Tacr3",
     "Trhde","Gas7","Cartpt","Kirrel3","Cplx2","Ndnf","Crhr2","Efemp1","Glra3","Ptgdr","Fibin","Tacstd2"]
CGE=["Gad1","Gad2","Slc32a1","Adarb2","Prox1","Nr2f2","Sp8","Htr3a","Lamp5","Vip","Sncg","Serpinf1","Cxcl14","Npy","Ndnf","Reln","Cnr1","Cck","Calb2","Lsp1","Dock5","Egln3","Pax6","Krt73","Piezo2","Csgalnact1","Ntng1","Chrna7","Id2","Chat","Slc17a8","Crh","Tac2","Penk","Nr2f1","Dcx","Sox11","Sox4","Neurod2","Mki67","Sox2","Dlx1","Dlx2","Snap25","Syt1","Rbfox3","Syp","Dnmt3a","Tet1","Fos","Egr1","Arc","Bdnf"]
EXC=["Slc17a7","Slc17a6","Satb2","Neurod2","Neurod6","Tbr1","Eomes","Fezf2","Bcl11b","Foxp2","Tle4","Sox5","Rorb","Cux1","Cux2","Pou3f2","Pou3f3","Lamp5","Rbp4","Ccn2","Nr4a2","Etv1","Rprm","Nxph3","Tshz2","Sla","Cdh13","Calb1","Gad1","Slc32a1","Dcx","Sox11","Sox4","Mki67","Sox2","Pax6","Hes1","Hes5","Nes","Snap25","Syt1","Rbfox3","Fos","Egr1","Arc","Bdnf","Nr4a3","Fosb","Fosl2"]
MIN=["Gad1","Gad2","Slc32a1","Pvalb","Sst","Vip","Lamp5","Sncg","Lhx6","Nkx2-1","Adarb2","Prox1","Calb1","Calb2","Reln","Npy","Cck","Cnr1","Chodl","Nos1","Cxcl14","Ndnf","Lsp1","Dock5","Egln3","Pax6","Krt73","Chrna2","Crhbp","Crh","Tac1","Th","Vipr2","Nr2f2","Id2","Chat","Htr3a","Sp8","Dcx","Sox11","Snap25","Syt1","Mki67"]

DEV="/home/inb539/LR_analysis/allen-long-dataset/DevVIS_scRNA_processed.h5ad"
MINF="/n/data1/hms/neurobio/fishell/jinoh/evodevo/MinLongitudinalRemapping.h5ad"

def dev_subset(classes, cache, n_max=None):
    cf=CACHE/cache
    if cf.exists(): return sc.read_h5ad(cf)
    big=ad.read_h5ad(DEV,backed="r"); mask=big.obs["class_label"].astype(str).isin(classes).values
    idx=np.where(mask)[0]
    if n_max and len(idx)>n_max:
        rs=np.random.RandomState(RNG); idx=np.sort(rs.choice(idx,n_max,replace=False))
    sub=big[idx].to_memory(); sub.write(cf); return sub

if __name__=="__main__":
    which=sys.argv[1] if len(sys.argv)>1 else "all"
    if which in ("cge","all"):
        print("CGE..."); d=sc.read_h5ad("/n/scratch/users/i/inb539/cge_cache/devvis_cge.h5ad")
        build(d,"devvis_cge.html","DevVIS — CGE interneurons","Lamp5 / Vip / Sncg + immature CGE (E13.5→P56)",
              ["subclass_label","cluster_label","class_label","Age"],CGE,age_field="age (d from birth)",age_from="Age")
    if which in ("mge","all"):
        print("MGE..."); d=dev_subset(["CTX-MGE GABA"],"devvis_mge.h5ad")
        build(d,"devvis_mge.html","DevVIS — MGE interneurons","Pvalb / Sst / Lamp5-Lhx6 + immature MGE (E11.5→P56)",
              ["subclass_label","cluster_label","class_label","Age"],MGE,age_field="age (d from birth)",age_from="Age")
    if which in ("exc","all"):
        print("EXC..."); d=dev_subset(["IT Glut","nonIT Glut"],"devvis_exc.h5ad",n_max=60000)
        build(d,"devvis_excitatory.html","DevVIS — excitatory neurons","IT + nonIT Glut (E11.5→P56, subsampled 60k)",
              ["subclass_label","cluster_label","class_label","Age"],EXC,age_field="age (d from birth)",age_from="Age")
    if which in ("min","all"):
        print("MIN..."); d=ad.read_h5ad(MINF,backed="r").to_memory()
        d.X=d.layers["log1p"].copy(); d.layers.clear()
        build(d,"min_longitudinal.html","Min Longitudinal — cortical interneurons","PV / SST / VIP / LAMP5 (P2→P28)",
              ["CellTypes_subclass","CellTypes","Sample"],MIN,age_field="age (postnatal day)",age_from="Sample")
    print("DONE")
