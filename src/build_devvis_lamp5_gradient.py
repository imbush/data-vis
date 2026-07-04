"""Build the DevVIS Lamp5 explorer v2 from lamp5_grad2.npz.
Adds: selectable continuum y-axis (per-age × PC/pseudotime + global + consensus),
dot-size slider, and a DevVIS-style NMF panel (component -> clickable top genes)."""
import numpy as np, json, base64, os
SP='/private/tmp/claude-501/-Users-inlebush-cs-lab-green-data-vis/95ef1b74-d552-4696-b236-9181bf853664/scratchpad'
OUT='/Users/inlebush/cs/lab/green/data-vis/devvis/devvis_lamp5_gradient.html'
d=np.load(f'{SP}/lamp5_grad2.npz', allow_pickle=True)
def b64(a): return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
mat=d['maturity'].astype(np.float32); Y=d['Y'].astype(np.float32); age=d['age'].astype(np.float32)
axis_names=[str(x) for x in d['axis_names']]; clu=d['clu_code'].astype(np.int16); clu_cats=[str(x) for x in d['clu_cats']]
genes=[str(g) for g in d['embed_genes']]; expr=d['expr_u8']; escale=int(d['expr_scale'])
W=d['nmf_W'].astype(np.float32); nmf_top=[str(x) for x in d['nmf_top']]; matf=int(d['mat_fac'])
gcls=[str(x) for x in d['g_cls']]; geta=d['g_eta'].astype(float); shared=[str(x) for x in d['shared_genes']]
N,G=expr.shape; NAX=len(axis_names); K=W.shape[1]
tbl=[{'g':genes[i],'i':i,'cls':gcls[i],'eta':round(geta[i],3)} for i in range(G) if gcls[i] in ('A-pole','B-pole','mid-peak','mid-dip')]
tbl.sort(key=lambda r:-r['eta'])
xed=[float(v) for v in np.quantile(mat,np.linspace(0,1,7))]
def ref_of(nm):   # cells in which to measure gradedness for each axis
    if nm.startswith('Global') or nm.startswith('Consensus'): return 'mat'
    if nm.startswith('P3'):  return [3,7]
    if nm.startswith('P8'):  return [8,14]
    if nm.startswith('P15'): return [15,28]
    if nm.startswith('P56'): return [56,56]
    return 'mat'
axis_ref=[ref_of(n) for n in axis_names]
META=dict(n=N,g=G,k=K,nax=NAX,genes=genes,escale=escale,clu_cats=clu_cats,axis_names=axis_names,
          nmf_top=nmf_top,mat_fac=matf,xed=xed,shared=shared,axis_ref=axis_ref)

TEMPLATE=r"""<!doctype html><html><head><meta charset="utf-8"><title>DevVIS Lamp5 — maturation × continuum</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 body{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#1a1a1a}
 #wrap{display:flex;height:100vh}
 #side{width:330px;padding:14px 16px;overflow-y:auto;border-right:1px solid #e3e3e3;background:#faf8f5}
 #main{flex:1;display:flex;flex-direction:column;min-width:0}
 #plot{flex:1;min-height:0} #emerge{height:250px;border-top:1px solid #e3e3e3}
 h1{font-size:15px;margin:0 0 2px} .sub{font-size:11px;color:#666;margin:0 0 8px;line-height:1.35}
 .sec{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;color:#8a7a5a;margin:13px 0 5px;border-top:1px solid #ece5da;padding-top:10px}
 select,input,button{font-size:12px;padding:5px 7px;border:1px solid #ccc;border-radius:5px;background:#fff}
 select,input.full{width:100%;box-sizing:border-box} button{cursor:pointer;margin:2px 3px 0 0}
 button:hover{background:#eee} button.on{background:#8a5a2b;color:#fff;border-color:#8a5a2b}
 .muted{color:#999;font-size:11px} .row{display:flex;gap:5px;align-items:center;flex-wrap:wrap}
 #glist,#nmfgenes{max-height:32vh;overflow-y:auto;border:1px solid #eadfce;border-radius:6px;background:#fff;margin-top:6px}
 .grow{display:flex;justify-content:space-between;align-items:center;padding:3px 7px;font-size:12px;cursor:pointer;border-bottom:1px solid #f2ece2}
 .grow:hover{background:#f6efe3} .grow.sel{background:#f0e2cc}
 .tag{font-size:9px;font-weight:700;padding:1px 5px;border-radius:8px;color:#fff}
 .clsbtn,.nmfb{font-size:10px;padding:3px 7px} #status{font-size:11px;color:#555;margin-top:7px;min-height:14px}
 label.sld{font-size:12px;display:block;margin-top:4px}
</style></head><body>
<div id="wrap">
 <div id="side">
  <h1>DevVIS Lamp5 · maturation × continuum</h1>
  <p class="sub" id="sub"></p>
  <div class="sec">Continuum y-axis</div>
  <select id="axis"></select>
  <div class="muted" id="axnote" style="margin-top:4px"></div>
  <div class="sec">Colour cells by</div>
  <select id="mode">
   <option value="gene">gene expression</option><option value="nmf">NMF factor</option>
   <option value="age">age (P days)</option><option value="cluster">cluster</option>
   <option value="maturity">maturity (x)</option><option value="continuum">continuum (y)</option></select>
  <div id="genebox" style="margin-top:7px"><input class="full" id="gin" list="gl" placeholder="type a gene (Enter)"><datalist id="gl"></datalist></div>
  <div id="nmfbox" style="margin-top:7px;display:none">
    <div class="muted">click a component, then a gene to colour by it (<b>*</b>=maturation)</div>
    <div id="nmfbtns" style="margin-top:4px"></div><div id="nmfgenes"></div></div>
  <div class="sec">Display</div>
  <label class="sld">dot size <input type="range" id="psz" min="1" max="9" step="0.5" value="4" style="width:150px;vertical-align:middle"> <span id="pszv">4</span></label>
  <div class="sec">Smoothly-graded genes along <span id="gtlabel" style="text-transform:none;color:#5a4a2a"></span></div>
  <div class="row" id="clsfilter">
   <button class="clsbtn on" data-c="all">all</button><button class="clsbtn" data-c="A-pole">A-pole</button>
   <button class="clsbtn" data-c="B-pole">B-pole</button><button class="clsbtn" data-c="mid-peak">mid-peak</button>
   <button class="clsbtn" data-c="mid-dip">mid-dip</button></div>
  <div id="glist"></div>
  <div id="status"></div>
 </div>
 <div id="main"><div id="plot"></div><div id="emerge"></div></div>
</div>
<script>
const META=__META__;
function b64b(s){const bin=atob(s),n=bin.length,u=new Uint8Array(n);for(let i=0;i<n;i++)u[i]=bin.charCodeAt(i);return u;}
function f32(s){const b=b64b(s);return new Float32Array(b.buffer,b.byteOffset,b.byteLength/4);}
const N=META.n,G=META.g,K=META.k,NAX=META.nax,ES=META.escale;
const EXPR=b64b("__EXPR__"),MAT=f32("__MAT__"),AGE=f32("__AGE__"),YALL=f32("__Y__"),NMFW=f32("__NMFW__");
const CLUb=b64b("__CLU__"),CLU=new Int16Array(CLUb.buffer,CLUb.byteOffset,CLUb.byteLength/2);
const PALETTE=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','#393b79','#637939'];
const CLSCOL={'A-pole':'#2166ac','B-pole':'#b2182b','mid-peak':'#7b3294','mid-dip':'#1b7837'};
const plotDiv=document.getElementById('plot'),emDiv=document.getElementById('emerge');
document.getElementById('sub').innerHTML=N.toLocaleString()+" Lamp5 · Gao 2025 · P3→P56 · "+G+" genes.<br><b>x</b>=maturity (Dcx/Sox11↓) · <b>y</b>=identity continuum (selectable below).";
let axisIdx=0, PSZ=4, curGene=-1, curMode='gene', curNMF=0;
function YC(){const c=new Float32Array(N);for(let i=0;i<N;i++)c[i]=YALL[i*NAX+axisIdx];return c;}
let yc=YC();
const gene=j=>{const o=new Float32Array(N);for(let i=0;i<N;i++)o[i]=EXPR[i*G+j]/ES;return o;};
function quant(a,q){const s=Float32Array.from(a).sort();return s[Math.floor(q*(s.length-1))];}
const baseLayout=()=>({margin:{l:44,r:8,t:8,b:38},hovermode:'closest',
 xaxis:{title:'maturity  (immature → mature)',zeroline:false},
 yaxis:{title:'continuum: '+META.axis_names[axisIdx]+'  (A-pole ↔ B-pole)',zeroline:false},
 paper_bgcolor:'#fff',plot_bgcolor:'#fff',uirevision:'k'});
function setStatus(t){document.getElementById('status').innerHTML=t;}

function scatter(color,cbar,cscale,discrete){
 if(discrete){const traces=META.clu_cats.map((nm,k)=>({type:'scattergl',mode:'markers',name:nm,x:[],y:[],marker:{size:PSZ,color:PALETTE[k%PALETTE.length]},hoverinfo:'name'}));
  for(let i=0;i<N;i++){const t=traces[CLU[i]];t.x.push(MAT[i]);t.y.push(yc[i]);}
  Plotly.react(plotDiv,traces,Object.assign(baseLayout(),{showlegend:true,legend:{font:{size:9},itemsizing:'constant'}}),{responsive:true,displaylogo:false});return;}
 const tr={type:'scattergl',mode:'markers',x:Array.from(MAT),y:Array.from(yc),
   marker:{size:PSZ,color:Array.from(color),colorscale:cscale,colorbar:{title:cbar,thickness:12,len:.7},cmin:quant(color,0.02),cmax:quant(color,0.98)},hoverinfo:'skip'};
 Plotly.react(plotDiv,[tr],Object.assign(baseLayout(),{showlegend:false}),{responsive:true,displaylogo:false});}

function xbin(v){const e=META.xed;for(let b=0;b<6;b++)if(v<=e[b+1]||b==5)return b;return 5;}
function emergence(j,label){const NY=12;const ye=[];const s=Float32Array.from(yc).sort();for(let b=0;b<=NY;b++)ye.push(s[Math.floor(b/NY*(N-1))]);
 const yb=v=>{for(let b=0;b<NY;b++)if(v<=ye[b+1]||b==NY-1)return b;return NY-1;};
 const sum=Array.from({length:NY},()=>new Float64Array(6)),cnt=Array.from({length:NY},()=>new Float64Array(6));
 for(let i=0;i<N;i++){const xb=xbin(MAT[i]),yy=yb(yc[i]);sum[yy][xb]+=EXPR[i*G+j]/ES;cnt[yy][xb]++;}
 const z=sum.map((r,y)=>r.map((s,x)=>cnt[y][x]?s/cnt[y][x]:null));
 Plotly.react(emDiv,[{type:'heatmap',z:z,x:[...Array(6)].map((_,b)=>'M'+(b+1)),y:[...Array(NY)].map((_,b)=>b+1),
   colorscale:'Magma',colorbar:{title:'mean',thickness:10,len:.9},hoverongaps:false,
   hovertemplate:'maturity %{x} · continuum %{y}<br>'+label+' = %{z:.2f}<extra></extra>'}],
  {margin:{l:34,r:8,t:22,b:30},title:{text:label+' — maturity (→) × continuum (↑) mean expression',font:{size:11}},
   xaxis:{title:'maturity bin',type:'category'},yaxis:{title:'continuum bin',type:'category'}},{responsive:true,displaylogo:false});}

// gradedness of every gene along the SELECTED y-axis, measured in that axis's
// reference cells (mature quartile for Global/Consensus, the age window otherwise).
function computeTable(){
 const ref=META.axis_ref[axisIdx], cells=[];
 if(ref==='mat'){const q=quant(MAT,0.75);for(let i=0;i<N;i++)if(MAT[i]>q)cells.push(i);}
 else{for(let i=0;i<N;i++)if(AGE[i]>=ref[0]&&AGE[i]<=ref[1])cells.push(i);}
 const nc=cells.length,B=12; if(nc<80)return [];
 const yv=cells.map(i=>yc[i]).sort((a,b)=>a-b); const ed=[];
 for(let b=0;b<=B;b++)ed.push(yv[Math.min(nc-1,Math.floor(b/B*(nc-1)))]);
 const cb=new Int8Array(nc);
 for(let t=0;t<nc;t++){const v=yc[cells[t]];let b=B-1;for(let k=0;k<B;k++){if(v<=ed[k+1]){b=k;break;}}cb[t]=b;}
 const im=(B-1)/2; let ivar=0;for(let b=0;b<B;b++)ivar+=(b-im)*(b-im);
 const out=[];
 for(let j=0;j<G;j++){
  const bs=new Float64Array(B),bc=new Int32Array(B);let s=0,ss=0;
  for(let t=0;t<nc;t++){const e=EXPR[cells[t]*G+j]/ES,b=cb[t];bs[b]+=e;bc[b]++;s+=e;ss+=e*e;}
  const grand=s/nc,tot=ss-nc*grand*grand+1e-9;let bet=0,mm=0;const m=new Float64Array(B);
  for(let b=0;b<B;b++){if(bc[b]>0)m[b]=bs[b]/bc[b];bet+=bc[b]*(m[b]-grand)*(m[b]-grand);mm+=m[b];}
  const eta=bet/tot; if(eta<0.06)continue; mm/=B;
  let mx=-1e9,mn=1e9,amax=0,amin=0,cov=0,mv=0;
  for(let b=0;b<B;b++){if(m[b]>mx){mx=m[b];amax=b;}if(m[b]<mn){mn=m[b];amin=b;}}
  for(let b=0;b<B;b++){cov+=(b-im)*(m[b]-mm);mv+=(m[b]-mm)*(m[b]-mm);}
  const lin=cov/(Math.sqrt(ivar*mv)+1e-9),rng=mx-mn+1e-9;
  const ppk=(m[amax]-Math.max(m[0],m[B-1]))/rng, pdp=(Math.min(m[0],m[B-1])-m[amin])/rng;
  let cls=null;
  if(Math.abs(lin)>=0.6) cls=lin>0?'B-pole':'A-pole';
  else if(amax>=2&&amax<=B-3&&ppk>0.35) cls='mid-peak';
  else if(amin>=2&&amin<=B-3&&pdp>0.35) cls='mid-dip';
  if(cls) out.push({g:META.genes[j],i:j,cls:cls,eta:+eta.toFixed(3)});
 }
 out.sort((a,b)=>b.eta-a.eta); return out;
}
let curTable=[];
function refreshTable(){curTable=computeTable();document.getElementById('gtlabel').textContent=META.axis_names[axisIdx];buildList();}
function colorByGene(j){curGene=j;curMode='gene';const g=META.genes[j];document.getElementById('gin').value=g;
 scatter(gene(j),g+' (log₂)','Magma',false);
 const t=curTable.find(r=>r.i===j); setStatus('<b>'+g+'</b>'+(t?' · '+t.cls+' · η²='+t.eta.toFixed(2)+' (along '+META.axis_names[axisIdx]+')':''));
 emergence(j,g);[...document.querySelectorAll('.grow')].forEach(r=>r.classList.toggle('sel',+r.dataset.j===j));}
function colorByNMF(k){curNMF=k;curMode='nmf';const col=new Float32Array(N);for(let i=0;i<N;i++)col[i]=NMFW[i*K+k];
 scatter(col,'C'+(k+1)+' loading','Viridis',false);
 const top=META.nmf_top[k].split('/');const el=document.getElementById('nmfgenes');el.innerHTML='';
 top.forEach(g=>{const j=META.genes.indexOf(g);const d=document.createElement('div');d.className='grow';
   d.innerHTML='<span>'+g+'</span><span class="muted">'+(j>=0?'show':'—')+'</span>';
   if(j>=0)d.onclick=()=>colorByGene(j); el.appendChild(d);});
 setStatus('NMF <b>C'+(k+1)+'</b>'+(k===META.mat_fac?' (maturation)':'')+' — top genes listed; click to colour');
 [...document.getElementById('nmfbtns').children].forEach((b,i)=>b.classList.toggle('on',i===k));}
function redraw(){ if(curMode==='gene')colorByGene(curGene>=0?curGene:META.genes.indexOf('Cxcl14'));
 else if(curMode==='nmf')colorByNMF(curNMF);
 else if(curMode==='age')scatter(AGE,'age (P)','Portland',false);
 else if(curMode==='cluster')scatter(null,null,null,true);
 else if(curMode==='maturity')scatter(MAT,'maturity','Viridis',false);
 else if(curMode==='continuum')scatter(yc,'continuum','RdBu',false);}

// axis selector
const axsel=document.getElementById('axis');META.axis_names.forEach((n,i)=>{const o=document.createElement('option');o.value=i;o.text=n;axsel.appendChild(o);});
document.getElementById('axnote').innerHTML='PC & pseudotime agree closely per age; the <b>P3–7</b> axis differs most from adult. "Consensus" = '+META.shared.length+' genes shared across all age axes.';
axsel.onchange=()=>{axisIdx=+axsel.value;yc=YC();refreshTable();redraw();};
// mode
document.getElementById('mode').onchange=()=>{const m=document.getElementById('mode').value;curMode=m;
 document.getElementById('genebox').style.display=m==='gene'?'block':'none';
 document.getElementById('nmfbox').style.display=m==='nmf'?'block':'none';
 if(m==='nmf')colorByNMF(curNMF); else redraw();};
// dot size
document.getElementById('psz').oninput=e=>{PSZ=+e.target.value;document.getElementById('pszv').textContent=PSZ;Plotly.restyle(plotDiv,{'marker.size':PSZ});};
// gene search
const gl=document.getElementById('gl');META.genes.forEach(g=>{const o=document.createElement('option');o.value=g;gl.appendChild(o);});
document.getElementById('gin').addEventListener('keydown',e=>{if(e.key==='Enter'){const j=META.genes.indexOf(e.target.value.trim());if(j>=0){document.getElementById('mode').value='gene';curMode='gene';document.getElementById('genebox').style.display='block';document.getElementById('nmfbox').style.display='none';colorByGene(j);}else setStatus('not embedded: '+e.target.value);}});
// nmf buttons
const nb=document.getElementById('nmfbtns');for(let k=0;k<K;k++){const b=document.createElement('button');b.className='nmfb';b.textContent='C'+(k+1)+(k===META.mat_fac?'*':'');b.onclick=()=>colorByNMF(k);nb.appendChild(b);}
// graded gene list
let clsFilter='all';
function buildList(){const el=document.getElementById('glist');el.innerHTML='';
 curTable.filter(r=>clsFilter==='all'||r.cls===clsFilter).slice(0,400).forEach(r=>{const d=document.createElement('div');d.className='grow';d.dataset.j=r.i;
  d.innerHTML='<span><span class="tag" style="background:'+CLSCOL[r.cls]+'">'+r.cls+'</span> '+r.g+'</span><span class="muted">η²'+r.eta.toFixed(2)+'</span>';
  d.onclick=()=>{document.getElementById('mode').value='gene';curMode='gene';document.getElementById('genebox').style.display='block';document.getElementById('nmfbox').style.display='none';colorByGene(r.i);};el.appendChild(d);});}
document.querySelectorAll('.clsbtn').forEach(b=>b.onclick=()=>{document.querySelectorAll('.clsbtn').forEach(x=>x.classList.remove('on'));b.classList.add('on');clsFilter=b.dataset.c;buildList();});
refreshTable();colorByGene(META.genes.indexOf('Cxcl14'));
</script></body></html>"""
# g_cls lookup for status (attach to table entries already); also expose gcls via table
html=TEMPLATE.replace("__META__",json.dumps(META)).replace("__EXPR__",b64(expr)).replace("__MAT__",b64(mat))\
    .replace("__AGE__",b64(age)).replace("__Y__",b64(Y)).replace("__CLU__",b64(clu)).replace("__NMFW__",b64(W))
open(OUT,'w').write(html)
print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB); axes={NAX}, genes={G}, nmf K={K}")
