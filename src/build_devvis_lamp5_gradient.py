"""Build the self-contained DevVIS Lamp5 maturation x continuum explorer from
lamp5_grad.npz. x = transcriptomic maturity, y = Lamp5 identity continuum.
Colour cells by gene / NMF factor / age / cluster; a graded-gene list (classified
by shape along the continuum) drives colouring; an emergence heatmap (maturity x
continuum, mean expression) shows how early each gene's gradient appears."""
import numpy as np, json, base64
SP='/private/tmp/claude-501/-Users-inlebush-cs-lab-green-data-vis/95ef1b74-d552-4696-b236-9181bf853664/scratchpad'
OUT='/Users/inlebush/cs/lab/green/data-vis/devvis/devvis_lamp5_gradient.html'
d=np.load(f'{SP}/lamp5_grad.npz', allow_pickle=True)

def b64(a): return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
mat=d['maturity'].astype(np.float32); yc=d['y'].astype(np.float32); age=d['age'].astype(np.float32)
clu=d['clu_code'].astype(np.int16); clu_cats=[str(x) for x in d['clu_cats']]
genes=[str(g) for g in d['embed_genes']]; expr=d['expr_u8']; escale=int(d['expr_scale'])
W=d['nmf_W'].astype(np.float32); nmf_top=[str(x) for x in d['nmf_top']]; mat_fac=int(d['mat_fac'])
g_cls=[str(x) for x in d['g_cls']]; g_eta=d['g_eta'].astype(float); g_spr=d['g_spr'].astype(float)
g_elo=d['g_elo'].astype(float); g_ehi=d['g_ehi'].astype(float)
N=len(mat); G=len(genes); K=W.shape[1]

# quantile bin edges for the emergence heatmap (maturity x continuum)
NX,NY=6,12
xed=[float(v) for v in np.quantile(mat,np.linspace(0,1,NX+1))]
yed=[float(v) for v in np.quantile(yc,np.linspace(0,1,NY+1))]

# graded-gene table (only classified genes), sorted by class then eta
tbl=[{'g':genes[i],'i':i,'cls':g_cls[i],'eta':round(g_eta[i],3),
      'elo':round(g_elo[i],3),'ehi':round(g_ehi[i],3)}
     for i in range(G) if g_cls[i] in ('A-pole','B-pole','mid-peak','mid-dip')]
tbl.sort(key=lambda r:(-r['eta']))

META=dict(n=N,g=G,k=K,genes=genes,escale=escale,clu_cats=clu_cats,
          nmf_top=nmf_top,mat_fac=mat_fac,xed=xed,yed=yed,nx=NX,ny=NY,
          gcls=g_cls,geta=[round(float(x),3) for x in g_eta],
          gelo=[round(float(x),3) for x in g_elo],gehi=[round(float(x),3) for x in g_ehi],
          table=tbl)

html=TEMPLATE=r"""<!doctype html><html><head><meta charset="utf-8"><title>DevVIS Lamp5 — maturation × continuum</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 body{margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#1a1a1a}
 #wrap{display:flex;height:100vh}
 #side{width:326px;padding:14px 16px;overflow-y:auto;border-right:1px solid #e3e3e3;background:#faf8f5}
 #main{flex:1;display:flex;flex-direction:column;min-width:0}
 #plot{flex:1;min-height:0}
 #emerge{height:270px;border-top:1px solid #e3e3e3}
 h1{font-size:15px;margin:0 0 2px} .sub{font-size:11px;color:#666;margin:0 0 10px;line-height:1.35}
 .sec{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;color:#8a7a5a;margin:14px 0 6px;border-top:1px solid #ece5da;padding-top:11px}
 select,input,button{font-size:12px;padding:5px 7px;border:1px solid #ccc;border-radius:5px;background:#fff}
 select,input.full{width:100%;box-sizing:border-box} button{cursor:pointer;margin:2px 3px 0 0}
 button:hover{background:#eee} button.on{background:#8a5a2b;color:#fff;border-color:#8a5a2b}
 .muted{color:#999;font-size:11px} .row{display:flex;gap:5px;align-items:center;flex-wrap:wrap}
 #glist{max-height:40vh;overflow-y:auto;border:1px solid #eadfce;border-radius:6px;background:#fff;margin-top:6px}
 .grow{display:flex;justify-content:space-between;align-items:center;padding:3px 7px;font-size:12px;cursor:pointer;border-bottom:1px solid #f2ece2}
 .grow:hover{background:#f6efe3} .grow.sel{background:#f0e2cc}
 .tag{font-size:9px;font-weight:700;padding:1px 5px;border-radius:8px;color:#fff}
 .emg{font-size:9px;color:#777;margin-left:5px}
 .clsbtn{font-size:10px;padding:3px 7px}
 #status{font-size:11px;color:#555;margin-top:7px;min-height:14px}
</style></head><body>
<div id="wrap">
 <div id="side">
  <h1>DevVIS Lamp5 · maturation × continuum</h1>
  <p class="sub" id="sub"></p>
  <div class="sec">Colour cells by</div>
  <select id="mode">
   <option value="gene">gene expression</option>
   <option value="nmf">NMF factor</option>
   <option value="age">age (P days)</option>
   <option value="cluster">cluster</option>
   <option value="maturity">maturity (x)</option>
   <option value="continuum">continuum (y)</option>
  </select>
  <div id="genebox" style="margin-top:7px"><input class="full" id="gin" list="gl" placeholder="type a gene (Enter)"><datalist id="gl"></datalist></div>
  <div id="nmfbox" style="margin-top:7px;display:none"></div>
  <div class="sec">Smoothly-graded genes along the continuum</div>
  <div class="row" id="clsfilter">
   <button class="clsbtn on" data-c="all">all</button>
   <button class="clsbtn" data-c="A-pole">A-pole</button>
   <button class="clsbtn" data-c="B-pole">B-pole</button>
   <button class="clsbtn" data-c="mid-peak">mid-peak</button>
   <button class="clsbtn" data-c="mid-dip">mid-dip</button>
  </div>
  <div class="muted" style="margin-top:5px">click a gene to colour · <b>early</b>/<b>late</b> = gradient present in immature / only mature cells</div>
  <div id="glist"></div>
  <div id="status"></div>
 </div>
 <div id="main"><div id="plot"></div><div id="emerge"></div></div>
</div>
<script>
const META=__META__;
function b64b(s){const bin=atob(s),n=bin.length,u=new Uint8Array(n);for(let i=0;i<n;i++)u[i]=bin.charCodeAt(i);return u;}
function f32(s){const b=b64b(s);return new Float32Array(b.buffer,b.byteOffset,b.byteLength/4);}
const N=META.n,G=META.g,K=META.k,ES=META.escale;
const EXPR=b64b("__EXPR__");                 // N*G uint8
const MAT=f32("__MAT__"),YC=f32("__YC__"),AGE=f32("__AGE__");
const CLUb=b64b("__CLU__"),CLU=new Int16Array(CLUb.buffer,CLUb.byteOffset,CLUb.byteLength/2);
const NMFW=f32("__NMFW__");                   // N*K
const PALETTE=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','#393b79','#637939'];
const CLSCOL={'A-pole':'#2166ac','B-pole':'#b2182b','mid-peak':'#7b3294','mid-dip':'#1b7837'};
const plotDiv=document.getElementById('plot'),emDiv=document.getElementById('emerge');
document.getElementById('sub').innerHTML=META.n.toLocaleString()+" Lamp5 interneurons · Gao 2025 · P3→P56 · full transcriptome ("+G+" genes embedded).<br><b>x</b> = transcriptomic maturity (Dcx/Sox11↓) · <b>y</b> = identity continuum (Cxcl14/Egln3 ↔ Nxph1/Npy), orthogonal to maturity.";
const gene=j=>{const o=new Float32Array(N);for(let i=0;i<N;i++)o[i]=EXPR[i*G+j]/ES;return o;};
const baseLayout={margin:{l:44,r:8,t:8,b:38},hovermode:'closest',
 xaxis:{title:'maturity  (immature → mature)',zeroline:false},
 yaxis:{title:'continuum  (A-pole Cxcl14 ↔ B-pole Nxph1)',zeroline:false},
 paper_bgcolor:'#fff',plot_bgcolor:'#fff',uirevision:'k'};
function setStatus(t){document.getElementById('status').innerHTML=t;}

function scatter(color,cbar,cscale,discrete){
 let tr;
 if(discrete){ // cluster: one trace per category for a legend
  const traces=META.clu_cats.map((nm,k)=>({type:'scattergl',mode:'markers',name:nm,
    x:[],y:[],marker:{size:4,color:PALETTE[k%PALETTE.length]},hoverinfo:'name'}));
  for(let i=0;i<N;i++){const t=traces[CLU[i]];t.x.push(MAT[i]);t.y.push(YC[i]);}
  Plotly.react(plotDiv,traces,Object.assign({},baseLayout,{showlegend:true,legend:{font:{size:9},itemsizing:'constant'}}),{responsive:true,displaylogo:false});
  return;
 }
 tr={type:'scattergl',mode:'markers',x:Array.from(MAT),y:Array.from(YC),
   marker:{size:4,color:Array.from(color),colorscale:cscale,colorbar:{title:cbar,thickness:12,len:.7},
   cmin:quant(color,0.02),cmax:quant(color,0.98)},hoverinfo:'skip'};
 Plotly.react(plotDiv,[tr],Object.assign({},baseLayout,{showlegend:false}),{responsive:true,displaylogo:false});
}
function quant(a,q){const s=Float32Array.from(a).sort();return s[Math.floor(q*(s.length-1))];}

// ---- emergence heatmap: maturity(x bins) × continuum(y bins) mean expression ----
function xbin(v){const e=META.xed;for(let b=0;b<META.nx;b++)if(v<=e[b+1]||b==META.nx-1)return b;return META.nx-1;}
function ybin(v){const e=META.yed;for(let b=0;b<META.ny;b++)if(v<=e[b+1]||b==META.ny-1)return b;return META.ny-1;}
function emergence(j,label){
 const sum=Array.from({length:META.ny},()=>new Float64Array(META.nx));
 const cnt=Array.from({length:META.ny},()=>new Float64Array(META.nx));
 for(let i=0;i<N;i++){const xb=xbin(MAT[i]),yb=ybin(YC[i]);sum[yb][xb]+=EXPR[i*G+j]/ES;cnt[yb][xb]++;}
 const z=sum.map((r,yb)=>r.map((s,xb)=>cnt[yb][xb]?s/cnt[yb][xb]:null));
 const xlab=[...Array(META.nx)].map((_,b)=>'M'+(b+1)), ylab=[...Array(META.ny)].map((_,b)=>b+1);
 Plotly.react(emDiv,[{type:'heatmap',z:z,x:xlab,y:ylab,colorscale:'Magma',
   colorbar:{title:'mean',thickness:10,len:.9},hoverongaps:false,
   hovertemplate:'maturity %{x} · continuum %{y}<br>'+label+' = %{z:.2f}<extra></extra>'}],
  {margin:{l:34,r:8,t:22,b:30},title:{text:label+' — emergence across maturity (→) × continuum (↑)',font:{size:11}},
   xaxis:{title:'maturity bin (immature→mature)',type:'category'},yaxis:{title:'continuum bin',type:'category'}},
  {responsive:true,displaylogo:false});
}

let curGene=-1;
function colorByGene(j){curGene=j;const g=META.genes[j];document.getElementById('gin').value=g;
 scatter(gene(j),g+' (log₂)','Magma',false);
 const cls=META.gcls[j],eta=META.geta[j];
 const cap=cls&&cls!='none'?(' · '+cls+' · gradedness η²='+eta.toFixed(2)+' · immature η²='+META.gelo[j].toFixed(2)+' / mature η²='+META.gehi[j].toFixed(2)):'';
 setStatus('<b>'+g+'</b>'+cap);
 emergence(j,g);
 [...document.querySelectorAll('.grow')].forEach(r=>r.classList.toggle('sel',+r.dataset.j===j));
}
function colorByNMF(k){scatter(NMFW.filter((_,i)=>i%K===k),'C'+(k+1)+' loading','Viridis',false);
 const g=META.nmf_top[k].split('/')[0]; if(META.genes.includes(g)){emergence(META.genes.indexOf(g),'C'+(k+1)+' top: '+g);}
 setStatus('NMF <b>C'+(k+1)+'</b>'+(k===META.mat_fac?' (maturation)':'')+' — top genes: '+META.nmf_top[k].replace(/\//g,', '));}

function applyMode(){const m=document.getElementById('mode').value;
 document.getElementById('genebox').style.display=m==='gene'?'block':'none';
 document.getElementById('nmfbox').style.display=m==='nmf'?'block':'none';
 if(m==='gene'){colorByGene(curGene>=0?curGene:META.genes.indexOf('Cxcl14'));}
 else if(m==='nmf'){colorByNMF(0);}
 else if(m==='age'){scatter(AGE,'age (P)','Portland',false);setStatus('coloured by age (postnatal day)');}
 else if(m==='cluster'){scatter(null,null,null,true);setStatus('coloured by cluster');}
 else if(m==='maturity'){scatter(MAT,'maturity','Viridis',false);setStatus('coloured by maturity (x-axis)');}
 else if(m==='continuum'){scatter(YC,'continuum','RdBu',false);setStatus('coloured by continuum (y-axis)');}
}
document.getElementById('mode').onchange=applyMode;

// gene search
const gl=document.getElementById('gl');META.genes.forEach(g=>{const o=document.createElement('option');o.value=g;gl.appendChild(o);});
document.getElementById('gin').addEventListener('keydown',e=>{if(e.key==='Enter'){const j=META.genes.indexOf(e.target.value.trim());if(j>=0){document.getElementById('mode').value='gene';applyMode();colorByGene(j);}else setStatus('gene not embedded: '+e.target.value);}});

// NMF buttons
const nb=document.getElementById('nmfbox');
for(let k=0;k<K;k++){const b=document.createElement('button');b.textContent='C'+(k+1)+(k===META.mat_fac?'*':'');b.title=META.nmf_top[k].replace(/\//g,', ');b.onclick=()=>{[...nb.children].forEach(x=>x.classList.remove('on'));b.classList.add('on');colorByNMF(k);};nb.appendChild(b);}

// graded-gene list
let clsFilter='all';
function buildList(){const el=document.getElementById('glist');el.innerHTML='';
 const rows=META.table.filter(r=>clsFilter==='all'||r.cls===clsFilter);
 rows.slice(0,400).forEach(r=>{const emg=r.elo>=0.6*r.ehi&&r.ehi>0.08?'early':(r.elo<0.3*r.ehi?'late':'gradual');
  const d=document.createElement('div');d.className='grow';d.dataset.j=r.i;
  d.innerHTML='<span><span class="tag" style="background:'+CLSCOL[r.cls]+'">'+r.cls+'</span> '+r.g+'</span>'+
   '<span class="emg">η²'+r.eta.toFixed(2)+' · '+emg+'</span>';
  d.onclick=()=>{document.getElementById('mode').value='gene';document.getElementById('genebox').style.display='block';document.getElementById('nmfbox').style.display='none';colorByGene(r.i);};
  el.appendChild(d);});
 setStatus(rows.length+' graded genes'+(clsFilter!=='all'?' · '+clsFilter:''));
}
document.querySelectorAll('.clsbtn').forEach(b=>b.onclick=()=>{document.querySelectorAll('.clsbtn').forEach(x=>x.classList.remove('on'));b.classList.add('on');clsFilter=b.dataset.c;buildList();});
buildList();
colorByGene(META.genes.indexOf('Cxcl14'));
</script></body></html>"""

html=(html.replace("__META__",json.dumps(META))
          .replace("__EXPR__",b64(expr)).replace("__MAT__",b64(mat)).replace("__YC__",b64(yc))
          .replace("__AGE__",b64(age)).replace("__CLU__",b64(clu)).replace("__NMFW__",b64(W)))
open(OUT,'w').write(html)
import os
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
