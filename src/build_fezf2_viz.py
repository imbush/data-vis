#!/usr/bin/env python
"""Self-contained Fezf2 interneuron visualizer (Wu/Fishell GSE272706, P20 cortex):
Control vs Fezf2-KO cortical interneurons.

  - Two side-by-side UMAPs (Control | Fezf2-KO) on the joint embedding, so the
    genotype-driven density/proportion shift is visible directly.
  - A proportions bar (Control vs Fezf2-KO per subclass) — the paper's headline
    (PV down, SST up in the mutant).
  - Colour by subclass / genotype / any gene (search); subclass filter.

Subclass labels are transferred from the Tasic adult V1 reference (ingest), not the
paper's own taxonomy (their annotated object is sign-in-walled on the Single Cell
Portal). At P20 there is one sample per genotype, so genotype is confounded with
batch — the shifts are descriptive (consistent with the paper's full design).
"""
import os, json, base64, warnings, numpy as np, scanpy as sc
warnings.filterwarnings('ignore'); sc.settings.verbosity = 0
from bokeh.palettes import Magma256
ROOT = '/Users/inlebush/cs/lab/green/sequencing/tasic2018_v1_merfish'
D = os.path.join(ROOT, 'data', 'fezf2_fishell')
OUT = '/Users/inlebush/cs/lab/green/data-vis/fezf2/fezf2_interneuron_explorer.html'

MARKERS = ['Fezf2', 'Gad1', 'Gad2', 'Slc32a1', 'Pvalb', 'Tac1', 'Cox6a2', 'Sst',
           'Chodl', 'Nos1', 'Vip', 'Calb2', 'Crh', 'Lamp5', 'Ndnf', 'Reln', 'Npy',
           'Sncg', 'Cck', 'Adarb2', 'Lhx6', 'Nkx2-1', 'Erbb4', 'Snap25', 'Bax']
SUB_ORDER = ['Pvalb', 'Sst', 'Vip', 'Lamp5', 'Sncg']
SUB_COLOR = {'Pvalb': '#d62728', 'Sst': '#1f77b4', 'Vip': '#2ca02c',
             'Lamp5': '#9467bd', 'Sncg': '#17becf'}
GENO_COLOR = {'Control': '#5b6770', 'Fezf2-KO': '#d62728'}


def main():
    a = sc.read_h5ad(os.path.join(D, 'p20_cIN_labeled.h5ad'))
    a = a[a.obs['subclass'].isin(SUB_ORDER)].copy()
    n = a.n_obs
    print(f'{n} cells; genotypes {dict(a.obs.genotype.value_counts())}')

    # gene panel: top HVG (already flagged) + curated markers, capped for size
    hvg = a.var_names[a.var['highly_variable']].tolist() if 'highly_variable' in a.var else []
    panel = list(dict.fromkeys([g for g in MARKERS if g in a.var_names] + hvg))[:600]
    Xp = a[:, panel].X
    Xp = Xp.toarray() if hasattr(Xp, 'toarray') else np.asarray(Xp)
    EXPR_SCALE = 16
    q = np.clip(np.round(Xp * EXPR_SCALE), 0, 255).astype(np.uint8)   # cells x panel
    expr_b64 = base64.b64encode(np.ascontiguousarray(q).tobytes()).decode()

    umap = a.obsm['X_umap']
    # normalize umap to a tidy range
    umap = (umap - umap.mean(0))
    geno = a.obs['genotype'].astype(str).values
    is_ko = (geno == 'Fezf2-KO').astype(np.uint8)
    sub = a.obs['subclass'].astype(str).values
    sub_idx = np.array([SUB_ORDER.index(s) for s in sub], dtype=np.uint8)

    # proportions per genotype
    import pandas as pd
    prop = (pd.crosstab(a.obs['subclass'], a.obs['genotype'], normalize='columns') * 100)
    prop = prop.reindex(SUB_ORDER)
    ctrl_prop = [round(float(prop.loc[s, 'Control']), 1) for s in SUB_ORDER]
    ko_prop = [round(float(prop.loc[s, 'Fezf2-KO']), 1) for s in SUB_ORDER]
    nctrl = int((geno == 'Control').sum()); nko = int((geno == 'Fezf2-KO').sum())

    js = (
        f"const N={n}, NG={len(panel)}, EXPR_SCALE={EXPR_SCALE};\n"
        f"const expr_b64={json.dumps(expr_b64)};\n"
        f"const gene_name={json.dumps(panel)};\n"
        f"const ux={json.dumps([round(float(v),3) for v in umap[:,0]])};\n"
        f"const uy={json.dumps([round(float(v),3) for v in umap[:,1]])};\n"
        f"const is_ko={json.dumps(is_ko.tolist())};\n"
        f"const sub_idx={json.dumps(sub_idx.tolist())};\n"
        f"const SUB_ORDER={json.dumps(SUB_ORDER)};\n"
        f"const SUB_COLOR={json.dumps([SUB_COLOR[s] for s in SUB_ORDER])};\n"
        f"const GENO_COLOR={json.dumps(GENO_COLOR)};\n"
        f"const ctrl_prop={json.dumps(ctrl_prop)};\n"
        f"const ko_prop={json.dumps(ko_prop)};\n"
        f"const NCTRL={nctrl}, NKO={nko};\n"
        f"const magma={json.dumps(list(Magma256))};\n"
    )
    datalist = '<datalist id="genes">' + ''.join(f'<option value="{g}">' for g in panel) + '</datalist>'

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Fezf2 interneuron explorer — Wu/Fishell</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 html,body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;color:#1a1a1a;background:#f7f5f1;}}
 .wrap{{max-width:1180px;margin:0 auto;padding:16px 20px 60px;}}
 h1{{font-size:24px;margin:0 0 2px;}} .tag{{color:#666;font-size:13px;}}
 .hint{{font-size:15px;margin:10px 0;color:#222;}} .hint b{{color:#1f77b4;}}
 details summary{{cursor:pointer;color:#666;font-size:12px;}}
 .ctrl{{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:10px 0;font-size:13px;}}
 .ctrl .label{{font-weight:600;color:#555;}}
 button{{font-size:12px;padding:4px 9px;border:1px solid #bbb;background:#fafafa;border-radius:5px;cursor:pointer;}}
 button.active{{background:#1f77b4;color:#fff;border-color:#1f77b4;}}
 input#gene{{width:130px;font-size:12px;padding:3px 6px;}}
 .chip{{display:inline-flex;align-items:center;gap:3px;padding:2px 6px;border-radius:10px;border:1px solid #ccc;cursor:pointer;user-select:none;}}
 .chip.off{{opacity:.35;}}
 .row{{display:flex;gap:12px;flex-wrap:wrap;}}
 .umap{{flex:1 1 380px;min-width:320px;}} .umap-title{{text-align:center;font-weight:600;font-size:13px;margin-bottom:2px;}}
 #bar{{height:300px;}} #status{{font-size:12px;color:#555;min-height:16px;}}
 .legend{{font-size:12px;color:#444;margin-top:6px;}}
</style></head><body><div class="wrap">
<h1>Fezf2 mutant cortical interneurons</h1>
<div class="tag">Wu et&nbsp;al. (Fishell lab), Nature 2026 · GSE272706 · P20 cortex snRNA-seq · {n:,} interneurons</div>
<div class="hint"><b>Left = Control, right = Fezf2-KO</b> on the same UMAP. The bar shows how
each interneuron subclass's share changes in the mutant — <b>PV down, SST up</b>.</div>
<details><summary>About / caveats</summary><div style="font-size:12px;color:#555;line-height:1.5;padding:6px 0;max-width:820px;">
Fezf2-KO loses L5b pyramidal neurons and expands L6 IT; the paper reports PV interneurons switch
subtype identity and SST adjust programmed cell death. Subclass labels here are transferred from the
Tasic adult V1 reference via scanpy ingest (the paper's own annotations are sign-in-walled on the
Broad Single Cell Portal SCP2716). At P20 there is one sample per genotype, so genotype is confounded
with batch — treat the shifts as descriptive. No cross-genotype batch correction was applied (it would
erase the effect). Colour by subclass / genotype / any gene; filter subclasses; hover for counts.
</div></details>
<div class="ctrl">
 <span class="label">Colour by:</span>
 <button id="cby-sub" class="active">Subclass</button>
 <button id="cby-geno">Genotype</button>
 <span class="label" style="margin-left:10px;">Find gene:</span>
 <input id="gene" list="genes" placeholder="e.g. Pvalb" autocomplete="off"><button id="gene-clear">clear</button>
 {datalist}
 <span class="label" style="margin-left:10px;">Subclasses:</span><span id="chips"></span>
</div>
<div class="row">
 <div class="umap"><div class="umap-title" id="t-ctrl">Control (n={nctrl:,})</div><div id="umap-ctrl" style="height:420px;"></div></div>
 <div class="umap"><div class="umap-title" id="t-ko">Fezf2-KO (n={nko:,})</div><div id="umap-ko" style="height:420px;"></div></div>
</div>
<div id="status"></div>
<div id="bar"></div>
<div class="legend" id="legend"></div>
<script>
{js}
const expr = new Uint8Array(Uint8Array.from(atob(expr_b64), c=>c.charCodeAt(0)).buffer);
const ctrlIdx=[], koIdx=[];
for(let i=0;i<N;i++){{ (is_ko[i]?koIdx:ctrlIdx).push(i); }}
let colorMode='sub';            // 'sub' | 'geno' | 'gene'
let geneCol=-1;
const subOn = new Set(SUB_ORDER);

function magmaArr(vals){{ let lo=Infinity,hi=-Infinity; for(const v of vals){{if(v<lo)lo=v;if(v>hi)hi=v;}}
  const r=(hi>lo)?(hi-lo):1; return vals.map(v=>magma[Math.max(0,Math.min(255,Math.round(255*(v-lo)/r)))]); }}

function colorsFor(idx){{
  if(colorMode==='gene' && geneCol>=0){{
    const vals=idx.map(i=>expr[i*NG+geneCol]/EXPR_SCALE); return magmaArr(vals);
  }}
  if(colorMode==='geno'){{ return idx.map(i=>is_ko[i]?GENO_COLOR['Fezf2-KO']:GENO_COLOR['Control']); }}
  return idx.map(i=>SUB_COLOR[sub_idx[i]]);
}}
function visIdx(side){{ const base=side==='ctrl'?ctrlIdx:koIdx;
  return base.filter(i=>subOn.has(SUB_ORDER[sub_idx[i]])); }}

function trace(idx){{ return {{x:idx.map(i=>ux[i]), y:idx.map(i=>uy[i]), mode:'markers', type:'scattergl',
  marker:{{size:2.5, color:colorsFor(idx), opacity:0.7}},
  text:idx.map(i=>SUB_ORDER[sub_idx[i]]), hoverinfo:'text', showlegend:false}}; }}
const LAYOUT={{margin:{{l:6,r:6,t:6,b:6}}, xaxis:{{visible:false}}, yaxis:{{visible:false,scaleanchor:'x'}},
  paper_bgcolor:'white', plot_bgcolor:'white', hovermode:'closest'}};
function draw(){{
  Plotly.react('umap-ctrl',[trace(visIdx('ctrl'))],LAYOUT,{{displayModeBar:false,responsive:true}});
  Plotly.react('umap-ko',[trace(visIdx('ko'))],LAYOUT,{{displayModeBar:false,responsive:true}});
}}

function drawBar(){{
  // proportions among CURRENTLY shown subclasses (renormalized), Control vs KO
  const shown=SUB_ORDER.filter(s=>subOn.has(s));
  const ci=shown.map(s=>ctrl_prop[SUB_ORDER.indexOf(s)]), ki=shown.map(s=>ko_prop[SUB_ORDER.indexOf(s)]);
  const cs=ci.reduce((a,b)=>a+b,0)||1, ks=ki.reduce((a,b)=>a+b,0)||1;
  const cN=ci.map(v=>+(100*v/cs).toFixed(1)), kN=ki.map(v=>+(100*v/ks).toFixed(1));
  Plotly.react('bar',[
    {{x:shown,y:cN,type:'bar',name:'Control',marker:{{color:GENO_COLOR['Control']}},
      text:cN.map(v=>v+'%'),textposition:'outside'}},
    {{x:shown,y:kN,type:'bar',name:'Fezf2-KO',marker:{{color:GENO_COLOR['Fezf2-KO']}},
      text:kN.map(v=>v+'%'),textposition:'outside'}}
  ],{{barmode:'group',margin:{{l:40,r:10,t:24,b:30}},height:300,
      title:{{text:'Interneuron subclass composition (% of shown INs)',x:0.5,font:{{size:13}}}},
      yaxis:{{title:'% of interneurons'}},legend:{{orientation:'h',y:1.12,x:0.5,xanchor:'center'}},
      paper_bgcolor:'white',plot_bgcolor:'white'}},{{displayModeBar:false,responsive:true}});
  // delta line
  const deltas=SUB_ORDER.map((s,k)=>{{const d=ko_prop[k]-ctrl_prop[k];
    return `<b style="color:${{SUB_COLOR[s]}}">${{s}}</b> ${{ctrl_prop[k]}}%→${{ko_prop[k]}}% (${{d>=0?'+':''}}${{d.toFixed(1)}})`;}});
  document.getElementById('legend').innerHTML='Δ in Fezf2-KO: '+deltas.join(' &nbsp; ');
}}

// chips
const chipBox=document.getElementById('chips');
SUB_ORDER.forEach(s=>{{ const el=document.createElement('span'); el.className='chip'; el.dataset.s=s;
  el.innerHTML=`<span style="width:9px;height:9px;border-radius:50%;background:${{SUB_COLOR[s]}};display:inline-block"></span>${{s}}`;
  el.onclick=()=>{{ if(subOn.has(s))subOn.delete(s); else subOn.add(s); el.classList.toggle('off'); draw(); drawBar(); }};
  chipBox.appendChild(el); }});

document.getElementById('cby-sub').onclick=()=>{{colorMode='sub';geneCol=-1;setActive('cby-sub');draw();status('coloured by subclass');}};
document.getElementById('cby-geno').onclick=()=>{{colorMode='geno';geneCol=-1;setActive('cby-geno');draw();status('coloured by genotype');}};
function setActive(id){{['cby-sub','cby-geno'].forEach(b=>document.getElementById(b).classList.toggle('active',b===id));}}
function status(m){{document.getElementById('status').innerHTML=m;}}
const gi=document.getElementById('gene');
function doGene(){{ const q=gi.value.trim().toLowerCase(); const j=gene_name.findIndex(g=>g.toLowerCase()===q);
  if(j<0){{status('gene <b>'+gi.value+'</b> not in panel');return;}} geneCol=j;colorMode='gene';setActive(null);draw();
  status('coloured by <b>'+gene_name[j]+'</b> expression (magma, per-panel range)'); }}
gi.addEventListener('change',doGene); gi.addEventListener('keydown',e=>{{if(e.key==='Enter')doGene();}});
document.getElementById('gene-clear').onclick=()=>{{gi.value='';colorMode='sub';geneCol=-1;setActive('cby-sub');draw();status('');}};

draw(); drawBar(); status('');
window.addEventListener('resize',()=>{{Plotly.Plots.resize('umap-ctrl');Plotly.Plots.resize('umap-ko');Plotly.Plots.resize('bar');}});
</script></div></body></html>"""

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f: f.write(html)
    print(f'wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
