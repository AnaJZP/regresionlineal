/* ── datos de riesgo crediticio ── */
const Y=[2.5,4.8,1.8,6.2,3.5];
const X1=[1.2,2.5,0.8,3.1,1.8];
const X2=[15,22,12,28,18];
const n=5, k=2;
const f=(v,d=4)=>typeof v==='number'?v.toFixed(d):v;

/* ── Matrices ── */
const X=Y.map((_,i)=>[1,X1[i],X2[i]]);

/* X'X (3x3) */
function matMul(A,B){
  const m=A.length,p=B[0].length,n2=B.length;
  const C=Array.from({length:m},()=>Array(p).fill(0));
  for(let i=0;i<m;i++)for(let j=0;j<p;j++)for(let l=0;l<n2;l++)C[i][j]+=A[i][l]*B[l][j];
  return C;
}
function transpose(A){return A[0].map((_,j)=>A.map(r=>r[j]));}

const Xt=transpose(X);
const XtX=matMul(Xt,X);
const Ymat=Y.map(v=>[v]);
const XtY=matMul(Xt,Ymat).map(r=>r[0]);

/* Inverse 3x3 */
function inv3(M){
  const [[a,b,c],[d,e,f2],[g,h,ii]]=M;
  const det=a*(e*ii-f2*h)-b*(d*ii-f2*g)+c*(d*h-e*g);
  const adj=[
    [e*ii-f2*h,c*h-b*ii,b*f2-c*e],
    [f2*g-d*ii,a*ii-c*g,c*d-a*f2],
    [d*h-e*g,b*g-a*h,a*e-b*d]
  ];
  return adj.map(r=>r.map(v=>v/det));
}

const XtXinv=inv3(XtX);
const beta=XtXinv.map((r,i)=>r.reduce((s,v,j)=>s+v*XtY[j],0));
const yhat=X.map(r=>r.reduce((s,v,j)=>s+v*beta[j],0));
const resid=Y.map((y,i)=>y-yhat[i]);
const SSE=resid.reduce((s,e)=>s+e*e,0);
const ybar=Y.reduce((a,b)=>a+b)/n;
const SST=Y.reduce((s,y)=>s+(y-ybar)**2,0);
const R2=1-SSE/SST;
const s2_mco=SSE/(n-k-1);
const s2_mv=SSE/n;

/* ── Tab logic ── */
function showTab(id,btn){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  if(btn)btn.classList.add('active');
  if(id==='resumen')buildResumen();
}

/* ── Panel 1: matrices ── */
(function(){
  const matY=Y.map(v=>`<tr><td>${v}</td></tr>`).join('');
  const matX=X.map(r=>`<tr>${r.map(v=>`<td>${v}</td>`).join('')}</tr>`).join('');
  document.getElementById('matrix-display').innerHTML=
    `<div class="mat-block"><span class="mat-lbl">y =</span><span class="mat-br">[</span><table class="mat">${matY}</table><span class="mat-br">]</span></div>`+
    `<div class="mat-block"><span class="mat-lbl">X =</span><span class="mat-br">[</span><table class="mat">${matX}</table><span class="mat-br">]</span></div>`+
    `<div class="mat-block"><span class="mat-lbl">β =</span><span class="mat-br">[</span><table class="mat">${['β₀','β₁','β₂'].map(v=>`<tr><td>${v}</td></tr>`).join('')}</table><span class="mat-br">]</span></div>`;
})();

/* ── Panel 2: MCO numérico ── */
(function(){
  document.getElementById('xtx-show').innerHTML=
    `<span class="h">X'X</span> (3×3):\n`+
    XtX.map(r=>'  ['+r.map(v=>f(v,2)).join(', ')+']').join('\n');

  document.getElementById('xty-show').innerHTML=
    `<span class="h">X'y</span> (3×1):\n`+
    '  ['+XtY.map(v=>f(v,4)).join(', ')+']';

  document.getElementById('beta-show').innerHTML=
    `<span class="h">β̂ = (X'X)⁻¹X'y</span>\n\n`+
    `  β̂₀ = <span class="h">${f(beta[0],6)}</span>  (intercepto)\n`+
    `  β̂₁ = <span class="h">${f(beta[1],6)}</span>  (efecto del apalancamiento)\n`+
    `  β̂₂ = <span class="h">${f(beta[2],6)}</span>  (efecto de la volatilidad)\n\n`+
    `Recta: Pérdida = ${f(beta[0],4)} + ${f(beta[1],4)}·Apal + ${f(beta[2],4)}·Vol`;

  let rows='',sseC=0;
  Y.forEach((y,i)=>{
    const e=resid[i];sseC+=e*e;
    rows+=`<tr><td>${i+1}</td><td>${f(y,2)}</td><td>${f(yhat[i],4)}</td>`+
      `<td style="color:${e>=0?'var(--blue)':'var(--red)'}">${f(e,4)}</td><td>${f(e*e,6)}</td></tr>`;
  });
  rows+=`<tr style="font-weight:600;background:var(--gl)"><td>Σ</td><td></td><td></td><td style="font-size:.8rem;color:var(--faint)">≈ 0</td><td>${f(sseC,6)}</td></tr>`;
  document.getElementById('resid-body').innerHTML=rows;

  document.getElementById('sse-show').innerHTML=
    `SSE = Σêᵢ² = <span class="h">${f(SSE,6)}</span>\n`+
    `SST = ${f(SST,6)}    R² = 1 − SSE/SST = <span class="h">${f(R2,4)}</span>`;
})();

/* ── Panel 3: MV numérico ── */
(function(){
  const ll_max=-(n/2)*Math.log(2*Math.PI)-(n/2)*Math.log(s2_mv)-(SSE/(2*s2_mv));
  document.getElementById('mv-numeric').textContent=
    `β̂₀_MV = ${f(beta[0],6)}  β̂₁_MV = ${f(beta[1],6)}  β̂₂_MV = ${f(beta[2],6)}\n`+
    `σ̂²_MV = SSE/n = ${f(SSE,6)}/${n} = ${f(s2_mv,6)}\n`+
    `ℓ_max = ${f(ll_max,4)}`;
})();

/* ── Panel 5: sigma ── */
(function(){
  document.getElementById('s2-mco-detail').innerHTML=
    `<div class="kv"><span class="kv-k">SSE</span><span class="kv-v" style="color:var(--blue)">${f(SSE,6)}</span></div>`+
    `<div class="kv"><span class="kv-k">n−k−1</span><span class="kv-v" style="color:var(--blue)">${n-k-1}</span></div>`+
    `<div class="kv"><span class="kv-k">s²</span><span class="kv-v" style="color:var(--blue)">${f(s2_mco,6)}</span></div>`;
  document.getElementById('s2-mv-detail').innerHTML=
    `<div class="kv"><span class="kv-k">SSE</span><span class="kv-v" style="color:var(--green)">${f(SSE,6)}</span></div>`+
    `<div class="kv"><span class="kv-k">n</span><span class="kv-v" style="color:var(--green)">${n}</span></div>`+
    `<div class="kv"><span class="kv-k">σ̂²</span><span class="kv-v" style="color:var(--green)">${f(s2_mv,6)}</span></div>`;
})();

/* ── Panel 6: resumen ── */
function buildResumen(){
  const rows=[
    ['Criterio','min SSE = Σêᵢ²','max ℓ(β,σ²|y)'],
    ['Supuesto distribucional','Gauss-Markov (1–5)','Normalidad'],
    ['β̂',`(X'X)⁻¹X'y`,`Idéntico`],
    ['β̂₀',f(beta[0],6),f(beta[0],6)],
    ['β̂₁',f(beta[1],6),f(beta[1],6)],
    ['β̂₂',f(beta[2],6),f(beta[2],6)],
    ['σ²',`s²=SSE/(n−k−1)=${f(s2_mco,6)}`,`σ̂²=SSE/n=${f(s2_mv,6)}`],
    ['Sesgo en σ²','0 (insesgado)',`−${f((k+1)/n*100,1)}%`],
    ['Inferencia','t y F exactos','Razón de verosimilitud'],
    ['Extensiones','WLS, GLS, robustos','Logit, Probit, Poisson'],
  ];
  document.getElementById('cmp-body').innerHTML=rows.map(([a,m,v])=>
    `<tr><td>${a}</td><td class="blue">${m}</td><td class="green">${v}</td></tr>`).join('');
  document.getElementById('final-mco').innerHTML=
    `<div class="kv"><span class="kv-k">β̂₀</span><span class="kv-v" style="color:var(--blue)">${f(beta[0],6)}</span></div>`+
    `<div class="kv"><span class="kv-k">β̂₁</span><span class="kv-v" style="color:var(--blue)">${f(beta[1],6)}</span></div>`+
    `<div class="kv"><span class="kv-k">β̂₂</span><span class="kv-v" style="color:var(--blue)">${f(beta[2],6)}</span></div>`+
    `<div class="kv"><span class="kv-k">s²</span><span class="kv-v" style="color:var(--blue)">${f(s2_mco,6)}</span></div>`+
    `<div class="kv"><span class="kv-k">R²</span><span class="kv-v" style="color:var(--blue)">${f(R2,4)}</span></div>`;
  document.getElementById('final-mv').innerHTML=
    `<div class="kv"><span class="kv-k">β̂₀</span><span class="kv-v" style="color:var(--green)">${f(beta[0],6)}</span></div>`+
    `<div class="kv"><span class="kv-k">β̂₁</span><span class="kv-v" style="color:var(--green)">${f(beta[1],6)}</span></div>`+
    `<div class="kv"><span class="kv-k">β̂₂</span><span class="kv-v" style="color:var(--green)">${f(beta[2],6)}</span></div>`+
    `<div class="kv"><span class="kv-k">σ̂²</span><span class="kv-v" style="color:var(--green)">${f(s2_mv,6)}</span></div>`+
    `<div class="kv"><span class="kv-k">ℓ_max</span><span class="kv-v" style="color:var(--green)">${f(-(n/2)*Math.log(2*Math.PI)-(n/2)*Math.log(s2_mv)-n/2,4)}</span></div>`;
}
