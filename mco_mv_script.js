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

/* ── Panel 2: Var/Cov ── */
(function(){
  const x1bar = X1.reduce((a,b)=>a+b)/n;
  const ybar2 = Y.reduce((a,b)=>a+b)/n;
  const x2bar = X2.reduce((a,b)=>a+b)/n;

  function buildVarCovTable(Xvar, xbar, varLabel, varSymbol, containerId, resultId){
    let rows = '';
    let sumCov = 0, sumVar = 0;
    Xvar.forEach((xi, i) => {
      const dx = xi - xbar;
      const dy = Y[i] - ybar2;
      const cov_i = dx * dy;
      const var_i = dx * dx;
      sumCov += cov_i;
      sumVar += var_i;
      rows += `<tr><td>${i+1}</td><td>${f(xi,1)}</td><td>${f(Y[i],1)}</td>` +
        `<td>${f(dx,4)}</td><td>${f(dy,4)}</td>` +
        `<td>${f(cov_i,4)}</td><td>${f(var_i,4)}</td></tr>`;
    });
    rows += `<tr style="font-weight:600;background:var(--gl)"><td>Σ</td><td></td><td></td>` +
      `<td></td><td></td><td>${f(sumCov,4)}</td><td>${f(sumVar,4)}</td></tr>`;

    document.getElementById(containerId).innerHTML = 
      `<div class="tbl-wrap" style="margin:1rem 0"><table class="data-table">` +
      `<thead><tr>` +
      `<th style="text-align:center">i</th>` +
      `<th>${varSymbol}</th>` +
      `<th>yᵢ</th>` +
      `<th>${varSymbol}−${varLabel.includes('1')?'x̄₁':'x̄₂'}</th>` +
      `<th>yᵢ−ȳ</th>` +
      `<th>(${varSymbol}−${varLabel.includes('1')?'x̄₁':'x̄₂'})(yᵢ−ȳ)</th>` +
      `<th>(${varSymbol}−${varLabel.includes('1')?'x̄₁':'x̄₂'})²</th>` +
      `</tr></thead><tbody>${rows}</tbody></table></div>`;

    const b1_vc = sumCov / sumVar;
    const b0_vc = ybar2 - b1_vc * xbar;

    document.getElementById(resultId).innerHTML =
      `<span class="h">Paso 1 — Medias:</span>\n` +
      `  ${varLabel.includes('1')?'x̄₁':'x̄₂'} = ${f(xbar,2)}    ȳ = ${f(ybar2,2)}\n\n` +
      `<span class="h">Paso 2 — Covarianza y Varianza:</span>\n` +
      `  Cov(${varLabel}, y) = Σ(${varSymbol}−${varLabel.includes('1')?'x̄₁':'x̄₂'})(yᵢ−ȳ) = ${f(sumCov,4)}\n` +
      `  Var(${varLabel})     = Σ(${varSymbol}−${varLabel.includes('1')?'x̄₁':'x̄₂'})²     = ${f(sumVar,4)}\n\n` +
      `<span class="h">Paso 3 — Pendiente:</span>\n` +
      `  β̂₁ = Cov/Var = ${f(sumCov,4)} / ${f(sumVar,4)} = <span class="h">${f(b1_vc,6)}</span>\n\n` +
      `<span class="h">Paso 4 — Intercepto:</span>\n` +
      `  β̂₀ = ȳ − β̂₁·${varLabel.includes('1')?'x̄₁':'x̄₂'} = ${f(ybar2,2)} − ${f(b1_vc,6)}×${f(xbar,2)} = <span class="h">${f(b0_vc,6)}</span>\n\n` +
      `<span class="h2">Modelo simple:</span>  Pérdida = ${f(b0_vc,4)} + ${f(b1_vc,4)}·${varLabel}`;
  }

  buildVarCovTable(X1, x1bar, 'x₁', 'x₁ᵢ', 'varcov-table-x1', 'varcov-result-x1');
  buildVarCovTable(X2, x2bar, 'x₂', 'x₂ᵢ', 'varcov-table-x2', 'varcov-result-x2');
})();

/* ── Panel 3: MCO numérico ── */
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

/* ── Panel 3: MCO sustituciones numéricas e interpretación ── */
(function(){
  // Paso 2: show X'y and X'X·β
  document.getElementById('paso2-numeric').innerHTML =
    `<span class="h">−2X'y + 2X'Xβ = 0</span>\n\n` +
    `X'y = [${XtY.map(v=>f(v,2)).join(', ')}]\n` +
    `X'X = \n` + XtX.map(r=>'  ['+r.map(v=>f(v,2)).join(', ')+']').join('\n') +
    `\n\nIgualando la derivada a cero, obtenemos las ecuaciones normales →`;

  // Paso 3: show normal equations
  document.getElementById('paso3-numeric').innerHTML =
    `<span class="h">X'X · β̂ = X'y</span>\n\n` +
    `  ${f(XtX[0][0],1)}·β̂₀ + ${f(XtX[0][1],1)}·β̂₁ + ${f(XtX[0][2],1)}·β̂₂ = ${f(XtY[0],2)}\n` +
    `  ${f(XtX[1][0],1)}·β̂₀ + ${f(XtX[1][1],1)}·β̂₁ + ${f(XtX[1][2],1)}·β̂₂ = ${f(XtY[1],2)}\n` +
    `  ${f(XtX[2][0],1)}·β̂₀ + ${f(XtX[2][1],1)}·β̂₁ + ${f(XtX[2][2],1)}·β̂₂ = ${f(XtY[2],2)}\n\n` +
    `Son 3 ecuaciones con 3 incógnitas → sistema determinado.`;

  // Paso 4: show inverse application
  document.getElementById('paso4-numeric').innerHTML =
    `<span class="h">β̂ = (X'X)⁻¹ · X'y</span>\n\n` +
    `(X'X)⁻¹ =\n` + XtXinv.map(r=>'  ['+r.map(v=>f(v,6)).join(', ')+']').join('\n') +
    `\n\n<span class="h">Resultado:</span>\n` +
    `  β̂₀ = <span class="h">${f(beta[0],6)}</span>  (intercepto)\n` +
    `  β̂₁ = <span class="h">${f(beta[1],6)}</span>  (por unidad de apalancamiento)\n` +
    `  β̂₂ = <span class="h">${f(beta[2],6)}</span>  (por punto de volatilidad)`;

  // Interpretation
  const sseRatio = SSE/SST;
  document.getElementById('mco-interpretation').innerHTML =
    `<div class="grid3" style="margin:.75rem 0">` +
    `<div class="stat-card"><div class="stat-val" style="font-size:1.3rem">${f(SST,4)}</div><div class="stat-label">SST</div></div>` +
    `<div class="stat-card"><div class="stat-val" style="font-size:1.3rem;color:var(--red)">${f(SSE,6)}</div><div class="stat-label">SSE</div></div>` +
    `<div class="stat-card"><div class="stat-val" style="font-size:1.3rem;color:var(--blue)">${f(R2,4)}</div><div class="stat-label">R²</div></div>` +
    `</div>` +
    `<div class="callout green">` +
    `<div class="callout-title">Interpretación en riesgo crediticio</div>` +
    `<p style="margin:0;font-size:.92rem"><strong>SST = ${f(SST,4)}</strong> — La variabilidad total de las pérdidas respecto a su media (ȳ = ${f(ybar,2)}%). Esto es todo el "riesgo" que queremos explicar.</p>` +
    `<p style="margin:.4rem 0 0;font-size:.92rem"><strong>SSE = ${f(SSE,6)}</strong> — Los errores al cuadrado que MCO minimizó. Este es el riesgo residual: lo que el modelo <em>no</em> puede predecir con apalancamiento y volatilidad.</p>` +
    `<p style="margin:.4rem 0 0;font-size:.92rem"><strong>SSE/SST = ${f(sseRatio,6)}</strong> — Solo el <strong>${f(sseRatio*100,4)}%</strong> de la variabilidad queda sin explicar.</p>` +
    `<p style="margin:.4rem 0 0;font-size:.92rem"><strong>R² = 1 − SSE/SST = ${f(R2,4)}</strong> — El modelo explica el <strong>${f(R2*100,2)}%</strong> de la variabilidad en las pérdidas. El apalancamiento y la volatilidad son predictores excelentes del riesgo crediticio en esta muestra.</p>` +
    `</div>`;
})();

/* ── Panel 4: MV evaluación numérica expandida ── */
(function(){
  const ll_max=-(n/2)*Math.log(2*Math.PI)-(n/2)*Math.log(s2_mv)-(SSE/(2*s2_mv));
  
  // Beta interpretation
  document.getElementById('mv-numeric-beta').innerHTML =
    `<div class="grid3" style="margin:.5rem 0">` +
    `<div class="stat-card"><div class="stat-val" style="font-size:1.1rem">${f(beta[0],4)}</div><div class="stat-label">β̂₀ (intercepto)</div></div>` +
    `<div class="stat-card"><div class="stat-val" style="font-size:1.1rem">${f(beta[1],4)}</div><div class="stat-label">β̂₁ (apalancamiento)</div></div>` +
    `<div class="stat-card"><div class="stat-val" style="font-size:1.1rem">${f(beta[2],4)}</div><div class="stat-label">β̂₂ (volatilidad)</div></div>` +
    `</div>`;

  // Sigma calculation step by step
  document.getElementById('mv-numeric-sigma').innerHTML =
    `<span class="h">Estimación de σ² — sustituyendo valores:</span>\n\n` +
    `  SSE = ${f(SSE,6)}\n` +
    `  n   = ${n}\n\n` +
    `  σ̂²_MV = SSE/n = ${f(SSE,6)} / ${n} = <span class="h">${f(s2_mv,6)}</span>\n` +
    `  σ̂_MV  = √(${f(s2_mv,6)}) = <span class="h">${f(Math.sqrt(s2_mv),6)}</span>  (desviación estándar del error)`;

  // Log-likelihood step by step  
  document.getElementById('mv-numeric-loglik').innerHTML =
    `<span class="h">Log-verosimilitud máxima:</span>\n\n` +
    `  ℓ = −(n/2)·ln(2π) − (n/2)·ln(σ̂²) − SSE/(2σ̂²)\n` +
    `  ℓ = −(${n}/2)·ln(2π) − (${n}/2)·ln(${f(s2_mv,6)}) − ${f(SSE,6)}/(2·${f(s2_mv,6)})\n` +
    `  ℓ = ${f(-(n/2)*Math.log(2*Math.PI),4)} − ${f((n/2)*Math.log(s2_mv),4)} − ${f(SSE/(2*s2_mv),4)}\n` +
    `  ℓ_max = <span class="h">${f(ll_max,4)}</span>`;

  // Interpretation
  document.getElementById('mv-interpretation').innerHTML =
    `<div class="callout green" style="margin-top:1rem">` +
    `<div class="callout-title">Interpretación de los resultados MV</div>` +
    `<p style="margin:0;font-size:.92rem"><strong>β̂₀ = ${f(beta[0],4)}</strong> — Pérdida base estimada cuando apalancamiento = 0 y volatilidad = 0%. El valor negativo es una extrapolación fuera del rango de datos (ninguna empresa tiene apalancamiento cero).</p>` +
    `<p style="margin:.4rem 0 0;font-size:.92rem"><strong>β̂₁ = ${f(beta[1],4)}</strong> — Por cada unidad adicional de apalancamiento, la pérdida aumenta en ${f(beta[1],2)} puntos porcentuales, <em>controlando</em> por volatilidad.</p>` +
    `<p style="margin:.4rem 0 0;font-size:.92rem"><strong>β̂₂ = ${f(beta[2],4)}</strong> — Por cada punto porcentual adicional de volatilidad, la pérdida aumenta en ${f(beta[2],4)} puntos porcentuales, <em>controlando</em> por apalancamiento.</p>` +
    `<p style="margin:.4rem 0 0;font-size:.92rem"><strong>σ̂²_MV = ${f(s2_mv,6)}</strong> — La varianza estimada del error es extremadamente pequeña, indicando que el modelo captura casi toda la variabilidad.</p>` +
    `<p style="margin:.4rem 0 0;font-size:.92rem"><strong>ℓ_max = ${f(ll_max,4)}</strong> — La log-verosimilitud máxima. Este valor se usa para comparar modelos: un modelo con mayor ℓ describe mejor los datos. Se emplea en el <em>Likelihood Ratio Test</em> y en criterios como AIC y BIC.</p>` +
    `</div>`;
})();

/* ── Panel 6: sigma ── */
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

/* ── Panel 7: resumen ── */
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
