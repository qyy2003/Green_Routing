'use strict';
// Prediction viewer: real 5-min series vs every model's forecasts, with zoom.
// Modes: block (window by window, every 5 min), rolling (fixed lead, every 5 min),
// sampled (the harness's sampled origins).

const COLORS = {
  persistence:'#8b95a5', seasonal_naive_day:'#6b7cff', seasonal_naive_week:'#9a6bff',
  historical_average:'#c78b3d', gbdt:'#e0b400', gru:'#2ecc71', tcn:'#00b8a9',
  dlinear:'#ff8a5c', patchtst:'#ff5c8a', stgnn:'#4ea3ff',
};
const REAL = '#e6e9ee';
const BASELINES = new Set(['persistence','seasonal_naive_day','seasonal_naive_week',
                           'historical_average']);
const MODE_LABEL = {
  block:'window by window (every 5 min)',
  rolling:'rolling, fixed lead (every 5 min)',
  sampled:'sampled origins (harness)',
};
const S = {
  meta:null, link:null, h:null, mode:null, data:null,
  off:new Set(),            // hidden models
  view:null,                // {t0,t1} visible time range
  full:null,                // full time range
  statsVisible:false,
  lb:{},                    // mode -> /api/leaderboard
  scores:{},                // `${mode}|${h}` -> link -> {model: [wape, ratio]}
};

const $ = s => document.querySelector(s);
const fmtT = t => new Date(t*1000).toISOString().slice(0,16).replace('T',' ');
const fmtN = (v,d=1) => v==null||!isFinite(v) ? '—' : v.toLocaleString('en-US',
  {minimumFractionDigits:d, maximumFractionDigits:d});
const models = () => S.meta.models[S.mode] || [];

function f32(b64){
  const bin = atob(b64), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf);
  for (let i=0;i<bin.length;i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}
function lowerBound(a, x){            // first index with a[i] >= x
  let lo=0, hi=a.length;
  while (lo<hi){ const m=(lo+hi)>>1; if (a[m]<x) lo=m+1; else hi=m; }
  return lo;
}
function grid(t0, step, n){
  const t = new Float64Array(n);
  for (let i=0;i<n;i++) t[i] = t0 + i*step;
  return t;
}

// ---------------------------------------------------------------- data load
async function boot(){
  S.meta = await (await fetch('/api/meta')).json();
  const c = S.meta.config;
  $('#runinfo').textContent =
    `${S.meta.run.split('/').slice(-2).join('/')} · ${c.n_links} links · test from ${c.test_start}`;
  const ms = $('#mode');
  for (const m of S.meta.modes){
    const o = document.createElement('option');
    o.value = m; o.textContent = MODE_LABEL[m] || m; ms.appendChild(o);
  }
  S.mode = S.meta.modes[0]; ms.value = S.mode;
  ms.onchange = async () => {
    S.mode = ms.value; S.off.clear();
    await Promise.all([ensureLB(), ensureScores()]);
    fillScoreModels(); renderList(); renderLeaderboard(); loadLink(S.link, true);
  };
  const hs = $('#horizon');
  S.meta.horizons.forEach(h => {
    const o = document.createElement('option');
    o.value = h.steps; o.textContent = `${h.name} (${h.steps} steps)`;
    hs.appendChild(o);
  });
  S.h = S.meta.horizons[0].steps;
  hs.value = S.h;
  hs.onchange = async () => {
    S.h = +hs.value; loadLink(S.link, true);
    await ensureScores(); renderList(); renderLeaderboard();
  };
  $('#statsVisible').onchange = e => { S.statsVisible = e.target.checked; renderStats(); };
  $('#search').oninput = renderList;
  $('#sort').onchange = async () => {
    const v = $('#sort').value;
    $('#scoreRow').hidden = !(v==='wape' || v==='ratio');
    await ensureScores(); renderList();
  };
  $('#scoreModel').onchange = renderList;
  $('#lbBtn').onclick = () => { $('#leaderboard').hidden = false; renderLeaderboard(); };
  $('#lbClose').onclick = () => { $('#leaderboard').hidden = true; draw(); };
  $('#lbMetric').onchange = renderLeaderboard;
  await ensureLB();
  fillScoreModels();
  renderList();
  loadLink(S.meta.links[0].id);
}

async function ensureLB(){
  if (!S.lb[S.mode]) S.lb[S.mode] = await (await fetch(`/api/leaderboard?mode=${S.mode}`)).json();
}
async function ensureScores(){
  const k = `${S.mode}|${S.h}`;
  if (!S.scores[k])
    S.scores[k] = await (await fetch(`/api/linkscores?h=${S.h}&mode=${S.mode}`)).json();
}

function fillScoreModels(){
  // default score model: the learned model with the best median ratio at this horizon
  const sm = $('#scoreModel'), prev = sm.value, lb = S.lb[S.mode];
  sm.innerHTML = '';
  let bestM = null, bestR = Infinity;
  for (const m of models()){
    const o = document.createElement('option');
    o.value = m.model; o.textContent = m.model; sm.appendChild(o);
    const c = lb && lb.cells[`${m.model}|${S.h}`];
    if (!BASELINES.has(m.model) && c && c.median_ratio!=null && c.median_ratio < bestR){
      bestR = c.median_ratio; bestM = m.model;
    }
  }
  if (prev && models().some(m=>m.model===prev)) sm.value = prev;
  else if (bestM) sm.value = bestM;
}

function renderList(){
  const q = $('#search').value.trim().toLowerCase();
  const by = $('#sort').value;
  let ls = S.meta.links.filter(l =>
    !q || l.id.toLowerCase().includes(q) || l.device.toLowerCase().includes(q));
  const sc = S.scores[`${S.mode}|${S.h}`], sm = $('#scoreModel').value;
  const score = l => {          // [wape, ratio] of the chosen model on this link
    const r = sc && sc[l.id] && sc[l.id][sm];
    return r ? (by==='wape' ? r[0] : r[1]) : null;
  };
  if (by==='mean') ls = ls.slice().sort((a,b)=>(b.mean_mbps??-1)-(a.mean_mbps??-1));
  else if (by==='peak') ls = ls.slice().sort((a,b)=>(b.peak_mbps??-1)-(a.peak_mbps??-1));
  else if ((by==='wape' || by==='ratio') && sc)
    ls = ls.slice().sort((a,b)=>(score(b)??-1)-(score(a)??-1));
  else ls = ls.slice().sort((a,b)=>a.id.localeCompare(b.id));
  $('#linkcount').textContent = `${ls.length} / ${S.meta.links.length}`;
  const ul = $('#links');
  ul.innerHTML = '';
  const frag = document.createDocumentFragment();
  for (const l of ls){
    const li = document.createElement('li');
    li.className = l.id===S.link ? 'sel' : '';
    const v = (by==='wape' || by==='ratio') && sc ? score(l) : null;
    const vtxt = by==='wape' ? (v==null?'—':v.toFixed(1)+'%')
               : by==='ratio' ? (v==null?'—':v.toFixed(2)+'×')
               : fmtN(l.mean_mbps,0);
    li.innerHTML = `<span class="nm">${l.id}</span>` +
                   `<span class="vl${by==='ratio'&&v!=null?(v<1?' good':' bad'):''}">${vtxt}</span>`;
    li.title = `${l.id} — mean ${fmtN(l.mean_mbps)} Mbps, peak ${fmtN(l.peak_mbps)} Mbps`;
    li.onclick = () => loadLink(l.id);
    frag.appendChild(li);
  }
  ul.appendChild(frag);
}

async function loadLink(id, keepView){
  if (!id) return;
  S.link = id;
  [...document.querySelectorAll('#links li')].forEach(li =>
    li.classList.toggle('sel', li.querySelector('.nm').textContent===id));
  $('#statstitle').textContent = `${id} — loading…`;
  const r = await fetch(`/api/link?id=${encodeURIComponent(id)}&h=${S.h}&mode=${S.mode}`);
  const d = await r.json();
  if (d.error){ $('#statstitle').textContent = d.error; return; }
  if (d.link !== S.link || d.mode !== S.mode || d.horizon !== S.h) return;  // stale
  // unify: every series -> {t: Float64Array, yp: Float32Array, yt: Float32Array}
  const ry = f32(d.real.b64);
  const real = {t: grid(d.real.t0, d.real.step, ry.length), y: ry,
                t0: d.real.t0, step: d.real.step};
  const ytAt = t => { const i = Math.round((t - real.t0)/real.step);
                      return i>=0 && i<ry.length ? ry[i] : NaN; };
  const series = {};
  for (const [name, m] of Object.entries(d.models)){
    let t, yp, yt;
    if (m.b64){
      yp = f32(m.b64); t = grid(m.t0, m.step, yp.length);
      yt = new Float32Array(yp.length);
      for (let i=0;i<yp.length;i++) yt[i] = ytAt(t[i]);
    } else {
      t = Float64Array.from(m.t);
      yp = Float32Array.from(m.y_pred, v => v==null ? NaN : v);
      yt = Float32Array.from(m.y_true, v => v==null ? NaN : v);
    }
    series[name] = {t, yp, yt};
  }
  S.data = {...d, real, series};
  const t0 = real.t[0], t1 = real.t[real.t.length-1];
  S.full = {t0, t1};
  if (!keepView || !S.view) S.view = {t0: Math.max(t0, d.lo_t - 3*86400), t1: Math.min(t1, d.lo_t + 11*86400)};
  renderLegend(); draw(); renderStats();
}

// ---------------------------------------------------------------- leaderboard
const LB_FMT = {
  median_ratio:  [v=>v.toFixed(3)+'×', 'low'],
  share_beats:   [v=>(100*v).toFixed(1)+'%', 'high'],
  median_wape:   [v=>v.toFixed(1)+'%', 'low'],
  pooled_wape:   [v=>v.toFixed(1)+'%', 'low'],
  pooled_mae:    [v=>fmtN(v,0), 'low'],
  peak_bias_pct: [v=>(v>0?'+':'')+v.toFixed(1)+'%', 'zero'],
  peak_wape:     [v=>v.toFixed(1)+'%', 'low'],
};
const LB_NOTE = {
  median_ratio: 'Each link scored as MAE(model) / MAE(persistence) on identical targets, then the median over links — one link, one vote. < 1 beats persistence. This is the report\'s headline metric.',
  share_beats: 'Share of links on which the model\'s MAE is below persistence\'s.',
  median_wape: 'WAPE = Σ|e| / Σ|y| per link, then the median over links: the typical link\'s error as a share of its own traffic.',
  pooled_wape: 'WAPE = Σ|e| / Σ|y| over every scored point of every link — volume-weighted, so the busiest links dominate (the carbon-bill view). On identical targets it ranks models exactly like pooled MAE.',
  pooled_mae: 'MAE in Mbps over every scored point of every link — dominated by the busiest links.',
  peak_bias_pct: 'Signed error on each link\'s top-decile actuals, % of peak volume (negative = shaving peaks). Closest to 0 is best; persistence shows the irreducible part.',
  peak_wape: 'WAPE restricted to each link\'s top-decile actuals.',
};
const MODE_NOTE = {
  block: 'Window by window: each horizon H cuts the test period into consecutive H-long windows, forecast in full from each window start — a prediction at every 5-min step; scores average over the window\'s leads 1…H.',
  rolling: 'Rolling, fixed lead: re-forecast at every 5-min origin, scored exactly H ahead — the report\'s horizon definition, at every origin.',
  sampled: 'Sampled: the harness\'s own records — one point per horizon at the run\'s sampled origins.',
};

function renderLeaderboard(){
  const lb = S.lb[S.mode];
  if (!lb || $('#leaderboard').hidden) return;
  const metric = $('#lbMetric').value, [fmt, dir] = LB_FMT[metric];
  const hs = lb.horizons;
  const val = (m,h) => { const c = lb.cells[`${m}|${h}`]; return c ? c[metric] : null; };
  const best = {};
  for (const h of hs){
    let bv = null;
    for (const m of lb.models){
      const v = val(m.model, h.steps);
      if (v==null || !isFinite(v)) continue;
      const score = dir==='low' ? v : dir==='high' ? -v : Math.abs(v);
      if (bv==null || score < bv.s) bv = {s:score, m:m.model};
    }
    best[h.steps] = bv && bv.m;
  }
  let html = '<table><thead><tr><th>model</th>' +
    hs.map(h=>`<th>${h.name}</th>`).join('') + '</tr></thead><tbody>';
  for (const m of lb.models){
    html += `<tr class="${BASELINES.has(m.model)?'base':''}"><td>` +
      `<span class="dot" style="background:${COLORS[m.model]||'#888'}"></span>${m.model}` +
      `<span class="muted small"> t${m.tier}</span></td>`;
    for (const h of hs){
      const v = val(m.model, h.steps);
      html += `<td class="${best[h.steps]===m.model?'bestc':''}">` +
              `${v==null||!isFinite(v)?'—':fmt(v)}</td>`;
    }
    html += '</tr>';
  }
  if (!lb.models.length) html += '<tr><td colspan="9">metrics not computed for this mode yet</td></tr>';
  $('#lbbody').innerHTML = html + '</tbody></table>';
  const c = S.meta.config;
  $('#lbsub').textContent = `· ${MODE_LABEL[S.mode]} · ${c.n_links} links · test from ${c.test_start}`;
  $('#lbnote').innerHTML = LB_NOTE[metric] + ' Best per horizon in green.<br>' + MODE_NOTE[S.mode];
}

// ---------------------------------------------------------------- rendering
function renderLegend(){
  const el = $('#legend');
  el.innerHTML = '';
  const mk = (name, color, tier) => {
    const d = document.createElement('div');
    d.className = 'item' + (S.off.has(name)?' off':'');
    d.innerHTML = `<span class="sw" style="background:${color}"></span>${name}` +
                  (tier!=null?`<span class="tier">t${tier}</span>`:'');
    d.onclick = () => { S.off.has(name) ? S.off.delete(name) : S.off.add(name);
                        renderLegend(); draw(); renderStats(); };
    el.appendChild(d);
  };
  mk('real', REAL, null);
  for (const m of models())
    if (S.data.series[m.model]) mk(m.model, COLORS[m.model]||'#888', m.tier);
}

function plotBox(cv){
  const r = cv.getBoundingClientRect(), dpr = devicePixelRatio||1;
  cv.width = r.width*dpr; cv.height = r.height*dpr;
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx, w:r.width, h:r.height, L:56, R:12, T:10, B:22};
}

function range(t){                    // visible index range [i0, i1)
  return [lowerBound(t, S.view.t0), lowerBound(t, S.view.t1 + 1e-6)];
}

function visible(){                   // {name: {t, yp, yt, i0, i1}} for shown models
  const out = {};
  for (const [name, s] of Object.entries(S.data.series)){
    if (S.off.has(name)) continue;
    const [i0, i1] = range(s.t);
    out[name] = {...s, i0, i1};
  }
  return out;
}

function draw(){
  if (!S.data) return;
  drawMain();
  drawErr();
}

function yScale(lo, hi, pad=0.08){
  if (!isFinite(lo)){ lo=0; hi=1; }
  if (lo===hi){ lo-=1; hi+=1; }
  const p=(hi-lo)*pad; return [lo-p, hi+p];
}

function axes(g, xr, yr, yfmt){
  const {ctx,w,h,L,R,T,B} = g;
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle='#2a313c'; ctx.fillStyle='#8b95a5';
  ctx.font='10.5px ui-sans-serif,system-ui,sans-serif';
  ctx.lineWidth=1;
  const ticks=5;
  for (let i=0;i<=ticks;i++){
    const v = yr[0] + (yr[1]-yr[0])*i/ticks;
    const y = h-B - (h-B-T)*i/ticks;
    ctx.globalAlpha=.35; ctx.beginPath(); ctx.moveTo(L,y); ctx.lineTo(w-R,y); ctx.stroke();
    ctx.globalAlpha=1; ctx.textAlign='right'; ctx.textBaseline='middle';
    ctx.fillText(yfmt(v), L-6, y);
  }
  const span = xr[1]-xr[0], n=6;
  ctx.textAlign='center'; ctx.textBaseline='top';
  for (let i=0;i<=n;i++){
    const t = xr[0] + span*i/n, x = L + (w-L-R)*i/n;
    const dt = new Date(t*1000);
    const lab = span > 3*86400
      ? dt.toISOString().slice(5,10)
      : dt.toISOString().slice(5,16).replace('T',' ');
    ctx.globalAlpha=.35; ctx.beginPath(); ctx.moveTo(x,T); ctx.lineTo(x,h-B); ctx.stroke();
    ctx.globalAlpha=1; ctx.fillText(lab, x, h-B+5);
  }
}

function mkProj(g, xr, yr){
  const {w,h,L,R,T,B} = g;
  return {
    x: t => L + (w-L-R)*(t-xr[0])/(xr[1]-xr[0]||1),
    y: v => (h-B) - (h-B-T)*(v-yr[0])/(yr[1]-yr[0]||1),
    inv: px => xr[0] + (px-L)/(w-L-R)*(xr[1]-xr[0]),
  };
}

// Line through (t[i], y[i]) for i in [i0, i1). NaN breaks the line. With more
// points than pixels, each pixel column draws its min..max (an envelope), so a
// 111-day window of 5-min data stays fast and faithful.
function polyline(ctx, P, t, y, i0, i1, width){
  const n = i1 - i0;
  ctx.beginPath();
  if (n <= 2*width){
    let pen=false;
    for (let i=i0;i<i1;i++){
      const v=y[i];
      if (!isFinite(v)){ pen=false; continue; }
      const X=P.x(t[i]), Y=P.y(v);
      pen ? ctx.lineTo(X,Y) : ctx.moveTo(X,Y); pen=true;
    }
  } else {
    let col=-1, mn=Infinity, mx=-Infinity, pen=false, gap=false;
    const flush = () => {
      if (col<0) return;
      if (mn===Infinity){ pen=false; return; }
      const a=P.y(mx), b=P.y(mn);
      if (!pen || gap){ ctx.moveTo(col, a); } else ctx.lineTo(col, a);
      ctx.lineTo(col, b); pen=true; gap=false;
    };
    for (let i=i0;i<i1;i++){
      const X=Math.floor(P.x(t[i])), v=y[i];
      if (X!==col){ flush(); col=X; mn=Infinity; mx=-Infinity; }
      if (!isFinite(v)){ gap=true; continue; }
      if (v<mn) mn=v; if (v>mx) mx=v;
    }
    flush();
  }
  ctx.stroke();
}

let MAIN=null, ERR=null;
function drawMain(){
  const g = plotBox($('#chart'));
  const width = g.w-g.L-g.R;
  const rr = S.data.real, [r0, r1] = range(rr.t);
  const vis = visible();
  let lo=Infinity, hi=-Infinity;
  const acc = (a,i0,i1) => { for (let i=i0;i<i1;i++){ const v=a[i]; if (v<lo) lo=v; if (v>hi) hi=v; } };
  if (!S.off.has('real')) acc(rr.y, r0, r1);
  for (const s of Object.values(vis)) acc(s.yp, s.i0, s.i1);
  const yr = yScale(lo, hi), xr=[S.view.t0,S.view.t1];
  axes(g, xr, yr, v=>fmtN(v, Math.abs(yr[1])<20?1:0));
  const P = mkProj(g,xr,yr);
  const {ctx} = g;
  ctx.save(); ctx.beginPath();
  ctx.rect(g.L, g.T, width, g.h-g.B-g.T); ctx.clip();

  // block mode: window seams (each re-forecast) when they are few enough to see
  const H = S.data.window_steps;
  if (H){
    const wsec = H*S.meta.step_seconds, n = (xr[1]-xr[0])/wsec;
    if (n <= 160){
      ctx.strokeStyle='#3a4352'; ctx.setLineDash([2,4]); ctx.lineWidth=1;
      let k = Math.ceil((xr[0]-S.data.lo_t)/wsec);
      for (let t=S.data.lo_t + k*wsec; t<=xr[1]; t+=wsec){
        const X=P.x(t); ctx.beginPath(); ctx.moveTo(X,g.T); ctx.lineTo(X,g.h-g.B); ctx.stroke();
      }
      ctx.setLineDash([]);
    }
  }
  // test start
  if (S.data.lo_t >= xr[0] && S.data.lo_t <= xr[1]){
    ctx.strokeStyle='#0c8f56'; ctx.lineWidth=1.2;
    const X=P.x(S.data.lo_t); ctx.beginPath(); ctx.moveTo(X,g.T); ctx.lineTo(X,g.h-g.B); ctx.stroke();
  }
  if (!S.off.has('real')){
    ctx.strokeStyle=REAL; ctx.lineWidth=1.25; ctx.globalAlpha=.85;
    polyline(ctx, P, rr.t, rr.y, r0, r1, width);
    ctx.globalAlpha=1;
  }
  for (const [name, s] of Object.entries(vis)){
    const col = COLORS[name]||'#888';
    ctx.strokeStyle=col; ctx.fillStyle=col; ctx.lineWidth=1.3; ctx.globalAlpha=.9;
    polyline(ctx, P, s.t, s.yp, s.i0, s.i1, width);
    ctx.globalAlpha=1;
    if (s.i1 - s.i0 <= 300){                       // markers only when sparse
      for (let i=s.i0;i<s.i1;i++){
        if (!isFinite(s.yp[i])) continue;
        ctx.beginPath(); ctx.arc(P.x(s.t[i]), P.y(s.yp[i]), 2.2, 0, 7); ctx.fill();
      }
    }
  }
  ctx.restore();
  MAIN = {g,P,xr,yr,vis};
  if (BOX) drawBox();
}

function drawErr(){
  const g = plotBox($('#errchart'));
  const width = g.w-g.L-g.R;
  const vis = visible();
  let lo=Infinity, hi=-Infinity;
  for (const s of Object.values(vis))
    for (let i=s.i0;i<s.i1;i++){ const e=s.yp[i]-s.yt[i]; if (e<lo) lo=e; if (e>hi) hi=e; }
  if (!isFinite(lo)){ lo=-1; hi=1; }
  const m = Math.max(Math.abs(lo), Math.abs(hi));
  const yr = yScale(-m, m), xr=[S.view.t0,S.view.t1];
  axes(g, xr, yr, v=>fmtN(v,0));
  const P = mkProj(g,xr,yr);
  const {ctx}=g;
  ctx.save(); ctx.beginPath(); ctx.rect(g.L,g.T,width,g.h-g.B-g.T); ctx.clip();
  ctx.strokeStyle='#5a6577'; ctx.setLineDash([3,3]); ctx.beginPath();
  ctx.moveTo(g.L,P.y(0)); ctx.lineTo(g.w-g.R,P.y(0)); ctx.stroke(); ctx.setLineDash([]);
  for (const [name,s] of Object.entries(vis)){
    const col = COLORS[name]||'#888';
    const n = s.i1 - s.i0;
    if (n <= 300){
      ctx.fillStyle = col;
      for (let i=s.i0;i<s.i1;i++){
        const e=s.yp[i]-s.yt[i]; if (!isFinite(e)) continue;
        ctx.beginPath(); ctx.arc(P.x(s.t[i]), P.y(e), 2, 0, 7); ctx.fill();
      }
    } else {
      const e = new Float32Array(n);
      for (let i=0;i<n;i++) e[i] = s.yp[s.i0+i]-s.yt[s.i0+i];
      ctx.strokeStyle = col; ctx.lineWidth=1; ctx.globalAlpha=.75;
      polyline(ctx, P, s.t.subarray(s.i0, s.i1), e, 0, n, width);
      ctx.globalAlpha=1;
    }
  }
  ctx.restore();
  ERR = {g,P};
}

// ---------------------------------------------------------------- stats
function computeVisibleStats(){
  const out = {};
  for (const [name,s] of Object.entries(S.data.series)){
    const [i0, i1] = range(s.t);
    let n=0, ae=0, se=0, sum=0, err=0, under=0;
    for (let i=i0;i<i1;i++){
      const p=s.yp[i], y=s.yt[i];
      if (!isFinite(p) || !isFinite(y)) continue;
      const e=p-y; n++;
      ae+=Math.abs(e); se+=e*e; sum+=Math.abs(y); err+=e; if(e<0) under++;
    }
    if (!n) continue;
    out[name] = {n, mae:ae/n, rmse:Math.sqrt(se/n),
      wape: sum>0 ? 100*ae/sum : null, bias_pct: sum>0 ? 100*err/sum : null,
      underpred_rate: under/n, peak_bias_pct:null};
  }
  const base = out['persistence'];
  for (const r of Object.values(out))
    r.skill = base && base.mae ? 1 - r.mae/base.mae : null;
  return out;
}

function renderStats(){
  if (!S.data) return;
  const d = S.data;
  const hname = S.meta.horizons.find(h=>h.steps===S.h)?.name || S.h;
  $('#statstitle').textContent = d.link;
  const lm = S.meta.links.find(l=>l.id===d.link) || {};
  $('#statsmeta').innerHTML =
    `device <b>${d.device}</b> · horizon <b>${hname}</b> · ${MODE_LABEL[S.mode]}<br>` +
    `mean ${fmtN(lm.mean_mbps)} Mbps · peak ${fmtN(lm.peak_mbps)} Mbps`;

  let rows = d.stats.map(r => ({...r}));
  if (S.statsVisible){
    const vis = computeVisibleStats();
    rows = rows.map(r => vis[r.model] ? {...r, ...vis[r.model], mase:null} : {...r, n:0, mae:null});
  }
  rows = rows.filter(r => !S.off.has(r.model));
  rows.sort((a,b)=>(a.mae??1e18)-(b.mae??1e18));
  const best = rows.length ? rows[0].model : null;

  const cell = (v,d2=1,cls='') => `<td class="${cls}">${fmtN(v,d2)}</td>`;
  let html = `<table><thead><tr><th>model</th><th>MAE</th><th>RMSE</th>` +
             `<th>WAPE%</th><th>skill</th><th>bias%</th><th>peak%</th></tr></thead><tbody>`;
  for (const r of rows){
    const sk = r.skill;
    html += `<tr class="${r.model===best?'best':''}">` +
      `<td><span class="dot" style="background:${COLORS[r.model]||'#888'}"></span>${r.model}` +
      `<span class="muted small"> t${r.tier}</span></td>` +
      cell(r.mae,1) + cell(r.rmse,1) + cell(r.wape,1) +
      `<td class="${sk>0?'pos':sk<0?'neg':''}">${sk==null?'—':(sk>0?'+':'')+sk.toFixed(3)}</td>` +
      `<td class="${r.bias_pct>0?'pos':'neg'}">${fmtN(r.bias_pct,1)}</td>` +
      `<td class="${r.peak_bias_pct>0?'pos':'neg'}">${fmtN(r.peak_bias_pct,1)}</td>` +
      `</tr>`;
  }
  html += '</tbody></table>';
  $('#statsbody').innerHTML = html;
  const n = rows.length ? (rows[0].n || 0) : 0;
  const scope = S.statsVisible ? 'in the visible window' : 'over the whole test period';
  $('#statsnote').innerHTML =
    `<b>${n.toLocaleString()}</b> scored forecasts ${scope}. ${MODE_NOTE[S.mode]} ` +
    `skill = 1 − MAE/MAE<sub>persistence</sub>; WAPE = Σ|e|/Σ|y|; bias% and peak% are ` +
    `signed — negative = under-forecast (peak% = this link's top-decile actuals` +
    `${S.statsVisible ? ', whole period only' : ''}).` +
    (S.statsVisible ? '' : ' Tick “stats over visible window” to rescore what you have zoomed into.');
}

// ---------------------------------------------------------------- interaction
let BOX=null, DRAG=null;
function drawBox(){
  if (!MAIN||!BOX) return;
  const {ctx,h,T,B} = MAIN.g;
  ctx.save();
  ctx.fillStyle='rgba(39,96,230,.18)'; ctx.strokeStyle='#2760e6';
  const x0=Math.min(BOX.x0,BOX.x1), x1=Math.max(BOX.x0,BOX.x1);
  ctx.fillRect(x0,T,x1-x0,h-B-T); ctx.strokeRect(x0,T,x1-x0,h-B-T);
  ctx.restore();
}

function setupChart(){
  const cv = $('#chart'), tip = $('#tooltip');
  cv.addEventListener('wheel', e => {
    if (!S.view) return;
    e.preventDefault();
    const r = cv.getBoundingClientRect();
    const t = MAIN.P.inv(e.clientX - r.left);
    const k = e.deltaY > 0 ? 1.2 : 1/1.2;
    let t0 = t - (t - S.view.t0)*k, t1 = t + (S.view.t1 - t)*k;
    t0 = Math.max(t0, S.full.t0); t1 = Math.min(t1, S.full.t1);
    if (t1 - t0 > 900){ S.view={t0,t1}; draw(); if(S.statsVisible) renderStats(); }
  }, {passive:false});

  cv.addEventListener('mousedown', e => {
    if (!S.view || !MAIN) return;
    const r = cv.getBoundingClientRect();
    DRAG = {x:e.clientX - r.left, t0:S.view.t0, t1:S.view.t1, shift:e.shiftKey};
    if (e.shiftKey){ BOX = {x0:DRAG.x, x1:DRAG.x}; }
  });
  addEventListener('mousemove', e => {
    const r = cv.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    if (DRAG && DRAG.shift){ BOX.x1 = px; draw(); return; }
    if (DRAG){
      const dt = (DRAG.x - px)/(MAIN.g.w-MAIN.g.L-MAIN.g.R)*(DRAG.t1-DRAG.t0);
      let t0=DRAG.t0+dt, t1=DRAG.t1+dt;
      if (t0 < S.full.t0){ t1 += S.full.t0-t0; t0 = S.full.t0; }
      if (t1 > S.full.t1){ t0 -= t1-S.full.t1; t1 = S.full.t1; }
      S.view={t0,t1}; draw(); return;
    }
    if (!MAIN || px<MAIN.g.L || px>MAIN.g.w-MAIN.g.R || py<0 || py>MAIN.g.h-MAIN.g.B){
      tip.classList.add('hidden'); return;
    }
    hover(px, py, r);
  });
  addEventListener('mouseup', () => {
    if (DRAG && DRAG.shift && BOX && Math.abs(BOX.x1-BOX.x0) > 8){
      const a=MAIN.P.inv(Math.min(BOX.x0,BOX.x1)), b=MAIN.P.inv(Math.max(BOX.x0,BOX.x1));
      S.view={t0:Math.max(a,S.full.t0), t1:Math.min(b,S.full.t1)};
      if (S.statsVisible) renderStats();
    }
    if (DRAG && S.statsVisible) renderStats();
    DRAG=null; BOX=null; draw();
  });
  cv.addEventListener('dblclick', () => {
    if (!S.full) return;
    S.view={...S.full}; draw(); if (S.statsVisible) renderStats();
  });
  cv.addEventListener('mouseleave', () => $('#tooltip').classList.add('hidden'));
  addEventListener('resize', () => draw());
}

function hover(px, py, rect){
  const tip = $('#tooltip');
  const t = MAIN.P.inv(px);
  const tol = (S.view.t1-S.view.t0)/60;
  let rows = [], best=Infinity, bestT=null, bestTrue=null;
  for (const [name,s] of Object.entries(MAIN.vis)){
    let i = lowerBound(s.t, t);
    if (i>0 && (i>=s.t.length || Math.abs(s.t[i-1]-t) < Math.abs(s.t[i]-t))) i--;
    if (i<0 || i>=s.t.length || !isFinite(s.yp[i])) continue;
    const cd = Math.abs(s.t[i]-t);
    if (cd < tol){
      rows.push([name, s.yp[i], s.yt[i]]);
      if (cd<best){ best=cd; bestT=s.t[i]; bestTrue=s.yt[i]; }
    }
  }
  if (!rows.length){ tip.classList.add('hidden'); return; }
  let html = `<b>${fmtT(bestT)}</b><br><span style="color:${REAL}">real</span> ` +
             `${fmtN(bestTrue)} Mbps<hr style="border:0;border-top:1px solid #2a313c;margin:4px 0">`;
  rows.sort((a,b)=>Math.abs(a[1]-a[2])-Math.abs(b[1]-b[2]));
  for (const [n,p,tr] of rows)
    html += `<span style="color:${COLORS[n]||'#888'}">■</span> ${n}: ${fmtN(p)} ` +
            `<span class="muted">(${p-tr>0?'+':''}${fmtN(p-tr)})</span><br>`;
  tip.innerHTML = html;
  tip.classList.remove('hidden');
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  tip.style.left = Math.min(px+14, rect.width-tw-6) + 'px';
  tip.style.top  = Math.max(6, Math.min(py+12, rect.height-th-6)) + 'px';
}

setupChart();
boot();
