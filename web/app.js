// SWITCHlan Backbone viewer — traffic & telemetry over the optical map.
//
// Links are coloured by a chosen link metric (utilization % or traffic Mbps)
// and their WIDTH encodes capacity. Nodes can be coloured by a device metric
// (CPU %, power W, temperature °C). Data comes on demand from server.py, which
// reads the compact per-month frames built from the Cricket dataset.

const SVG_NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);

const stage = $("stage"), viewport = $("viewport"), paper = $("paper");
const loading = $("loading"), tooltip = $("tooltip"), subtitle = $("subtitle");
const slider = $("time-slider"), timeInput = $("time-input"), timeLabel = $("time-label");
const matchedEl = $("matched"), playBtn = $("play"), windowSel = $("window-size"), aggSel = $("agg-fn");

async function getJSON(url){
  const r = await fetch(url, { cache: "no-store" });
  const text = await r.text();
  if(!r.ok) throw new Error(`${url} → HTTP ${r.status}`);
  try { return JSON.parse(text); }
  catch(_) { throw new Error(`${url} did not return JSON — is server.py running (not "python -m http.server")? Got: ${text.slice(0,60)}`); }
}

const view = { scale: 1, x: 0, y: 0, min: 0.05, max: 12 };
let meta = null, mapSvg = null;
let linkEls = {}, linkById = {}, nodeEls = {};
let curFrame = null, playing = false, playTimer = null;

const state = {
  linksOn: true, linkMetric: "util",
  nodesOn: false, nodeMetric: "cpu",
  window: 300, agg: "mean",   // aggregation window (s) and function
};
const WINDOW_LABEL = { 300:"5 min", 3600:"1 hour", 21600:"6 hours", 86400:"1 day",
  259200:"3 days", 604800:"1 week", 1209600:"2 weeks", 2592000:"1 month" };

// ---- metric definitions: scale + colour ----
const RAMP = [ [0,[51,187,102]], [0.4,[255,221,0]], [0.7,[255,153,51]], [1,[226,34,34]] ];
function ramp(f) {
  f = Math.max(0, Math.min(1, f));
  for (let i = 1; i < RAMP.length; i++) {
    if (f <= RAMP[i][0]) {
      const [f0,c0]=RAMP[i-1],[f1,c1]=RAMP[i], k=(f-f0)/(f1-f0);
      const c=c0.map((v,j)=>Math.round(v+(c1[j]-v)*k));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
  }
  return "rgb(226,34,34)";
}
const logNorm = (v, max) => Math.log10(1 + Math.max(0,v)) / Math.log10(1 + max);

const METRICS = {
  util:    { title:"Utilization (%)",    min:"0", max:"100", norm:(v)=>v/100,            fmt:(v)=>v.toFixed(1)+"%" },
  traffic: { title:"Traffic (Mbps, log)",min:"0", max:"100 G", norm:(v)=>logNorm(v,100000), fmt:fmtMbps },
  cpu:     { title:"CPU (%)",            min:"0", max:"100", norm:(v)=>v/100,            fmt:(v)=>v.toFixed(1)+"%" },
  pw:      { title:"Power (W)",          min:"0", max:"4 kW", norm:(v)=>v/4000,          fmt:(v)=>v.toFixed(0)+" W" },
  temp:    { title:"Temperature (°C)",   min:"20",max:"75",  norm:(v)=>(v-20)/55,        fmt:(v)=>v.toFixed(1)+" °C" },
};

function fmtMbps(v){ return v>=1000 ? (v/1000).toFixed(2)+" Gbps" : v.toFixed(1)+" Mbps"; }

// width from capacity (bps): 1G..200G -> 2.5..13 px on a log scale
function capWidth(bps){
  const f = (Math.log10(bps) - 9) / (Math.log10(2e11) - 9); // 1e9..2e11
  return 2.5 + Math.max(0, Math.min(1, f)) * 10.5;
}

// ---- pan / zoom ----
function applyTransform(){ viewport.style.transform = `translate(${view.x}px,${view.y}px) scale(${view.scale})`; }
function fit(){
  const r = stage.getBoundingClientRect(), pw=paper.offsetWidth, ph=paper.offsetHeight;
  if(!pw||!ph) return;
  const m=40, s=Math.min((r.width-m)/pw,(r.height-m)/ph);
  view.scale=Math.max(view.min,Math.min(view.max,s));
  view.x=(r.width-pw*view.scale)/2; view.y=(r.height-ph*view.scale)/2; applyTransform();
}
function zoomAt(fx,fy,factor){
  const next=Math.max(view.min,Math.min(view.max,view.scale*factor)), k=next/view.scale;
  view.x=fx-(fx-view.x)*k; view.y=fy-(fy-view.y)*k; view.scale=next; applyTransform();
}
stage.addEventListener("wheel",(e)=>{ e.preventDefault(); const r=stage.getBoundingClientRect();
  zoomAt(e.clientX-r.left,e.clientY-r.top, e.deltaY<0?1.12:1/1.12); },{passive:false});
let dragging=false,last={x:0,y:0},gStart={x:0,y:0},gMoved=false,pressTarget=null;
stage.addEventListener("pointerdown",(e)=>{ pressTarget=null; if(e.target.closest(".panel")) return;
  pressTarget=e.target;
  dragging=true; last={x:e.clientX,y:e.clientY}; gStart={x:e.clientX,y:e.clientY}; gMoved=false;
  stage.classList.add("panning"); stage.setPointerCapture(e.pointerId); });
stage.addEventListener("pointermove",(e)=>{ if(!dragging) return;
  view.x+=e.clientX-last.x; view.y+=e.clientY-last.y; last={x:e.clientX,y:e.clientY};
  if(Math.hypot(e.clientX-gStart.x,e.clientY-gStart.y)>4) gMoved=true; applyTransform(); });
function endDrag(e){
  dragging=false; stage.classList.remove("panning");
  try{stage.releasePointerCapture(e.pointerId);}catch(_){}
  // A press with no drag = a click: toggle the pin on the link/node under it.
  if(!gMoved && pressTarget){
    if(pressTarget.classList && pressTarget.classList.contains("traffic-link"))
      togglePin("link", pressTarget.dataset.id);
    else if(pressTarget.classList && pressTarget.classList.contains("node-marker"))
      togglePin("node", pressTarget.dataset.code);
  }
  pressTarget=null;
}
stage.addEventListener("pointerup",endDrag); stage.addEventListener("pointercancel",endDrag);
$("zoom-in").onclick=()=>{const r=stage.getBoundingClientRect(); zoomAt(r.width/2,r.height/2,1.25);};
$("zoom-out").onclick=()=>{const r=stage.getBoundingClientRect(); zoomAt(r.width/2,r.height/2,1/1.25);};
$("zoom-reset").onclick=fit;
window.addEventListener("resize",fit);

// ---- overlay build ----
function buildOverlay(){
  const g=document.createElementNS(SVG_NS,"g"); g.setAttribute("id","overlay");
  // links first (under nodes)
  for(const link of meta.links){
    const [a,b]=link.endpoints, pa=meta.nodes[a], pb=meta.nodes[b];
    if(!pa||!pb) continue;
    const line=document.createElementNS(SVG_NS,"line");
    line.setAttribute("x1",pa.x); line.setAttribute("y1",pa.y);
    line.setAttribute("x2",pb.x); line.setAttribute("y2",pb.y);
    line.setAttribute("class","traffic-link nodata");
    line.setAttribute("stroke-width", capWidth(link.capacity_bps).toFixed(1));
    line.dataset.id=link.id;
    line.addEventListener("pointerenter",(e)=>{ showLinkTip(link,e); onHover("link",link.id); });
    line.addEventListener("pointermove",positionTip);
    line.addEventListener("pointerleave",()=>{ hideTip(); onLeave(); });
    g.appendChild(line); linkEls[link.id]=line; linkById[link.id]=link;
  }
  // node markers on top
  for(const code in meta.nodes){
    const p=meta.nodes[code];
    const c=document.createElementNS(SVG_NS,"circle");
    c.setAttribute("cx",p.x); c.setAttribute("cy",p.y); c.setAttribute("r",11);
    c.setAttribute("class","node-marker hidden");
    c.dataset.code=code;
    c.addEventListener("pointerenter",(e)=>{ showNodeTip(code,e); onHover("node",code); });
    c.addEventListener("pointermove",positionTip);
    c.addEventListener("pointerleave",()=>{ hideTip(); onLeave(); });
    g.appendChild(c); nodeEls[code]=c;
  }
  mapSvg.appendChild(g);
}

// ---- tooltip ----
const aggWord=()=> state.window<=300 ? "" : `${state.agg} over ${WINDOW_LABEL[state.window]}`;
function showLinkTip(link,e){
  const s=curFrame&&curFrame.links[link.id], [a,b]=link.endpoints;
  let h=`<b>${a} ↔ ${b}</b> · cap ${fmtCap(link.capacity_bps)}`;
  if(s){
    const load=Math.max(s.in||0,s.out||0), util=link.capacity_bps? load/(link.capacity_bps/1e6)*100:0;
    h+=`<br>in ${fmtMbps(s.in||0)} · out ${fmtMbps(s.out||0)}<br>utilization ${util.toFixed(1)}%`;
    if(aggWord()) h+=`<br><span style="color:var(--muted)">${aggWord()}</span>`;
  } else h+=`<br>no data at this time`;
  tooltip.innerHTML=h; tooltip.style.display="block"; positionTip(e);
}
function showNodeTip(code,e){
  const s=curFrame&&curFrame.nodes[code];
  let h=`<b>${code}</b>`;
  if(s){
    if(s.cpu!=null) h+=`<br>CPU ${s.cpu.toFixed(1)}%`;
    if(s.pw!=null) h+=`<br>Power ${s.pw.toFixed(0)} W`;
    if(s.temp!=null) h+=`<br>Temp ${s.temp.toFixed(1)} °C`;
    if(aggWord()) h+=`<br><span style="color:var(--muted)">${aggWord()}</span>`;
  } else h+=`<br>no data at this time`;
  tooltip.innerHTML=h; tooltip.style.display="block"; positionTip(e);
}
function positionTip(e){ const r=stage.getBoundingClientRect();
  tooltip.style.left=(e.clientX-r.left+14)+"px"; tooltip.style.top=(e.clientY-r.top+14)+"px"; }
function hideTip(){ tooltip.style.display="none"; }
function fmtCap(bps){ return bps>=1e9? (bps/1e9)+"G" : (bps/1e6)+"M"; }

// ---- inspector (right panel): time-series plot + stats + description ----
const IN_COL="#4aa3ff", OUT_COL="#ff9a3c", METRIC_COL="#59d98e";
let pinned=null, hover=null;         // each {kind, id}
let inspSeq=0, inspTimer=null;

function curTarget(){ return hover || pinned; }
function onHover(kind,id){ hover={kind,id}; refreshInspector(true); }
function onLeave(){ hover=null; if(pinned) refreshInspector(true); else clearInspector(); }
function togglePin(kind,id){
  pinned = (pinned && pinned.kind===kind && pinned.id===id) ? null : {kind,id};
  updatePinHighlight(); refreshInspector(true);
}
function pinTarget(kind,id){ pinned={kind,id}; updatePinHighlight(); refreshInspector(true); }
function updatePinHighlight(){
  for(const k in linkEls) linkEls[k].classList.remove("pinned");
  for(const c in nodeEls) nodeEls[c].classList.remove("pinned");
  if(pinned){ const el = pinned.kind==="link" ? linkEls[pinned.id] : nodeEls[pinned.id];
    if(el) el.classList.add("pinned"); }
}
function clearInspector(){ $("insp-empty").style.display=""; $("insp-body").style.display="none"; }
function inspWindow(){
  // instant window -> show the last 24h so the plot is still meaningful
  return state.window>300 ? {t: currentTs(), w: state.window, label: WINDOW_LABEL[state.window]}
                          : {t: currentTs()-86400, w: 86400, label: "last 24 h"};
}
function refreshInspector(immediate){
  const tgt=curTarget();
  if(!tgt) return;
  const {kind,id}=tgt, {t,w}=inspWindow();
  const metric = kind==="node" ? state.nodeMetric : "in";
  const run=async()=>{ const seq=++inspSeq;
    try{ const d=await getJSON(`/api/series?kind=${kind}&id=${encodeURIComponent(id)}&t=${t}&w=${w}&metric=${metric}`);
      if(seq!==inspSeq) return; renderInspector(d);
    }catch(_){} };
  clearTimeout(inspTimer);
  if(immediate) run(); else inspTimer=setTimeout(run,220);
}

const NODE_UNIT={cpu:"%",pw:" W",temp:" °C"}, NODE_NAME={cpu:"CPU",pw:"Power",temp:"Temp"};
function fmtVal(kind,v){ if(v==null) return "–";
  return kind==="link" ? fmtMbps(v) : v.toFixed(kind==="node"&&state.nodeMetric==="pw"?0:1)+NODE_UNIT[state.nodeMetric]; }

function renderInspector(d){
  $("insp-empty").style.display="none"; $("insp-body").style.display="";
  const win=inspWindow();
  const rangeStr=`${fmtTime(win.t)} → ${fmtTime(win.t+win.w)}`;
  if(d.kind==="link"){
    const link=linkById[d.id], [a,b]=link.endpoints;
    $("insp-title").innerHTML=`<b>${a} ↔ ${b}</b>`;
    $("insp-sub").textContent=`capacity ${fmtCap(d.capacity_bps)} · ${win.label} · ${rangeStr}`;
    drawPlot(d.series, ["in","out"], [IN_COL,OUT_COL], Math.max(d.stats.in?.max||0, d.stats.out?.max||0));
    $("insp-legend").innerHTML=`<span style="color:${IN_COL}">▬ in</span> &nbsp; <span style="color:${OUT_COL}">▬ out</span>`;
    const capMbps=d.capacity_bps/1e6;
    const rows=[["", "min","mean","median","max"]];
    for(const k of ["in","out"]){ const s=d.stats[k];
      rows.push([k, s?fmtMbps(s.min):"–", s?fmtMbps(s.mean):"–", s?fmtMbps(s.median):"–", s?fmtMbps(s.max):"–"]); }
    renderStatsTable(rows);
    let desc="";
    const so=d.stats.out, si=d.stats.in;
    if(so&&si){
      const pk=Math.max(so.max,si.max), pkk=so.max>=si.max?"out":"in", pkt=so.max>=si.max?so.peak_t:si.peak_t;
      const pkUtil=capMbps?pk/capMbps*100:0;
      desc=`Mean load ${fmtMbps(Math.max(si.mean,so.mean))} (${(Math.max(si.mean,so.mean)/capMbps*100).toFixed(0)}% of capacity). `+
           `Peak ${fmtMbps(pk)} ${pkk} (${pkUtil.toFixed(0)}%) at ${fmtTime(pkt)}.`;
    } else desc="No traffic data for this link in the selected window.";
    $("insp-desc").textContent=desc;
  } else {
    $("insp-title").innerHTML=`<b>${d.id}</b> — ${NODE_NAME[state.nodeMetric]}`;
    $("insp-sub").textContent=`${win.label} · ${rangeStr}`;
    drawPlot(d.series, [state.nodeMetric], [METRIC_COL], d.stats[state.nodeMetric]?.max||0);
    $("insp-legend").innerHTML=`<span style="color:${METRIC_COL}">▬ ${NODE_NAME[state.nodeMetric]}</span>`;
    const s=d.stats[state.nodeMetric];
    renderStatsTable([["", "min","mean","median","max"],
      [NODE_NAME[state.nodeMetric], s?fmtVal("node",s.min):"–", s?fmtVal("node",s.mean):"–", s?fmtVal("node",s.median):"–", s?fmtVal("node",s.max):"–"]]);
    $("insp-desc").textContent = s ? `Mean ${fmtVal("node",s.mean)}, peak ${fmtVal("node",s.max)} at ${fmtTime(s.peak_t)}.`
                                   : "No data for this node in the selected window.";
  }
  const onPinned = pinned && (!hover || (hover.kind===pinned.kind && hover.id===pinned.id));
  const pinNote = onPinned ? " · 📌 pinned (click to unpin)"
                : (hover && pinned) ? ` · previewing (pinned ${pinned.id})` : "";
  if(pinNote) $("insp-sub").textContent += pinNote;
}
function renderStatsTable(rows){
  const t=$("insp-stats"); t.innerHTML="";
  rows.forEach((r,ri)=>{ const tr=document.createElement("tr");
    r.forEach(c=>{ const cell=document.createElement(ri===0?"th":"td"); cell.textContent=c; tr.appendChild(cell); });
    t.appendChild(tr); });
}
function drawPlot(series, keys, colors, yMax){
  const svg=$("insp-plot"); while(svg.firstChild) svg.removeChild(svg.firstChild);
  const W=340,H=132,pad={l:62,r:12,t:10,b:20};
  const ts=series.t; if(!ts||!ts.length) return;
  let vmax=yMax||0;
  if(vmax<=0){ keys.forEach(k=>series[k].forEach(v=>{ if(v!=null&&v>vmax) vmax=v; })); }
  if(vmax<=0) vmax=1;
  const isLink = keys.includes("in")||keys.includes("out");
  const unitFmt=(v)=> isLink ? fmtMbps(v)
      : (state.nodeMetric==="pw"? v.toFixed(0): v.toFixed(1))+NODE_UNIT[state.nodeMetric].trim();
  const plotW=W-pad.l-pad.r, plotH=H-pad.t-pad.b;
  const x=(i)=> pad.l + plotW*(i/(ts.length-1||1));
  const y=(v)=> pad.t + plotH*(1 - v/vmax);
  const mk=(n)=>document.createElementNS(SVG_NS,n);
  const add=(el)=>svg.appendChild(el);
  const txt=(x0,y0,s,anchor,color)=>{ const t=mk("text"); t.setAttribute("x",x0); t.setAttribute("y",y0);
    t.setAttribute("fill",color||"#9aa0ab"); t.setAttribute("font-size","9");
    if(anchor)t.setAttribute("text-anchor",anchor); t.textContent=s; add(t); };
  // y-axis: gridlines + tick labels with units (0, ½, max)
  [0,0.5,1].forEach(f=>{ const yy=y(vmax*f);
    const gl=mk("line"); gl.setAttribute("x1",pad.l); gl.setAttribute("x2",W-pad.r);
    gl.setAttribute("y1",yy); gl.setAttribute("y2",yy);
    gl.setAttribute("stroke", f===0?"#3a3d46":"#2b2e36"); add(gl);
    txt(pad.l-5, yy+3, unitFmt(vmax*f), "end");
  });
  // series
  keys.forEach((k,ki)=>{ let d="",pen=false;
    series[k].forEach((v,i)=>{ if(v==null){pen=false;return;} d+=(pen?" L":" M")+x(i).toFixed(1)+" "+y(v).toFixed(1); pen=true; });
    const p=mk("path"); p.setAttribute("d",d.trim()); p.setAttribute("fill","none");
    p.setAttribute("stroke",colors[ki]); p.setAttribute("stroke-width","1.6"); add(p);
  });
  // x-axis time labels
  const d0=new Date(ts[0]*1000), d1=new Date(ts[ts.length-1]*1000), p=(n)=>String(n).padStart(2,"0");
  const tstr=(d)=>`${p(d.getMonth()+1)}/${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
  txt(pad.l, H-6, tstr(d0)); txt(W-pad.r, H-6, tstr(d1), "end");
}

// ---- render ----
function renderFrame(frame){
  curFrame=frame;
  renderLinks(); renderNodes();
  const nl=Object.keys(frame.links).length, nn=Object.keys(frame.nodes).length;
  matchedEl.textContent=`${nl} links · ${nn} nodes`;
}
function renderLinks(){
  const M=METRICS[state.linkMetric];
  for(const id in linkEls){
    const el=linkEls[id], link=linkById[id], s=curFrame&&curFrame.links[id];
    if(!state.linksOn){ el.style.display="none"; continue; }
    el.style.display="";
    if(s){
      const load=Math.max(s.in||0,s.out||0);
      const val = state.linkMetric==="util" ? (link.capacity_bps? load/(link.capacity_bps/1e6)*100 : 0) : load;
      el.classList.remove("nodata");
      el.setAttribute("stroke", ramp(M.norm(val)));
    } else {
      el.classList.add("nodata"); el.removeAttribute("stroke");
    }
  }
}
function renderNodes(){
  const M=METRICS[state.nodeMetric];
  for(const code in nodeEls){
    const el=nodeEls[code], s=curFrame&&curFrame.nodes[code];
    const v = s ? s[state.nodeMetric] : null;
    if(!state.nodesOn){ el.classList.add("hidden"); continue; }
    el.classList.remove("hidden");
    if(v!=null){ el.setAttribute("fill", ramp(M.norm(v))); el.setAttribute("fill-opacity","0.95"); }
    else { el.setAttribute("fill","#6b7280"); el.setAttribute("fill-opacity","0.5"); }
  }
}
function updateLegend(){
  const grad = `linear-gradient(90deg, ${ramp(0)}, ${ramp(0.4)}, ${ramp(0.7)}, ${ramp(1)})`;
  const L=METRICS[state.linkMetric], N=METRICS[state.nodeMetric];
  $("legend-link").style.display = state.linksOn ? "" : "none";
  $("legend-node").style.display = state.nodesOn ? "" : "none";
  $("legend-link-title").textContent = L.title;
  $("legend-link-min").textContent = L.min; $("legend-link-max").textContent = L.max;
  $("legend-bar-link").style.background = grad;
  $("legend-node-title").textContent = N.title;
  $("legend-node-min").textContent = N.min; $("legend-node-max").textContent = N.max;
  $("legend-bar-node").style.background = grad;
}

// ---- time ----
function fmtTime(ts){ const d=new Date(ts*1000),p=(n)=>String(n).padStart(2,"0");
  return `${d.getFullYear()}-${p(d.getMonth()+1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`; }
function toInputValue(ts){ const d=new Date(ts*1000),p=(n)=>String(n).padStart(2,"0");
  return `${d.getFullYear()}-${p(d.getMonth()+1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`; }

let fetchSeq=0, fetchTimer=null;
function fetchAt(ts, immediate=false){
  timeLabel.textContent=fmtTime(ts);
  const run=async()=>{ const seq=++fetchSeq;
    try{ const f=await getJSON(`/api/agg?t=${ts}&w=${state.window}&agg=${state.agg}`);
      if(seq!==fetchSeq) return; renderFrame(f);
    }catch(_){} };
  clearTimeout(fetchTimer);
  if(immediate) run(); else fetchTimer=setTimeout(run,90);
}
function currentTs(){ return parseInt(slider.value,10); }
function setTs(ts, immediate=false){
  ts=Math.max(meta.time_span.first,Math.min(meta.time_span.last,ts));
  slider.value=ts; timeInput.value=toInputValue(ts); fetchAt(ts,immediate); refreshInspector(immediate);
}
slider.addEventListener("input",()=>setTs(currentTs()));
timeInput.addEventListener("change",()=>{ const ts=Math.floor(new Date(timeInput.value).getTime()/1000);
  if(!isNaN(ts)) setTs(ts,true); });
playBtn.onclick=()=>{ playing=!playing; playBtn.textContent=playing?"⏸":"▶";
  if(playing){ playTimer=setInterval(()=>{ const step = state.window>300 ? state.window : 3600;
      let n=currentTs()+step; if(n>meta.time_span.last) n=meta.time_span.first; setTs(n,true); },700);
  } else clearInterval(playTimer);
};
windowSel.onchange=(e)=>{ state.window=parseInt(e.target.value,10);
  aggSel.disabled = state.window<=300; refreshInspector(); setTs(currentTs(),true); };
aggSel.onchange=(e)=>{ state.agg=e.target.value; refreshInspector(); setTs(currentTs(),true); };

// ---- layer controls ----
$("links-on").onchange=(e)=>{ state.linksOn=e.target.checked; renderLinks(); updateLegend(); };
$("link-metric").onchange=(e)=>{ state.linkMetric=e.target.value; renderLinks(); updateLegend(); };
$("nodes-on").onchange=(e)=>{ state.nodesOn=e.target.checked; renderNodes(); updateLegend(); };
$("node-metric").onchange=(e)=>{ state.nodeMetric=e.target.value; renderNodes(); updateLegend(); refreshInspector(true); };

function drawCapBars(){
  const caps=[1e9,1e10,1e11,2e11], box=$("capbars");
  box.innerHTML="";
  caps.forEach(c=>{
    const col=document.createElement("div");
    col.style.cssText="display:flex;flex-direction:column;align-items:center;gap:3px;";
    const bar=document.createElement("i");
    bar.style.cssText=`width:28px;height:${capWidth(c).toFixed(1)}px;background:#8892a0;border-radius:3px;`;
    const lab=document.createElement("span");
    lab.style.cssText="font-size:10px;color:var(--muted);"; lab.textContent=fmtCap(c);
    col.appendChild(bar); col.appendChild(lab); box.appendChild(col);
  });
}

// ---- bootstrap ----
async function init(){
  const svgText=await (await fetch("assets/switchlan.svg",{cache:"no-store"})).text();
  paper.innerHTML=svgText; mapSvg=paper.querySelector("svg");
  mapSvg.removeAttribute("width"); mapSvg.removeAttribute("height");
  const [,,w,h]=mapSvg.getAttribute("viewBox").split(/\s+/).map(Number);
  mapSvg.style.width=w+"px"; mapSvg.style.height=h+"px";

  meta=await getJSON("/api/meta");
  const drawable=meta.links.filter(l=>meta.nodes[l.endpoints[0]]&&meta.nodes[l.endpoints[1]]).length;
  subtitle.textContent=`${drawable}/${meta.links.length} links · ${fmtTime(meta.time_span.first)} → ${fmtTime(meta.time_span.last)} · ${meta.interval_seconds/60} min · Cricket dataset`;

  buildOverlay(); drawCapBars();

  // Optional URL params (?links=0/1&link=util|traffic&nodes=0/1&node=cpu|pw|temp&t=unix)
  const q=new URLSearchParams(location.search);
  if(q.has("links")){ state.linksOn=q.get("links")!=="0"; $("links-on").checked=state.linksOn; }
  if(q.has("link")){ state.linkMetric=q.get("link"); $("link-metric").value=state.linkMetric; }
  if(q.has("nodes")){ state.nodesOn=q.get("nodes")==="1"; $("nodes-on").checked=state.nodesOn; }
  if(q.has("node")){ state.nodeMetric=q.get("node"); $("node-metric").value=state.nodeMetric; }
  if(q.has("w")){ state.window=parseInt(q.get("w"),10); windowSel.value=String(state.window); }
  if(q.has("agg")){ state.agg=q.get("agg"); aggSel.value=state.agg; }
  aggSel.disabled = state.window<=300;
  updateLegend();

  slider.min=meta.time_span.first; slider.max=meta.time_span.last; slider.step=meta.interval_seconds;
  loading.remove(); fit();

  // Start at a time that has data (coverage begins ~end of Dec 2023).
  const mid = Math.floor((meta.time_span.first + meta.time_span.last) / 2);
  const start = q.has("t") ? parseInt(q.get("t"),10) : Math.min(mid, 1718452800); // 2024-06-15 12:00 UTC
  setTs(start, true);

  // deep-link to an inspector (share a link/node view): ?inspect=BA-EZ or ?inspect=AG
  if(q.has("inspect")){ const id=q.get("inspect");
    pinTarget(linkById[id] ? "link" : "node", id); }
}
init().catch(err=>{ loading.textContent="Error: "+err.message; });
