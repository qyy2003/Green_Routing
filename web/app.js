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
const matchedEl = $("matched"), playBtn = $("play"), stepSel = $("step-size");

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
};

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
let dragging=false,last={x:0,y:0};
stage.addEventListener("pointerdown",(e)=>{ if(e.target.closest(".panel")) return;
  dragging=true; last={x:e.clientX,y:e.clientY}; stage.classList.add("panning"); stage.setPointerCapture(e.pointerId); });
stage.addEventListener("pointermove",(e)=>{ if(!dragging) return;
  view.x+=e.clientX-last.x; view.y+=e.clientY-last.y; last={x:e.clientX,y:e.clientY}; applyTransform(); });
function endDrag(e){ dragging=false; stage.classList.remove("panning"); try{stage.releasePointerCapture(e.pointerId);}catch(_){} }
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
    line.addEventListener("pointerenter",(e)=>showLinkTip(link,e));
    line.addEventListener("pointermove",positionTip);
    line.addEventListener("pointerleave",hideTip);
    g.appendChild(line); linkEls[link.id]=line; linkById[link.id]=link;
  }
  // node markers on top
  for(const code in meta.nodes){
    const p=meta.nodes[code];
    const c=document.createElementNS(SVG_NS,"circle");
    c.setAttribute("cx",p.x); c.setAttribute("cy",p.y); c.setAttribute("r",11);
    c.setAttribute("class","node-marker hidden");
    c.dataset.code=code;
    c.addEventListener("pointerenter",(e)=>showNodeTip(code,e));
    c.addEventListener("pointermove",positionTip);
    c.addEventListener("pointerleave",hideTip);
    g.appendChild(c); nodeEls[code]=c;
  }
  mapSvg.appendChild(g);
}

// ---- tooltip ----
function showLinkTip(link,e){
  const s=curFrame&&curFrame.links[link.id], [a,b]=link.endpoints;
  let h=`<b>${a} ↔ ${b}</b> · cap ${fmtCap(link.capacity_bps)}`;
  if(s){
    const load=Math.max(s.in||0,s.out||0), util=link.capacity_bps? load/(link.capacity_bps/1e6)*100:0;
    h+=`<br>in ${fmtMbps(s.in||0)} · out ${fmtMbps(s.out||0)}<br>utilization ${util.toFixed(1)}%`;
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
  } else h+=`<br>no data at this time`;
  tooltip.innerHTML=h; tooltip.style.display="block"; positionTip(e);
}
function positionTip(e){ const r=stage.getBoundingClientRect();
  tooltip.style.left=(e.clientX-r.left+14)+"px"; tooltip.style.top=(e.clientY-r.top+14)+"px"; }
function hideTip(){ tooltip.style.display="none"; }
function fmtCap(bps){ return bps>=1e9? (bps/1e9)+"G" : (bps/1e6)+"M"; }

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
    try{ const f=await getJSON(`/api/frame?t=${ts}`);
      if(seq!==fetchSeq) return; renderFrame(f);
    }catch(_){} };
  clearTimeout(fetchTimer);
  if(immediate) run(); else fetchTimer=setTimeout(run,90);
}
function currentTs(){ return parseInt(slider.value,10); }
function setTs(ts, immediate=false){
  ts=Math.max(meta.time_span.first,Math.min(meta.time_span.last,ts));
  slider.value=ts; timeInput.value=toInputValue(ts); fetchAt(ts,immediate);
}
slider.addEventListener("input",()=>setTs(currentTs()));
timeInput.addEventListener("change",()=>{ const ts=Math.floor(new Date(timeInput.value).getTime()/1000);
  if(!isNaN(ts)) setTs(ts,true); });
playBtn.onclick=()=>{ playing=!playing; playBtn.textContent=playing?"⏸":"▶";
  if(playing){ playTimer=setInterval(()=>{ const step=parseInt(stepSel.value,10);
      let n=currentTs()+step; if(n>meta.time_span.last) n=meta.time_span.first; setTs(n,true); },700);
  } else clearInterval(playTimer);
};

// ---- layer controls ----
$("links-on").onchange=(e)=>{ state.linksOn=e.target.checked; renderLinks(); updateLegend(); };
$("link-metric").onchange=(e)=>{ state.linkMetric=e.target.value; renderLinks(); updateLegend(); };
$("nodes-on").onchange=(e)=>{ state.nodesOn=e.target.checked; renderNodes(); updateLegend(); };
$("node-metric").onchange=(e)=>{ state.nodeMetric=e.target.value; renderNodes(); updateLegend(); };

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
  updateLegend();

  slider.min=meta.time_span.first; slider.max=meta.time_span.last; slider.step=meta.interval_seconds;
  loading.remove(); fit();

  // Start at a time that has data (coverage begins ~end of Dec 2023).
  const mid = Math.floor((meta.time_span.first + meta.time_span.last) / 2);
  const start = q.has("t") ? parseInt(q.get("t"),10) : Math.min(mid, 1718452800); // 2024-06-15 12:00 UTC
  setTs(start, true);
}
init().catch(err=>{ loading.textContent="Error: "+err.message; });
