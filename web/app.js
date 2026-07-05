// SWITCHlan Traffic Map — Step 3: interactive traffic overlay.
//
// Loads the vector backbone map, overlays one line per backbone link using the
// node coordinates extracted from the PDF, and colours/sizes each line by the
// traffic at a user-chosen time. Traffic is fetched on demand from the backend
// (server.py), which reads the monthly usage logs lazily.

const SVG_NS = "http://www.w3.org/2000/svg";

const stage = document.getElementById("stage");
const viewport = document.getElementById("viewport");
const paper = document.getElementById("paper");
const loading = document.getElementById("loading");
const tooltip = document.getElementById("tooltip");
const subtitle = document.getElementById("subtitle");

const slider = document.getElementById("time-slider");
const timeInput = document.getElementById("time-input");
const timeLabel = document.getElementById("time-label");
const matchedEl = document.getElementById("matched");
const playBtn = document.getElementById("play");
const stepSel = document.getElementById("step-size");
const legendMax = document.getElementById("legend-max");

const view = { scale: 1, x: 0, y: 0, min: 0.05, max: 12 };
let meta = null, nodes = null;
let mapSvg = null;
let linkEls = {};          // link id -> <line>
let curFrame = null;       // last traffic frame
let playing = false, playTimer = null;

// Colour scale: max(in,out) Mbps -> colour. Fixed reference at 10 Gbps so
// colours are comparable across times. Backbone links are mostly 10G/100G.
const SCALE_MAX_MBPS = 10000;
legendMax.textContent = (SCALE_MAX_MBPS / 1000) + " Gbps";

function loadColor(mbps) {
  // 0..1 across green -> yellow -> orange -> red, log-ish so low loads show.
  const f = Math.max(0, Math.min(1, Math.log10(1 + mbps) / Math.log10(1 + SCALE_MAX_MBPS)));
  const stops = [
    [0.0, [51, 187, 102]],
    [0.4, [255, 221, 0]],
    [0.7, [255, 153, 51]],
    [1.0, [226, 34, 34]],
  ];
  for (let i = 1; i < stops.length; i++) {
    if (f <= stops[i][0]) {
      const [f0, c0] = stops[i - 1], [f1, c1] = stops[i];
      const k = (f - f0) / (f1 - f0);
      const c = c0.map((v, j) => Math.round(v + (c1[j] - v) * k));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
  }
  return "rgb(226,34,34)";
}

function loadWidth(mbps) {
  const f = Math.max(0, Math.min(1, Math.log10(1 + mbps) / Math.log10(1 + SCALE_MAX_MBPS)));
  return 2.5 + f * 9;
}

// ---------- pan / zoom ----------
function applyTransform() {
  viewport.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale})`;
}
function fit() {
  const r = stage.getBoundingClientRect();
  const pw = paper.offsetWidth, ph = paper.offsetHeight;
  if (!pw || !ph) return;
  const m = 40;
  const s = Math.min((r.width - m) / pw, (r.height - m) / ph);
  view.scale = Math.max(view.min, Math.min(view.max, s));
  view.x = (r.width - pw * view.scale) / 2;
  view.y = (r.height - ph * view.scale) / 2;
  applyTransform();
}
function zoomAt(fx, fy, factor) {
  const next = Math.max(view.min, Math.min(view.max, view.scale * factor));
  const k = next / view.scale;
  view.x = fx - (fx - view.x) * k;
  view.y = fy - (fy - view.y) * k;
  view.scale = next;
  applyTransform();
}
stage.addEventListener("wheel", (e) => {
  e.preventDefault();
  const r = stage.getBoundingClientRect();
  zoomAt(e.clientX - r.left, e.clientY - r.top, e.deltaY < 0 ? 1.12 : 1 / 1.12);
}, { passive: false });

let dragging = false, last = { x: 0, y: 0 }, moved = false;
stage.addEventListener("pointerdown", (e) => {
  dragging = true; moved = false; last = { x: e.clientX, y: e.clientY };
  stage.classList.add("panning"); stage.setPointerCapture(e.pointerId);
});
stage.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  view.x += e.clientX - last.x; view.y += e.clientY - last.y;
  last = { x: e.clientX, y: e.clientY }; moved = true; applyTransform();
});
function endDrag(e) {
  dragging = false; stage.classList.remove("panning");
  try { stage.releasePointerCapture(e.pointerId); } catch (_) {}
}
stage.addEventListener("pointerup", endDrag);
stage.addEventListener("pointercancel", endDrag);

document.getElementById("zoom-in").onclick = () => { const r = stage.getBoundingClientRect(); zoomAt(r.width/2, r.height/2, 1.25); };
document.getElementById("zoom-out").onclick = () => { const r = stage.getBoundingClientRect(); zoomAt(r.width/2, r.height/2, 1/1.25); };
document.getElementById("zoom-reset").onclick = fit;
window.addEventListener("resize", fit);

// ---------- build overlay ----------
function buildOverlay() {
  const g = document.createElementNS(SVG_NS, "g");
  g.setAttribute("id", "traffic-overlay");
  for (const link of meta.links) {
    const [a, b] = link.endpoints;
    const pa = nodes[a], pb = nodes[b];
    if (!pa || !pb) continue;          // unmapped endpoint (CHU/GR/SLF)
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", pa.x); line.setAttribute("y1", pa.y);
    line.setAttribute("x2", pb.x); line.setAttribute("y2", pb.y);
    line.setAttribute("class", "traffic-link nodata");
    line.setAttribute("stroke-width", "3");
    line.dataset.id = link.id;
    line.addEventListener("pointerenter", (e) => showTip(link, e));
    line.addEventListener("pointermove", (e) => positionTip(e));
    line.addEventListener("pointerleave", hideTip);
    g.appendChild(line);
    linkEls[link.id] = line;
  }
  mapSvg.appendChild(g);
}

function showTip(link, e) {
  const s = curFrame && curFrame.links[link.id];
  const [a, b] = link.endpoints;
  let html = `<b>${a} ↔ ${b}</b>${link.instance ? " #" + link.instance : ""}`;
  if (s) {
    html += `<br>in: ${fmtMbps(s.in)} &nbsp; out: ${fmtMbps(s.out)}`;
  } else {
    html += `<br>no data at this time`;
  }
  tooltip.innerHTML = html;
  tooltip.style.display = "block";
  positionTip(e);
}
function positionTip(e) {
  const r = stage.getBoundingClientRect();
  tooltip.style.left = (e.clientX - r.left + 14) + "px";
  tooltip.style.top = (e.clientY - r.top + 14) + "px";
}
function hideTip() { tooltip.style.display = "none"; }

function fmtMbps(v) {
  if (v >= 1000) return (v / 1000).toFixed(2) + " Gbps";
  return v.toFixed(1) + " Mbps";
}

// ---------- render a frame ----------
function renderFrame(frame) {
  curFrame = frame;
  for (const id in linkEls) {
    const el = linkEls[id];
    const s = frame.links[id];
    if (s) {
      const load = Math.max(s.in, s.out);
      el.classList.remove("nodata");
      el.setAttribute("stroke", loadColor(load));
      el.setAttribute("stroke-width", loadWidth(load));
    } else {
      el.classList.add("nodata");
      el.removeAttribute("stroke");
      el.setAttribute("stroke-width", "3");
    }
  }
  matchedEl.textContent = `${frame.matched}/${Object.keys(linkEls).length} links`;
}

// ---------- time control ----------
function fmtTime(ts) {
  const d = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth()+1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}
function toInputValue(ts) {
  const d = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth()+1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
}

let fetchSeq = 0;
let fetchTimer = null;
async function fetchAt(ts, immediate = false) {
  timeLabel.textContent = fmtTime(ts);
  const run = async () => {
    const seq = ++fetchSeq;
    try {
      const res = await fetch(`/api/traffic?t=${ts}`);
      const frame = await res.json();
      if (seq !== fetchSeq) return;      // a newer request superseded this one
      renderFrame(frame);
    } catch (err) { /* keep previous frame */ }
  };
  clearTimeout(fetchTimer);
  if (immediate) run(); else fetchTimer = setTimeout(run, 90);
}

function currentTs() { return parseInt(slider.value, 10); }
function setTs(ts, immediate = false) {
  ts = Math.max(meta.time_span.first, Math.min(meta.time_span.last, ts));
  slider.value = ts;
  timeInput.value = toInputValue(ts);
  fetchAt(ts, immediate);
}

slider.addEventListener("input", () => setTs(currentTs()));
timeInput.addEventListener("change", () => {
  const ts = Math.floor(new Date(timeInput.value).getTime() / 1000);
  if (!isNaN(ts)) setTs(ts, true);
});

playBtn.onclick = () => {
  playing = !playing;
  playBtn.textContent = playing ? "⏸" : "▶";
  if (playing) {
    playTimer = setInterval(() => {
      const step = parseInt(stepSel.value, 10);
      let next = currentTs() + step;
      if (next > meta.time_span.last) next = meta.time_span.first;
      setTs(next, true);
    }, 700);
  } else {
    clearInterval(playTimer);
  }
};

// ---------- bootstrap ----------
async function init() {
  // map
  const svgText = await (await fetch("assets/switchlan.svg")).text();
  paper.innerHTML = svgText;
  mapSvg = paper.querySelector("svg");
  mapSvg.removeAttribute("width");
  mapSvg.removeAttribute("height");
  const [, , w, h] = mapSvg.getAttribute("viewBox").split(/\s+/).map(Number);
  mapSvg.style.width = w + "px";
  mapSvg.style.height = h + "px";

  // data
  [meta, nodes] = await Promise.all([
    fetch("/api/meta").then(r => r.json()),
    fetch("/api/nodes").then(r => r.json()),
  ]);

  const drawable = meta.links.filter(l => nodes[l.endpoints[0]] && nodes[l.endpoints[1]]).length;
  subtitle.textContent = `${drawable}/${meta.links.length} links · ${fmtTime(meta.time_span.first)} → ${fmtTime(meta.time_span.last)} · ${meta.interval_seconds/60} min samples`;

  buildOverlay();

  slider.min = meta.time_span.first;
  slider.max = meta.time_span.last;
  slider.step = meta.interval_seconds;

  loading.remove();
  fit();

  // start at a populated time: 12:00 on the first full day of coverage
  const start = meta.time_span.first + 12 * 3600;
  setTs(start, true);
}

init().catch(err => { loading.textContent = "Error: " + err.message; });
