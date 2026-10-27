/* Generative vs. Discriminative Model Lab — frontend (no external libraries) */
const $ = (s) => document.querySelector(s);
const api = async (url, body) => {
  const r = await fetch(url, body ? {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)} : {});
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "Request failed");
  return j;
};
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), 3200); }

/* ---------------------------------------------------------------- tabs */
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "report") loadReport();
  if (b.dataset.tab === "studio" && !studioReady) initStudio();
}));

/* ---------------------------------------------------------------- canvas */
const pad = $("#pad"), ctx = pad.getContext("2d");
let strokes = [], cur = null, drawing = false, uploaded = null;
function resetPad() { ctx.fillStyle = "#000"; ctx.fillRect(0, 0, pad.width, pad.height); }
function redraw() {
  resetPad();
  ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.strokeStyle = "#fff";
  for (const s of strokes) {
    ctx.lineWidth = s.w; ctx.beginPath();
    s.pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
    if (s.pts.length === 1) ctx.lineTo(s.pts[0][0] + 0.1, s.pts[0][1]);
    ctx.stroke();
  }
  $("#padHint").style.display = strokes.length || uploaded ? "none" : "grid";
}
function pos(e) {
  const r = pad.getBoundingClientRect(), t = e.touches ? e.touches[0] : e;
  return [(t.clientX - r.left) * pad.width / r.width, (t.clientY - r.top) * pad.height / r.height];
}
function start(e) {
  e.preventDefault();
  if (uploaded) { uploaded = null; strokes = []; }
  drawing = true; cur = {w: +$("#brush").value, pts: [pos(e)]}; strokes.push(cur); redraw();
}
function move(e) { if (!drawing) return; e.preventDefault(); cur.pts.push(pos(e)); redraw(); }
function end() { if (!drawing) return; drawing = false; if ($("#auto").checked) analyze(); }
pad.addEventListener("mousedown", start); pad.addEventListener("mousemove", move); window.addEventListener("mouseup", end);
pad.addEventListener("touchstart", start, {passive: false}); pad.addEventListener("touchmove", move, {passive: false}); pad.addEventListener("touchend", end);
$("#btnClear").onclick = () => { strokes = []; uploaded = null; redraw(); };
$("#btnUndo").onclick = () => { if (uploaded) { uploaded = null; } else strokes.pop(); redraw(); if (strokes.length && $("#auto").checked) analyze(); };
$("#btnAnalyze").onclick = () => analyze();

function showImageOnPad(src, after) {
  const img = new Image();
  img.onload = () => {
    resetPad();
    const s = Math.min(pad.width / img.width, pad.height / img.height);
    const w = img.width * s, h = img.height * s;
    ctx.drawImage(img, (pad.width - w) / 2, (pad.height - h) / 2, w, h);
    $("#padHint").style.display = "none";
    after && after();
  };
  img.src = src;
}
$("#file").addEventListener("change", (e) => {
  const f = e.target.files[0]; if (!f) return;
  const rd = new FileReader();
  rd.onload = () => { strokes = []; uploaded = rd.result; showImageOnPad(uploaded, () => analyze()); };
  rd.readAsDataURL(f); e.target.value = "";
});

/* ---------------------------------------------------------------- samples */
api("/api/samples").then((d) => {
  const box = $("#samples");
  // show one of each digit first row, then the rest
  const order = [];
  for (let k = 0; k < 5; k++) for (let c = 0; c < 10; c++) order.push(c * 5 + k);
  order.forEach((i) => {
    const s = d.samples[i], im = document.createElement("img");
    im.src = s.image; im.title = "MNIST test digit (true label " + s.label + ")"; im.alt = "sample " + s.label;
    im.onclick = () => { strokes = []; uploaded = s.raw; showImageOnPad(s.raw, () => analyze(s.label)); };
    box.appendChild(im);
  });
}).catch(() => toast("Could not load sample digits"));

/* ---------------------------------------------------------------- render helpers */
function bars(el, probs) {
  const top = probs.indexOf(Math.max(...probs));
  el.innerHTML = probs.map((p, i) => `
    <div class="bar ${i === top ? "best" : ""}">
      <span class="lab">${i}</span>
      <div class="track"><div class="fill" style="width:${Math.max(p * 100, 0.4)}%"></div></div>
      <span class="val">${(p * 100).toFixed(p >= 0.995 || p < 0.001 ? 1 : 1)}%</span>
    </div>`).join("");
}
const pct = (p) => (p * 100).toFixed(p > 0.999 ? 2 : 1) + "%";

/* ---------------------------------------------------------------- analyze */
let busy = false, lastPred = null, trueLabel = null;
async function analyze(label) {
  if (busy) return;
  if (!strokes.length && !uploaded) { toast("Draw or upload a digit first"); return; }
  trueLabel = typeof label === "number" ? label : null;
  busy = true; $("#btnAnalyze").disabled = true; $("#btnAnalyze").textContent = "Analyzing…";
  try {
    // uploaded file / sample -> send original pixels; drawing -> send the canvas
    const r = await api("/api/analyze", {image: uploaded || pad.toDataURL("image/png")});
    render(r);
  } catch (e) { toast(e.message); }
  finally { busy = false; $("#btnAnalyze").disabled = false; $("#btnAnalyze").textContent = "Analyze digit"; }
}

function render(r) {
  $("#emptyState").classList.add("hidden");
  $("#resultsBody").classList.remove("hidden");
  $("#seen28").src = r.input28; $("#inSmall").src = r.input28;
  $("#invNote").textContent = r.inverted ? "Image was inverted (dark ink on light background detected)." : "";

  const c = r.classifier, g = r.generative;
  $("#dPred").textContent = c.prediction; $("#dConf").textContent = pct(c.confidence);
  $("#dTime").textContent = `1 forward pass · ${c.inference_ms} ms`;
  bars($("#dBars"), c.probabilities);
  $("#fmaps").innerHTML = c.feature_maps.map((s) => `<img src="${s}" alt="feature map">`).join("");

  $("#gPred").textContent = g.prediction; $("#gConf").textContent = pct(g.confidence);
  $("#gTime").textContent = `10 encode/decode passes · ${g.inference_ms} ms`;
  bars($("#gBars"), g.probabilities);
  $("#typPct").textContent = `${g.typicality_percentile} percentile`;
  $("#typBar").style.left = `calc(${Math.min(Math.max(g.typicality_percentile, 0), 100)}% - 2px)`;
  $("#typLabel").textContent = g.typicality_label;

  $("#recon").src = g.reconstruction; $("#reconL").textContent = `using label ${g.recon_label}`;
  $("#sampD").textContent = c.prediction;
  $("#gSamples").innerHTML = g.samples.map((s) => `<img src="${s}" alt="generated">`).join("");

  const tl = trueLabel !== null ? ` True label: <b>${trueLabel}</b>.` : "";
  const a = $("#agree");
  if (r.agree) {
    a.className = "agree yes";
    a.innerHTML = `✓ Both models say <b>${c.prediction}</b>. The classifier is ${pct(c.confidence)} sure; the generative model agrees through Bayes' rule.${tl}`;
  } else {
    a.className = "agree no";
    a.innerHTML = `⚠ The models disagree — classifier says <b>${c.prediction}</b>, generative model says <b>${g.prediction}</b>. This usually means the digit is ambiguous or unusual.${tl}`;
  }
  if (g.typicality_percentile < 1) { a.className = "agree no"; a.innerHTML += ` <br><span class="small">Note: the generative model rates this image as very unlikely to be a real digit, yet the classifier still reports ${pct(c.confidence)} confidence — a classic weakness of purely discriminative models. (The generative posterior is <i>relative</i> between digits; the typicality score is the <i>absolute</i> check.)</span>`; }

  lastPred = c.prediction;
  loadLatent(c.prediction, g.latent);
}

async function loadLatent(d, z) {
  try {
    const r = await api("/api/latent_grid?digit=" + d);
    $("#lgrid").src = r.image;
    const span = r.span, n = 10, step = (2 * span) / (n - 1), clamp = (v) => Math.max(0.02, Math.min(0.98, v));
    const fx = clamp((0.5 + (z[0] + span) / step) / n), fy = clamp((0.5 + (span - z[1]) / step) / n);
    $("#ldot").style.left = fx * 100 + "%"; $("#ldot").style.top = fy * 100 + "%";
    $("#lcoord").textContent = `z = (${z[0].toFixed(2)}, ${z[1].toFixed(2)})`;
  } catch (e) { /* non-critical */ }
}

$("#btnMore").onclick = async () => {
  if (lastPred === null) return;
  const r = await api("/api/generate", {digit: lastPred, n: 8});
  $("#gSamples").innerHTML = r.images.map((s) => `<img src="${s}" alt="generated">`).join("");
};

/* ---------------------------------------------------------------- studio */
let studioReady = false, studioDigit = 3;
function initStudio() {
  studioReady = true;
  const box = $("#digitPick");
  for (let i = 0; i < 10; i++) {
    const b = document.createElement("button"); b.textContent = i; b.className = i === studioDigit ? "on" : "";
    b.onclick = () => { studioDigit = i; box.querySelectorAll("button").forEach((x, j) => x.classList.toggle("on", j === i)); generate(); };
    box.appendChild(b);
  }
  $("#nSamp").oninput = (e) => ($("#nVal").textContent = e.target.value);
  $("#temp").oninput = (e) => ($("#tVal").textContent = (+e.target.value).toFixed(1));
  $("#btnGen").onclick = generate;
  generate();
}
async function generate() {
  const body = {digit: studioDigit, n: +$("#nSamp").value, temperature: +$("#temp").value, seed: $("#seed").value};
  try {
    const r = await api("/api/generate", body);
    $("#genGrid").innerHTML = r.images.map((s, i) => {
      const ck = r.checks[i], ok = ck.pred === r.digit;
      return `<div class="gcell"><img src="${s}" alt="generated ${r.digit}"><span class="chip ${ok ? "ok" : "bad"}">${ok ? "✓" : "✗ read as " + ck.pred} · ${(ck.conf * 100).toFixed(0)}%</span></div>`;
    }).join("");
    $("#agreeRate").textContent = pct(r.classifier_agreement);
    $("#genTime").textContent = `${r.images.length} samples in ${r.inference_ms} ms · T = ${r.temperature}`;
    const lg = await api("/api/latent_grid?digit=" + r.digit);
    $("#lmap").src = lg.image; $("#lmD").textContent = r.digit;
  } catch (e) { toast(e.message); }
}

/* ---------------------------------------------------------------- report */
let reportReady = false;
function lineChart(el, xs, series, opts = {}) {
  const W = 520, H = 250, L = 46, R = opts.right ? 46 : 16, T = 16, B = 34;
  const iw = W - L - R, ih = H - T - B;
  const sx = (i) => L + (xs.length === 1 ? iw / 2 : (i / (xs.length - 1)) * iw);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
  const axisFor = (s) => {
    const all = opts.shared ? series.flatMap((q) => q.data) : s.data; let lo = Math.min(...all), hi = Math.max(...all);
    if (s.range) [lo, hi] = s.range; const pad = (hi - lo) * 0.08 || 1; return [lo - (s.range ? 0 : pad), hi + (s.range ? 0 : pad)];
  };
  series.forEach((s, k) => {
    const [lo, hi] = axisFor(s), sy = (v) => T + ih - ((v - lo) / (hi - lo)) * ih;
    if (k === 0 || (k === 1 && opts.right && !opts.shared)) {
      for (let t = 0; t <= 4; t++) {
        const v = lo + ((hi - lo) * t) / 4, y = sy(v);
        if (k === 0) svg += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="#eceef2"/>`;
        svg += `<text x="${k === 0 ? L - 6 : W - R + 6}" y="${y + 4}" text-anchor="${k === 0 ? "end" : "start"}">${s.fmt ? s.fmt(v) : v.toFixed(2)}</text>`;
      }
    }
    const pts = s.data.map((v, i) => `${sx(i)},${sy(v)}`).join(" ");
    svg += `<polyline fill="none" stroke="${s.color}" stroke-width="2.5" points="${pts}" ${s.dash ? 'stroke-dasharray="5 4"' : ""}/>`;
    s.data.forEach((v, i) => { if (xs.length <= 12) svg += `<circle cx="${sx(i)}" cy="${sy(v)}" r="3.5" fill="${s.color}"/>`; });
  });
  const every = Math.ceil(xs.length / 10);
  xs.forEach((x, i) => { if (i % every === 0 || i === xs.length - 1) svg += `<text x="${sx(i)}" y="${H - 14}" text-anchor="middle">${x}</text>`; });
  svg += `<text x="${L + iw / 2}" y="${H - 1}" text-anchor="middle">epoch</text>`;
  let lx = L + 8;
  series.forEach((s) => { svg += `<rect x="${lx}" y="${T}" width="12" height="4" fill="${s.color}" rx="2"/><text x="${lx + 16}" y="${T + 6}" style="fill:#151821">${s.name}</text>`; lx += 24 + s.name.length * 6.4; });
  el.innerHTML = svg + "</svg>";
}
async function loadReport() {
  if (reportReady) return; reportReady = true;
  const m = await api("/api/metrics");
  const c = m.classifier, v = m.cvae;
  $("#kpis").innerHTML = [
    ["d", "Classifier test accuracy", pct(c.test_accuracy)],
    ["g", "Generative (Bayes-rule) accuracy", pct(v.generative_classification_accuracy)],
    ["d", "Classifier parameters", c.params.toLocaleString()],
    ["g", "CVAE parameters", v.params.toLocaleString()],
    ["", "MNIST train / test images", `${m.dataset.train / 1000}k / ${m.dataset.test / 1000}k`],
  ].map(([k, a, b]) => `<div class="kpi ${k}"><div class="k">${a}</div><div class="v">${b}</div></div>`).join("");

  const ep = c.history.test_acc.map((_, i) => i + 1);
  lineChart($("#chClf"), ep, [
    {name: "train loss", data: c.history.train_loss, color: "#9aa5bd", dash: true},
    {name: "test accuracy", data: c.history.test_acc, color: "#2f5bea", fmt: (x) => (x * 100).toFixed(1) + "%"},
  ], {right: true});
  const ev = v.history.loss.map((_, i) => i + 1);
  lineChart($("#chVae"), ev, [
    {name: "−ELBO (total)", data: v.history.loss, color: "#d9480f"},
    {name: "reconstruction term", data: v.history.recon, color: "#f59f73", dash: true},
  ], {shared: true});

  const cm = c.confusion_matrix; let html = `<div class="h"></div>`;
  for (let j = 0; j < 10; j++) html += `<div class="h">${j}</div>`;
  cm.forEach((row, i) => {
    const tot = row.reduce((a, b) => a + b, 0);
    html += `<div class="h">${i}</div>`;
    row.forEach((n, j) => {
      const f = n / tot, diag = i === j;
      const bg = diag ? `rgba(47,91,234,${0.15 + 0.85 * f})` : n ? `rgba(217,72,15,${Math.min(0.15 + n / 25, 0.9)})` : "#f6f7f9";
      html += `<div title="true ${i}, predicted ${j}: ${n}" style="background:${bg};color:${diag && f > 0.5 ? "#fff" : "#151821"}">${n || ""}</div>`;
    });
  });
  $("#cm").innerHTML = html + "";
  $("#cm").insertAdjacentHTML("afterend", `<div class="cm-axis"><span>rows = true digit</span><span>columns = predicted digit</span></div>`);

  $("#h2h").innerHTML = `<thead><tr><th></th><th class="c-d">Classifier (CNN)</th><th class="c-g">CVAE</th></tr></thead><tbody>
    <tr><td>Learns</td><td>p(y | x)</td><td>p(x | y)</td></tr>
    <tr><td>Accuracy on test set</td><td>${pct(c.test_accuracy)}</td><td>${pct(v.generative_classification_accuracy)}</td></tr>
    <tr><td>Parameters</td><td>${c.params.toLocaleString()}</td><td>${v.params.toLocaleString()}</td></tr>
    <tr><td>Epochs</td><td>${c.epochs}</td><td>${v.epochs}</td></tr>
    <tr><td>Can generate digits</td><td>No</td><td>Yes</td></tr>
    <tr><td>Median test ELBO</td><td>—</td><td>${v.test_elbo_percentiles["50"]} nats</td></tr></tbody>`;
}

redraw();
