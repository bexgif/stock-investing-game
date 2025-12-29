// ===== STATE =====
let prices = [];
let dates = []; // parallel to prices; each item is a Date or null
let day = 1;
let cash = 10000;
let shares = 0;
let currentPrice = 100;
let viewOffset = 0; // 0 = most recent window; larger moves left (earlier history)

// Full uploaded series (oldest → newest)
let fullPrices = [];
let fullDates  = [];
let cursor = 0;                 // next index in full series to reveal
const DISPLAY_WINDOW = 220;   // how many past days we keep on screen
const SEED_WINDOW    = 50;     // show ONLY the very first row at start

// AI virtual portfolio to compare vs player
let aiCash = 10000;
let aiShares = 0;
const aiStartCapital = 10000;
let aiTradesMade = 0; // for ROI placeholder

// ML stuff
let model;                  // TensorFlow.js model
let replayBuffer = [];
const MAX_BUFFER = 200;

// metrics
let aiCorrect = 0;
let aiTotal = 0;

// face (BlazeFace)
let blazeModel = null;
let facePresent = false;

// affective scores (smoothed)
let engagementEMA = 0;      // 0..1
let confidenceEMA = 0;      // 0..1 (from model)
let moodLabel = "-";

// motion buffers
const boxHistory = [];
const MAX_BOX_HISTORY = 12;

// hover state for chart
let hoverIndex = -1;

// UI elements created dynamically
let hoverInfoEl = null;

// ===== DOM =====
const cashEl         = document.getElementById("cash");
const sharesEl       = document.getElementById("shares");
const portfolioEl    = document.getElementById("portfolio");
const priceEl        = document.getElementById("price");
const dayEl          = document.getElementById("day");
const aiActionEl     = document.getElementById("ai-action");
const aiReasonEl     = document.getElementById("ai-reason");
const moodMsgEl      = document.getElementById("mood-msg");
const aiRoiEl        = document.getElementById("ai-roi");
const playerRoiEl    = document.getElementById("player-roi");
const aiAccEl        = document.getElementById("ai-acc");
const statusEl       = document.getElementById("status");
const logList        = document.getElementById("logList");
const aiModeEl       = document.getElementById("aiMode");
const videoEl        = document.getElementById("video");
const faceStatusEl   = document.getElementById("face-status");
const faceHintEl     = document.getElementById("face-hint");
const engagementEl   = document.getElementById("engagement");
const moodLabelEl    = document.getElementById("mood-label");
const aiEngagementEl = document.getElementById("ai-engagement");
const aiMoodEl       = document.getElementById("ai-mood");
const yMinEl         = document.getElementById("y-min");
const yMaxEl         = document.getElementById("y-max");
const ctx            = document.getElementById("priceChart").getContext("2d");
const csvFileEl      = document.getElementById("csvFile");

// buttons
const buyBtn  = document.getElementById("buyBtn");
const sellBtn = document.getElementById("sellBtn");
const holdBtn = document.getElementById("holdBtn");

// ===== FORMATTERS =====
const usd = (n) => n.toLocaleString("en-GB", { style: "currency", currency: "GBP" });
const pct = (n) => `${n.toFixed(1)}%`;
function fmtDate(d) {
  if (!(d instanceof Date) || isNaN(d)) return "";
  return d.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
}

// ===== SCRIPT LOADER (robust) =====
function loadScript(url) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = url;
    s.async = true;
    s.onload = () => resolve(url);
    s.onerror = () => reject(new Error("Failed to load " + url));
    document.head.appendChild(s);
  });
}

function getChartSeries() {
  if (fullPrices.length) {
    const end = Math.min(cursor, fullPrices.length);      // up to “today” in the sim
    const maxBack = Math.max(0, end - DISPLAY_WINDOW);    // furthest we can pan left
    const clampedOffset = Math.max(0, Math.min(viewOffset, maxBack));

    const visibleEnd = end - clampedOffset;
    const visibleStart = Math.max(0, visibleEnd - DISPLAY_WINDOW);
    return {
      P: fullPrices.slice(visibleStart, visibleEnd),
      D: fullDates.slice(visibleStart, visibleEnd),
    };
  }
  // before CSV is loaded we just show the rolling synthetic window
  return { P: prices, D: dates };
}

async function ensureDeps() {
  // TensorFlow.js
  if (!window.tf) {
    const tfCandidates = [
      "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.21.0/dist/tf.min.js",
      "https://unpkg.com/@tensorflow/tfjs@4.21.0/dist/tf.min.js"
    ];
    let ok = false, lastErr = null;
    for (const u of tfCandidates) {
      try { await loadScript(u); ok = true; break; } catch (e) { lastErr = e; }
    }
    if (!ok || !window.tf) {
      throw new Error("TensorFlow.js failed to load. " + (lastErr ? lastErr.message : ""));
    }
  }
  // BlazeFace wrapper
  if (!window.blazeface) {
    const bfCandidates = [
      "https://cdn.jsdelivr.net/npm/@tensorflow-models/blazeface@0.0.7/dist/blazeface.min.js",
      "https://unpkg.com/@tensorflow-models/blazeface@0.0.7/dist/blazeface.min.js"
    ];
    let ok = false, lastErr = null;
    for (const u of bfCandidates) {
      try { await loadScript(u); ok = true; break; } catch (e) { lastErr = e; }
    }
    if (!ok || !window.blazeface) {
      throw new Error("BlazeFace library failed to load. " + (lastErr ? lastErr.message : ""));
    }
  }
}

function pctReturn(p0, p1) {
  return (p1 - p0) / p0;
}

function mean(arr) {
  return arr.reduce((a,b) => a + b, 0) / Math.max(1, arr.length);
}

function std(arr) {
  const m = mean(arr);
  const v = mean(arr.map(x => (x - m) ** 2));
  return Math.sqrt(v);
}

function buildDatasetFromPrices(pricesArr, lookback = 30) {
  // pricesArr should be oldest → newest
  if (!pricesArr || pricesArr.length < lookback + 2) {
    throw new Error("Not enough rows for dataset.");
  }

  // daily returns aligned to price index
  const rets = [];
  for (let i = 1; i < pricesArr.length; i++) {
    rets.push(pctReturn(pricesArr[i - 1], pricesArr[i]));
  }
  // rets length = pricesArr.length - 1

  const X = [];
  const y = [];

  // t indexes the "current day" in price index space
  // We need lookback returns ending at day t, and label from t->t+1
  for (let t = lookback; t < pricesArr.length - 1; t++) {
    // window of returns uses rets indices [t-lookback .. t-1]
    const win = rets.slice(t - lookback, t);

    const m = mean(win);
    const s = std(win) || 1e-8;

    // z-score normalise returns window (helps training)
    const zwin = win.map(r => (r - m) / s);

    // summary features
    const vol = s;
    const mom10 = (t >= 10) ? pctReturn(pricesArr[t - 10], pricesArr[t]) : 0;

    const feats = [...zwin, m, vol, mom10];

    // label based on next-day return
    const nextR = pctReturn(pricesArr[t], pricesArr[t + 1]);
    const thr = 0.002; // 0.2%
    let label = 1; // HOLD
    if (nextR > thr) label = 2;      // BUY
    else if (nextR < -thr) label = 0; // SELL

    X.push(feats);
    y.push(label);
  }

  return { X, y, inputSize: lookback + 3 };
}

function timeSeriesSplit(X, y, trainFrac = 0.8) {
  const n = X.length;
  const cut = Math.max(1, Math.floor(n * trainFrac));
  return {
    Xtr: X.slice(0, cut),
    ytr: y.slice(0, cut),
    Xte: X.slice(cut),
    yte: y.slice(cut),
  };
}

function confusionMatrix(yTrue, yPred, k = 3) {
  const cm = Array.from({ length: k }, () => Array(k).fill(0));
  for (let i = 0; i < yTrue.length; i++) {
    cm[yTrue[i]][yPred[i]]++;
  }
  return cm;
}

// ===== INIT =====
init().catch((e) => {
  console.error("Init fatal:", e);
  statusEl.textContent = "Init failed: " + (e && e.message ? e.message : String(e));
});

async function init() {
  await ensureDeps();


  // seed synthetic data so the app is interactive before CSV upload
  prices = generateInitialPrices(50, currentPrice);
  dates = new Array(prices.length).fill(null);
  drawChart();

  model = buildModel(7);
  await trainModel(model);


  updateUI();
  setupBlazeFace();

  const suggestion = await getAISuggestion();
  renderAISuggestion(suggestion);

  // bind buttons after everything is ready
  buyBtn.addEventListener("click", () => playerAction("BUY"));
  sellBtn.addEventListener("click", () => playerAction("SELL"));
  holdBtn.addEventListener("click", () => playerAction("HOLD"));

  // === CSV wiring ===
  if (csvFileEl) {
    csvFileEl.addEventListener("change", () => {
      const f = csvFileEl.files && csvFileEl.files[0];
      if (!f) return;
      const reader = new FileReader();
      reader.onload = () => {
        try { loadPricesFromCSVText(String(reader.result)); }
        catch (e) { statusEl.textContent = "CSV error: " + e.message; }
      };
      reader.onerror = () => statusEl.textContent = "Failed to read file.";
      reader.readAsText(f);
    });
  }

  // === dynamic UI under the chart ===
  const canvas = ctx.canvas;

  // live hover label
  hoverInfoEl = document.createElement("p");
  hoverInfoEl.id = "hoverInfo";
  hoverInfoEl.className = "hint";
  hoverInfoEl.style.margin = "6px 0 12px";
  hoverInfoEl.textContent = "—";
  canvas.parentNode.insertBefore(hoverInfoEl, canvas.nextSibling);

  // scrollback with mouse wheel
  canvas.addEventListener("wheel", (e) => {
    if (!fullPrices.length) return;
    e.preventDefault();
    const step = e.deltaY > 0 ? 5 : -5; // pan ~5 trading days per notch
    const end = Math.min(cursor, fullPrices.length);
    const maxBack = Math.max(0, end - DISPLAY_WINDOW);
    viewOffset = Math.max(0, Math.min(maxBack, viewOffset + step));
    drawChart();
  });

  // double-click to jump back to “today”
  canvas.addEventListener("dblclick", () => {
    viewOffset = 0;
    drawChart();
  });

  // hover (use visible window)
  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;

    const { P: chartPrices, D: chartDates } = getChartSeries();
    const n = chartPrices.length;

    const i = Math.round(xToIndex(x, canvas.width, n));
    hoverIndex = Math.max(0, Math.min(n - 1, i));
    drawChart();

    if (hoverInfoEl) {
      const d = chartDates[hoverIndex];
      hoverInfoEl.textContent = `${d ? fmtDate(d) : "Day " + (hoverIndex + 1)} · ${usd(chartPrices[hoverIndex])}`;
    }
  });

  canvas.addEventListener("mouseleave", () => {
    hoverIndex = -1;
    drawChart();
    if (hoverInfoEl) hoverInfoEl.textContent = latestLabel();
  });
} // <— this closes init()


// ===== MARKET SIM =====
function generateInitialPrices(n, start) {
  const arr = [];
  let p = start;
  for (let i = 0; i < n; i++) {
    p = nextPrice(p);
    arr.push(p);
  }
  currentPrice = arr[arr.length - 1];
  return arr;
}
function nextPrice(prev) {
  const drift = 0.03;
  const noise = (Math.random() - 0.5) * 2; // -1..1
  const pctChange = drift + noise * 0.7;
  const newPrice = Math.max(10, prev * (1 + pctChange / 100));
  return +newPrice.toFixed(2);
}
function advanceDay() {
  // If we have an uploaded series, step through that instead of generating noise
  if (fullPrices.length) {
    if (cursor < fullPrices.length) {
      const newP = fullPrices[cursor];
      const newD = fullDates[cursor];
      cursor++;

      prices.push(newP);
      dates.push(newD);

      if (prices.length > DISPLAY_WINDOW) { prices.shift(); dates.shift(); }

      currentPrice = newP;
      day++;
      drawChart();
    }
    return;
  }

  // Fallback (no CSV loaded yet): synthetic data as before
  const newP = nextPrice(currentPrice);
  prices.push(newP);
  dates.push(null);
  if (prices.length > 50) { prices.shift(); dates.shift(); }
  currentPrice = newP;
  day++;
  drawChart();
}


// ===== CHART UTILS =====
function chartX(i, w, n) {
  return (i / Math.max(1, n - 1)) * (w - 20) + 10;
}
function chartY(p, h, min, range) {
  return h - 20 - ((p - min) / range) * (h - 40);
}
function xToIndex(x, w, n) {
  const t = (x - 10) / (w - 20);
  return t * (n - 1);
}

// ===== DRAW CHART =====
function drawChart() {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.clearRect(0, 0, w, h);

  // === use visible window ===
  const { P: chartPrices, D: chartDates } = getChartSeries();
  const n = chartPrices.length;
  if (!n) return;

  const max = Math.max(...chartPrices);
  const min = Math.min(...chartPrices);
  const range = max - min || 1;

  // grid
  ctx.strokeStyle = "rgba(148,163,184,0.15)";
  ctx.lineWidth = 1;
  const rows = 4;
  for (let i = 0; i <= rows; i++) {
    const y = h - 20 - (i / rows) * (h - 40);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  // x-axis
  ctx.strokeStyle = "#1f2937";
  ctx.beginPath(); ctx.moveTo(0, h - 20); ctx.lineTo(w, h - 20); ctx.stroke();

  // price line
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2;
  ctx.beginPath();
  chartPrices.forEach((p, i) => {
    const x = chartX(i, w, n);
    const y = chartY(p, h, min, range);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // last point (right edge of visible window)
  const lastX = chartX(n - 1, w, n);
  const lastY = chartY(chartPrices[n - 1], h, min, range);
  ctx.fillStyle = "#f97316";
  ctx.beginPath(); ctx.arc(lastX, lastY, 4, 0, Math.PI * 2); ctx.fill();

  // y labels
  if (yMinEl) yMinEl.textContent = usd(min);
  if (yMaxEl) yMaxEl.textContent = usd(max);

  // === sparse x labels: ~6 evenly spaced ===
  const labelCount = 6;
  ctx.fillStyle = "rgba(226,232,240,0.85)";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.font = "12px system-ui";
  for (let i = 0; i < labelCount; i++) {
    const idx = Math.round((i / (labelCount - 1)) * (n - 1));
    const x = chartX(idx, w, n);
    const label = chartDates[idx] ? fmtDate(chartDates[idx]) : String(idx + 1);
    ctx.fillText(label, x, h - 18);
  }

  // === hover state is now over the visible window ===
  const globalToLocal = (gi) => gi; // we hover only within the window now
  if (hoverIndex >= n) hoverIndex = n - 1;

  // hover tooltip (uses improved box)
  // hover tooltip (auto-size to text metrics)
if (hoverIndex >= 0 && hoverIndex < n) {
  const px = chartX(hoverIndex, w, n);
  const py = chartY(chartPrices[hoverIndex], h, min, range);

  // point marker
  ctx.beginPath();
  ctx.arc(px, py, 3, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();

  // text + box
  const text = `${chartDates[hoverIndex] ? fmtDate(chartDates[hoverIndex]) : "Day " + (hoverIndex + 1)}  ·  ${usd(chartPrices[hoverIndex])}`;

  ctx.font = "12px system-ui";
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";

  const pad = 8;
  const tm = ctx.measureText(text);
  const textW = Math.ceil(tm.width);
  // Use real ascent/descent where available; fall back to a safe height
  const ascent  = Math.ceil(tm.actualBoundingBoxAscent  || 9);
  const descent = Math.ceil(tm.actualBoundingBoxDescent || 3);
  const textH = ascent + descent;

  const boxW = textW + pad * 2;
  const boxH = textH + pad * 2;

  // default position: right-above the point
  let bx = Math.round(px + 10);
  let by = Math.round(py - boxH - 8);

  // clamp to canvas; if too high, flip below
  if (bx + boxW > w - 8) bx = w - 8 - boxW;
  if (bx < 8) bx = 8;
  if (by < 8) by = Math.round(py + 8);

  // box
  ctx.fillStyle = "rgba(15,23,42,0.92)";
  ctx.fillRect(bx, by, boxW, boxH);
  ctx.strokeStyle = "rgba(148,163,184,0.35)";
  ctx.strokeRect(bx + 0.5, by + 0.5, boxW - 1, boxH - 1);

  // text (baseline inside the box)
  ctx.fillStyle = "rgba(226,232,240,0.96)";
  const tx = bx + pad;
  const ty = by + pad + ascent; // baseline position
  ctx.fillText(text, tx, ty);
}

  // static latest label
  if (hoverInfoEl && hoverIndex < 0 && n) {
    const i = n - 1;
    const d = chartDates[i];
    hoverInfoEl.textContent = `${d ? fmtDate(d) : "Day " + (i + 1)} · ${usd(chartPrices[i])}`;
  }
}

// ===== FEATURES =====
function buildFeatures() {
  const last5 = prices.slice(-5);
  const ma5 = last5.reduce((a, b) => a + b, 0) / last5.length;
  const change = (last5[last5.length - 1] - last5[0]) / last5[0];
  return [...last5, ma5, change];
}

// ===== TFJS MODEL =====
function buildModel(inputSize = 33) {
  const m = tf.sequential();
  m.add(tf.layers.dense({ inputShape: [inputSize], units: 48, activation: "relu" }));
  m.add(tf.layers.dense({ units: 24, activation: "relu" }));
  m.add(tf.layers.dense({ units: 3, activation: "softmax" }));
  m.compile({
    optimizer: tf.train.adam(0.001),
    loss: "sparseCategoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return m;
}

async function trainModel(m) {
  const xs = [];
  const ys = [];

  const sim = generateInitialPrices(150, 100);
  for (let i = 7; i < sim.length; i++) {
    const win = sim.slice(i - 5, i);
    const ma = win.reduce((a, b) => a + b, 0) / win.length;
    const change = (win[win.length - 1] - win[0]) / win[0];
    const feats = [...win, ma, change];

    const label = labelFromWindow(win);
    xs.push(feats);
    ys.push(oneHot(label, 3));
  }

  const xTensor = tf.tensor2d(xs);
  const xNorm = xTensor.div(xTensor.max());
  const yTensor = tf.tensor2d(ys);

  await m.fit(xNorm, yTensor, {
    epochs: 35,
    batchSize: 20,
    shuffle: true,
  });

  xTensor.dispose();
  xNorm.dispose();
  yTensor.dispose();
}
function labelFromWindow(win) {
  const last = win[win.length - 1];
  const first = win[0];
  const change = ((last - first) / first) * 100;
  if (change > 0.5) return 2;   // BUY
  if (change < -0.5) return 0;  // SELL
  return 1;                     // HOLD
}
function oneHot(i, len) { const a = new Array(len).fill(0); a[i] = 1; return a; }

async function trainOnCSVSeries() {
  if (!fullPrices || fullPrices.length < 100) {
    statusEl.textContent = "Upload a CSV first (needs enough rows).";
    return;
  }

  statusEl.textContent = "Building dataset from CSV...";
  const { X, y, inputSize } = buildDatasetFromPrices(fullPrices, 30);
  const { Xtr, ytr, Xte, yte } = timeSeriesSplit(X, y, 0.8);

  // tensors
  const xTr = tf.tensor2d(Xtr);
  const yTr = tf.tensor1d(ytr, "int32");
  const xTe = tf.tensor2d(Xte);
  const yTe = tf.tensor1d(yte, "int32");

  // rebuild model for correct input shape
  model = buildModel(inputSize);

  statusEl.textContent = `Training on real CSV data… (train ${Xtr.length}, test ${Xte.length})`;

  await model.fit(xTr, yTr, {
    epochs: 25,
    batchSize: 32,
    shuffle: false, // keep time order (safer)
    validationData: [xTe, yTe],
  });

  // Evaluate + confusion matrix
  const preds = model.predict(xTe);
  const yHat = Array.from(preds.argMax(-1).dataSync());

  let correct = 0;
  for (let i = 0; i < yHat.length; i++) if (yHat[i] === yte[i]) correct++;
  const acc = correct / Math.max(1, yHat.length);

  const cm = confusionMatrix(yte, yHat, 3);

  statusEl.textContent =
    `Trained on CSV. Test accuracy: ${(acc * 100).toFixed(1)}%. ` +
    `CM: [[${cm[0].join(",")}],[${cm[1].join(",")}],[${cm[2].join(",")}]]`;

  // cleanup
  xTr.dispose(); yTr.dispose(); xTe.dispose(); yTe.dispose();
  preds.dispose();
}

// ===== AI SUGGESTION =====
async function getAISuggestion() {
  const mode = aiModeEl.value;
  const feats = buildFeatures();

  if (mode === "rule") {
    return { action: ruleBasedAction(), probs: [0.33,0.33,0.33], origin: "rule", message: "Rule says do this." };
  }

  const base = await predictFromModel(feats);

  if (mode === "face") {
    const msg = buildCoachMessage(base, facePresent, engagementEMA, moodLabel);
    const adjusted = adjustForFaceV2(base, facePresent, engagementEMA, moodLabel);
    return { action: adjusted, probs: base.probs, origin: facePresent ? "Face(" + moodLabel + ")" : "Face(absent)", message: msg };
    }

  return base;
}
async function predictFromModel(feats) {
  const max = Math.max(...feats);
  const input = tf.tensor2d([feats]).div(max);
  const pred = model.predict(input);
  const data = await pred.data();
  input.dispose(); pred.dispose();

  const actions = ["SELL", "HOLD", "BUY"];

  let topIdx = 0, secondIdx = 1;
  for (let i = 1; i < data.length; i++) {
    if (data[i] > data[topIdx]) { secondIdx = topIdx; topIdx = i; }
    else if (i !== topIdx && data[i] > data[secondIdx]) { secondIdx = i; }
  }

  const action = actions[topIdx];
  const conf = data[topIdx];
  const second = actions[secondIdx];
  const secondConf = data[secondIdx];

  if (conf < 0.25) {
    const ruleAct = ruleBasedAction();
    return { action: ruleAct, probs: data, origin: "nn(low-conf->rule)", conf, second, secondConf };
  }

  confidenceEMA = ema(confidenceEMA, conf, 0.25);
  return { action, probs: data, origin: "nn", conf, second, secondConf };
}
function ruleBasedAction() {
  const last5 = prices.slice(-5);
  const ma5 = last5.reduce((a, b) => a + b, 0) / last5.length;
  const last = last5[last5.length - 1];
  if (last < ma5 * 0.995) return "BUY";
  if (last > ma5 * 1.005) return "SELL";
  return "HOLD";
}

// ===== COACH MESSAGES =====
function buildCoachMessage(pred, present, engage, mood) {
  const conf = pred.conf ?? 0.3;
  const top = pred.action;
  const second = pred.second || (top === "BUY" ? "HOLD" : "BUY");
  const gap = conf - (pred.secondConf ?? (conf - 0.05));

  if (!present) return "You are not fully engaged - safest is to HOLD this round.";

  if (engage >= 0.6 && (mood === "positive" || "focused")) {
    if (conf >= 0.60 && gap >= 0.10) {
      if (top === "BUY")  return "You seem positive and the model agrees - buying is reasonable here.";
      if (top === "SELL") return "You look confident and the signal tilts down - selling makes sense.";
      return "You are focused and the model likes your position - HOLD is fine.";
    }
    return "You are engaged - model leans " + top + "; " + second + " is also defensible if you want less risk.";
  }

  if (engage >= 0.35 && mood === "unsure") {
    if (conf >= 0.55 && top !== "SELL") {
      return "You look a bit unsure; the model still prefers " + top + ". Start small or set a stop.";
    }
    return "You seem unsure - HOLD or take a smaller position until the signal firms up.";
  }

  if (top === "HOLD")
    return "Signal and focus are not strong - HOLD is safer; BUY offers more upside but also more risk.";

  return "Signal and focus are not strong - HOLD is the safer option until you are ready.";
}

// ===== FACE-AWARE ADJUSTMENT =====
function adjustForFaceV2(base, present, engage, mood) {
  const conf = base.conf ?? 0.3;
  let action = base.action;

  if (!present) return action === "BUY" ? "HOLD" : action;

  if (engage >= 0.6 && conf >= 0.45) return action;

  if (mood === "unsure" && conf < 0.6) {
    if (action === "SELL" || action === "BUY") return "HOLD";
  }

  if (engage < 0.35 && action === "BUY" && conf < 0.55) return "HOLD";

  return action;
}
// ===== PLAYER ACTION =====
async function playerAction(action) {
  const prevPrice = currentPrice;
  const stateFeats = buildFeatures();

  const msg = executePlayerTrade(action, currentPrice);
  log(msg);
  statusEl.textContent = msg;

  const aiSuggestion = await getAISuggestion();
  const beforeAICash = aiCash, beforeAIShares = aiShares;
  executeAITrade(aiSuggestion.action, currentPrice);
  if (aiCash !== beforeAICash || aiShares !== beforeAIShares) aiTradesMade++;

  advanceDay();
  updateUI();

  const reward = currentPrice - prevPrice;
  storeExperience(stateFeats, aiSuggestion.action, reward);

  aiTotal++;
  const aiWasRight =
    (reward > 0 && aiSuggestion.action === "BUY") ||
    (reward < 0 && aiSuggestion.action === "SELL") ||
    (Math.abs(reward) < 0.01 && aiSuggestion.action === "HOLD");
  if (aiWasRight) aiCorrect++;

  if (replayBuffer.length >= 30 && day % 5 === 0) {
    await retrainOnBuffer();
  }

  const nextS = await getAISuggestion();
  renderAISuggestion(nextS);
}
function executePlayerTrade(action, price) {
  if (action === "BUY") {
    if (cash >= price) {
      cash -= price; shares += 1;
      return "You bought 1 share at " + usd(price) + ".";
    }
    return "Not enough cash to buy.";
  }
  if (action === "SELL") {
    if (shares > 0) {
      shares -= 1; cash += price;
      return "You sold 1 share at " + usd(price) + ".";
    }
    return "No shares to sell.";
  }
  return "You held your position.";
}
function executeAITrade(action, price) {
  if (action === "BUY" && aiCash >= price) { aiCash -= price; aiShares += 1; }
  else if (action === "SELL" && aiShares > 0) { aiShares -= 1; aiCash += price; }
}

// ===== RL-ish ONLINE UPDATE =====
function storeExperience(state, action, reward) {
  const actionIdx = action === "SELL" ? 0 : action === "HOLD" ? 1 : 2;
  replayBuffer.push({ state, actionIdx, reward });
  if (replayBuffer.length > MAX_BUFFER) replayBuffer.shift();
}
async function retrainOnBuffer() {
  const xs = [], ys = [];
  replayBuffer.forEach((exp) => {
    xs.push(exp.state);
    const label = exp.reward > 0 ? exp.actionIdx : 1;
    ys.push(oneHot(label, 3));
  });

  const xTensor = tf.tensor2d(xs);
  const xNorm = xTensor.div(xTensor.max());
  const yTensor = tf.tensor2d(ys);

  await model.fit(xNorm, yTensor, { epochs: 10, batchSize: 8, shuffle: true });

  xTensor.dispose(); xNorm.dispose(); yTensor.dispose();
  log("AI fine-tuned on recent gameplay data.");
}

// ===== UI =====
function refreshActionButtons(){
  sellBtn.disabled = (shares === 0);
  sellBtn.title = sellBtn.disabled ? "You have no shares to sell" : "Sell 1 share";

  buyBtn.disabled = (cash < currentPrice);
  buyBtn.title = buyBtn.disabled ? "Not enough cash to buy" : "Buy 1 share";
}
function updateUI() {
  const portfolioValue = cash + shares * currentPrice;
  cashEl.textContent = usd(cash);
  sharesEl.textContent = shares;
  portfolioEl.textContent = usd(portfolioValue);
  priceEl.textContent = usd(currentPrice);
  dayEl.textContent = day;

  const playerRoi = ((portfolioValue - 10000) / 10000) * 100;
  playerRoiEl.textContent = pct(playerRoi);

  const aiPortfolio = aiCash + aiShares * currentPrice;
  const aiRoi = ((aiPortfolio - aiStartCapital) / aiStartCapital) * 100;
  aiRoiEl.textContent = aiTradesMade ? pct(aiRoi) : "—";

  const aiAcc = aiTotal > 0 ? (aiCorrect / aiTotal) * 100 : 0;
  aiAccEl.textContent = aiTotal ? pct(aiAcc) : "—";

  const engPct = Math.round(engagementEMA * 100);
  engagementEl.textContent = engPct + "%";
  aiEngagementEl.textContent = engPct + "%";
  moodLabelEl.textContent = facePresent ? moodLabel : "—";
  aiMoodEl.textContent = facePresent ? moodLabel : "—";

  refreshActionButtons();
}
function renderAISuggestion({ action, probs, origin, message }) {
  aiActionEl.textContent = action;

  if (probs) {
    aiReasonEl.textContent =
      (origin || "NN") + " · face=" + (facePresent ? "yes" : "no") +
      " · sell " + (probs[0]*100).toFixed(0) + "%" +
      " · hold " + (probs[1]*100).toFixed(0) + "%" +
      " · buy "  + (probs[2]*100).toFixed(0) + "%";
  } else {
    aiReasonEl.textContent = (origin || "rule") + " · face=" + (facePresent ? "yes" : "no");
  }

  moodMsgEl.textContent = message || "-";
  log("AI suggests (" + (origin || "nn") + "): " + action);
}
function log(text) {
  const li = document.createElement("li");
  li.textContent = "[Day " + day + "] " + text;
  logList.prepend(li);
}
// ===== BLAZEFACE + AFFECT =====
async function setupBlazeFace() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoEl.srcObject = stream;

    videoEl.setAttribute("playsinline", "true");
    await new Promise((resolve) => {
      if (videoEl.readyState >= 1) return resolve();
      videoEl.onloadedmetadata = () => resolve();
    });
    await videoEl.play();

    faceHintEl.textContent = "Camera ready. Loading face model...";
  } catch (err) {
    console.warn("camera error:", err);
    faceHintEl.textContent = "Camera blocked - face mode disabled.";
    return;
  }

  try {
    blazeModel = await blazeface.load();
    faceHintEl.textContent = "BlazeFace loaded (choose Face-aware coach).";
  } catch (err) {
    console.error("blazeface load error:", err);
    faceHintEl.textContent = "Could not load face model - use NN/Rule.";
    return;
  }

  setInterval(async () => {
    if (!blazeModel) return;
    if (!videoEl || videoEl.readyState < 2) return; // HAVE_CURRENT_DATA

    try {
      const preds = await blazeModel.estimateFaces(videoEl, false);
      facePresent = !!(preds && preds.length > 0);
      faceStatusEl.textContent = facePresent ? "yes" : "no";

      if (facePresent) {
        const f = preds[0];
        const box = normalizeBox(f);
        updateAffect(box, f.probability || 0.9);
      } else {
        decayAffectWhenAbsent();
      }
      updateUI();
    } catch (err) {
      console.warn("blazeface detect error:", err);
      faceHintEl.textContent = "Face detect error - using default AI.";
      blazeModel = null;
    }
  }, 1000);
}
function normalizeBox(facePred) {
  const tl = facePred.topLeft;
  const br = facePred.bottomRight;
  const x1 = Array.isArray(tl) ? tl[0] : tl.x;
  const y1 = Array.isArray(tl) ? tl[1] : tl.y;
  const x2 = Array.isArray(br) ? br[0] : br.x;
  const y2 = Array.isArray(br) ? br[1] : br.y;

  const w = videoEl.videoWidth || 1;
  const h = videoEl.videoHeight || 1;
  return {
    cx: ((x1 + x2) / 2) / w,
    cy: ((y1 + y2) / 2) / h,
    size: ((x2 - x1) * (y2 - y1)) / (w * h),
  };
}
function updateAffect(box, presenceProb) {
  boxHistory.push(box);
  if (boxHistory.length > MAX_BOX_HISTORY) boxHistory.shift();

  let dx = 0, dy = 0, ds = 0;
  for (let i = 1; i < boxHistory.length; i++) {
    dx += Math.abs(boxHistory[i].cx - boxHistory[i-1].cx);
    dy += Math.abs(boxHistory[i].cy - boxHistory[i-1].cy);
    ds += Math.abs(boxHistory[i].size - boxHistory[i-1].size);
  }
  const steps = Math.max(1, boxHistory.length - 1);
  dx/=steps; dy/=steps; ds/=steps;

  const jitter = Math.min(1, (dx + dy) * 6);
  const steadiness = 1 - jitter;

  const proximity = clamp(scale(box.size, 0.02, 0.20));
  const sizeStability = 1 - clamp(ds * 40);

  const rawEngagement = clamp(0.55 * proximity + 0.35 * steadiness + 0.10 * sizeStability);
  engagementEMA = ema(engagementEMA, rawEngagement * presenceProb, 0.25);

  let mood = "focused";
  if (engagementEMA < 0.35) mood = "distracted";
  else if (jitter > 0.08) mood = "unsure";
  else if (proximity > 0.5 && steadiness > 0.6) mood = "positive";
  moodLabel = mood;
}
function decayAffectWhenAbsent() {
  engagementEMA = ema(engagementEMA, 0, 0.10);
  moodLabel = "-";
  boxHistory.length = 0;
}

// ===== HELPERS =====
function ema(prev, value, alpha = 0.2) { return prev * (1 - alpha) + value * alpha; }
function clamp(v, lo = 0, hi = 1) { return Math.max(lo, Math.min(hi, v)); }
function scale(v, min, max) { return clamp((v - min) / (max - min)); }

// ===== CSV LOADER (AAPL: Date,Price) =====
function parseDateStrict(raw) {
  if (!raw) return null;
  const s = String(raw).trim().replace(/^\uFEFF/, "").replace(/,+/g, ""); // strip BOM/commas

  // yyyy-mm-dd or yyyy/mm/dd
  let m = s.match(/^(\d{4})[\/-](\d{1,2})[\/-](\d{1,2})$/);
  if (m) return new Date(+m[1], +m[2] - 1, +m[3]);

  // dd/mm/yyyy or dd-mm-yyyy (day-first)
  m = s.match(/^(\d{1,2})[\/-](\d{1,2})[\/-](\d{2,4})$/);
  if (m) {
    const dd = +m[1], mm = +m[2], yy = +m[3];
    const yyyy = yy < 100 ? (2000 + yy) : yy;
    return new Date(yyyy, mm - 1, dd);
  }

  // 23 Sep 2025 / 23 September 2025 / Sep 23 2025
  const months = {jan:0,feb:1,mar:2,apr:3,may:4,jun:5,jul:6,aug:7,sep:8,sept:8,oct:9,nov:10,dec:11,
                  january:0,february:1,march:2,april:3,june:5,july:6,august:7,september:8,october:9,november:10,december:11};

  m = s.match(/^(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$/); // day month year
  if (m && months[m[2].toLowerCase()] !== undefined)
    return new Date(+m[3], months[m[2].toLowerCase()], +m[1]);

  m = s.match(/^([A-Za-z]+)\s+(\d{1,2})\s+(\d{4})$/); // month day year
  if (m && months[m[1].toLowerCase()] !== undefined)
    return new Date(+m[3], months[m[1].toLowerCase()], +m[2]);

  return null;
}


function parseCSVToSeries(text) {
  const clean = text.replace(/\r/g, "\n");
  const rows = clean.split("\n").map(r => r.trim()).filter(Boolean);
  if (!rows.length) throw new Error("CSV appears to be empty.");

  // Header (strip BOM)
  let first = rows[0].replace(/^\uFEFF/, "");
  const hasHeader = /[A-Za-z]/.test(first);
  const dataRows = hasHeader ? rows.slice(1) : rows;
  const header = hasHeader ? first.split(",").map(h => h.trim().toLowerCase()) : [];

  const priceCandidates = ["price","close","adj close","adj_close","value","rate"];
  const dateCandidates  = ["date","time","timestamp","datetime"];

  let priceCol = -1, dateCol = -1;
  if (hasHeader) {
    priceCol = header.findIndex(h => priceCandidates.includes(h));
    dateCol  = header.findIndex(h => dateCandidates.includes(h));
    if (priceCol === -1) priceCol = header.length - 1; // default: last col
    if (dateCol === -1)  dateCol = 0;                  // default: first col
  }

  const outPrices = [];
  const outDates  = [];

  for (const row of dataRows) {
    const cols = row.split(",").map(x => x.trim());
    if (!cols.length) continue;

    // price
    let v = (priceCol >= 0 && priceCol < cols.length) ? cols[priceCol] : cols[cols.length - 1];
    v = v.replace(/["']/g, "").replace(/\s/g, "").replace(/,/g, "");
    const num = parseFloat(v);
    if (!isFinite(num)) continue;

    // date
    const rawDate = (dateCol >= 0 && dateCol < cols.length) ? cols[dateCol] : "";
    const d = parseDateStrict(rawDate);

    outPrices.push(+num.toFixed(2));
    outDates.push(d);
  }

  if (outPrices.length < 5) throw new Error("Not enough rows detected.");
  return { prices: outPrices, dates: outDates };
}


function loadPricesFromCSVText(text) {
  const { prices: Praw, dates: Draw } = parseCSVToSeries(text);

  // Zip rows, keep only valid dates, sort earliest → latest
  const rows = [];
  for (let i = 0; i < Praw.length; i++) {
    const d = Draw[i];
    if (d instanceof Date && !isNaN(d)) rows.push({ d, p: +Praw[i] });
  }

  if (rows.length < 50) {
    statusEl.textContent = "Not enough valid dated rows in CSV.";
    return;
  }

  rows.sort((a, b) => a.d - b.d);

  // Unzip to canonical series
  fullPrices = rows.map(r => +r.p.toFixed(2));
  fullDates  = rows.map(r => r.d);

  // Reset sim + view so we always start at the beginning
  cursor = 0;
  viewOffset = 0;
  prices = [];
  dates  = [];

  const SEED_WINDOW = 50;
  const start = 0;
  const end = Math.min(SEED_WINDOW, fullPrices.length);

  // Fill initial visible window from earliest date
  for (let i = start; i < end; i++) {
    prices.push(fullPrices[i]);
    dates.push(fullDates[i]);
  }

  cursor = end;
  currentPrice = prices[prices.length - 1];
  day = prices.length;

  drawChart();
  updateUI();

  log(
    "Loaded " + fullPrices.length +
    " rows; visible " + fmtDate(fullDates[start]) +
    " → " + fmtDate(fullDates[end - 1]) + "."
  );
  statusEl.textContent = "CSV loaded. Training model on this CSV…";

  // Train once, after data is fully loaded
  trainOnCSVSeries().catch(e => {
    console.error(e);
    statusEl.textContent = "Training error: " + e.message;
  });
}

// ===== DATE PANEL HELPERS =====
function latestLabel() {
  const i = prices.length - 1;
  const d = dates[i];
  const dateTxt = d ? fmtDate(d) : "Day " + (i + 1);
  return `${dateTxt} · ${usd(prices[i])}`;
}

