// STATE
let prices = [];
let dates = []; // parallel to prices; each item is a Date or null
let day = 1;
let cash = 10000;
let shares = 0;
let currentPrice = 100;
let viewOffset = 0; // 0 = most recent window; larger moves left (earlier history)

// Full uploaded series
let fullPrices = [];
let fullDates  = [];
let cursor = 0;                 // next index in full series to reveal
const DISPLAY_WINDOW = 220;     // how many past days we keep on screen
const SEED_WINDOW    = 50;      // show ONLY the very first rows at start

// AI virtual portfolio to compare vs player
let aiCash = 10000;
let aiShares = 0;
const aiStartCapital = 10000;
let aiTradesMade = 0; // for ROI placeholder

// metrics
let aiCorrect = 0;
let aiTotal = 0;

// face (BlazeFace)
let blazeModel = null;
let facePresent = false;

// affective scores
let engagementEMA = 0;      // 0..1
let confidenceEMA = 0;      // 0..1
let moodLabel = "-";

// motion buffers
const boxHistory = [];
const MAX_BOX_HISTORY = 12;

// hover state for chart
let hoverIndex = -1;

// UI elements created dynamically
let hoverInfoEl = null;

// DOM
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

// FORMATTERS
const usd = (n) => n.toLocaleString("en-GB", { style: "currency", currency: "GBP" });
const pct = (n) => `${n.toFixed(1)}%`;
function fmtDate(d) {
  if (!(d instanceof Date) || isNaN(d)) return "";
  return d.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
}

// SCRIPT LOADER
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
    const end = Math.min(cursor, fullPrices.length);
    const maxBack = Math.max(0, end - DISPLAY_WINDOW);
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
  // BlazeFace wrapper only (NO TFJS / NO NN)
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

function pctReturn(p0, p1) { return (p1 - p0) / p0; }
function mean(arr) { return arr.reduce((a,b) => a + b, 0) / Math.max(1, arr.length); }
function std(arr) {
  const m = mean(arr);
  const v = mean(arr.map(x => (x - m) ** 2));
  return Math.sqrt(v);
}

function confusionMatrix(yTrue, yPred, k = 3) {
  const cm = Array.from({ length: k }, () => Array(k).fill(0));
  for (let i = 0; i < yTrue.length; i++) cm[yTrue[i]][yPred[i]]++;
  return cm;
}

// INIT
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

  updateUI();
  setupBlazeFace();

  const suggestion = await getAISuggestion();
  renderAISuggestion(suggestion);

  // bind buttons after everything is ready
  buyBtn.addEventListener("click", () => playerAction("BUY"));
  sellBtn.addEventListener("click", () => playerAction("SELL"));
  holdBtn.addEventListener("click", () => playerAction("HOLD"));

  // CSV wiring
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

// MARKET SIM
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

// CHART UTILS
function chartX(i, w, n) { return (i / Math.max(1, n - 1)) * (w - 20) + 10; }
function chartY(p, h, min, range) { return h - 20 - ((p - min) / range) * (h - 40); }
function xToIndex(x, w, n) {
  const t = (x - 10) / (w - 20);
  return t * (n - 1);
}

// DRAW CHART 
function drawChart() {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.clearRect(0, 0, w, h);

  // use visible window
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

  // last point
  const lastX = chartX(n - 1, w, n);
  const lastY = chartY(chartPrices[n - 1], h, min, range);
  ctx.fillStyle = "#f97316";
  ctx.beginPath(); ctx.arc(lastX, lastY, 4, 0, Math.PI * 2); ctx.fill();

  // y labels
  if (yMinEl) yMinEl.textContent = usd(min);
  if (yMaxEl) yMaxEl.textContent = usd(max);

  // x labels
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

  // hover tooltip (auto-size)
  if (hoverIndex >= n) hoverIndex = n - 1;

  if (hoverIndex >= 0 && hoverIndex < n) {
    const px = chartX(hoverIndex, w, n);
    const py = chartY(chartPrices[hoverIndex], h, min, range);

    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();

    const text = `${chartDates[hoverIndex] ? fmtDate(chartDates[hoverIndex]) : "Day " + (hoverIndex + 1)}  ·  ${usd(chartPrices[hoverIndex])}`;

    ctx.font = "12px system-ui";
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";

    const pad = 8;
    const tm = ctx.measureText(text);
    const textW = Math.ceil(tm.width);
    const ascent  = Math.ceil(tm.actualBoundingBoxAscent  || 9);
    const descent = Math.ceil(tm.actualBoundingBoxDescent || 3);
    const textH = ascent + descent;

    const boxW = textW + pad * 2;
    const boxH = textH + pad * 2;

    let bx = Math.round(px + 10);
    let by = Math.round(py - boxH - 8);

    if (bx + boxW > w - 8) bx = w - 8 - boxW;
    if (bx < 8) bx = 8;
    if (by < 8) by = Math.round(py + 8);

    ctx.fillStyle = "rgba(15,23,42,0.92)";
    ctx.fillRect(bx, by, boxW, boxH);
    ctx.strokeStyle = "rgba(148,163,184,0.35)";
    ctx.strokeRect(bx + 0.5, by + 0.5, boxW - 1, boxH - 1);

    ctx.fillStyle = "rgba(226,232,240,0.96)";
    const tx = bx + pad;
    const ty = by + pad + ascent;
    ctx.fillText(text, tx, ty);
  }

  // static latest label
  if (hoverInfoEl && hoverIndex < 0 && n) {
    const i = n - 1;
    const d = chartDates[i];
    hoverInfoEl.textContent = `${d ? fmtDate(d) : "Day " + (i + 1)} · ${usd(chartPrices[i])}`;
  }
}

function buildFeatures() {
  const last5 = prices.slice(-5);
  const ma5 = last5.reduce((a, b) => a + b, 0) / last5.length;
  const change = (last5[last5.length - 1] - last5[0]) / last5[0];
  return { last5, ma5, change };
}

// RULE-BASED AI
function ruleBasedAction() {
  const { last5, ma5 } = buildFeatures();
  const last = last5[last5.length - 1];

  // Same thresholds you had:
  if (last < ma5 * 0.995) return "BUY";
  if (last > ma5 * 1.005) return "SELL";
  return "HOLD";
}

// Derive a simple confidence from how far price is from the MA band.
// This keeps the UI message formatting (sell/hold/buy %) and “confidenceEMA” meaningful.
function ruleProbs() {
  const { last5, ma5 } = buildFeatures();
  const last = last5[last5.length - 1];

  // distance from MA as a fraction
  const d = Math.abs(last - ma5) / Math.max(1e-8, ma5);

  // map distance to 0..1 strength
  const strength = clamp(scale(d, 0.0005, 0.01)); // ~0.05% to 1%

  const action = ruleBasedAction();

  // baseline distribution
  let sell = 0.33, hold = 0.34, buy = 0.33;

  if (action === "BUY") {
    buy = 0.34 + 0.56 * strength;
    hold = 0.34 - 0.28 * strength;
    sell = 1 - buy - hold;
  } else if (action === "SELL") {
    sell = 0.34 + 0.56 * strength;
    hold = 0.34 - 0.28 * strength;
    buy = 1 - sell - hold;
  } else {
    // HOLD gets stronger when distance is small
    hold = 0.40 + 0.50 * (1 - strength);
    const rem = 1 - hold;
    sell = rem / 2;
    buy = rem / 2;
  }

  // clamp + renormalise defensively
  sell = clamp(sell, 0.01, 0.98);
  hold = clamp(hold, 0.01, 0.98);
  buy  = clamp(buy,  0.01, 0.98);
  const s = sell + hold + buy;
  sell /= s; hold /= s; buy /= s;

  return [sell, hold, buy];
}

function pickTopTwo(probs) {
  let topIdx = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[topIdx]) topIdx = i;

  let secondIdx = topIdx === 0 ? 1 : 0;
  for (let i = 0; i < probs.length; i++) {
    if (i === topIdx) continue;
    if (probs[i] > probs[secondIdx]) secondIdx = i;
  }
  return { topIdx, secondIdx };
}

async function getAISuggestion() {
  const mode = (aiModeEl && aiModeEl.value) ? aiModeEl.value : "rule";

  // Any “nn” mode from old UI falls back to rule now
  const effectiveMode = (mode === "face" || mode === "rule") ? mode : "rule";

  const probs = ruleProbs();
  const actions = ["SELL", "HOLD", "BUY"];
  const { topIdx, secondIdx } = pickTopTwo(probs);

  const action = actions[topIdx];
  const conf = probs[topIdx];
  const second = actions[secondIdx];
  const secondConf = probs[secondIdx];

  confidenceEMA = ema(confidenceEMA, conf, 0.25);

  const base = { action, probs, origin: "rule", conf, second, secondConf };

  if (effectiveMode === "face") {
    const msg = buildCoachMessage(base, facePresent, engagementEMA, moodLabel);
    const adjusted = adjustForFaceV2(base, facePresent, engagementEMA, moodLabel);
    return {
      action: adjusted,
      probs,
      origin: facePresent ? "Face(" + moodLabel + ")" : "Face(absent)",
      message: msg,
      conf,
      second,
      secondConf
    };
  }

  return { ...base, origin: "rule", message: "Rule-based suggestion." };
}

// ===== COACH MESSAGES =====
function buildCoachMessage(pred, present, engage, mood) {
  const conf = pred.conf ?? 0.3;
  const top = pred.action;
  const second = pred.second || (top === "BUY" ? "HOLD" : "BUY");
  const gap = conf - (pred.secondConf ?? (conf - 0.05));

  if (!present) return "You are not fully engaged — safest is to HOLD this round.";

  if (engage >= 0.6 && (mood === "positive" || mood === "focused")) {
    if (conf >= 0.60 && gap >= 0.10) {
      if (top === "BUY")  return "You seem positive and the signal agrees — buying is reasonable here.";
      if (top === "SELL") return "You look confident and the signal tilts down — selling makes sense.";
      return "You are focused and the signal likes your position — HOLD is fine.";
    }
    return "You are engaged — signal leans " + top + "; " + second + " is also defensible if you want less risk.";
  }

  if (engage >= 0.35 && mood === "unsure") {
    if (conf >= 0.55 && top !== "SELL") {
      return "You look a bit unsure; the signal still prefers " + top + ". Start small or set a stop.";
    }
    return "You seem unsure — HOLD or take a smaller position until the signal firms up.";
  }

  if (top === "HOLD")
    return "Signal and focus are not strong — HOLD is safer; BUY offers more upside but also more risk.";

  return "Signal and focus are not strong — HOLD is the safer option until you are ready.";
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

  const msg = executePlayerTrade(action, currentPrice);
  log(msg);
  statusEl.textContent = msg;

  const aiSuggestion = await getAISuggestion();
  const beforeAICash = aiCash, beforeAIShares = aiShares;
  executeAITrade(aiSuggestion.action, currentPrice);
  if (aiCash !== beforeAICash || aiShares !== beforeAIShares) aiTradesMade++;

  advanceDay();
  updateUI();

  // accuracy scoring (same logic as before)
  const reward = currentPrice - prevPrice;
  aiTotal++;
  const aiWasRight =
    (reward > 0 && aiSuggestion.action === "BUY") ||
    (reward < 0 && aiSuggestion.action === "SELL") ||
    (Math.abs(reward) < 0.01 && aiSuggestion.action === "HOLD");
  if (aiWasRight) aiCorrect++;

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
      (origin || "rule") + " · face=" + (facePresent ? "yes" : "no") +
      " · sell " + (probs[0]*100).toFixed(0) + "%" +
      " · hold " + (probs[1]*100).toFixed(0) + "%" +
      " · buy "  + (probs[2]*100).toFixed(0) + "%";
  } else {
    aiReasonEl.textContent = (origin || "rule") + " · face=" + (facePresent ? "yes" : "no");
  }

  moodMsgEl.textContent = message || "-";
  log("AI suggests (" + (origin || "rule") + "): " + action);
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
    faceHintEl.textContent = "Camera blocked — face mode disabled.";
    return;
  }

  try {
    blazeModel = await blazeface.load();
    faceHintEl.textContent = "BlazeFace loaded (choose Face-aware coach).";
  } catch (err) {
    console.error("blazeface load error:", err);
    faceHintEl.textContent = "Could not load face model — use Rule mode.";
    return;
  }

  setInterval(async () => {
    if (!blazeModel) return;
    if (!videoEl || videoEl.readyState < 2) return;

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
      faceHintEl.textContent = "Face detect error — using default AI.";
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
  const s = String(raw).trim().replace(/^\uFEFF/, "").replace(/,+/g, "");

  // yyyy-mm-dd or yyyy/mm/dd
  let m = s.match(/^(\d{4})[\/-](\d{1,2})[\/-](\d{1,2})$/);
  if (m) return new Date(+m[1], +m[2] - 1, +m[3]);

  // dd/mm/yyyy or dd-mm-yyyy
  m = s.match(/^(\d{1,2})[\/-](\d{1,2})[\/-](\d{2,4})$/);
  if (m) {
    const dd = +m[1], mm = +m[2], yy = +m[3];
    const yyyy = yy < 100 ? (2000 + yy) : yy;
    return new Date(yyyy, mm - 1, dd);
  }

  // 23 Sep 2025 / Sep 23 2025
  const months = {
    jan:0,feb:1,mar:2,apr:3,may:4,jun:5,jul:6,aug:7,sep:8,sept:8,oct:9,nov:10,dec:11,
    january:0,february:1,march:2,april:3,june:5,july:6,august:7,september:8,october:9,november:10,december:11
  };

  m = s.match(/^(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$/);
  if (m && months[m[2].toLowerCase()] !== undefined)
    return new Date(+m[3], months[m[2].toLowerCase()], +m[1]);

  m = s.match(/^([A-Za-z]+)\s+(\d{1,2})\s+(\d{4})$/);
  if (m && months[m[1].toLowerCase()] !== undefined)
    return new Date(+m[3], months[m[1].toLowerCase()], +m[2]);

  return null;
}

function parseCSVToSeries(text) {
  const clean = text.replace(/\r/g, "\n");
  const rows = clean.split("\n").map(r => r.trim()).filter(Boolean);
  if (!rows.length) throw new Error("CSV appears to be empty.");

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
    if (priceCol === -1) priceCol = header.length - 1;
    if (dateCol === -1)  dateCol = 0;
  }

  const outPrices = [];
  const outDates  = [];

  for (const row of dataRows) {
    const cols = row.split(",").map(x => x.trim());
    if (!cols.length) continue;

    let v = (priceCol >= 0 && priceCol < cols.length) ? cols[priceCol] : cols[cols.length - 1];
    v = v.replace(/["']/g, "").replace(/\s/g, "").replace(/,/g, "");
    const num = parseFloat(v);
    if (!isFinite(num)) continue;

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

  fullPrices = rows.map(r => +r.p.toFixed(2));
  fullDates  = rows.map(r => r.d);

  cursor = 0;
  viewOffset = 0;
  prices = [];
  dates  = [];

  const start = 0;
  const end = Math.min(SEED_WINDOW, fullPrices.length);

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

  // No training step anymore
  statusEl.textContent = "CSV loaded. EMPATH is running in rule-based mode.";
}

// ===== DATE PANEL HELPERS =====
function latestLabel() {
  const i = prices.length - 1;
  const d = dates[i];
  const dateTxt = d ? fmtDate(d) : "Day " + (i + 1);
  return `${dateTxt} · ${usd(prices[i])}`;
}
