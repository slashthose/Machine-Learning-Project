// ══════════════════════════════════════════════════════════
// SAKSHI SECURE BANK — frontend logic (no framework, vanilla JS)
// ══════════════════════════════════════════════════════════

const API = ""; // same-origin, FastAPI serves both API and static files

let SAMPLES = {};
let currentV = new Array(28).fill(0);

const el = (id) => document.getElementById(id);

// ── Load samples + populate dropdown ─────────────────────
async function loadSamples() {
  try {
    const res = await fetch(`${API}/api/samples`);
    SAMPLES = await res.json();
    const select = el("sampleSelect");
    Object.keys(SAMPLES).forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      select.appendChild(opt);
    });
  } catch (err) {
    showNotice(
      "Could not reach the API to load sample transactions. Is the FastAPI server running?",
    );
  }
}

function showNotice(msg) {
  const n = el("apiNotice");
  n.textContent = msg;
  n.classList.remove("hidden");
}

// FastAPI returns `detail` as a plain string for our own HTTPException calls,
// but as an array of {loc, msg, type} objects for Pydantic validation errors
// (e.g. amount/time out of range). Handle both so we never render "[object Object]".
function formatApiError(err) {
  if (!err || !err.detail) return "Prediction failed.";
  if (typeof err.detail === "string") return err.detail;
  if (Array.isArray(err.detail)) {
    return err.detail
      .map((e) => {
        const field = Array.isArray(e.loc) ? e.loc[e.loc.length - 1] : "input";
        return `${field}: ${e.msg}`;
      })
      .join(" — ");
  }
  return "Prediction failed.";
}

// ── Sample selection autofills amount/time/V-features ────
el("sampleSelect").addEventListener("change", (e) => {
  const name = e.target.value;
  const tag = el("sampleTag");
  if (!name) {
    currentV = new Array(28).fill(0);
    tag.classList.add("hidden");
    return;
  }
  const sample = SAMPLES[name];
  el("amountInput").value = sample.amount;
  el("timeInput").value = sample.time;
  currentV = sample.v;

  tag.classList.remove("hidden");
  tag.className = `sample-tag ${sample.label}`;
  tag.textContent =
    sample.label === "fraud"
      ? "Known fraud pattern from the dataset"
      : "Known legitimate transaction from the dataset";
});

// ── Floating card tilt on mousemove, click to flip ───────
const stage = el("cardStage");
const cardFlip = el("cardFlip");
let isFlipped = false;

stage.addEventListener("mousemove", (e) => {
  if (isFlipped) return;
  const r = stage.getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width - 0.5;
  const y = (e.clientY - r.top) / r.height - 0.5;
  cardFlip.style.transform = `rotateY(${x * 22}deg) rotateX(${-y * 22}deg) translateZ(10px)`;
});
stage.addEventListener("mouseleave", () => {
  if (!isFlipped) cardFlip.style.transform = "";
});
cardFlip.addEventListener("click", () => {
  isFlipped = !isFlipped;
  cardFlip.style.transform = "";
  cardFlip.classList.toggle("flipped", isFlipped);
});

// ── Masked card number + CVV derived from the transaction ─
function digitsFromSeed(seedText, count) {
  let hash = 0;
  for (let i = 0; i < seedText.length; i++) {
    hash = (hash * 31 + seedText.charCodeAt(i)) >>> 0;
  }
  return String(hash % Math.pow(10, count)).padStart(count, "7");
}

// ── Analyze button ───────────────────────────────────────
el("analyzeBtn").addEventListener("click", async () => {
  const amount = parseFloat(el("amountInput").value);
  const time = parseFloat(el("timeInput").value);
  const sampleName = el("sampleSelect").value;

  if (isNaN(amount) || isNaN(time)) {
    showNotice("Enter a valid amount and time before analyzing.");
    return;
  }

  el("analyzeBtn").disabled = true;
  el("analyzeBtn").querySelector("svg") || null;
  const btnLabel = el("analyzeBtn");
  const originalHTML = btnLabel.innerHTML;
  btnLabel.textContent = "Scanning…";

  // make sure the front face is showing, then trigger the scan sweep
  if (isFlipped) {
    isFlipped = false;
    cardFlip.classList.remove("flipped");
  }
  const scanline = el("scanline");
  scanline.classList.remove("active");
  void scanline.offsetWidth; // restart animation
  scanline.classList.add("active");

  try {
    const res = await fetch(`${API}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ time, amount, v: currentV }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(formatApiError(err));
    }

    const data = await res.json();

    if (data.demo_mode) {
      showNotice(
        "model.pkl not found on the server — showing a rough heuristic estimate, not a real prediction.",
      );
    } else if (data.using_fallback_scaler) {
      showNotice(
        "scalers.pkl not found — using approximate reference statistics to scale Time/Amount. Rerun the training notebook to generate exact scalers.",
      );
    } else {
      el("apiNotice").classList.add("hidden");
    }

    renderResult(data, amount, sampleName || `${amount}-${time}`);
  } catch (err) {
    showNotice(err.message || "Something went wrong reaching the API.");
  } finally {
    el("analyzeBtn").disabled = false;
    btnLabel.innerHTML = originalHTML;
  }
});

function renderResult(data, amount, seed) {
  el("resultEmpty").classList.add("hidden");
  el("resultBody").classList.remove("hidden");

  const isFraud = data.prediction === 1;

  const verdictBadge = el("verdictBadge");
  verdictBadge.textContent = isFraud ? "FRAUD" : "LEGITIMATE";
  verdictBadge.className = `verdict-badge ${isFraud ? "fraud" : "legit"}`;

  const riskPill = el("riskPill");
  riskPill.textContent = `${data.risk_level} RISK`;
  riskPill.className = `risk-pill ${data.risk_level}`;

  el("fraudPct").textContent = `${data.fraud_prob}%`;
  el("legitPct").textContent = `${data.legit_prob}%`;
  el("fraudBar").style.width = `${data.fraud_prob}%`;
  el("legitBar").style.width = `${data.legit_prob}%`;

  el("recommendationText").textContent = isFraud
    ? "Block the transaction and alert the customer immediately."
    : "Approve and process the transaction.";

  // update the floating card
  cardFlip.classList.remove("fraud", "legit");
  cardFlip.classList.add(isFraud ? "fraud" : "legit");

  const badge = el("cardBadge");
  badge.textContent = isFraud ? "FRAUD DETECTED" : "LEGITIMATE";
  badge.className = `card-badge ${isFraud ? "fraud" : "legit"}`;

  el("cardAmount").textContent = `$${amount.toFixed(2)}`;
  el("cardRisk").textContent = `${data.fraud_prob}%`;
  el("cardNumbers").textContent =
    `•••• •••• •••• ${digitsFromSeed(String(seed), 4)}`;
  el("cvvBox").textContent = digitsFromSeed(String(seed) + "-cvv", 3);
}

// ── Results tab: model comparison table + chart ──────────
async function loadModelResults() {
  try {
    const res = await fetch(`${API}/api/model-results`);
    const results = await res.json();

    const tbody = document.querySelector("#resultsTable tbody");
    results.forEach((r) => {
      const tr = document.createElement("tr");
      if (r.best) tr.className = "best";
      tr.innerHTML = `<td>${r.model}${r.best ? " ★" : ""}</td><td>${r.accuracy.toFixed(2)}%</td><td>${r.roc_auc.toFixed(4)}</td>`;
      tbody.appendChild(tr);
    });

    const ctx = el("resultsChart").getContext("2d");
    new Chart(ctx, {
      type: "bar",
      data: {
        labels: results.map((r) => r.model),
        datasets: [
          {
            label: "Accuracy (%)",
            data: results.map((r) => r.accuracy),
            backgroundColor: "#3E92CC",
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          y: {
            min: 92,
            max: 101,
            ticks: { color: "#9AA6C4" },
            grid: { color: "rgba(255,255,255,0.08)" },
          },
          x: { ticks: { color: "#9AA6C4" }, grid: { display: false } },
        },
      },
    });
  } catch (err) {
    // results section degrades gracefully if API is unreachable
  }
}

loadSamples();
loadModelResults();
