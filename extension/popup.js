// Popup logic: grab the selected text from the current tab, send it to the
// FastAPI /predict endpoint and render the risk score + red flags.

const DEFAULT_API = "http://localhost:8000";
const $ = (id) => document.getElementById(id);

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function getApiUrl() {
  const { apiUrl } = await chrome.storage.sync.get("apiUrl");
  return (apiUrl || DEFAULT_API).replace(/\/+$/, "");
}

async function getSelectedText() {
  // 1) text saved by the right-click menu
  const { pendingText } = await chrome.storage.local.get("pendingText");
  if (pendingText) {
    await chrome.storage.local.remove("pendingText");
    chrome.action.setBadgeText({ text: "" });
    return pendingText;
  }
  // 2) current selection in the active tab
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || !tab.id || /^chrome:|^edge:|^about:/.test(tab.url || "")) return "";
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => window.getSelection().toString()
    });
    return result || "";
  } catch (e) {
    return ""; // pages like the Chrome Web Store block scripting
  }
}

function render(res) {
  const flags = res.red_flags
    .map((f) => {
      const ev = f.triggered && f.evidence.length
        ? ` <span class="ev">“${escapeHtml(f.evidence.join("”, “"))}”</span>` : "";
      return `<li>${f.triggered ? "❌" : "✅"} ${escapeHtml(f.name)}${ev}</li>`;
    })
    .join("");
  const words = res.highlighted_words.slice(0, 8)
    .map((w) => `<mark>${escapeHtml(w.term)}</mark>`).join(" ");
  const warns = res.warnings.map((w) => `<div class="warn">⚠️ ${escapeHtml(w)}</div>`).join("");
  const verdict = { high: "Likely FAKE", medium: "Be careful", low: "Looks legitimate" }[res.risk_level];
  const color = { high: "#dc2626", medium: "#ca8a04", low: "#16a34a" }[res.risk_level];

  $("result").innerHTML = `
    <div class="score ${res.risk_level}">${res.risk_score.toFixed(0)}/100 · ${verdict}</div>
    <div class="bar"><div style="width:${res.risk_score}%;background:${color}"></div></div>
    <div class="muted">Prediction: <b>${res.label}</b> · model: ${escapeHtml(res.model_name)}</div>
    ${warns}
    <ul>${flags}</ul>
    ${words ? `<div class="muted">Suspicious words:</div><div>${words}</div>` : ""}
  `;
}

async function check() {
  const text = $("text").value.trim();
  if (!text) {
    $("result").innerHTML = `<div class="error">Select or paste a job post first.</div>`;
    return;
  }
  $("check").disabled = true;
  $("check").textContent = "Checking…";
  try {
    const api = await getApiUrl();
    const resp = await fetch(`${api}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, company: $("company").value, email: $("email").value, explain: "auto" })
    });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(typeof body.detail === "string" ? body.detail : `API error ${resp.status}`);
    }
    render(await resp.json());
  } catch (e) {
    $("result").innerHTML = `<div class="error">${escapeHtml(e.message)}<br>
      Is the API running? <code>uvicorn api.main:app --port 8000</code></div>`;
  } finally {
    $("check").disabled = false;
    $("check").textContent = "Check post";
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  $("apiUrl").value = await getApiUrl();
  $("apiUrl").addEventListener("change", (e) =>
    chrome.storage.sync.set({ apiUrl: e.target.value.trim() || DEFAULT_API }));
  $("check").addEventListener("click", check);
  const selected = await getSelectedText();
  if (selected) {
    $("text").value = selected.slice(0, 20000);
    check(); // auto-check when text was selected
  }
});
