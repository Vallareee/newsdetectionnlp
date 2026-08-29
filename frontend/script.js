const newsInput = document.getElementById("newsInput");
const imageInput = document.getElementById("imageInput");
const imagePanel = document.getElementById("imagePanel");
const textLabel = document.getElementById("textLabel");
const checkBtn = document.getElementById("checkBtn");
const resultDiv = document.getElementById("result");
const statusEl = document.getElementById("status");
const preview = document.getElementById("preview");
const dropLabel = document.getElementById("dropLabel");

let mode = "text";

document.querySelectorAll(".mode").forEach((btn) => {
  btn.addEventListener("click", () => {
    mode = btn.dataset.mode;
    document.querySelectorAll(".mode").forEach((b) => b.classList.toggle("active", b === btn));
    const needsImage = mode !== "text";
    const needsText = mode !== "image";
    imagePanel.classList.toggle("hidden", !needsImage);
    newsInput.classList.toggle("hidden", !needsText);
    textLabel.classList.toggle("hidden", !needsText);
    textLabel.textContent = mode === "both" ? "Caption / claim to match with the image:" : "Enter to check:";
  });
});

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) {
    preview.classList.add("hidden");
    dropLabel.textContent = "Tap to upload a photo, or take one on mobile";
    return;
  }
  dropLabel.textContent = file.name;
  preview.src = URL.createObjectURL(file);
  preview.classList.remove("hidden");
});

checkBtn.addEventListener("click", checkNews);

function setBusy(busy, message) {
  checkBtn.disabled = busy;
  statusEl.hidden = !message;
  statusEl.textContent = message || "";
}

async function checkNews() {
  resultDiv.className = "result";
  resultDiv.innerHTML = "";

  try {
    if (mode === "text") {
      const text = newsInput.value.trim();
      if (!text) {
        resultDiv.textContent = "Please enter some text.";
        return;
      }
      setBusy(true, "Scoring with TF-IDF + LR / Naive Bayes / Random Forest…");
      const data = await postJSON("/api/check-text", { text });
      renderResult(data);
      return;
    }

    const file = imageInput.files[0];
    if (!file) {
      resultDiv.textContent = "Please upload an image.";
      return;
    }
    if (mode === "both" && !newsInput.value.trim()) {
      resultDiv.textContent = "Add the caption or claim that was posted with this image.";
      return;
    }

    setBusy(true, "Running reverse image search, then the college scoring formula…");
    const body = new FormData();
    body.append("image", file);
    body.append("text", newsInput.value.trim());
    const res = await fetch("/api/check-image", { method: "POST", body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    renderResult(data);
  } catch (err) {
    resultDiv.className = "result false";
    resultDiv.textContent = err.message || "Something went wrong.";
  } finally {
    setBusy(false);
  }
}

async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data;
}

function renderResult(data) {
  const label = data.label || "Misleading";
  resultDiv.className = "result " + label.toLowerCase();

  const first = data.reverse_search?.first_appearance || data.first_appearance;
  const searchMsg = data.reverse_search?.message;
  const pct = data.likelihood_true_pct;

  let html = `<div class="result-label">Result: ${label}</div>`;
  html += `<div class="meta">`;
  if (typeof pct === "number") {
    html += `<p>Likelihood of being true: <strong>${pct}%</strong></p>`;
  }
  if (data.note) {
    html += `<p>${escapeHtml(data.note)}</p>`;
  }
  if (typeof data.match_score === "number") {
    html += `<p>Caption ↔ first-appearance match: <strong>${Math.round(data.match_score * 100)}%</strong></p>`;
  }
  if (first?.url) {
    const when = first.date ? ` (${escapeHtml(first.date)})` : "";
    html += `<p>First appearance${when}: <a href="${escapeAttr(first.url)}" target="_blank" rel="noopener">${escapeHtml(first.title || first.url)}</a></p>`;
    if (first.snippet) html += `<p>${escapeHtml(first.snippet)}</p>`;
  } else if (searchMsg) {
    html += `<p>${escapeHtml(searchMsg)}</p>`;
  }
  html += `</div>`;
  resultDiv.innerHTML = html;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll("'", "&#39;");
}
