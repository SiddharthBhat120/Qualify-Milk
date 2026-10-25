/* Analyser: live predictions as you move the controls */
(() => {
  const form = document.getElementById("lab-form");
  if (!form) return;
  const glass = document.querySelector(".result [data-glass]").glass;
  const verdict = document.getElementById("verdict");
  const conf = document.getElementById("confidence");
  const notes = document.getElementById("notes");
  const live = document.getElementById("live");
  const errBox = document.getElementById("result-error");
  const saveBtn = document.getElementById("save-btn");
  const labelInput = document.getElementById("sample-label");
  const presets = JSON.parse(document.getElementById("presets").textContent);
  const GRADE_TEXT = { high: "High", medium: "Medium", low: "Low" };
  let lastGrade = null, timer = null, ctrl = null;

  // Milk colour 240 (yellowish) … 255 (pure white)
  const milkColour = (v) => {
    const t = (v - 240) / 15;
    const mix = (a, b) => Math.round(a + (b - a) * t);
    return `rgb(${mix(242, 255)}, ${mix(222, 255)}, ${mix(170, 252)})`;
  };

  const values = () => {
    const fd = new FormData(form), out = {};
    for (const [k, v] of fd.entries()) if (k !== "label") out[k] = Number(v);
    return out;
  };

  function paintControls() {
    form.querySelectorAll('input[type="range"]').forEach((r) => {
      const p = ((r.value - r.min) / (r.max - r.min)) * 100;
      r.style.setProperty("--p", `${p}%`);
      const out = document.getElementById(`${r.name}-out`);
      if (out) out.textContent = r.name === "pH" ? Number(r.value).toFixed(1) : r.name === "Temperature" ? `${r.value}°` : r.value;
    });
    form.querySelectorAll(".seg-track").forEach((t) => {
      const checked = t.querySelector("input:checked");
      const second = checked && checked === t.querySelectorAll("input")[1];
      t.classList.toggle("second", !!second);
      t.classList.toggle("bad", checked?.dataset.bad === "1");
    });
    const c = form.elements.Colour.value;
    document.getElementById("colour-swatch").style.background = milkColour(c);
    glass.set({ color: milkColour(c) });
  }

  function show(r) {
    errBox.hidden = true;
    if (r.grade !== lastGrade) {
      verdict.className = `verdict grade-${r.grade}`;
      void verdict.offsetWidth; // restart animation
      verdict.classList.add("pop");
      verdict.textContent = GRADE_TEXT[r.grade];
      lastGrade = r.grade;
    }
    conf.textContent = `${r.confidence}% confident`;
    for (const g of ["high", "medium", "low"]) {
      const row = document.querySelector(`.prob[data-g="${g}"]`);
      row.querySelector("i").style.width = `${r.probabilities[g]}%`;
      row.querySelector("span:last-child").textContent = `${Math.round(r.probabilities[g])}%`;
    }
    const level = { high: 0.86, medium: 0.6, low: 0.32 }[r.grade];
    glass.set({ level: level * (0.75 + (r.confidence / 100) * 0.25) });
    notes.innerHTML = r.insights.map((n, i) => `<li class="${n.level}" style="--i:${i}">${escapeHtml(n.text)}</li>`).join("");
  }

  const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  async function run(save = false) {
    ctrl?.abort();
    ctrl = new AbortController();
    live.classList.add("busy");
    live.lastChild.textContent = save ? " Saving…" : " Analysing…";
    try {
      const body = { ...values(), save, label: labelInput.value };
      const res = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: ctrl.signal });
      const data = await res.json();
      if (res.status === 401) { location.href = "/auth?next=/analyse"; return; }
      if (!res.ok) throw new Error(data.error || "The prediction failed.");
      show(data);
      if (save && data.saved_id) {
        toast(`Saved to history as ${GRADE_TEXT[data.grade]} quality.`, "success");
        labelInput.value = "";
      }
    } catch (e) {
      if (e.name === "AbortError") return;
      errBox.hidden = false;
      errBox.textContent = e.message === "Failed to fetch" ? "Can't reach the server. Is app.py running?" : e.message;
    } finally {
      live.classList.remove("busy");
      live.lastChild.textContent = " Live";
    }
  }

  function schedule() {
    paintControls();
    clearTimeout(timer);
    timer = setTimeout(() => run(false), 140);
  }

  form.addEventListener("input", schedule);
  form.addEventListener("submit", (e) => { e.preventDefault(); run(true); });

  document.querySelectorAll(".preset").forEach((b) =>
    b.addEventListener("click", () => {
      const p = presets[b.dataset.grade];
      for (const [k, v] of Object.entries(p)) {
        const el = form.elements[k];
        if (el instanceof RadioNodeList) [...el].forEach((r) => (r.checked = Number(r.value) === v));
        else el.value = v;
      }
      labelInput.value = `${b.textContent.trim()} sample`;
      schedule();
    })
  );

  function toast(msg, kind) {
    let box = document.querySelector(".toasts");
    if (!box) { box = document.createElement("div"); box.className = "toasts"; document.body.append(box); }
    const t = document.createElement("div");
    t.className = `toast ${kind}`;
    t.setAttribute("role", "status");
    t.textContent = msg;
    box.append(t);
    setTimeout(() => t.remove(), 5200);
  }

  paintControls();
  run(false);
})();
