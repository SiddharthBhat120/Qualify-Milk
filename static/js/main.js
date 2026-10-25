/* Qualify Milk: shared UI behaviour */
(() => {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ── Animated milk glass ─────────────────────────────────────────────
     <div data-glass data-level="0.7" data-color="#fff"></div>
     el.glass.set({ level, color }) changes it with a smooth slosh.     */
  let uid = 0;
  class MilkGlass {
    constructor(el) {
      this.el = el;
      this.level = this.target = parseFloat(el.dataset.level || "0.72");
      this.tilt = 0;
      this.slosh = 0;
      const id = `g${++uid}`;
      const color = el.dataset.color || "#ffffff";
      el.classList.add("glass-wrap");
      el.innerHTML = `
        <svg class="glass-svg" viewBox="0 0 200 260" role="img" aria-label="Glass of milk">
          <defs><clipPath id="${id}"><path d="M28 18 L172 18 L156 244 Q155 252 146 252 L54 252 Q45 252 44 244 Z"/></clipPath></defs>
          <g clip-path="url(#${id})">
            <path class="liquid back" fill="${color}" opacity="0.55"/>
            <path class="liquid front" fill="${color}"/>
            ${Array.from({ length: 7 }, (_, i) => `<circle class="bubble" r="${2 + (i % 3)}" data-i="${i}"/>`).join("")}
          </g>
          <path class="glass-outline" d="M28 18 L172 18 L156 244 Q155 252 146 252 L54 252 Q45 252 44 244 Z" stroke-linejoin="round"/>
          <path d="M150 40 L140 200" stroke="rgba(255,255,255,.55)" stroke-width="6" stroke-linecap="round" fill="none"/>
        </svg>`;
      this.back = el.querySelector(".back");
      this.front = el.querySelector(".front");
      this.bubbles = [...el.querySelectorAll(".bubble")].map((b, i) => ({ b, x: 60 + ((i * 37) % 90), y: 250 - i * 30, s: 0.4 + (i % 4) * 0.2 }));
      this.t = Math.random() * 10;
      el.glass = this;
      this.draw = this.draw.bind(this);
      if (reduce) this.render(); else requestAnimationFrame(this.draw);
    }
    set({ level, color }) {
      if (typeof level === "number") {
        this.slosh = Math.min(1, this.slosh + Math.abs(level - this.target) * 3 + 0.35);
        this.target = Math.max(0.08, Math.min(0.95, level));
      }
      if (color) { this.back.setAttribute("fill", color); this.front.setAttribute("fill", color); }
      if (reduce) { this.level = this.target; this.render(); }
    }
    wave(offset, amp, phase) {
      const top = 252 - this.level * 234;
      let d = `M0 ${top}`;
      for (let x = 0; x <= 200; x += 10) {
        const y = top + Math.sin(x / 26 + phase) * amp + (x - 100) * this.tilt * 0.12 + offset;
        d += ` L${x} ${y.toFixed(1)}`;
      }
      return `${d} L200 260 L0 260 Z`;
    }
    render() {
      const amp = 3 + this.slosh * 10;
      this.back.setAttribute("d", this.wave(-3, amp, this.t * 1.3 + 2));
      this.front.setAttribute("d", this.wave(0, amp * 0.8, this.t * 1.8));
      const top = 252 - this.level * 234;
      for (const o of this.bubbles) {
        o.y -= o.s;
        if (o.y < top + 6) o.y = 250;
        o.b.setAttribute("cx", o.x + Math.sin(this.t * 2 + o.x) * 3);
        o.b.setAttribute("cy", o.y);
        o.b.setAttribute("opacity", o.y < top + 14 ? 0 : 0.8);
      }
    }
    draw() {
      this.t += 0.022;
      this.level += (this.target - this.level) * 0.06;
      this.slosh *= 0.975;
      this.render();
      requestAnimationFrame(this.draw);
    }
  }
  document.querySelectorAll("[data-glass]").forEach((el) => new MilkGlass(el));
  window.MilkGlass = MilkGlass;

  // Hero glass leans toward the cursor
  const heroGlass = document.querySelector(".hero [data-glass]");
  if (heroGlass && !reduce) {
    window.addEventListener("pointermove", (e) => {
      heroGlass.glass.tilt = (e.clientX / innerWidth - 0.5) * 0.8;
      heroGlass.glass.slosh = Math.min(1, heroGlass.glass.slosh + 0.01);
    });
  }

  /* ── Reveal on scroll ── */
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      e.target.classList.add("in");
      e.target.querySelectorAll("[data-w]").forEach((b) => (b.style.width = b.dataset.w));
      e.target.querySelectorAll("[data-count]").forEach(countUp);
      io.unobserve(e.target);
    });
  }, { threshold: 0.15 });
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));

  /* ── Count-up numbers: <b data-count="99.81" data-suffix="%"> ── */
  function countUp(el) {
    const end = parseFloat(el.dataset.count);
    const dec = (el.dataset.count.split(".")[1] || "").length;
    const suffix = el.dataset.suffix || "";
    if (reduce) { el.textContent = end.toFixed(dec) + suffix; return; }
    const t0 = performance.now();
    const step = (t) => {
      const p = Math.min(1, (t - t0) / 1400);
      const v = end * (1 - Math.pow(1 - p, 4));
      el.textContent = v.toLocaleString("en-IN", { minimumFractionDigits: dec, maximumFractionDigits: dec }) + suffix;
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  document.querySelectorAll(".hero [data-count]").forEach((el) => setTimeout(() => countUp(el), 600));

  /* ── Nav: hide on scroll down, burger menu ── */
  const nav = document.querySelector(".nav");
  if (nav) {
    let last = scrollY;
    addEventListener("scroll", () => {
      const y = scrollY;
      nav.classList.toggle("hide", y > last && y > 200 && !nav.classList.contains("open"));
      last = y;
    }, { passive: true });
    const burger = nav.querySelector(".nav-burger");
    burger?.addEventListener("click", () => {
      const open = nav.classList.toggle("open");
      burger.setAttribute("aria-expanded", open);
    });
  }
})();
