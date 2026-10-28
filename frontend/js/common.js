/* ══════════════════════════════════════════════════════════
   WaveDoc — shared helpers
   ══════════════════════════════════════════════════════════ */

/* Where the backend lives.
   - same server (python backend/app.py)      → ''  (relative /api)
   - frontend on Vercel, backend elsewhere    → open the site once with
       https://your-site.vercel.app/?api=https://your-backend-url
     The address is remembered in this browser; ?api= again to change it,
     ?api=reset to clear it. */
const API = (() => {
  let saved = '';
  try {
    const q = new URLSearchParams(location.search).get('api');
    if (q === 'reset') localStorage.removeItem('wavedoc_api');
    else if (q) localStorage.setItem('wavedoc_api', q.replace(/\/+$/, ''));
    saved = localStorage.getItem('wavedoc_api') || '';
  } catch (e) {}
  if (saved) return saved;
  return (location.port === '8000' || location.port === '')
    ? '' : `${location.protocol}//${location.hostname}:8000`;
})();

async function api(path, opts) {
  const res = await fetch(`${API}/api${path}`, opts);
  if (!res.ok) {
    let msg = `${res.status}`;
    try { const j = await res.json(); msg = j.detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return res.json();
}

/* ── toast ─────────────────────────────────────────────── */
let toastTimer;
function toast(msg, isError) {
  let el = document.querySelector('.toast');
  if (!el) {
    el = document.createElement('div');
    el.className = 'toast';
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.classList.toggle('err', !!isError);
  requestAnimationFrame(() => el.classList.add('show'));
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 3600);
}

/* ── colour ────────────────────────────────────────────── */
const PALETTE = ['#5AC8FA', '#A78BFA', '#FF7AB6', '#50FFD6', '#FFB84D'];

function hsvToRgb(h, s, v) {
  const c = v * s, x = c * (1 - Math.abs(((h / 60) % 2) - 1)), m = v - c;
  let r, g, b;
  if (h < 60)       { r = c; g = x; b = 0; }
  else if (h < 120) { r = x; g = c; b = 0; }
  else if (h < 180) { r = 0; g = c; b = x; }
  else if (h < 240) { r = 0; g = x; b = c; }
  else if (h < 300) { r = x; g = 0; b = c; }
  else              { r = c; g = 0; b = x; }
  return [(r + m) * 255, (g + m) * 255, (b + m) * 255];
}

/* Phase in [-pi, pi] → colour wheel */
function phaseColour(p) {
  const h = (((p + Math.PI) / (2 * Math.PI)) * 360) % 360;
  return hsvToRgb(h, 0.74, 0.94);
}

/* ── canvas helpers ────────────────────────────────────── */
function fitCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const r = cv.getBoundingClientRect();
  cv.width = Math.max(1, Math.round(r.width * dpr));
  cv.height = Math.max(1, Math.round(r.height * dpr));
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: r.width, h: r.height };
}

/* Draw a matrix as a heat image */
function drawMatrix(cv, mat, colourFn) {
  if (!mat || !mat.length) return;
  const rows = mat.length, cols = mat[0].length;
  const off = document.createElement('canvas');
  off.width = cols; off.height = rows;
  const octx = off.getContext('2d');
  const img = octx.createImageData(cols, rows);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const [r, g, b] = colourFn(mat[y][x]);
      const i = (y * cols + x) * 4;
      img.data[i] = r; img.data[i + 1] = g;
      img.data[i + 2] = b; img.data[i + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);
  const { ctx, w, h } = fitCanvas(cv);
  ctx.imageSmoothingEnabled = true;
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(off, 0, 0, w, h);
}

/* Greyscale colour map for ultrasound frames */
function greyColour(min, max) {
  const span = (max - min) || 1;
  return v => {
    const t = Math.max(0, Math.min(1, (v - min) / span));
    const g = 12 + t * 233;
    return [g, g, g];
  };
}

/* Discrete mask colours: background, LV, myocardium, LA */
const MASK_COLOURS = [[10, 14, 26], [90, 200, 250],
                      [167, 139, 250], [255, 122, 182]];
const maskColour = v => MASK_COLOURS[v] || MASK_COLOURS[0];

/* ══════════════════════════════════════════════════════════
   Live monitor — a scrolling trace like a bedside display
   ══════════════════════════════════════════════════════════ */
class LiveMonitor {
  constructor(canvas, opts = {}) {
    this.cv = canvas;
    this.speed = opts.speed || 2.2;       // px per frame
    this.colour = opts.colour || '#5AC8FA';
    this.glow = opts.glow !== false;
    this.buffer = [];                     // source waveform, one cycle
    this.cursor = 0;
    this.trail = [];                      // points currently on screen
    this.running = false;
    this.bpmEl = opts.bpmEl || null;
    this.rate = opts.rate || 360;
    this._boundResize = () => this.resize();
    window.addEventListener('resize', this._boundResize);
  }

  /* Feed it a real recording; it loops seamlessly */
  setWaveform(arr) {
    if (!arr || !arr.length) return;
    const min = Math.min(...arr), max = Math.max(...arr);
    const span = (max - min) || 1;
    this.buffer = arr.map(v => (v - min) / span * 2 - 1);  // → [-1, 1]
    this.cursor = 0;
    this.trail = [];
  }

  /* A synthetic ECG so the display is never blank */
  useDemoWaveform(n = 900) {
    const s = new Array(n);
    const beat = 150;
    for (let i = 0; i < n; i++) {
      const p = i % beat;
      let v = 0.05 * Math.sin(i * 0.05);
      v += 0.11 * Math.exp(-Math.pow(p - 30, 2) / 34);
      v -= 0.09 * Math.exp(-Math.pow(p - 56, 2) / 6);
      v += 1.00 * Math.exp(-Math.pow(p - 62, 2) / 7);
      v -= 0.24 * Math.exp(-Math.pow(p - 70, 2) / 11);
      v += 0.20 * Math.exp(-Math.pow(p - 104, 2) / 150);
      s[i] = v;
    }
    this.setWaveform(s);
  }

  resize() {
    if (!this.cv) return;
    const g = fitCanvas(this.cv);
    this.w = g.w; this.h = g.h; this.ctx = g.ctx;
    this.trail = this.trail.filter(p => p.x <= this.w);
  }

  start() {
    if (this.running) return;
    this.resize();
    if (!this.buffer.length) this.useDemoWaveform();
    this.running = true;
    this._loop();
  }

  stop() {
    this.running = false;
    if (this._raf) cancelAnimationFrame(this._raf);
  }

  destroy() {
    this.stop();
    window.removeEventListener('resize', this._boundResize);
  }

  _grid() {
    const { ctx, w, h } = this;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,.055)';
    ctx.lineWidth = 1;
    const step = 34;
    for (let x = (this._gridOff || 0) % step; x < w; x += step) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += step) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
    ctx.restore();
  }

  _loop() {
    if (!this.running) return;
    const { ctx, w, h } = this;
    const mid = h * 0.54, amp = h * 0.34;

    /* advance: pull the next few samples into the trail */
    const perFrame = Math.max(1, Math.round(this.speed));
    for (let k = 0; k < perFrame; k++) {
      const v = this.buffer[this.cursor % this.buffer.length];
      this.cursor++;
      this.trail.push({ x: w, y: mid - v * amp });
    }
    /* scroll everything left */
    for (const p of this.trail) p.x -= this.speed;
    while (this.trail.length && this.trail[0].x < -4) this.trail.shift();
    this._gridOff = (this._gridOff || 0) - this.speed * 0.35;

    /* paint */
    ctx.clearRect(0, 0, w, h);
    this._grid();

    if (this.trail.length > 1) {
      /* soft fill under the trace */
      ctx.beginPath();
      ctx.moveTo(this.trail[0].x, mid);
      for (const p of this.trail) ctx.lineTo(p.x, p.y);
      ctx.lineTo(this.trail[this.trail.length - 1].x, mid);
      ctx.closePath();
      const grad = ctx.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, 'rgba(90,200,250,.22)');
      grad.addColorStop(1, 'rgba(90,200,250,0)');
      ctx.fillStyle = grad;
      ctx.fill();

      /* the trace itself */
      ctx.beginPath();
      ctx.moveTo(this.trail[0].x, this.trail[0].y);
      for (const p of this.trail) ctx.lineTo(p.x, p.y);
      ctx.strokeStyle = this.colour;
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      if (this.glow) {
        ctx.shadowColor = this.colour;
        ctx.shadowBlur = 11;
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

      /* leading dot */
      const tip = this.trail[this.trail.length - 1];
      ctx.beginPath();
      ctx.arc(tip.x, tip.y, 3.4, 0, Math.PI * 2);
      ctx.fillStyle = '#FFFFFF';
      ctx.shadowColor = this.colour;
      ctx.shadowBlur = 14;
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    this._raf = requestAnimationFrame(() => this._loop());
  }
}

/* ── nav highlight ─────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  const here = location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-links a').forEach(a => {
    const href = a.getAttribute('href');
    if (href === here || (here === '' && href === 'index.html')) {
      a.classList.add('on');
    }
  });
});
