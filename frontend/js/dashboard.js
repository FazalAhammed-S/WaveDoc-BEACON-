/* ══════════════════════════════════════════════════════════
   WaveDoc — dashboard
   ══════════════════════════════════════════════════════════ */

let MODELS = [];
let current = null;
let monitor = null;

const el = id => document.getElementById(id);

/* ── boot ──────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', async () => {
  monitor = new LiveMonitor(el('live'), { speed: 2.2 });
  monitor.useDemoWaveform();
  monitor.start();
  setBpm(72);

  try {
    const health = await api('/health');
    const ready = Object.values(health.available)
      .filter(v => v.weights).length;
    el('health').className = ready ? 'pill ok' : 'pill warn';
    el('health').innerHTML =
      `<span class="dot"></span>${health.device.toUpperCase()} · ${ready}/5 models`;
  } catch (e) {
    el('health').className = 'pill warn';
    el('health').textContent = 'API offline';
    toast('Cannot reach the API server. Is backend/app.py running?', true);
    return;
  }

  try {
    MODELS = await api('/models');
  } catch (e) {
    toast('Could not load the model list', true);
    return;
  }

  const sel = el('model-select');
  sel.innerHTML = MODELS.map(m =>
    `<option value="${m.key}"${m.has_weights ? '' : ' disabled'}>
       ${m.title} — ${m.subtitle}${m.has_weights ? '' : '  (weights missing)'}
     </option>`).join('');
  sel.addEventListener('change', () => selectModel(sel.value));
  selectModel(MODELS.find(m => m.has_weights)?.key || MODELS[0].key);

  el('btn-random').addEventListener('click', () => drawSample(-1));
  el('file').addEventListener('change', e => {
    if (e.target.files[0]) upload(e.target.files[0]);
  });

  const drop = el('drop');
  ['dragenter', 'dragover'].forEach(ev =>
    drop.addEventListener(ev, e => {
      e.preventDefault(); drop.classList.add('over');
    }));
  ['dragleave', 'drop'].forEach(ev =>
    drop.addEventListener(ev, e => {
      e.preventDefault(); drop.classList.remove('over');
    }));
  drop.addEventListener('drop', e => {
    const f = e.dataTransfer.files[0];
    if (f) upload(f);
  });
});

function setBpm(v) {
  el('bpm').innerHTML = `${v}<small>bpm</small>`;
}

/* ── model selection ──────────────────────────────────── */
function selectModel(key) {
  current = MODELS.find(m => m.key === key);
  if (!current) return;
  el('model-select').value = key;

  el('sp-shape').textContent = current.shape_text;
  el('sp-chan').textContent = current.channel_text;
  el('sp-rate').textContent = current.rate;
  el('sp-dur').textContent = current.duration;
  el('model-link').href = `models.html#${key}`;
  el('drop-hint').textContent =
    `${current.shape_text} · csv, txt or npy`;

  const b = current.baseline;
  el('model-note').innerHTML =
    `${current.dataset} · ${current.primary[0]}
     <strong style="color:var(--text)">${current.primary[1].toFixed(4)}</strong>
     vs ${b.name} ${b.value}`;

  el('class-chips').innerHTML = current.kind === 'signal'
    ? current.classes.map((c, i) =>
        `<button class="chip" data-cls="${i}">${c.split('—')[0].trim()}</button>`
      ).join('')
    : '';
  el('class-chips').querySelectorAll('.chip').forEach(c =>
    c.addEventListener('click', () => drawSample(+c.dataset.cls)));

  el('result').innerHTML = '';
  showEmpty();
}

function showEmpty() {
  el('result').innerHTML = `
    <div class="glass mt empty">
      <div class="icon">◈</div>
      <div class="t">Draw a random sample to begin</div>
      <div class="s">or drop your own recording above</div>
    </div>`;
}

/* ── fetch a sample ───────────────────────────────────── */
async function drawSample(cls) {
  if (!current) return;
  if (!current.has_data) {
    toast('No dataset on the server for this model — upload a file instead',
          true);
    return;
  }
  busy('Running inference');
  try {
    const q = cls >= 0 ? `?cls=${cls}` : '';
    const r = await api(`/sample/${current.key}${q}`);
    render(r);
  } catch (e) {
    toast(`Sample failed — ${e.message}`, true);
    showEmpty();
  }
}

async function upload(file) {
  if (!current) return;
  busy(`Reading ${file.name}`);
  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await api(`/predict/${current.key}`,
                        { method: 'POST', body: fd });
    if (r.adjusted) toast('Input was padded or cropped to the expected shape');
    render(r);
  } catch (e) {
    toast(`Upload failed — ${e.message}`, true);
    showEmpty();
  }
}

function busy(msg) {
  el('result').innerHTML = `
    <div class="glass mt empty">
      <div style="margin-bottom:14px"><span class="spin"></span></div>
      <div class="t">${msg}…</div>
    </div>`;
}

/* ── render ───────────────────────────────────────────── */
function render(r) {
  if (r.kind === 'image') return renderImage(r);
  renderSignal(r);
}

function renderSignal(r) {
  const top = r.prediction;
  const bars = r.probabilities.map((p, i) => `
    <div class="pbar ${i === top ? 'top' : ''}">
      <span class="lb">${r.classes[i]}</span>
      <div class="tr"><div class="fl" style="width:0%" data-w="${p * 100}"></div></div>
      <span class="vl">${(p * 100).toFixed(1)}%</span>
    </div>`).join('');

  let badge = '';
  if (r.correct === true) {
    badge = `<div class="badge ok">matches recorded label</div>`;
  } else if (r.correct === false) {
    badge = `<div class="badge no">recorded: ${r.truth_class}</div>`;
  }

  el('result').innerHTML = `
    <div class="row c2 mt">
      <div class="glass">
        <div class="step">SUBMITTED SIGNAL</div>
        <div class="monitor" style="height:250px">
          <canvas id="static-trace"></canvas>
        </div>
        <p class="hint" id="chan-note"></p>
      </div>

      <div class="glass">
        <div class="step">CLASS</div>
        <div class="verdict">${r.predicted_class}</div>
        <div class="conf">${(r.confidence * 100).toFixed(1)}% confidence</div>
        ${badge}
        <div style="margin-top:20px">${bars}</div>
      </div>
    </div>

    ${r.phase_map ? `
    <div class="glass mt">
      <div class="step">PHASE COHERENCE MAP</div>
      <canvas id="phase" style="width:100%;height:250px;border-radius:14px;
              border:1px solid var(--edge-2);display:block"></canvas>
      <p class="hint">
        Hue is phase — one full turn of colour is one full turn of phase.
        Amplitude-only models cannot produce this reading.
      </p>
    </div>` : ''}
  `;

  /* animate the bars in */
  requestAnimationFrame(() => {
    document.querySelectorAll('#result .fl').forEach(f => {
      f.style.width = f.dataset.w + '%';
    });
  });

  /* the live monitor now plays this recording */
  if (r.channels && r.channels.length) {
    monitor.setWaveform(r.channels[0]);
    el('live-label').textContent =
      `${current.title} · ${r.predicted_class}` +
      (r.index !== undefined ? ` · record #${r.index}` : '');
    el('live-rate').textContent = current.rate;
    setBpm(estimateBpm(r.channels[0], current));
    drawStatic(r);
  }

  if (r.phase_map) {
    drawMatrix(el('phase'), r.phase_map, phaseColour);
  }

  const note = el('chan-note');
  if (note) {
    note.textContent = r.n_channels > r.channels.length
      ? `Showing ${r.channels.length} of ${r.n_channels} channels.`
      : (r.n_channels > 1 ? `${r.n_channels} channels, offset for clarity.` : '');
  }
}

/* stacked multi-channel static plot */
function drawStatic(r) {
  const cv = el('static-trace');
  if (!cv) return;
  const { ctx, w, h } = fitCanvas(cv);
  ctx.clearRect(0, 0, w, h);

  ctx.strokeStyle = 'rgba(255,255,255,.05)';
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 34) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  for (let y = 0; y < h; y += 34) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  const chans = r.channels;
  const band = h / chans.length;
  chans.forEach((ch, k) => {
    const min = Math.min(...ch), max = Math.max(...ch);
    const span = (max - min) || 1;
    const mid = band * (k + 0.5);
    const amp = band * 0.38;
    ctx.beginPath();
    ch.forEach((v, i) => {
      const x = (i / (ch.length - 1)) * w;
      const y = mid - (((v - min) / span) * 2 - 1) * amp;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.strokeStyle = PALETTE[k % PALETTE.length];
    ctx.lineWidth = 1.7;
    ctx.lineJoin = 'round';
    ctx.stroke();
  });
}

/* rough rate estimate from peak spacing */
function estimateBpm(sig, model) {
  const fs = parseFloat(model.rate) || 360;
  if (!fs || model.kind !== 'signal') return '—';
  const max = Math.max(...sig), min = Math.min(...sig);
  const thresh = min + (max - min) * 0.72;
  const peaks = [];
  for (let i = 1; i < sig.length - 1; i++) {
    if (sig[i] > thresh && sig[i] >= sig[i - 1] && sig[i] > sig[i + 1]) {
      if (!peaks.length || i - peaks[peaks.length - 1] > fs * 0.2) peaks.push(i);
    }
  }
  if (peaks.length < 2) return '—';
  let sum = 0;
  for (let i = 1; i < peaks.length; i++) sum += peaks[i] - peaks[i - 1];
  const meanGap = sum / (peaks.length - 1);
  const bpm = Math.round((60 * fs) / meanGap);
  return (bpm > 25 && bpm < 260) ? bpm : '—';
}

/* ── segmentation view ────────────────────────────────── */
function renderImage(r) {
  const cov = r.coverage.map((c, i) => `
    <div>
      <div style="font-size:12.5px;color:var(--text-2);margin-bottom:6px">
        ${r.classes[i]}</div>
      <div style="font-size:25px;font-weight:600;
                  color:rgb(${MASK_COLOURS[i].join(',')});
                  font-variant-numeric:tabular-nums">
        ${(c * 100).toFixed(1)}%</div>
    </div>`).join('');

  const dice = r.dice ? `
    <div class="badge ok" style="margin-top:18px">
      Dice vs radiologist — LV ${r.dice[0].toFixed(3)} ·
      Myocardium ${r.dice[1].toFixed(3)} · LA ${r.dice[2].toFixed(3)}
    </div>` : '';

  el('result').innerHTML = `
    <div class="glass mt">
      <div class="step">SEGMENTATION</div>
      <div class="tri">
        <div>
          <p class="hint" style="margin:0 0 9px">Ultrasound frame</p>
          <canvas id="usg-frame"></canvas>
        </div>
        <div>
          <p class="hint" style="margin:0 0 9px">Structures found</p>
          <canvas id="usg-mask"></canvas>
        </div>
        <div>
          <p class="hint" style="margin:0 0 9px">Phase overlay</p>
          <canvas id="usg-phase"></canvas>
        </div>
      </div>
      <div class="legend">
        ${r.classes.map((c, i) =>
          `<span><i style="background:rgb(${MASK_COLOURS[i].join(',')})"></i>${c}</span>`
        ).join('')}
      </div>
    </div>

    <div class="glass mt">
      <div class="step">PIXEL COVERAGE</div>
      <div class="row c4">${cov}</div>
      ${dice}
      <p class="hint">
        Left ventricle area at end-diastole and end-systole gives the volume
        change per beat — the basis of ejection fraction.
      </p>
    </div>`;

  const flat = r.frame.flat();
  drawMatrix(el('usg-frame'), r.frame,
             greyColour(Math.min(...flat), Math.max(...flat)));
  drawMatrix(el('usg-mask'), r.mask, maskColour);
  drawMatrix(el('usg-phase'), r.phase, phaseColour);

  el('live-label').textContent =
    `${current.title} · frame #${r.index ?? '—'}`;
  setBpm('—');
}
