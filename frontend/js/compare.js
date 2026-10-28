/* ══════════════════════════════════════════════════════════
   WaveDoc — benchmark comparison
   ══════════════════════════════════════════════════════════ */

let ROWS = [];

document.addEventListener('DOMContentLoaded', async () => {
  try {
    const h = await api('/health');
    const ready = Object.values(h.available).filter(v => v.weights).length;
    document.getElementById('health').className = ready ? 'pill ok' : 'pill warn';
    document.getElementById('health').innerHTML =
      `<span class="dot"></span>${h.device.toUpperCase()} · ${ready}/5 models`;
  } catch (e) {
    document.getElementById('health').className = 'pill warn';
    document.getElementById('health').textContent = 'API offline';
  }

  let data, abl, models;
  try {
    [data, abl, models] = await Promise.all([
      api('/benchmarks'), api('/ablation'), api('/models')
    ]);
  } catch (e) {
    toast('Could not load benchmark data', true);
    return;
  }

  ROWS = data.rows;
  document.getElementById('tally').innerHTML =
    `<span class="dot"></span>leads on ${data.wins} of ${data.total} benchmarks`;

  renderTable();
  renderAblation(abl);
  drawChart();
  await renderGammas(models);

  window.addEventListener('resize', drawChart);
});

/* ── table ─────────────────────────────────────────────── */
function renderTable() {
  document.getElementById('table').innerHTML = `
    <thead>
      <tr><th>Benchmark</th><th>Metric</th><th>Baseline</th>
          <th>WaveDoc</th><th>Difference</th></tr>
    </thead>
    <tbody>
      ${ROWS.map(r => `
        <tr class="${r.leads ? 'leads' : ''}">
          <td class="nm">${r.benchmark}<span>${r.reference}</span></td>
          <td>${r.metric}</td>
          <td>${r.baseline.toFixed(4)}</td>
          <td class="ours">${r.wavedoc.toFixed(4)}</td>
          <td class="${r.leads ? 'up' : 'down'}">
            ${r.delta >= 0 ? '+' : ''}${r.delta.toFixed(4)}</td>
        </tr>`).join('')}
    </tbody>`;
}

function renderAblation(rows) {
  const tick = b => b
    ? '<span style="color:#6FE58A">✓</span>'
    : '<span style="color:var(--text-3)">—</span>';
  document.getElementById('ablation').innerHTML = `
    <thead>
      <tr><th>Variant</th><th>Complex</th><th>TunnelOptim</th><th>γ</th>
          <th>Macro F1</th><th>Accuracy</th><th>AUC</th></tr>
    </thead>
    <tbody>
      ${rows.map((r, i) => `
        <tr class="${i === rows.length - 1 ? 'leads' : ''}">
          <td class="nm">${r.variant}</td>
          <td>${tick(r.complex)}</td>
          <td>${tick(r.tunnel)}</td>
          <td>${tick(r.gamma)}</td>
          <td class="${i === rows.length - 1 ? 'ours' : ''}">${r.f1.toFixed(4)}</td>
          <td>${r.acc.toFixed(4)}</td>
          <td>${r.auc.toFixed(4)}</td>
        </tr>`).join('')}
    </tbody>`;
}

/* ── grouped bar chart on canvas ──────────────────────── */
function drawChart() {
  const cv = document.getElementById('chart');
  if (!cv || !ROWS.length) return;
  const { ctx, w, h } = fitCanvas(cv);
  ctx.clearRect(0, 0, w, h);

  const padL = 44, padR = 16, padT = 34, padB = 66;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const lo = 0.6, hi = 1.02;
  const yOf = v => padT + plotH - ((v - lo) / (hi - lo)) * plotH;

  /* horizontal guides */
  ctx.strokeStyle = 'rgba(255,255,255,.07)';
  ctx.fillStyle = 'rgba(242,245,250,.4)';
  ctx.font = '10px Inter, sans-serif';
  ctx.lineWidth = 1;
  for (let v = 0.6; v <= 1.001; v += 0.1) {
    const y = yOf(v);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
    ctx.textAlign = 'right';
    ctx.fillText(v.toFixed(1), padL - 8, y + 3);
  }

  const slot = plotW / ROWS.length;
  const bw = Math.min(34, slot * 0.3);
  const gap = 6;

  ROWS.forEach((r, i) => {
    const cx = padL + slot * (i + 0.5);
    const xb = cx - bw - gap / 2;
    const xw = cx + gap / 2;

    /* baseline bar */
    const yb = yOf(r.baseline);
    ctx.fillStyle = 'rgba(255,255,255,.24)';
    roundRect(ctx, xb, yb, bw, padT + plotH - yb, 5);
    ctx.fill();

    /* wavedoc bar */
    const yw = yOf(r.wavedoc);
    const grad = ctx.createLinearGradient(0, yw, 0, padT + plotH);
    grad.addColorStop(0, '#5AC8FA');
    grad.addColorStop(1, 'rgba(167,139,250,.75)');
    ctx.fillStyle = grad;
    roundRect(ctx, xw, yw, bw, padT + plotH - yw, 5);
    ctx.fill();
    if (r.leads) {
      ctx.strokeStyle = 'rgba(90,200,250,.85)';
      ctx.lineWidth = 1.4;
      roundRect(ctx, xw, yw, bw, padT + plotH - yw, 5);
      ctx.stroke();
    }

    /* values */
    ctx.textAlign = 'center';
    ctx.font = '9.5px Inter, sans-serif';
    ctx.fillStyle = 'rgba(242,245,250,.45)';
    ctx.fillText(r.baseline.toFixed(3), xb + bw / 2, yb - 6);
    ctx.fillStyle = '#F2F5FA';
    ctx.font = '600 10px Inter, sans-serif';
    ctx.fillText(r.wavedoc.toFixed(3), xw + bw / 2, yw - 6);

    /* delta */
    ctx.font = '600 11px Inter, sans-serif';
    ctx.fillStyle = r.leads ? '#6FE58A' : '#FF8F9C';
    ctx.fillText(`${r.delta >= 0 ? '+' : ''}${r.delta.toFixed(3)}`,
                 cx, Math.min(yb, yw) - 21);

    /* x labels */
    ctx.fillStyle = 'rgba(242,245,250,.7)';
    ctx.font = '10.5px Inter, sans-serif';
    wrapText(ctx, r.benchmark, cx, h - padB + 18, slot - 6, 12);
    ctx.fillStyle = 'rgba(242,245,250,.35)';
    ctx.font = '9px Inter, sans-serif';
    ctx.fillText(r.metric, cx, h - padB + 44);
  });
}

function roundRect(ctx, x, y, w, h, r) {
  if (h <= 0) return;
  r = Math.min(r, w / 2, h);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function wrapText(ctx, text, x, y, maxW, lh) {
  const words = text.split(' ');
  let line = '', lines = [];
  for (const word of words) {
    const test = line ? `${line} ${word}` : word;
    if (ctx.measureText(test).width > maxW && line) {
      lines.push(line); line = word;
    } else line = test;
  }
  lines.push(line);
  lines.slice(0, 2).forEach((l, i) => ctx.fillText(l, x, y + i * lh));
}

/* ── gamma across modalities ──────────────────────────── */
async function renderGammas(models) {
  const host = document.getElementById('gammas');
  const cards = [];
  for (const m of models) {
    if (!m.has_weights) continue;
    let d;
    try { d = await api(`/models/${m.key}`); } catch (e) { continue; }
    const g = d.gamma_live || d.gamma || [];
    if (!g.length) continue;
    const max = Math.max(...g);
    const mono = g.every((v, i) => i === 0 || g[i - 1] > v);
    cards.push(`
      <div>
        <h3 style="font-size:14px">${d.title}</h3>
        <div class="hint" style="margin:2px 0 0">${g.length} layers</div>
        <div class="gbars">
          ${g.map(v => `<div class="gbar" style="height:${(v / max * 100).toFixed(1)}%">
             <span>${v.toFixed(3)}</span></div>`).join('')}
        </div>
        <div class="glabels">
          ${g.map((_, i) => `<div>L${i + 1}</div>`).join('')}
        </div>
        <div style="font-size:11.5px;margin-top:11px;
             color:${mono ? '#6FE58A' : '#FF8F9C'}">
          ${g[0].toFixed(3)} → ${g[g.length - 1].toFixed(3)}
          ${mono ? '· falls' : '· check'}
        </div>
      </div>`);
  }
  host.innerHTML = cards.join('') ||
    '<p class="hint">No model weights loaded on the server.</p>';
}
