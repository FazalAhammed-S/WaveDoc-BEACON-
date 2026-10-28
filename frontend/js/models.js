/* ══════════════════════════════════════════════════════════
   WaveDoc — model catalogue and detail
   ══════════════════════════════════════════════════════════ */

let LIST = [];

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
    toast('Cannot reach the API server', true);
    return;
  }

  LIST = await api('/models');
  renderCards();

  if (location.hash) openDetail(location.hash.slice(1));
  window.addEventListener('hashchange', () => {
    if (location.hash) openDetail(location.hash.slice(1));
  });
});

function renderCards() {
  document.getElementById('cards').innerHTML = LIST.map(m => {
    const delta = m.primary[1] - m.baseline.value;
    const leads = delta > 0;
    return `
    <a class="glass mcard" href="#${m.key}">
      <div class="top">
        <div>
          <h3>${m.title}</h3>
          <div class="ds">${m.subtitle}</div>
          <div class="ds">${m.dataset}</div>
        </div>
        <div style="text-align:right">
          <div class="num" style="color:${leads ? '#6FE58A' : 'var(--text)'}">
            ${m.primary[1].toFixed(3)}</div>
          <div class="metric">${m.primary[0]}</div>
        </div>
      </div>
      <div style="font-size:12.5px;color:var(--text-3)">
        ${m.shape_text} · ${m.channel_text} · ${m.duration}
      </div>
      <div class="foot">
        <span class="${leads ? 'up' : 'down'}">
          ${delta >= 0 ? '+' : ''}${delta.toFixed(3)} vs ${m.baseline.name.split(' et')[0]}
        </span>
        <span class="go">Details →</span>
      </div>
    </a>`;
  }).join('');
}

async function openDetail(key) {
  const host = document.getElementById('detail');
  host.innerHTML = `<div class="glass mt empty">
      <div><span class="spin"></span></div></div>`;

  let d;
  try {
    d = await api(`/models/${key}`);
  } catch (e) {
    host.innerHTML = '';
    toast('Could not load that model', true);
    return;
  }

  const gamma = d.gamma_live || d.gamma || [];
  const gmax = Math.max(...gamma, 0.0001);
  const monotone = gamma.every((v, i) => i === 0 || gamma[i - 1] > v);

  const gbars = gamma.map(v => `
    <div class="gbar" style="height:${(v / gmax * 100).toFixed(1)}%">
      <span>${v.toFixed(3)}</span>
    </div>`).join('');
  const glabels = gamma.map((_, i) =>
    `<div>L${i + 1}</div>`).join('');

  const metrics = Object.entries(d.metrics).map(([k, v]) => `
    <div class="spec">
      <div class="k">${k.toUpperCase()}</div>
      <div class="v">${v.toFixed(4)}</div>
    </div>`).join('');

  const arch = Object.entries(d.arch).map(([k, v]) => `
    <tr>
      <td class="nm" style="text-transform:capitalize">${k}</td>
      <td style="text-align:right;color:var(--text-2)">${v}</td>
    </tr>`).join('');

  const classes = d.classes.map((c, i) => `
    <span class="chip" style="cursor:default">
      <span style="color:var(--text-3);margin-right:6px">${i}</span>${c}
    </span>`).join('');

  const delta = d.primary[1] - d.baseline.value;

  host.innerHTML = `
    <div class="glass mt-lg" style="scroll-margin-top:90px" id="detail-panel">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;
                  gap:20px;flex-wrap:wrap;margin-bottom:20px">
        <div>
          <div class="step">MODEL DETAIL</div>
          <h2>${d.title}</h2>
          <p class="sub" style="margin-top:6px">${d.subtitle} · ${d.dataset}</p>
        </div>
        <a class="btn primary" href="index.html">Try it on the dashboard →</a>
      </div>

      <div class="specs">${metrics}</div>

      <div class="row c2 mt-lg">
        <div>
          <div class="step">ARCHITECTURE</div>
          <table>
            <tr><td class="nm">Input shape</td>
                <td style="text-align:right;color:var(--text-2)">${d.shape_text}</td></tr>
            <tr><td class="nm">Sampling</td>
                <td style="text-align:right;color:var(--text-2)">${d.rate} · ${d.duration}</td></tr>
            ${arch}
          </table>
        </div>

        <div>
          <div class="step">DECOHERENCE GRADIENT γ</div>
          <div class="gbars">${gbars}</div>
          <div class="glabels">${glabels}</div>
          <div class="badge ${monotone ? 'ok' : 'no'}" style="margin-top:16px">
            ${monotone
              ? 'falls monotonically — validated'
              : 'not monotonic in this checkpoint'}
          </div>
          <p class="hint">
            γ is initialised uniformly and optimised only by backpropagation.
            Nothing specifies this ordering.
          </p>
        </div>
      </div>

      <div class="mt-lg">
        <div class="step">CLASSES</div>
        <div class="chips">${classes}</div>
      </div>

      <div class="mt-lg">
        <div class="step">AGAINST THE PUBLISHED BASELINE</div>
        <table>
          <thead><tr><th>Source</th><th>Metric</th><th>Value</th></tr></thead>
          <tbody>
            <tr><td class="nm">${d.baseline.name}</td>
                <td>${d.baseline.metric}</td>
                <td>${d.baseline.value.toFixed(4)}</td></tr>
            <tr class="${delta > 0 ? 'leads' : ''}">
                <td class="nm ours">WaveDoc</td>
                <td>${d.primary[0]}</td>
                <td class="ours">${d.primary[1].toFixed(4)}</td></tr>
            <tr><td class="nm">Difference</td><td></td>
                <td class="${delta > 0 ? 'up' : 'down'}">
                  ${delta >= 0 ? '+' : ''}${delta.toFixed(4)}</td></tr>
          </tbody>
        </table>
      </div>

      ${d.notes ? `
      <div class="mt-lg" style="padding:16px 18px;border-radius:16px;
           background:var(--glass-2);border:1px solid var(--edge-2)">
        <div class="step" style="margin-bottom:8px">NOTE</div>
        <p style="font-size:13.5px;color:var(--text-2);margin:0">${d.notes}</p>
      </div>` : ''}

      <div class="mt-lg" style="display:flex;gap:10px;flex-wrap:wrap">
        <a class="btn" href="compare.html">See all benchmarks →</a>
        <a class="btn" href="#" onclick="history.pushState('','',location.pathname);
           document.getElementById('detail').innerHTML='';return false">Close</a>
      </div>
    </div>`;

  document.getElementById('detail-panel')
    .scrollIntoView({ behavior: 'smooth', block: 'start' });
}
