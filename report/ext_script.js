
  // ---- Extended study (Amendment 3) ----
  dash.calc('ext-mech', {
    title: 'Stuck variants and recovery',
    description: 'Counts explored variants where 32 agents end at least 2 points below 4, how many coupling helped, and the median share of the loss it recovered.',
    inputs: ['ext_mechanism'],
    fn: (ext_mechanism) => {
      const stuck = ext_mechanism.filter((r) => r.responsive);
      const rec = stuck.map((r) => -r.coupling_pp / r.size_pp).sort((a, b) => a - b);
      const n = rec.length;
      const median = n ? (n % 2 ? rec[(n - 1) / 2] : (rec[n / 2 - 1] + rec[n / 2]) / 2) : null;
      return [{variants: ext_mechanism.length, stuck: n, helped: stuck.filter((r) => r.coupling_pp > 0).length,
        median_recovered: median}];
    },
  });

  dash.calc('ext-setcount', {
    title: 'Adaption sets with an effect',
    description: 'Counts Adaption confirmation sets with a bootstrap interval, and how many exclude zero.',
    inputs: ['ext_sets'],
    fn: (ext_sets) => {
      const b = ext_sets.filter((r) => r.l2_pp != null);
      return [{effect: b.filter((r) => r.decision === 'effect').length, total: b.length}];
    },
  });

  const sxCalcCell = (id, field, fmt) => (el) => {
    const d = dash.data(id);
    el.textContent = d.status === 'ok' && d.data[0] && d.data[0][field] != null ? fmt(d.data[0][field]) : '—';
  };
  dash.draw('ext-n-variants', sxCalcCell('ext-mech', 'variants', d3.format(',')));
  dash.draw('ext-n-stuck', sxCalcCell('ext-mech', 'stuck', d3.format(',')));
  dash.draw('ext-n-helped', sxCalcCell('ext-mech', 'helped', d3.format(',')));
  dash.draw('ext-median', sxCalcCell('ext-mech', 'median_recovered', d3.format('.0%')));
  dash.draw('ext-t-sets', sxCalcCell('ext-setcount', 'effect', String));
  dash.draw('ext-t-sets-n', sxCalcCell('ext-setcount', 'total', String));

  const sxRowCell = (source, pred, field, fmt) => (el) => {
    const d = dash.data(source);
    const r = d.status === 'ok' ? d.data.find(pred) : null;
    el.textContent = r && r[field] != null ? fmt(r[field]) : '—';
  };
  const sxGame = (lvl) => (r) => r.factor === 'Game' && r.level === lvl;
  dash.draw('ext-t-stag', sxRowCell('ext_factors', sxGame('Stag hunt'), 'responsive', String));
  dash.draw('ext-t-coord', sxRowCell('ext_factors', sxGame('Original game'), 'responsive', String));
  dash.draw('ext-t-gamen', sxRowCell('ext_factors', sxGame('Stag hunt'), 'variants', String));
  const sxSet = (s) => (r) => r.set === s;
  dash.draw('ext-t-qwen', sxRowCell('ext_sets', sxSet('qwen3.8-27b_pooled'), 'l2_pp', (v) => sxSigned(v)));
  dash.draw('ext-t-qwen-lo', sxRowCell('ext_sets', sxSet('qwen3.8-27b_pooled'), 'l2_lo_pp', (v) => sxSigned(v)));
  dash.draw('ext-t-qwen-hi', sxRowCell('ext_sets', sxSet('qwen3.8-27b_pooled'), 'l2_hi_pp', (v) => sxSigned(v)));
  dash.draw('ext-t-glm', sxRowCell('ext_sets', sxSet('glm-5.2_pooled'), 'l2_pp', (v) => sxSigned(v)));
  dash.draw('ext-t-glm-lo', sxRowCell('ext_sets', sxSet('glm-5.2_pooled'), 'l2_lo_pp', (v) => sxSigned(v)));
  dash.draw('ext-t-glm-hi', sxRowCell('ext_sets', sxSet('glm-5.2_pooled'), 'l2_hi_pp', (v) => sxSigned(v)));

  dash.draw('ext-scatter', (box) => {
    box.replaceChildren();
    const d = dash.data('ext_mechanism');
    if (d.status !== 'ok' || !d.data.length) { box.textContent = '—'; return; }
    const rows = d.data;
    const W = Math.max(300, box.clientWidth || 600), H = 340, mb = 34, mt = 8, mr = 10;
    const svg = d3.select(box).append('svg').attr('width', '100%').attr('viewBox', `0 0 ${W} ${H}`);
    const xe = d3.extent(rows, (r) => r.size_pp), ye = d3.extent(rows, (r) => r.coupling_pp);
    const y = d3.scaleLinear([Math.min(ye[0], -2), Math.max(ye[1], 2)], [H - mb, mt]).nice(6);
    const yt = y.ticks(6);
    const probe = svg.append('text').attr('font-size', 11).text(sxSigned(yt[0], 0));
    const ml = Math.ceil(probe.node().getComputedTextLength()) + 10; probe.remove();
    const x = d3.scaleLinear([Math.min(xe[0], -2), Math.max(xe[1], 2)], [ml, W - mr]).nice(6);
    const xt = x.ticks(Math.max(3, Math.floor(W / 90)));
    svg.append('g').selectAll('line').data(yt).join('line').attr('x1', ml).attr('x2', W - mr)
      .attr('y1', (t) => y(t)).attr('y2', (t) => y(t)).attr('stroke', 'var(--cds-chart-grid)');
    svg.append('g').selectAll('text').data(yt).join('text').attr('x', ml - 6).attr('y', (t) => y(t)).attr('dy', '0.32em')
      .attr('text-anchor', 'end').attr('font-size', 11).attr('fill', 'var(--color-fg-muted)').text((t) => sxSigned(t, 0));
    svg.append('g').selectAll('text').data(xt).join('text').attr('x', (t) => x(t)).attr('y', H - mb + 16)
      .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', 'var(--color-fg-muted)').text((t) => sxSigned(t, 0));
    svg.append('text').attr('x', W - mr).attr('y', H - 4).attr('text-anchor', 'end').attr('font-size', 11)
      .attr('fill', 'var(--color-fg-muted)').text('Size effect, points');
    svg.append('line').attr('x1', x(0)).attr('x2', x(0)).attr('y1', mt).attr('y2', H - mb).attr('stroke', 'var(--cds-chart-axis)');
    svg.append('line').attr('x1', ml).attr('x2', W - mr).attr('y1', y(0)).attr('y2', y(0)).attr('stroke', 'var(--cds-chart-axis)');
    const lo = Math.max(x.domain()[0], -y.domain()[1]);
    svg.append('line').attr('x1', x(lo)).attr('y1', y(-lo)).attr('x2', x(0)).attr('y2', y(0))
      .attr('stroke', 'var(--cds-chart-reference)').attr('stroke-dasharray', '4 3');
    const col = (r) => (r.game === 'Stag hunt' ? dash.colors[1] : dash.colors[0]);
    svg.append('g').selectAll('circle').data(rows).join('circle').attr('cx', (r) => x(r.size_pp)).attr('cy', (r) => y(r.coupling_pp))
      .attr('r', 3).attr('fill', col).attr('fill-opacity', 0.6);
    const hl = svg.append('circle').attr('r', 5).attr('fill', 'none').attr('stroke-width', 2).attr('opacity', 0);
    const tip = sxTip(box);
    const del = d3.Delaunay.from(rows, (r) => x(r.size_pp), (r) => y(r.coupling_pp));
    svg.append('rect').attr('x', ml).attr('y', mt).attr('width', W - ml - mr).attr('height', H - mb - mt).attr('fill', 'transparent')
      .on('pointermove', (ev) => {
        const [px, py] = d3.pointer(ev);
        const r = rows[del.find(px, py)];
        hl.attr('cx', x(r.size_pp)).attr('cy', y(r.coupling_pp)).attr('stroke', col(r)).attr('opacity', 1);
        const rect = box.getBoundingClientRect(), sr = svg.node().getBoundingClientRect(), s = sr.width / W;
        tip.show([['t', `${r.model} · ${r.deployment}`], [null, `${r.game}, ${r.wording}, ${r.variant}`],
          [null, `Size ${sxSigned(r.size_pp)} pts, coupling ${sxSigned(r.coupling_pp)} pts`, col(r)]],
        x(r.size_pp) * s + (sr.left - rect.left), y(r.coupling_pp) * s + (sr.top - rect.top));
      })
      .on('pointerleave', () => { hl.attr('opacity', 0); tip.hide(); });
  });

  dash.draw('ext-factors', (box) => {
    box.replaceChildren();
    const d = dash.data('ext_factors');
    if (d.status !== 'ok' || !d.data.length) { box.textContent = '—'; return; }
    const maxShare = d3.max(d.data, (r) => r.share) || 1;
    [...new Set(d.data.map((r) => r.factor))].forEach((f) => {
      const panel = document.createElement('div');
      const t = document.createElement('div'); t.className = 'panel-title'; t.textContent = f; panel.appendChild(t);
      d.data.filter((r) => r.factor === f).sort((a, b) => a.order - b.order).forEach((r) => {
        const row = document.createElement('div');
        row.style.cssText = 'display:grid;grid-template-columns:minmax(6rem,9rem) 1fr auto;gap:var(--cds-gap-sm);align-items:center;font-size:var(--cds-font-size-caption);margin-block:2px';
        const lab = document.createElement('span'); lab.textContent = r.level; lab.style.color = 'var(--color-fg-muted)';
        const track = document.createElement('span'); track.style.cssText = 'height:10px;position:relative';
        const bar = document.createElement('span');
        bar.style.cssText = `position:absolute;left:0;top:0;bottom:0;border-radius:2px;width:${(100 * r.share) / maxShare}%;background:${dash.colors[0]}`;
        track.appendChild(bar);
        const val = document.createElement('span'); val.className = 'sx-num'; val.textContent = `${r.responsive} of ${r.variants}`;
        row.append(lab, track, val);
        panel.appendChild(row);
      });
      box.appendChild(panel);
    });
  });

  window.sx.extSet = window.sx.extSet || 'glm-5.2_pooled';
  dash.draw('ext-set-pick', (sel) => {
    const d = dash.data('ext_ncurve');
    const sets = d.status === 'ok' ? [...new Map(d.data.map((r) => [r.set, `${r.model}: ${r.kind}`])).entries()] : [];
    sel.replaceChildren(...sets.map(([v, l]) => new Option(l, v)));
    if (sets.length && !sets.some(([v]) => v === window.sx.extSet)) window.sx.extSet = sets[0][0];
    sel.value = window.sx.extSet;
    sel.onchange = () => { window.sx.extSet = sel.value; dash.draw('ext-ncurve'); dash.draw('ext-dose'); };
  });

  dash.draw('ext-ncurve', (box) => {
    box.replaceChildren();
    const d = dash.data('ext_ncurve');
    if (d.status !== 'ok') { box.textContent = '—'; return; }
    const rows = d.data.filter((r) => r.set === window.sx.extSet).sort((a, b) => a.N - b.N);
    if (!rows.length) { box.textContent = '—'; return; }
    const Ns = rows.map((r) => r.N);
    sxLinePanel(box, rows[0].kind, [
      {name: 'Independent', color: dash.colors[0], values: rows.map((r) => ({x: r.N, y: r.x_independent}))},
      {name: 'Coupled', color: dash.colors[1], values: rows.map((r) => ({x: r.N, y: r.x_coupled}))},
    ], Ns, 'Agents', (v, tipMode) => d3.format(tipMode ? '.1%' : '.0%')(v), {band: true, dots: true});
  });

  dash.draw('ext-dose', (box) => {
    box.replaceChildren();
    const d = dash.data('ext_dose');
    if (d.status !== 'ok') { box.textContent = '—'; return; }
    const rows = d.data.filter((r) => r.set === window.sx.extSet && r.alpha !== 'restore');
    if (!rows.length) { box.textContent = '—'; return; }
    const alphas = [...new Set(rows.map((r) => +r.alpha))].sort((a, b) => a - b);
    const series = [32, 64].map((N, i) => ({name: `${N} agents`, color: dash.colors[i],
      values: alphas.map((a) => { const r = rows.find((q) => q.N === N && +q.alpha === a); return {x: a, y: r ? r.coupling_pp : null}; })}));
    sxLinePanel(box, 'Probability of sharing randomness', series, alphas, 'Sharing probability',
      (v, tipMode) => sxSigned(v, tipMode ? 1 : 0) + (tipMode ? ' pts' : ''), {dots: true, xTicks: [0.1, 0.3, 0.5, 0.75, 1]});
  });

  function sxTableF(name, source, keyCols, cols) {
    dash.draw(name, (tbody) => {
      const d = dash.data(source);
      tbody.replaceChildren();
      if (d.status !== 'ok') return;
      d.data.forEach((r) => {
        const tr = document.createElement('tr');
        tr.dataset.key = keyCols.map((k) => encodeURIComponent(String(r[k]))).join(',');
        cols.forEach(([field, fmt, numeric]) => {
          const td = document.createElement('td');
          if (numeric) td.className = 'n';
          const v = r[field];
          td.textContent = v == null ? '—' : fmt ? fmt(v) : String(v);
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
    });
  }
  const sxPts = (v) => sxSigned(v);
  const sxPts2 = (v) => sxSigned(v, 2);  // interval bounds: some sit within 0.05 of zero
  sxTableF('ext-sets-table', 'ext_sets', ['set'], [['model'], ['kind'], ['variants', String, 1], ['size32_pp', sxPts, 1],
    ['size64_pp', sxPts, 1], ['coupling32_pp', sxPts, 1], ['coupling64_pp', sxPts, 1], ['l2_pp', sxPts, 1],
    ['l2_lo_pp', sxPts2, 1], ['l2_hi_pp', sxPts2, 1], ['decision']]);
  sxTableF('ext-labels-table', 'ext_labels', ['model', 'game'], [['model'], ['game'], ['letter_J', d3.format('.2f'), 1],
    ['letter_K', d3.format('.2f'), 1], ['listed_first', d3.format('.2f'), 1], ['listed_second', d3.format('.2f'), 1]]);
  sxTableF('ext-providers-table', 'ext_providers', ['model', 'provider'], [['model'], ['provider'], ['pinned', (v) => (v ? 'Yes' : '')],
    ['mad_vs_adaption', d3.format('.3f'), 1], ['size_pp', sxPts, 1], ['coupling_pp', sxPts, 1]]);
  sxTableF('ext-retest-table', 'ext_retest', ['model'], [['model'], ['sum_z2', d3.format(',.0f'), 1], ['critical_99', d3.format(',.0f'), 1],
    ['abs_z_gt_2', String, 1], ['prompts', String, 1], ['mean_abs_diff', d3.format('.3f'), 1]]);
