// The "Inspect" tab: shows, with the real numbers, how the selected neuron's value was
// computed from the layer before it.
import { LAYER_BY_ID, unpack, pack } from '../viz/layout.js';
import { kernelIndex, flattenIndex } from '../model/cnn.js';
import { ACTIVATION, GRAY, cssColor, signedCss } from '../viz/colors.js';

// ------------------------------------------------------------------ formatting helpers

const fmt = (v, d = 2) => (Object.is(Math.round(v * 10 ** d), -0) ? 0 : v).toFixed(d);
const signed = (v, d = 2) => (Math.abs(v) < 0.5 * 10 ** -d ? '' : v > 0 ? '+' : '−') + Math.abs(v).toFixed(d);
/** Very short number for inside tiny grid cells: 0.35 -> ".35", -0.35 -> "-.35". */
const tiny = (v) => {
  if (Math.abs(v) < 0.005) return '0';
  if (Math.abs(v) >= 0.995) return v.toFixed(1);
  return v.toFixed(2).replace(/^(-?)0\./, '$1.');
};
const maxOf = (arr) => arr.reduce((m, v) => (v > m ? v : m), 1e-9);
const maxAbs = (arr) => arr.reduce((m, v) => (Math.abs(v) > m ? Math.abs(v) : m), 1e-9);

const actColor = (max) => (v) => cssColor(ACTIVATION, v / max);
const grayColor = (v) => cssColor(GRAY, v);
const signColor = (max) => (v) => signedCss(v, max);

/** Dark text on light cells, light text on dark ones. */
function textColor(rgb) {
  const [r, g, b] = rgb.match(/\d+/g).map(Number);
  return 0.299 * r + 0.587 * g + 0.114 * b > 150 ? '#0b0f18' : 'rgba(255,255,255,0.9)';
}

/** A small heatmap made of divs. NaN cells are drawn as padding. */
function heat(values, cols, color, { cell = 22, numbers = false, hl = -1 } = {}) {
  const cells = Array.from(values, (v, i) => {
    if (Number.isNaN(v)) return `<div style="width:${cell}px;height:${cell}px;background:#0d1220" title="zero padding (outside the image)"></div>`;
    const bg = color(v);
    return `<div style="width:${cell}px;height:${cell}px;background:${bg};color:${textColor(bg)}"${i === hl ? ' class="hl"' : ''} title="${fmt(v, 3)}">${numbers ? tiny(v) : ''}</div>`;
  }).join('');
  return `<div class="heat" style="grid-template-columns:repeat(${cols},${cell}px)">${cells}</div>`;
}

const block = (html, cap) => `<div class="grid-block">${html}<div class="cap">${cap}</div></div>`;

function sumTable(rows) {
  return `<table class="sum-table">${rows.map(([k, v, cls]) =>
    `<tr${cls ? ` class="${cls}"` : ''}><td>${k}</td><td>${v}</td></tr>`).join('')}</table>`;
}

function contribBar(v, max, width = 90) {
  const w = Math.max(1, (Math.abs(v) / (max || 1)) * width);
  return `<div class="contrib"><span class="cbar" style="width:${w}px;background:${v >= 0 ? 'var(--pos)' : 'var(--neg)'}"></span>${signed(v)}</div>`;
}

function hbars(items, max) {
  return `<div class="hbar-list">${items.map(({ label, v, action = '' }) => {
    const pct = Math.min(50, (Math.abs(v) / (max || 1)) * 50);
    const bar = v >= 0
      ? `<span style="left:50%;width:${pct}%;background:var(--pos)"></span>`
      : `<span style="right:50%;width:${pct}%;background:var(--neg)"></span>`;
    return `<div class="hbar"${action}><span>${label}</span><div class="track"><span class="axis"></span>${bar}</div><span class="v">${signed(v)}</span></div>`;
  }).join('')}</div>`;
}

const reluRows = (sum, bias) => {
  const pre = sum + bias;
  return [
    ['+ bias', signed(bias)],
    ['= weighted sum', fmt(pre)],
    [`ReLU: max(0, ${fmt(pre)})`, fmt(Math.max(0, pre)), 'total'],
  ];
};

const sweepButton = (label) => `<p><button class="primary" data-action="sweep">▶ ${label}</button></p>`;

// ------------------------------------------------------------------ per-layer views

export function describe(ref, res) {
  const L = LAYER_BY_ID[ref.layer];
  if (L.id === 'output') return `digit ${ref.index}: <b>${(res.probs[ref.index] * 100).toFixed(1)}%</b>`;
  if (L.id === 'dense1') return `Dense neuron #${ref.index} = <b>${fmt(res.dense1[ref.index])}</b>`;
  const { map, y, x } = unpack(L, ref.index);
  const value = (L.id === 'input' ? res.input : res[L.id])[ref.index];
  const what = L.id === 'input' ? 'pixel' : `map #${map}`;
  return `${L.title} · ${what} · row ${y}, col ${x} = <b>${fmt(value)}</b>`;
}

const VIEWS = {
  input(ref, net, res) {
    const { y, x } = unpack(LAYER_BY_ID.input, ref.index);
    const v = res.input[ref.index];
    return `
      <h3>Input pixel · row ${y}, col ${x}</h3>
      <p class="lede">The brightness of one pixel of the 28×28 image, from 0 (black) to 1 (white).</p>
      ${sumTable([['Value', fmt(v, 3), 'total']])}
      <p>This is all the network gets: 784 numbers. It has no idea it is looking at a digit.
      Every Conv 1 neuron within 2 pixels of this one sees it, through its own 5×5 kernel.</p>`;
  },

  conv1(ref, net, res) {
    const layer = net.conv1;
    const { map: o, y, x } = unpack(LAYER_BY_ID.conv1, ref.index);
    const { k } = layer;
    const pad = (k - 1) >> 1;
    const patch = [], kernel = [], product = [];
    for (let ky = 0; ky < k; ky++)
      for (let kx = 0; kx < k; kx++) {
        const iy = y + ky - pad, ix = x + kx - pad;
        const inside = iy >= 0 && iy < 28 && ix >= 0 && ix < 28;
        const p = inside ? res.input[iy * 28 + ix] : NaN;
        const w = layer.w[kernelIndex(layer, o, 0, ky, kx)];
        patch.push(p);
        kernel.push(w);
        product.push(inside ? p * w : NaN);
      }
    const sum = product.reduce((s, v) => s + (Number.isNaN(v) ? 0 : v), 0);
    const kMax = maxAbs(kernel);
    const pMax = maxAbs(product.filter((v) => !Number.isNaN(v)));
    const value = res.conv1[ref.index];
    return `
      <h3>Conv 1 · filter #${o} · row ${y}, col ${x}</h3>
      <p class="lede">One neuron of feature map #${o}. It looks at the 5×5 patch of the input centered on
      this position, multiplies it cell by cell with filter #${o}'s kernel, and adds everything up.</p>
      <div class="equation">
        ${block(heat(patch, k, grayColor, { numbers: true, cell: 19 }), 'input patch')}
        <span class="op">×</span>
        ${block(heat(kernel, k, signColor(kMax), { numbers: true, cell: 19 }), `kernel #${o}`)}
        <span class="op">=</span>
        ${block(heat(product, k, signColor(pMax), { numbers: true, cell: 19 }), 'products')}
      </div>
      ${sumTable([['Sum of the 25 products', fmt(sum)], ...reluRows(sum, layer.b[o])])}
      <div class="callout">${value > 0
        ? `Orange kernel weights reward bright pixels and blue ones penalize them. The result is large
           where the patch looks like the kernel's orange pattern: that is what "detecting a feature" means.`
        : `The weighted sum is negative, so ReLU outputs 0: filter #${o}'s pattern is not present here.`}</div>
      ${sweepButton('Slide this kernel across the image')}
      <p class="muted">The same 25 weights are reused at all 784 positions. That weight sharing is what makes
      this a convolution. At the border the kernel hangs over the edge, and the missing pixels count as 0
      ("same" padding), so the output stays 28×28.</p>`;
  },

  conv2(ref, net, res) {
    const layer = net.conv2;
    const P = LAYER_BY_ID.pool1;
    const { map: o, y, x } = unpack(LAYER_BY_ID.conv2, ref.index);
    const { k, inC } = layer;
    const pad = (k - 1) >> 1;
    const pMax = maxOf(res.pool1);
    let wMax = 1e-9;
    for (let i = 0; i < inC * k * k; i++) wMax = Math.max(wMax, Math.abs(layer.w[o * inC * k * k + i]));

    const rows = [];
    for (let i = 0; i < inC; i++) {
      const patch = [], kernel = [];
      let c = 0;
      for (let ky = 0; ky < k; ky++)
        for (let kx = 0; kx < k; kx++) {
          const iy = y + ky - pad, ix = x + kx - pad;
          const inside = iy >= 0 && iy < P.size && ix >= 0 && ix < P.size;
          const p = inside ? res.pool1[pack(P, i, iy, ix)] : NaN;
          const w = layer.w[kernelIndex(layer, o, i, ky, kx)];
          patch.push(p);
          kernel.push(w);
          if (inside) c += p * w;
        }
      rows.push({ i, patch, kernel, c });
    }
    const sum = rows.reduce((s, r) => s + r.c, 0);
    const cMax = maxAbs(rows.map((r) => r.c));
    return `
      <h3>Conv 2 · filter #${o} · row ${y}, col ${x}</h3>
      <p class="lede">Conv 2 kernels are 3D: 3×3 in space and 8 deep, one slice for each feature map
      coming out of Max pool 1. This neuron adds up what all 8 slices see.</p>
      <div class="chan-rows">
        <div class="chan-row muted"><span>from</span><span>patch</span><span>kernel slice</span><span>contribution</span></div>
        ${rows.map((r) => `
          <div class="chan-row" data-action="select" data-layer="pool1" data-index="${pack(P, r.i, y, x)}" style="cursor:pointer" title="Select the pool 1 neuron under the center of this window">
            <span>map #${r.i}</span>
            ${heat(r.patch, k, actColor(pMax), { cell: 14 })}
            ${heat(r.kernel, k, signColor(wMax), { cell: 14 })}
            ${contribBar(r.c, cMax)}
          </div>`).join('')}
      </div>
      ${sumTable([['Sum of the 8 contributions (72 products)', fmt(sum)], ...reluRows(sum, layer.b[o])])}
      <div class="callout">Each 3×3 window here covers a 10×10 pixel area of the original image (its
      <i>receptive field</i>), because the maps were already pooled. So Conv 2 finds larger shapes such as
      curves, corners and loops, by combining the simple strokes and edges that Conv 1 found.</div>
      ${sweepButton('Slide this kernel across the maps')}`;
  },

  pool1: (ref, net, res) => poolView(ref, res, 'pool1', 'conv1', 28),
  pool2: (ref, net, res) => poolView(ref, res, 'pool2', 'conv2', 14),

  dense1(ref, net, res) {
    const j = ref.index;
    const P = LAYER_BY_ID.pool2;
    const { outN, w, b } = net.dense1;
    const weights = [], contrib = [];
    for (let c = 0; c < P.maps; c++) {
      const wm = [], cm = [];
      for (let y = 0; y < P.size; y++)
        for (let x = 0; x < P.size; x++) {
          const wv = w[flattenIndex(c, y, x, P.size, P.maps) * outN + j];
          wm.push(wv);
          cm.push(wv * res.pool2[pack(P, c, y, x)]);
        }
      weights.push(wm);
      contrib.push(cm);
    }
    const wMax = maxAbs(weights.flat());
    const cMax = maxAbs(contrib.flat());
    const sum = contrib.flat().reduce((s, v) => s + v, 0);
    const maps = (arr, color) => `<div class="map-grid" style="grid-template-columns:repeat(4,max-content)">
      ${arr.map((m) => heat(m, 7, color, { cell: 7 })).join('')}</div>`;
    const votes = Array.from({ length: 10 }, (_, d) => ({ label: `digit ${d}`, v: net.dense2.w[j * 10 + d] }));
    return `
      <h3>Dense · neuron #${j}</h3>
      <p class="lede">Connected to all 784 values of Max pool 2 (16 maps of 7×7), each with its own weight.</p>
      <h4>Its weights: what it looks for</h4>
      ${maps(weights, signColor(wMax))}
      <p class="muted" style="margin-top:6px">One 7×7 grid per pool 2 map (#0 to #15, row by row). Orange: "I want activity here".
      Blue: "activity here argues against me".</p>
      <h4>Input × weight: what it found in this image</h4>
      ${maps(contrib, signColor(cMax))}
      ${sumTable([['Sum of the 784 products', fmt(sum)], ...reluRows(sum, b[j])])}
      <h4>How it votes for each digit (its output weights)</h4>
      ${hbars(votes, maxAbs(votes.map((v) => v.v)))}
      <div class="callout">From here on the network no longer works with pictures. The 784 numbers are
      just a list, and each dense neuron is a weighted vote over all of them.</div>`;
  },

  output(ref, net, res) {
    const d = ref.index;
    const { w, b } = net.dense2;
    const items = [];
    for (let j = 0; j < 64; j++) {
      const c = res.dense1[j] * w[j * 10 + d];
      if (c !== 0) items.push({ label: `neuron #${j}`, v: c, action: ` data-action="select" data-layer="dense1" data-index="${j}" style="cursor:pointer"` });
    }
    items.sort((a, b2) => Math.abs(b2.v) - Math.abs(a.v));
    const sum = items.reduce((s, it) => s + it.v, 0);
    const z = res.logits[d];
    const expSum = res.logits.reduce((s, l) => s + Math.exp(l), 0);
    return `
      <h3>Output · digit ${d}</h3>
      <p class="lede">A weighted sum of the 64 dense neurons gives a score (the <i>logit</i>), and softmax
      turns the 10 scores into probabilities that add up to 100%.</p>
      ${sumTable([
        ['Σ activation × weight (64 neurons)', fmt(sum)],
        ['+ bias', signed(b[d])],
        [`logit z<sub>${d}</sub>`, fmt(z)],
        [`softmax: e<sup>z${d}</sup> / Σ e<sup>z</sup>`, `${(res.probs[d] * 100).toFixed(2)}%`, 'total'],
      ])}
      <h4>Biggest contributions (activation × weight)</h4>
      ${items.length ? hbars(items.slice(0, 12), maxAbs(items.map((i) => i.v))) : '<p class="muted">No dense neuron is active.</p>'}
      <p class="muted" style="margin-top:6px">Click a row to inspect that dense neuron.</p>
      <h4>Softmax across all digits</h4>
      ${sumTable(Array.from({ length: 10 }, (_, i) => [
        `${i === res.prediction ? '▶ ' : ''}digit ${i}: z = ${fmt(res.logits[i])}, e<sup>z</sup> = ${fmt(Math.exp(res.logits[i]), 1)}`,
        `${(Math.exp(res.logits[i]) / expSum * 100).toFixed(1)}%`,
        i === d ? 'total' : '',
      ]))}
      <p class="muted">Softmax exaggerates differences: a logit a few points higher than the rest ends up with
      almost all the probability.</p>`;
  },
};

function poolView(ref, res, id, srcId, srcSize) {
  const L = LAYER_BY_ID[id];
  const S = LAYER_BY_ID[srcId];
  const { map, y, x } = unpack(L, ref.index);
  const vals = [];
  for (let dy = 0; dy < 2; dy++)
    for (let dx = 0; dx < 2; dx++) vals.push(res[srcId][pack(S, map, 2 * y + dy, 2 * x + dx)]);
  const best = vals.indexOf(Math.max(...vals));
  const by = 2 * y + (best >> 1), bx = 2 * x + (best & 1);
  const out = srcSize / 2;
  return `
    <h3>${L.title} · map #${map} · row ${y}, col ${x}</h3>
    <p class="lede">Looks at a 2×2 block of ${S.title} map #${map} (rows ${2 * y}–${2 * y + 1},
    cols ${2 * x}–${2 * x + 1}) and keeps only the largest value.</p>
    <div class="equation">
      ${block(heat(vals, 2, actColor(maxOf(res[srcId])), { cell: 44, numbers: true, hl: best }), `${S.title} block`)}
      <span class="op">→ max →</span>
      ${block(heat([vals[best]], 1, actColor(maxOf(res[srcId])), { cell: 44, numbers: true }), 'output')}
    </div>
    <p>Pooling halves the width and height (${srcSize}×${srcSize} → ${out}×${out}), so the next layer has 4× fewer
    positions to work on, and a stroke that moves by a pixel barely changes the result. The network keeps
    <i>that</i> a feature is present, not its exact position. There are no weights to learn here.</p>
    <p><button data-action="select" data-layer="${srcId}" data-index="${pack(S, map, by, bx)}">Inspect the winning ${S.title} neuron</button></p>
    ${sweepButton('Watch pooling sweep this map')}`;
}

export class Inspector {
  constructor(root, { onAction }) {
    this.root = root;
    root.addEventListener('click', (e) => {
      const el = e.target.closest('[data-action]');
      if (!el) return;
      onAction({ action: el.dataset.action, layer: el.dataset.layer, index: Number(el.dataset.index) });
    });
    this.clear();
  }

  clear() {
    this.root.innerHTML = `<div class="empty-state">
      <p><b>Click any cube</b> in the 3D view to see exactly how its value is computed from the layer before it.</p>
      <p>Good places to start: a bright cube in <b>Conv 1</b>, or the tallest bar in <b>Output</b>.</p></div>`;
  }

  show(ref, net, res) {
    const scroll = this.root.scrollTop;
    this.root.innerHTML = VIEWS[ref.layer](ref, net, res);
    this.root.scrollTop = scroll;
  }
}
