// The "Kernels" tab: every learned filter, drawn as a small image, next to the feature map
// it produces for the current input.
import { ACTIVATION, DIVERGING, paintPixels } from '../viz/colors.js';
import { kernelIndex } from '../model/cnn.js';

function makeCanvas(size, title) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  if (title) c.title = title;
  return c;
}

function paint(canvas, values, map, toT) {
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(canvas.width, canvas.height);
  paintPixels(img.data, values, map, toT);
  ctx.putImageData(img, 0, 0);
}

const maxAbs = (arr) => arr.reduce((m, v) => Math.max(m, Math.abs(v)), 1e-9);

export class KernelPanel {
  constructor(root, { onSelectMap }) {
    this.root = root;
    root.innerHTML = `
      <h3>What the filters learned</h3>
      <p class="lede">Each kernel is a small grid of weights learned during training.
      <span style="color:var(--pos)">Orange</span> = positive weight,
      <span style="color:var(--neg)">blue</span> = negative. Nobody designed these patterns: they emerged
      from seeing thousands of digits.</p>
      <h4>Conv 1 · 8 kernels of 5×5 → their feature maps</h4>
      <div class="kernel-cards" id="k1"></div>
      <p class="muted" style="margin-top:8px">Many act as <b>edge and stroke detectors</b>: an orange band next
      to a blue band responds to a bright line with a dark side, at a particular angle. The map below each
      kernel shows where that pattern appears in your digit. Click one to inspect it in 3D.</p>
      <h4>Conv 2 · 16 kernels of 3×3×8</h4>
      <p class="muted">Each row is one 3D kernel, cut into its 8 slices, one per Conv 1 / pool 1 map (columns).
      It builds a new feature by mixing the 8 simpler ones, e.g. "vertical stroke here <i>and</i> horizontal
      stroke above it" = a corner.</p>
      <div class="k2-grid" id="k2"></div>
      <div class="callout" style="margin-top:14px">Want to see these appear? Open the <b>Train</b> tab and train
      the network from scratch: the kernels start as random noise and settle into these kinds of patterns
      within a few hundred steps.</div>`;

    this.k1 = [];
    const k1Root = root.querySelector('#k1');
    for (let o = 0; o < 8; o++) {
      const card = document.createElement('div');
      card.className = 'kcard';
      const kernel = makeCanvas(5, `Conv 1 kernel #${o}`);
      const fmap = makeCanvas(28, `Feature map #${o} for the current input`);
      card.append(kernel, Object.assign(document.createElement('div'), { className: 'arrow', textContent: '↓' }), fmap,
        Object.assign(document.createElement('div'), { className: 'cap', textContent: `#${o}` }));
      card.addEventListener('click', () => onSelectMap('conv1', o));
      k1Root.append(card);
      this.k1.push({ kernel, fmap });
    }

    const k2Root = root.querySelector('#k2');
    k2Root.append(Object.assign(document.createElement('div'), { className: 'cl' }));
    for (let i = 0; i < 8; i++) k2Root.append(Object.assign(document.createElement('div'), { className: 'cl', textContent: `in ${i}` }));
    this.k2 = [];
    for (let o = 0; o < 16; o++) {
      k2Root.append(Object.assign(document.createElement('div'), { className: 'rl', textContent: `#${o}` }));
      const row = [];
      for (let i = 0; i < 8; i++) {
        const c = makeCanvas(3, `Conv 2 kernel #${o}, slice for input map ${i}`);
        c.addEventListener('click', () => onSelectMap('conv2', o));
        k2Root.append(c);
        row.push(c);
      }
      this.k2.push(row);
    }
  }

  update(net, res) {
    const c1 = net.conv1;
    const m1 = maxAbs(c1.w);
    const fMax = Math.max(1e-9, ...res.conv1);
    this.k1.forEach(({ kernel, fmap }, o) => {
      const w = c1.w.subarray(o * 25, (o + 1) * 25);
      paint(kernel, w, DIVERGING, (v) => 0.5 + 0.5 * v / m1);
      paint(fmap, res.conv1.subarray(o * 784, (o + 1) * 784), ACTIVATION, (v) => v / fMax);
    });

    const c2 = net.conv2;
    const m2 = maxAbs(c2.w);
    const slice = new Float32Array(9);
    this.k2.forEach((row, o) => row.forEach((canvas, i) => {
      for (let ky = 0; ky < 3; ky++)
        for (let kx = 0; kx < 3; kx++) slice[ky * 3 + kx] = c2.w[kernelIndex(c2, o, i, ky, kx)];
      paint(canvas, slice, DIVERGING, (v) => 0.5 + 0.5 * v / m2);
    }));
  }
}
