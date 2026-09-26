// For a selected neuron, works out which neurons in the previous layer feed it and how much
// each one contributes. The 3D view draws these as lines, and for convolutions it also floats
// the kernel above the patch it is looking at (the "ghost").
import { LAYER_BY_ID, unpack, pack } from './layout.js';
import { kernelIndex, flattenIndex } from '../model/cnn.js';

/**
 * @returns {{ links: {layer, index, s, max?}[], ghost: {layer, map, y, x, t}[] }}
 *   s: signed strength in [-1, 1] (contribution relative to the largest one)
 *   t: kernel weight relative to the largest |weight| of that filter
 */
export function receptiveField(ref, net, res) {
  switch (ref.layer) {
    case 'conv1': return convField(ref, net.conv1, 'input', res.input);
    case 'conv2': return convField(ref, net.conv2, 'pool1', res.pool1);
    case 'pool1': return poolField(ref, 'conv1', res.conv1);
    case 'pool2': return poolField(ref, 'conv2', res.conv2);
    case 'dense1': return denseField(ref, net, res);
    case 'output': return outputField(ref, net, res);
    default: return { links: [], ghost: [] };
  }
}

function convField(ref, layer, srcId, src) {
  const L = LAYER_BY_ID[ref.layer];
  const S = LAYER_BY_ID[srcId];
  const { map: o, y, x } = unpack(L, ref.index);
  const { k, inC } = layer;
  const pad = (k - 1) >> 1;

  let maxW = 1e-9;
  for (let i = 0; i < inC * k * k; i++) maxW = Math.max(maxW, Math.abs(layer.w[o * inC * k * k + i]));

  const ghost = [];
  const raw = [];
  for (let i = 0; i < inC; i++) {
    for (let ky = 0; ky < k; ky++) {
      for (let kx = 0; kx < k; kx++) {
        const w = layer.w[kernelIndex(layer, o, i, ky, kx)];
        const iy = y + ky - pad;
        const ix = x + kx - pad;
        ghost.push({ layer: srcId, map: i, y: iy, x: ix, t: w / maxW });
        if (iy < 0 || iy >= S.size || ix < 0 || ix >= S.size) continue; // zero padding
        const index = pack(S, i, iy, ix);
        raw.push({ layer: srcId, index, c: src[index] * w, w });
      }
    }
  }
  const maxC = Math.max(1e-9, ...raw.map((r) => Math.abs(r.c)));
  // Inputs that are zero contribute nothing; draw them faintly with the weight's sign.
  const links = raw.map((r) => ({
    layer: r.layer, index: r.index,
    s: r.c !== 0 ? r.c / maxC : Math.sign(r.w) * 0.04,
  }));
  return { links, ghost };
}

function poolField(ref, srcId, src) {
  const L = LAYER_BY_ID[ref.layer];
  const S = LAYER_BY_ID[srcId];
  const { map, y, x } = unpack(L, ref.index);
  const cells = [];
  for (let dy = 0; dy < 2; dy++)
    for (let dx = 0; dx < 2; dx++) {
      const index = pack(S, map, 2 * y + dy, 2 * x + dx);
      cells.push({ index, v: src[index] });
    }
  const best = cells.reduce((a, b) => (b.v > a.v ? b : a));
  return {
    links: cells.map((c) => ({ layer: srcId, index: c.index, s: c === best ? 1 : 0.12, max: c === best })),
    ghost: [],
  };
}

function denseField(ref, net, res) {
  const j = ref.index;
  const P = LAYER_BY_ID.pool2;
  const { outN, w } = net.dense1;
  const raw = [];
  for (let c = 0; c < P.maps; c++)
    for (let y = 0; y < P.size; y++)
      for (let x = 0; x < P.size; x++) {
        const index = pack(P, c, y, x);
        const a = res.pool2[index];
        if (a === 0) continue;
        raw.push({ index, c: a * w[flattenIndex(c, y, x, P.size, P.maps) * outN + j] });
      }
  raw.sort((a, b) => Math.abs(b.c) - Math.abs(a.c));
  const top = raw.slice(0, 90);
  const maxC = Math.max(1e-9, ...top.map((r) => Math.abs(r.c)));
  return { links: top.map((r) => ({ layer: 'pool2', index: r.index, s: r.c / maxC })), ghost: [] };
}

function outputField(ref, net, res) {
  const d = ref.index;
  const { outN, w } = net.dense2;
  const raw = [];
  for (let j = 0; j < res.dense1.length; j++) {
    const a = res.dense1[j];
    if (a !== 0) raw.push({ index: j, c: a * w[j * outN + d] });
  }
  const maxC = Math.max(1e-9, ...raw.map((r) => Math.abs(r.c)));
  return { links: raw.map((r) => ({ layer: 'dense1', index: r.index, s: r.c / maxC })), ghost: [] };
}
