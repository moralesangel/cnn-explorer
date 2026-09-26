// Where every neuron sits in 3D space.
//
// The network flows left to right along +x. Each layer lies flat on the floor (the x/z
// plane) and every neuron is a cube whose height (y) and colour show its activation.
// Feature maps are laid out as a grid of images; pixel row y runs towards the camera (+z)
// so the maps read like pictures when seen from the default viewpoint.
//
// Within a layer, neurons are numbered exactly like the activation arrays in cnn.js:
// index = map * size * size + y * size + x (maps) or just the unit number (vectors).

export const LAYERS = [
  { id: 'input', kind: 'map', maps: 1, size: 28, cols: 1, heightScale: 3,
    title: 'Input', sub: '28 × 28 pixels' },
  { id: 'conv1', kind: 'map', maps: 8, size: 28, cols: 2, heightScale: 4,
    title: 'Conv 1', sub: '8 filters · 5×5 · ReLU' },
  { id: 'pool1', kind: 'map', maps: 8, size: 14, cols: 2, heightScale: 4,
    title: 'Max pool 1', sub: '2×2 → 14 × 14' },
  { id: 'conv2', kind: 'map', maps: 16, size: 14, cols: 2, heightScale: 4,
    title: 'Conv 2', sub: '16 filters · 3×3×8 · ReLU' },
  { id: 'pool2', kind: 'map', maps: 16, size: 7, cols: 2, heightScale: 4,
    title: 'Max pool 2', sub: '2×2 → 7 × 7 · flatten → 784' },
  { id: 'dense1', kind: 'vector', units: 64, pitch: 2, cube: 1.5, heightScale: 8,
    title: 'Dense', sub: '64 · ReLU' },
  { id: 'output', kind: 'vector', units: 10, pitch: 9, cube: 6, heightScale: 26,
    title: 'Output', sub: 'softmax' },
];

export const LAYER_BY_ID = Object.fromEntries(LAYERS.map((L) => [L.id, L]));
const LAYER_GAP = 26;
export const CELL = 0.9; // cube footprint inside a 1x1 pixel slot

(function computeLayout() {
  let x = 0;
  for (const L of LAYERS) {
    if (L.kind === 'map') {
      L.gap = L.size >= 28 ? 4 : L.size >= 14 ? 3 : 2;
      L.rows = Math.ceil(L.maps / L.cols);
      L.width = L.cols * L.size + (L.cols - 1) * L.gap;
      L.depth = L.rows * L.size + (L.rows - 1) * L.gap;
      L.count = L.maps * L.size * L.size;
    } else {
      L.width = L.cube;
      L.depth = L.units * L.pitch;
      L.count = L.units;
    }
    L.x0 = x;
    L.z0 = -L.depth / 2;
    x += L.width + LAYER_GAP;
  }
  const total = x - LAYER_GAP;
  for (const L of LAYERS) {
    L.x0 -= total / 2;
    L.cx = L.x0 + L.width / 2;
  }
})();

export const NETWORK_WIDTH = LAYERS.at(-1).x0 + LAYERS.at(-1).width - LAYERS[0].x0;
export const NETWORK_DEPTH = Math.max(...LAYERS.map((L) => L.depth));

/** Top-left corner (x, z) of feature map m. */
export function mapOrigin(L, m) {
  const col = m % L.cols;
  const row = Math.floor(m / L.cols);
  return { x: L.x0 + col * (L.size + L.gap), z: L.z0 + row * (L.size + L.gap) };
}

/** Centre (x, z) of pixel (y, x) in map m. Works outside the map too (for padding). */
export function pixelCenter(L, m, y, x) {
  const o = mapOrigin(L, m);
  return { x: o.x + x + 0.5, z: o.z + y + 0.5 };
}

export function cellCenter(L, index) {
  if (L.kind === 'vector') return { x: L.x0 + L.cube / 2, z: L.z0 + index * L.pitch + L.pitch / 2 };
  const { map, y, x } = unpack(L, index);
  return pixelCenter(L, map, y, x);
}

export function cellFootprint(L) {
  return L.kind === 'vector' ? L.cube : CELL;
}

/** Layer-local index -> { map, y, x } (for feature-map layers). */
export function unpack(L, index) {
  const plane = L.size * L.size;
  const map = Math.floor(index / plane);
  const r = index % plane;
  return { map, y: Math.floor(r / L.size), x: r % L.size };
}

export const pack = (L, map, y, x) => map * L.size * L.size + y * L.size + x;
