// Colormaps as 256-entry lookup tables. `srgb` tables are for CSS/canvas; `linear` tables
// are for three.js instance colors (which are interpreted in linear space).

function parseHex(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

const toLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

function makeMap(stops) {
  const parsed = stops.map(([t, hex]) => [t, parseHex(hex)]);
  const srgb = new Float32Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let s = 0;
    while (s < parsed.length - 2 && t > parsed[s + 1][0]) s++;
    const [t0, c0] = parsed[s];
    const [t1, c1] = parsed[s + 1];
    const f = Math.min(1, Math.max(0, (t - t0) / (t1 - t0)));
    for (let k = 0; k < 3; k++) srgb[i * 3 + k] = c0[k] + (c1[k] - c0[k]) * f;
  }
  return { srgb, linear: srgb.map(toLinear) };
}

/** Activations (>= 0): dark navy -> indigo -> magenta -> orange -> pale yellow. */
export const ACTIVATION = makeMap([[0, '#161c2c'], [0.2, '#2d2a6e'], [0.45, '#8a2f8f'], [0.72, '#e8604a'], [1, '#fde9a0']]);
/** Raw pixel brightness. */
export const GRAY = makeMap([[0, '#161c2c'], [1, '#ffffff']]);
/** Signed values (weights, contributions): blue < 0 < orange. Index with (v + 1) / 2. */
export const DIVERGING = makeMap([[0, '#2f7bff'], [0.5, '#1b2130'], [1, '#ffab2e']]);

const idx = (t) => Math.max(0, Math.min(255, Math.round(t * 255))) * 3;

/** Writes the colour for t in [0, 1] into out[offset..offset+2]. */
export function writeColor(table, t, out, offset) {
  const i = idx(t);
  out[offset] = table[i];
  out[offset + 1] = table[i + 1];
  out[offset + 2] = table[i + 2];
}

export function cssColor(map, t) {
  const i = idx(t);
  const c = map.srgb;
  return `rgb(${Math.round(c[i] * 255)},${Math.round(c[i + 1] * 255)},${Math.round(c[i + 2] * 255)})`;
}

/** Colour for a signed value v scaled by `max` (|v| >= max saturates). */
export const signedCss = (v, max) => cssColor(DIVERGING, 0.5 + 0.5 * Math.max(-1, Math.min(1, v / (max || 1))));

/** Paints values into an ImageData-sized RGBA buffer using a colormap. */
export function paintPixels(rgba, values, map, toT) {
  const c = map.srgb;
  for (let p = 0; p < values.length; p++) {
    const i = idx(toT(values[p]));
    rgba[p * 4] = c[i] * 255;
    rgba[p * 4 + 1] = c[i + 1] * 255;
    rgba[p * 4 + 2] = c[i + 2] * 255;
    rgba[p * 4 + 3] = 255;
  }
}
