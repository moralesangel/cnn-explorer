// The drawing canvas, and the preprocessing that turns a drawing into an MNIST-like input.
//
// MNIST digits were size-normalized to fit a 20x20 box and then centered in a 28x28 image by
// their center of mass. A network trained on them expects exactly that, so drawings get the
// same treatment; without it, a digit drawn in a corner or very small is often misread.
import { GRAY, paintPixels } from '../viz/colors.js';

const SIZE = 280;
const STROKE = 20;

export class DrawPad {
  constructor(canvas, preview, { onChange, onStart }) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { willReadFrequently: true });
    this.preview = preview;
    this.previewCtx = preview.getContext('2d');
    this.onChange = onChange;
    this.onStart = onStart;
    this.showingExample = false;
    this.pending = false;
    this.clear(false);

    let last = null;
    const point = (e) => {
      const r = canvas.getBoundingClientRect();
      return { x: ((e.clientX - r.left) / r.width) * SIZE, y: ((e.clientY - r.top) / r.height) * SIZE };
    };
    canvas.addEventListener('pointerdown', (e) => {
      canvas.setPointerCapture(e.pointerId);
      if (this.showingExample) this.clear(false);
      this.showingExample = false;
      this.onStart?.();
      last = point(e);
      this._stroke(last, last);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!last) return;
      const p = point(e);
      this._stroke(last, p);
      last = p;
    });
    const end = () => {
      if (!last) return;
      last = null;
      this._emit();
    };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
  }

  _stroke(a, b) {
    const { ctx } = this;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = STROKE;
    ctx.lineCap = ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    // Coalesce many pointer events into at most one network update per frame.
    if (!this.pending) {
      this.pending = true;
      requestAnimationFrame(() => {
        this.pending = false;
        this._emit();
      });
    }
  }

  _emit() {
    const image = preprocess(this.canvas, this.ctx);
    this._showPreview(image);
    this.onChange(image, { drawn: true });
  }

  clear(emit = true) {
    this.ctx.fillStyle = '#000';
    this.ctx.fillRect(0, 0, SIZE, SIZE);
    this.showingExample = false;
    const empty = new Float32Array(784);
    this._showPreview(empty);
    if (emit) this.onChange(empty, { drawn: true });
  }

  /** Shows an MNIST digit (784 values in 0..1) on the pad and sends it unchanged to the net. */
  showExample(pixels, meta) {
    const small = document.createElement('canvas');
    small.width = small.height = 28;
    const sctx = small.getContext('2d');
    const img = sctx.createImageData(28, 28);
    pixels.forEach((v, i) => {
      img.data.fill(v * 255, i * 4, i * 4 + 3);
      img.data[i * 4 + 3] = 255;
    });
    sctx.putImageData(img, 0, 0);
    this.ctx.fillStyle = '#000';
    this.ctx.fillRect(0, 0, SIZE, SIZE);
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.drawImage(small, 0, 0, SIZE, SIZE);
    this.showingExample = true;
    this._showPreview(pixels);
    this.onChange(pixels, meta);
  }

  _showPreview(pixels) {
    const img = this.previewCtx.createImageData(28, 28);
    paintPixels(img.data, pixels, GRAY, (v) => v);
    this.previewCtx.putImageData(img, 0, 0);
  }
}

/** Canvas drawing -> Float32Array(784) in 0..1, MNIST style. */
export function preprocess(canvas, ctx) {
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
  let minX = SIZE, minY = SIZE, maxX = -1, maxY = -1;
  for (let y = 0; y < SIZE; y++)
    for (let x = 0; x < SIZE; x++)
      if (data[(y * SIZE + x) * 4] > 30) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
  const out = new Float32Array(784);
  if (maxX < 0) return out;

  // 1. Crop to the ink and scale so the longer side is 20 px (aspect ratio kept).
  const bw = maxX - minX + 1;
  const bh = maxY - minY + 1;
  const scale = 20 / Math.max(bw, bh);
  const tw = Math.max(1, Math.round(bw * scale));
  const th = Math.max(1, Math.round(bh * scale));

  // Downscale in two steps (canvas to 4x, then a 4x4 box average) for smooth anti-aliasing.
  const S = 4;
  const tmp = document.createElement('canvas');
  tmp.width = tw * S;
  tmp.height = th * S;
  const tctx = tmp.getContext('2d', { willReadFrequently: true });
  tctx.imageSmoothingEnabled = true;
  tctx.imageSmoothingQuality = 'high';
  tctx.drawImage(canvas, minX, minY, bw, bh, 0, 0, tw * S, th * S);
  const big = tctx.getImageData(0, 0, tw * S, th * S).data;
  const small = new Float32Array(tw * th);
  let peak = 0;
  for (let y = 0; y < th; y++)
    for (let x = 0; x < tw; x++) {
      let sum = 0;
      for (let dy = 0; dy < S; dy++)
        for (let dx = 0; dx < S; dx++) sum += big[((y * S + dy) * tw * S + x * S + dx) * 4];
      const v = sum / (S * S * 255);
      small[y * tw + x] = v;
      peak = Math.max(peak, v);
    }

  // 2. Place it so its center of mass lands on the center of the 28x28 image.
  let mass = 0, mx = 0, my = 0;
  for (let y = 0; y < th; y++)
    for (let x = 0; x < tw; x++) {
      const v = small[y * tw + x];
      mass += v;
      mx += v * (x + 0.5);
      my += v * (y + 0.5);
    }
  const clamp = (v, hi) => Math.max(0, Math.min(hi, v));
  const ox = clamp(Math.round(14 - mx / mass), 28 - tw);
  const oy = clamp(Math.round(14 - my / mass), 28 - th);
  for (let y = 0; y < th; y++)
    for (let x = 0; x < tw; x++) out[(oy + y) * 28 + ox + x] = Math.min(1, small[y * tw + x] / peak);
  return out;
}
