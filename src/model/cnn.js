// A plain-JavaScript forward pass of the network in arch.js.
//
// TF.js could do this too, but writing it out by hand keeps every intermediate number
// available for the visualization, and it doubles as readable reference code for what each
// layer actually computes. Feature maps are stored channel-major: value (c, y, x) of a
// layer with height h and width w lives at data[c*h*w + y*w + x].

/**
 * Converts weights exported by tf-model.js (TF layout) into the layout used here:
 * conv kernels become [filter][inChannel][ky][kx], dense kernels stay [input][unit].
 */
export function loadNetwork(json) {
  const conv = ({ shape: [k, , inC, outC], kernel, bias }) => {
    const w = new Float32Array(outC * inC * k * k);
    for (let ky = 0; ky < k; ky++)
      for (let kx = 0; kx < k; kx++)
        for (let i = 0; i < inC; i++)
          for (let o = 0; o < outC; o++)
            w[((o * inC + i) * k + ky) * k + kx] = kernel[((ky * k + kx) * inC + i) * outC + o];
    return { k, inC, outC, w, b: Float32Array.from(bias) };
  };
  const dense = ({ shape: [inN, outN], kernel, bias }) =>
    ({ inN, outN, w: Float32Array.from(kernel), b: Float32Array.from(bias) });
  const L = json.layers;
  return {
    meta: json.meta ?? {},
    conv1: conv(L.conv1),
    conv2: conv(L.conv2),
    dense1: dense(L.dense1),
    dense2: dense(L.dense2),
  };
}

/** Index of the kernel weight for (filter o, input channel i, ky, kx). */
export const kernelIndex = (layer, o, i, ky, kx) => ((o * layer.inC + i) * layer.k + ky) * layer.k + kx;

/**
 * TF's Flatten walks the pooled volume in (y, x, channel) order, so this is where value
 * (c, y, x) of a 7x7xC map ends up in the 784-long vector fed to dense1.
 */
export const flattenIndex = (c, y, x, w, channels) => (y * w + x) * channels + c;

function conv2dSame(input, inC, size, layer) {
  const { k, outC, w, b } = layer;
  const pad = (k - 1) >> 1;
  const plane = size * size;
  const pre = new Float32Array(outC * plane);
  for (let o = 0; o < outC; o++) {
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        let sum = b[o];
        for (let i = 0; i < inC; i++) {
          for (let ky = 0; ky < k; ky++) {
            const iy = y + ky - pad;
            if (iy < 0 || iy >= size) continue; // zero padding
            for (let kx = 0; kx < k; kx++) {
              const ix = x + kx - pad;
              if (ix < 0 || ix >= size) continue;
              sum += input[i * plane + iy * size + ix] * w[((o * inC + i) * k + ky) * k + kx];
            }
          }
        }
        pre[o * plane + y * size + x] = sum;
      }
    }
  }
  return pre;
}

const relu = (arr) => arr.map((v) => (v > 0 ? v : 0));

function maxPool2(input, channels, size) {
  const out = size >> 1;
  const res = new Float32Array(channels * out * out);
  for (let c = 0; c < channels; c++)
    for (let y = 0; y < out; y++)
      for (let x = 0; x < out; x++) {
        const base = c * size * size + 2 * y * size + 2 * x;
        res[c * out * out + y * out + x] = Math.max(
          input[base], input[base + 1], input[base + size], input[base + size + 1]);
      }
  return res;
}

function denseLayer(input, layer) {
  const { inN, outN, w, b } = layer;
  const out = Float32Array.from(b);
  for (let i = 0; i < inN; i++) {
    const v = input[i];
    if (v === 0) continue;
    for (let o = 0; o < outN; o++) out[o] += v * w[i * outN + o];
  }
  return out;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / total);
}

/**
 * Runs one 28x28 image (Float32Array of 784 values in 0..1) through the network and
 * returns every intermediate result.
 */
export function forward(net, image) {
  const s1 = 28, s2 = 14, s3 = 7;
  const c1 = net.conv1.outC, c2 = net.conv2.outC;

  const conv1Pre = conv2dSame(image, 1, s1, net.conv1);
  const conv1 = relu(conv1Pre);
  const pool1 = maxPool2(conv1, c1, s1);
  const conv2Pre = conv2dSame(pool1, c1, s2, net.conv2);
  const conv2 = relu(conv2Pre);
  const pool2 = maxPool2(conv2, c2, s2);

  // Channel-major -> TF flatten order.
  const flat = new Float32Array(c2 * s3 * s3);
  for (let c = 0; c < c2; c++)
    for (let y = 0; y < s3; y++)
      for (let x = 0; x < s3; x++)
        flat[flattenIndex(c, y, x, s3, c2)] = pool2[c * s3 * s3 + y * s3 + x];

  const dense1Pre = denseLayer(flat, net.dense1);
  const dense1 = relu(dense1Pre);
  const logits = denseLayer(dense1, net.dense2);
  const probs = softmax(logits);

  return {
    input: image,
    conv1Pre, conv1, pool1, conv2Pre, conv2, pool2, flat,
    dense1Pre, dense1, logits, probs,
    prediction: probs.indexOf(Math.max(...probs)),
  };
}
