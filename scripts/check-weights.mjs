// Verifies public/model/weights.json against the full MNIST test set using the
// hand-written forward pass in src/model/cnn.js (the one the visualization uses), and checks
// that loading the same weights into the TF.js model gives identical predictions.
//
//   npm run check
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as tf from '@tensorflow/tfjs';
import { ensureMnist } from './prepare-data.mjs';
import { loadNetwork, forward } from '../src/model/cnn.js';
import { buildModel, importWeights } from '../src/model/tf-model.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const json = JSON.parse(fs.readFileSync(path.join(ROOT, 'public', 'model', 'weights.json'), 'utf8'));
const net = loadNetwork(json);
const { test } = await ensureMnist();

const n = test.labels.dims[0];
const pixels = Float32Array.from(test.images.data, (v) => v / 255);
let correct = 0;
const confusion = Array.from({ length: 10 }, () => new Array(10).fill(0));
for (let i = 0; i < n; i++) {
  const { prediction } = forward(net, pixels.subarray(i * 784, (i + 1) * 784));
  confusion[test.labels.data[i]][prediction]++;
  if (prediction === test.labels.data[i]) correct++;
}
const acc = correct / n;
console.log(`JS forward pass: ${(acc * 100).toFixed(2)}% on ${n} test digits (weights report ${(json.meta.testAccuracy * 100).toFixed(2)}%)`);
console.log('confusion (rows = true digit, cols = predicted):');
confusion.forEach((row, d) => console.log(`  ${d}: ${row.map((v) => String(v).padStart(5)).join('')}`));

await tf.setBackend('cpu');
const model = buildModel(tf);
importWeights(tf, model, json);
const sample = 100;
const tfProbs = model.predict(tf.tensor4d(pixels.subarray(0, sample * 784), [sample, 28, 28, 1])).arraySync();
let maxDiff = 0;
for (let i = 0; i < sample; i++) {
  forward(net, pixels.subarray(i * 784, (i + 1) * 784)).probs
    .forEach((p, j) => { maxDiff = Math.max(maxDiff, Math.abs(p - tfProbs[i][j])); });
}
console.log(`max |JS forward - TF.js| on ${sample} digits: ${maxDiff.toExponential(2)}`);

if (Math.abs(acc - json.meta.testAccuracy) > 0.002 || maxDiff > 1e-3) {
  console.error('MISMATCH: the JS forward pass does not reproduce the trained model');
  process.exit(1);
}
console.log('ok');
