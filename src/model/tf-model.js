// TF.js version of the network, used only for training (in Node and in the browser).
// `tf` is passed in rather than imported so the same file works with whichever TF.js
// build the caller has loaded.
import { ARCH } from './arch.js';

export const WEIGHT_LAYERS = ['conv1', 'conv2', 'dense1', 'dense2'];

export function buildModel(tf, learningRate = 0.001) {
  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    name: 'conv1', inputShape: [ARCH.inputSize, ARCH.inputSize, 1],
    filters: ARCH.conv1.filters, kernelSize: ARCH.conv1.kernel, padding: 'same', activation: 'relu',
  }));
  model.add(tf.layers.maxPooling2d({ name: 'pool1', poolSize: 2 }));
  model.add(tf.layers.conv2d({
    name: 'conv2', filters: ARCH.conv2.filters, kernelSize: ARCH.conv2.kernel, padding: 'same', activation: 'relu',
  }));
  model.add(tf.layers.maxPooling2d({ name: 'pool2', poolSize: 2 }));
  model.add(tf.layers.flatten({ name: 'flatten' }));
  model.add(tf.layers.dense({ name: 'dense1', units: ARCH.dense1.units, activation: 'relu' }));
  model.add(tf.layers.dense({ name: 'dense2', units: ARCH.classes, activation: 'softmax' }));
  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
}

const round = (arr) => Array.from(arr, (v) => Math.round(v * 1e5) / 1e5);

/**
 * Serializes the trained weights as plain JSON, in TF.js layout:
 * conv kernels are [kh, kw, inChannels, filters], dense kernels are [inputs, units].
 */
export function exportWeights(model, meta = {}) {
  const out = { format: 'cnn-explorer/1', meta, layers: {} };
  for (const name of WEIGHT_LAYERS) {
    const [kernel, bias] = model.getLayer(name).getWeights();
    out.layers[name] = { shape: kernel.shape, kernel: round(kernel.dataSync()), bias: round(bias.dataSync()) };
  }
  return out;
}

export function importWeights(tf, model, json) {
  for (const name of WEIGHT_LAYERS) {
    const w = json.layers[name];
    model.getLayer(name).setWeights([tf.tensor(w.kernel, w.shape), tf.tensor1d(w.bias)]);
  }
}
