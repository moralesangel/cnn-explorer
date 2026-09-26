// Trains the network from scratch in the browser with TF.js (WebGL when available).
// This module and TF.js itself are only downloaded when the user starts training.
import * as tf from '@tensorflow/tfjs';
import { buildModel, exportWeights } from '../model/tf-model.js';
import { loadMnist } from '../data/mnist.js';

/**
 * @param {object} o
 * @param {number} o.epochs
 * @param {number} o.learningRate
 * @param {(msg: string) => void} o.onStatus
 * @param {(s: {epoch, batch, batches, step, loss, acc}) => void} o.onBatch
 * @param {(json: object, info: {step, epoch}) => void} o.onWeights   snapshot for the visualization
 * @param {(s: {epoch, testAcc, step}) => void} o.onEpoch
 * @param {() => boolean} o.shouldStop
 */
export async function trainFromScratch({ epochs, learningRate, batchSize = 32, onStatus, onBatch, onWeights, onEpoch, shouldStop }) {
  onStatus('Starting TensorFlow.js…');
  await tf.ready();
  onStatus(`Loading MNIST digits… (backend: ${tf.getBackend()})`);
  const [train, test] = await Promise.all([loadMnist('train'), loadMnist('test')]);

  const xs = tf.tensor4d(train.pixels, [train.count, 28, 28, 1]);
  const ys = tf.oneHot(tf.tensor1d(train.labels, 'int32'), 10).cast('float32');
  const testXs = tf.tensor4d(test.pixels, [test.count, 28, 28, 1]);
  const testYs = tf.oneHot(tf.tensor1d(test.labels, 'int32'), 10).cast('float32');

  const model = buildModel(tf, learningRate);
  const batches = Math.ceil(train.count / batchSize);
  let step = 0;
  onWeights(exportWeights(model), { step, epoch: 0 });
  onStatus('Training…');

  try {
    for (let epoch = 0; epoch < epochs; epoch++) {
      const order = tf.util.createShuffledIndices(train.count);
      for (let b = 0; b < batches; b++) {
        if (shouldStop()) {
          onStatus('Stopped.');
          return;
        }
        const idx = tf.tensor1d(Int32Array.from(order.subarray(b * batchSize, (b + 1) * batchSize)), 'int32');
        const bx = tf.gather(xs, idx);
        const by = tf.gather(ys, idx);
        const [loss, acc] = await model.trainOnBatch(bx, by);
        tf.dispose([idx, bx, by]);
        step++;
        onBatch({ epoch, batch: b, batches, step, loss, acc });

        // Frequent snapshots early on, when the kernels change the most.
        if (step <= 60 ? step % 3 === 0 : step % 12 === 0) {
          onWeights(exportWeights(model), { step, epoch });
          await tf.nextFrame();
        }
      }
      const evaluation = model.evaluate(testXs, testYs, { batchSize: 500 }); // [loss, accuracy]
      const testAcc = (await evaluation[1].data())[0];
      tf.dispose(evaluation);
      onWeights(exportWeights(model, { trainedAt: 'in your browser', testAccuracy: testAcc, epochs: epoch + 1, trainSamples: train.count }), { step, epoch });
      onEpoch({ epoch, testAcc, step });
    }
    onStatus('Done! These are now the weights you see everywhere on the page.');
  } finally {
    tf.dispose([xs, ys, testXs, testYs]);
    model.dispose();
  }
}
