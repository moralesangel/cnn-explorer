// The "Guide" tab: a walk through the network, one layer at a time. Clicking a step flies the
// camera to that layer.

const STEPS = [
  {
    layers: ['input'],
    title: 'The input: 784 numbers',
    body: `<p>Your drawing is cropped, scaled to fit a 20×20 box and centered in a 28×28 grid, exactly how
      the MNIST digits were prepared. Each pixel becomes a number from 0 (black) to 1 (white). In 3D, taller
      and brighter cubes are brighter pixels.</p>`,
  },
  {
    layers: ['input', 'conv1'],
    title: 'Convolution: sliding a small pattern',
    body: `<p>A <b>kernel</b> is a 5×5 grid of weights. It slides over the image; at every position the patch
      under it is multiplied cell by cell with the kernel and summed, plus a bias. Doing this at all 784 positions
      produces a <b>feature map</b>. The map is bright wherever the image locally looks like the kernel.</p>
      <p>Conv 1 has 8 kernels, so it produces 8 feature maps. Click a bright cube in Conv 1 to see the kernel
      float over the patch it is reading, then press "Slide this kernel".</p>`,
    formula: 'out(y,x) = b + Σᵢⱼ in(y+i, x+j) · k(i,j)',
  },
  {
    layers: ['conv1'],
    title: 'ReLU: keep only the positive evidence',
    body: `<p>Each sum then goes through <b>ReLU</b>, which replaces negative numbers with 0. A neuron either
      reports "I found my pattern, this strongly" or stays silent. This simple bend is what lets stacked layers
      learn things a single weighted sum never could.</p>`,
    formula: 'ReLU(x) = max(0, x)',
  },
  {
    layers: ['conv1', 'pool1'],
    title: 'Max pooling: shrink, keep the strongest',
    body: `<p>Each map is cut into 2×2 blocks and only the largest value in each block survives: 28×28 becomes
      14×14. The network keeps <i>that</i> a stroke is present in a region and forgets its exact pixel, which
      makes it tolerant to small shifts and 4× cheaper for the next layer. Pooling has no weights.</p>`,
  },
  {
    layers: ['pool1', 'conv2'],
    title: 'A second convolution: patterns of patterns',
    body: `<p>Conv 2's kernels are <b>3×3×8</b> blocks: they read all 8 pooled maps at once and combine them.
      "A vertical stroke with a horizontal one above it" is a corner; several curves make a loop. Because the input
      was pooled, each neuron here "sees" a 10×10 pixel region of the original drawing.</p>`,
  },
  {
    layers: ['conv2', 'pool2'],
    title: 'Pool again, then flatten',
    body: `<p>Another 2×2 max pool gives 16 maps of 7×7 = <b>784 numbers</b>. These are then lined up into one
      long list ("flattened"). Each number now means something like "a loop-ish shape in the upper-left area".</p>`,
  },
  {
    layers: ['pool2', 'dense1'],
    title: 'Dense layer: weighing all the evidence',
    body: `<p>Each of the 64 neurons is connected to <b>all 784</b> inputs with its own weight (50,176 weights in
      total, most of the network's parameters). It computes one weighted sum plus a bias, then ReLU. Click one to
      see its weights drawn over the 16 maps: the layout of features it is looking for.</p>`,
    formula: 'h = ReLU(W · x + b)',
  },
  {
    layers: ['dense1', 'output'],
    title: 'Output: softmax turns scores into probabilities',
    body: `<p>10 neurons, one per digit, each a weighted sum of the 64 dense neurons: its <b>logit</b>. Softmax
      exponentiates the logits and divides by their total, so they become positive and add up to 100%. The tallest
      bar is the prediction.</p>`,
    formula: 'p(d) = e^(z_d) / Σⱼ e^(z_j)',
  },
  {
    layers: null,
    title: 'Where do the weights come from? Training',
    body: `<p>Nobody sets the 52,266 weights by hand. Training shows the network batches of labelled digits,
      measures how wrong the output probabilities are (<b>cross-entropy loss</b>), and uses
      <b>backpropagation</b> to work out how each weight contributed to that error. Every weight is then nudged a
      little in the direction that reduces it (<b>gradient descent</b>, here the Adam optimizer). After a few
      thousand nudges the kernels have turned into edge detectors.</p>
      <p>Open the <b>Train</b> tab to watch it happen from random weights.</p>`,
  },
];

export class Guide {
  constructor(root, { onStep }) {
    root.innerHTML = `
      <h3>How a CNN reads a digit</h3>
      <p class="lede">A convolutional neural network (CNN) turns 784 pixel values into 10 probabilities, one layer
      at a time. Draw a digit on the left, then follow the steps. Each one moves the camera to that part of the
      network.</p>
      ${STEPS.map((s, i) => `
        <div class="guide-step" data-step="${i}">
          <h3><span class="step">${i + 1}</span>${s.title}</h3>
          ${s.body}
          ${s.formula ? `<div class="formula">${s.formula}</div>` : ''}
        </div>`).join('')}
      <p class="muted">This network is tiny on purpose (about 52k weights) so every neuron fits on screen. It still
      gets about 99% of the MNIST test digits right.</p>`;

    root.addEventListener('click', (e) => {
      const el = e.target.closest('.guide-step');
      if (!el) return;
      root.querySelectorAll('.guide-step').forEach((s) => s.classList.toggle('active', s === el));
      onStep(STEPS[Number(el.dataset.step)]);
    });
  }
}
