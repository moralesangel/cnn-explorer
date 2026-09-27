import './style.css';
import { loadNetwork, forward } from './model/cnn.js';
import { NetworkView } from './viz/scene.js';
import { LAYER_BY_ID, unpack, pack } from './viz/layout.js';
import { receptiveField } from './viz/connections.js';
import { DrawPad } from './ui/draw-pad.js';
import { OutputBars } from './ui/output.js';
import { Inspector, describe } from './ui/inspector.js';
import { KernelPanel } from './ui/kernels.js';
import { Guide } from './ui/guide.js';
import { TrainPanel } from './ui/train-panel.js';
import { loadMnist } from './data/mnist.js';

const $ = (sel) => document.querySelector(sel);

/** Which layer feeds each layer (for framing the camera on a layer and its input). */
const PREVIOUS = { input: 'input', conv1: 'input', pool1: 'conv1', conv2: 'pool1', pool2: 'conv2', dense1: 'pool2', output: 'dense1' };

const state = {
  pretrained: null, // weights JSON as downloaded
  net: null, // weights currently in use (pretrained, or being trained in the browser)
  trainedInBrowser: null, // { step, testAcc } while/after training here
  image: new Float32Array(784),
  blank: true,
  res: null, // result of the last forward pass
  selection: null, // { layer, index }
  sweep: null,
};

// ------------------------------------------------------------------ tabs

function showTab(name) {
  document.querySelectorAll('.tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab').forEach((t) => { t.hidden = t.id !== `tab-${name}`; });
}
document.querySelectorAll('.tabs button').forEach((b) => b.addEventListener('click', () => showTab(b.dataset.tab)));

// ------------------------------------------------------------------ components

const tooltip = $('#tooltip');
const view = new NetworkView($('#viewport'), {
  onHover(ref, x, y) {
    if (!ref || !state.res) {
      tooltip.hidden = true;
      return;
    }
    tooltip.innerHTML = describe(ref, state.res);
    tooltip.style.left = `${x}px`;
    tooltip.style.top = `${y}px`;
    tooltip.classList.toggle('flip', x > tooltip.parentElement.clientWidth - 260);
    tooltip.hidden = false;
  },
  onSelect(ref) {
    stopSweep();
    select(ref);
  },
  onLayerClick(id) {
    view.focusLayers(id === 'input' ? ['input'] : [PREVIOUS[id], id]);
    view.setActiveLayer(id);
  },
});

const inspector = new Inspector($('#tab-inspect'), {
  onAction({ action, layer, index }) {
    if (action === 'sweep') startSweep(state.selection);
    if (action === 'select') {
      stopSweep();
      select({ layer, index });
    }
  },
});

const kernels = new KernelPanel($('#tab-kernels'), {
  onSelectMap(layerId, map) {
    // Pick the strongest neuron in that feature map: the place where the filter fired most.
    const L = LAYER_BY_ID[layerId];
    const plane = L.size * L.size;
    const values = state.res[layerId].subarray(map * plane, (map + 1) * plane);
    const best = values.indexOf(Math.max(...values));
    stopSweep();
    select({ layer: layerId, index: map * plane + best });
    view.focusLayers([PREVIOUS[layerId], layerId]);
  },
});

new Guide($('#tab-guide'), {
  onStep(step) {
    if (!step.layers) {
      view.overview();
      view.setActiveLayer(null);
      return;
    }
    view.focusLayers(step.layers);
    view.setActiveLayer(step.layers.at(-1));
  },
});

new TrainPanel($('#tab-train'), {
  onWeights(json, info) {
    state.net = loadNetwork(json);
    state.trainedInBrowser = { step: info.step, testAcc: json.meta?.testAccuracy ?? state.trainedInBrowser?.testAcc };
    updateModelInfo();
    run({ stagger: 0 });
  },
  onRestore() {
    state.net = loadNetwork(state.pretrained);
    state.trainedInBrowser = null;
    updateModelInfo();
    run();
  },
  onShowKernels: () => showTab('kernels'),
});

const bars = new OutputBars({ bars: $('#bars'), digit: $('#pred-digit'), conf: $('#pred-conf') }, {
  onSelect(d) {
    stopSweep();
    select({ layer: 'output', index: d });
    view.focusLayers(['dense1', 'output']);
  },
});

const pad = new DrawPad($('#pad'), $('#seen'), {
  onStart() {
    $('#pad-hint').style.opacity = 0;
  },
  onChange(image, meta = {}) {
    stopSweep();
    state.image = image;
    state.blank = !image.some((v) => v > 0);
    $('#true-label').textContent = meta.label !== undefined ? `True label: ${meta.label}.` : '';
    $('#pad-hint').style.opacity = state.blank ? 1 : 0;
    if (state.net) run();
  },
});

$('#clear').addEventListener('click', () => pad.clear());
$('#example').addEventListener('click', showRandomExample);
$('#overview').addEventListener('click', () => {
  view.overview();
  view.setActiveLayer(null);
});
$('#replay').addEventListener('click', () => view.replay());

// Collapsible side panels. The choice is remembered per browser (storage may be unavailable,
// e.g. in private windows, in which case the panels simply start open).
const PANELS_KEY = 'cnn-explorer:collapsed-panels';
function setPanelCollapsed(side, collapsed, save = true) {
  $('#app').classList.toggle(`${side}-collapsed`, collapsed);
  const name = side === 'left' ? 'drawing panel' : 'side panel';
  document.querySelectorAll(`[data-panel-toggle="${side}"]`).forEach((button) => {
    button.setAttribute('aria-expanded', String(!collapsed));
    button.title = `${collapsed ? 'Show' : 'Hide'} the ${name}`;
  });
  if (!save) return;
  try {
    const app = $('#app').classList;
    localStorage.setItem(PANELS_KEY, JSON.stringify({ left: app.contains('left-collapsed'), right: app.contains('right-collapsed') }));
  } catch { /* not persisted */ }
}
document.querySelectorAll('[data-panel-toggle]').forEach((button) => {
  const side = button.dataset.panelToggle;
  button.addEventListener('click', () => setPanelCollapsed(side, !$('#app').classList.contains(`${side}-collapsed`)));
});
try {
  const saved = JSON.parse(localStorage.getItem(PANELS_KEY) ?? '{}');
  if (saved.left) setPanelCollapsed('left', true, false);
  if (saved.right) setPanelCollapsed('right', true, false);
} catch { /* start with both panels open */ }
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    stopSweep();
    select(null);
  }
});

// ------------------------------------------------------------------ core loop

function normalize(arr) {
  let max = 1e-6;
  for (const v of arr) if (v > max) max = v;
  return arr.map((v) => v / max);
}

/** Per-layer 0..1 values for the 3D view (each layer scaled by its own maximum). */
function displayValues(res) {
  // With nothing drawn, the only activity left is the biases; scaled to the layer maximum that
  // would light whole maps up, so show an idle network instead.
  if (state.blank) {
    return Object.fromEntries(['input', 'conv1', 'pool1', 'conv2', 'pool2', 'dense1', 'output']
      .map((id) => [id, new Float32Array(id === 'output' ? res.probs.length : res[id].length)]));
  }
  return {
    input: res.input,
    conv1: normalize(res.conv1),
    pool1: normalize(res.pool1),
    conv2: normalize(res.conv2),
    pool2: normalize(res.pool2),
    dense1: normalize(res.dense1),
    output: res.probs,
  };
}

function run({ stagger } = {}) {
  state.res = forward(state.net, state.image);
  view.setValues(displayValues(state.res), { stagger });
  view.setTopDigit(state.blank ? -1 : state.res.prediction);
  bars.update(state.res, state.blank);
  kernels.update(state.net, state.res);
  if (state.selection) {
    view.setSelection(state.selection, receptiveField(state.selection, state.net, state.res));
    scheduleInspector();
  }
}

let inspectorTimer = null;
function scheduleInspector() {
  if (inspectorTimer) return;
  inspectorTimer = setTimeout(() => {
    inspectorTimer = null;
    if (state.selection) inspector.show(state.selection, state.net, state.res);
  }, 90);
}

function select(ref, { quiet = false } = {}) {
  state.selection = ref;
  if (!ref) {
    view.setSelection(null);
    inspector.clear();
    return;
  }
  view.setSelection(ref, receptiveField(ref, state.net, state.res));
  if (quiet) {
    scheduleInspector();
  } else {
    inspector.show(ref, state.net, state.res);
    showTab('inspect');
  }
}

// ------------------------------------------------------------------ sweep animation

/** Animates a kernel (or pooling window) sliding over its input, revealing the map as it goes. */
function startSweep(ref) {
  if (!ref) return;
  stopSweep();
  const L = LAYER_BY_ID[ref.layer];
  const { map } = unpack(L, ref.index);
  const plane = L.size * L.size;
  const full = displayValues(state.res)[L.id];
  const perFrame = L.size >= 28 ? 3 : 1;
  view.focusLayers([PREVIOUS[L.id], L.id]);
  state.sweep = { L, t: 0 };

  const tick = () => {
    const s = state.sweep;
    if (!s || s.L !== L) return;
    const masked = full.slice();
    masked.fill(0, map * plane + s.t + 1, (map + 1) * plane);
    view.setValues({ [L.id]: masked }, { immediate: true });
    select({ layer: L.id, index: pack(L, map, Math.floor(s.t / L.size), s.t % L.size) }, { quiet: true });
    if (s.t >= plane - 1) {
      stopSweep();
      return;
    }
    s.t = Math.min(plane - 1, s.t + perFrame);
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function stopSweep() {
  if (!state.sweep) return;
  const { L } = state.sweep;
  state.sweep = null;
  view.setValues({ [L.id]: displayValues(state.res)[L.id] }, { immediate: true });
}

// ------------------------------------------------------------------ startup

function updateModelInfo() {
  const params = '52,266 parameters';
  const t = state.trainedInBrowser;
  if (t) {
    const acc = t.testAcc !== undefined ? ` · <b>${(t.testAcc * 100).toFixed(1)}%</b> test accuracy` : '';
    $('#model-info').innerHTML = `Weights trained <b>in your browser</b> · step ${t.step}${acc}`;
    return;
  }
  const m = state.pretrained.meta;
  $('#model-info').innerHTML = `Pretrained on ${m.trainSamples.toLocaleString('en-US')} digits · <b>${(m.testAccuracy * 100).toFixed(2)}%</b> test accuracy · ${params}`;
}

async function showRandomExample() {
  const test = await loadMnist('test');
  const i = Math.floor(Math.random() * test.count);
  pad.showExample(test.image(i), { label: test.labels[i] });
}

async function init() {
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}model/weights.json`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.pretrained = await res.json();
  } catch (err) {
    $('#model-info').textContent = `Could not load model/weights.json (${err.message}). Run "npm run train" first.`;
    return;
  }
  state.net = loadNetwork(state.pretrained);
  updateModelInfo();
  run();

  // Start with a real MNIST digit so the network has something to show.
  const test = await loadMnist('test');
  pad.showExample(test.image(0), { label: test.labels[0] });
  view.replay();
}

init();

// Handy for poking at the model from the browser console while developing.
if (import.meta.env.DEV) window.cnn = { state, view, select, forward };
