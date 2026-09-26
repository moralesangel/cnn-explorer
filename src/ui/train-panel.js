// The "Train" tab: controls, live statistics and a loss/accuracy chart for in-browser training.

export class TrainPanel {
  constructor(root, { onWeights, onRestore, onShowKernels }) {
    this.root = root;
    this.onWeights = onWeights;
    root.innerHTML = `
      <h3>Train it yourself</h3>
      <p class="lede">The pretrained weights were learned from 60,000 digits. Here you can throw them away and
      train the same network from <b>random numbers</b>, right in your browser, on 12,000 MNIST digits.</p>
      <div class="train-controls">
        <label>Epochs <select id="t-epochs"><option>1</option><option selected>2</option><option>3</option><option>5</option></select></label>
        <label>Learning rate <select id="t-lr"><option>0.0005</option><option>0.001</option><option selected>0.002</option><option>0.005</option></select></label>
      </div>
      <div class="train-controls">
        <button class="primary" id="t-start">Train from scratch</button>
        <button id="t-stop" disabled>Stop</button>
        <button id="t-restore">Restore pretrained</button>
      </div>
      <p class="muted" id="t-status">Tip: keep the <a href="#" id="t-kernels" style="color:var(--accent)">Kernels</a>
      tab or the 3D view in sight while it trains.</p>
      <div class="stats">
        <div class="stat"><div class="k">Epoch</div><div class="v" id="t-epoch">–</div></div>
        <div class="stat"><div class="k">Step</div><div class="v" id="t-step">–</div></div>
        <div class="stat"><div class="k">Loss</div><div class="v" id="t-loss">–</div></div>
        <div class="stat"><div class="k">Test acc</div><div class="v" id="t-test">–</div></div>
      </div>
      <canvas id="train-chart" width="720" height="300"></canvas>
      <div class="chart-legend">
        <span><i style="background:var(--accent)"></i>loss (lower is better)</span>
        <span><i style="background:var(--accent-2)"></i>batch accuracy</span>
        <span><i style="background:#6ee7b7"></i>test accuracy</span>
      </div>
      <div class="progress"><div id="t-progress"></div></div>
      <h4>What happens at every step</h4>
      <ol style="padding-left:18px;margin:0;color:#c3cbe0">
        <li>Take a batch of 32 labelled digits.</li>
        <li><b>Forward pass</b>: run them through the network to get 10 probabilities each.</li>
        <li><b>Loss</b>: cross-entropy, −log(probability given to the correct digit). 2.3 means random guessing
          (10% each); near 0 means confident and right.</li>
        <li><b>Backpropagation</b>: the chain rule, applied backwards through every layer, gives how much each of
          the 52,266 weights pushed the loss up or down.</li>
        <li><b>Update</b>: nudge every weight a tiny bit against its gradient (Adam optimizer).</li>
      </ol>
      <p class="muted" style="margin-top:10px">One <b>epoch</b> is one pass over all 12,000 training digits (375
      steps). Test accuracy is measured on 2,000 digits the network never trains on.</p>`;

    this.$ = (id) => root.querySelector(`#${id}`);
    this.chart = this.$('train-chart');
    this.running = false;
    this.stopRequested = false;
    this.history = [];
    this.tests = [];

    this.$('t-start').addEventListener('click', () => this.start());
    this.$('t-stop').addEventListener('click', () => { this.stopRequested = true; });
    this.$('t-restore').addEventListener('click', () => {
      this.stopRequested = true;
      onRestore();
      this.status('Pretrained weights restored.');
    });
    this.$('t-kernels').addEventListener('click', (e) => {
      e.preventDefault();
      onShowKernels();
    });
    this.drawChart();
  }

  status(msg) {
    this.$('t-status').textContent = msg;
  }

  async start() {
    if (this.running) return;
    this.running = true;
    this.stopRequested = false;
    this.history = [];
    this.tests = [];
    this.$('t-start').disabled = true;
    this.$('t-stop').disabled = false;
    const epochs = Number(this.$('t-epochs').value);
    const learningRate = Number(this.$('t-lr').value);
    let ema = null;

    try {
      const { trainFromScratch } = await import('../train/trainer.js');
      await trainFromScratch({
        epochs,
        learningRate,
        onStatus: (m) => this.status(m),
        shouldStop: () => this.stopRequested,
        onWeights: (json, info) => this.onWeights(json, info),
        onBatch: ({ epoch, batch, batches, step, loss, acc }) => {
          ema = ema ? { loss: ema.loss * 0.9 + loss * 0.1, acc: ema.acc * 0.9 + acc * 0.1 } : { loss, acc };
          this.history.push({ step, ...ema });
          this.totalSteps = batches * epochs;
          this.$('t-epoch').textContent = `${epoch + 1}/${epochs}`;
          this.$('t-step').textContent = step;
          this.$('t-loss').textContent = ema.loss.toFixed(3);
          this.$('t-progress').style.width = `${(step / this.totalSteps) * 100}%`;
          if (step % 3 === 0) this.drawChart();
        },
        onEpoch: ({ testAcc, step }) => {
          this.tests.push({ step, acc: testAcc });
          this.$('t-test').textContent = `${(testAcc * 100).toFixed(1)}%`;
          this.drawChart();
        },
      });
    } catch (err) {
      console.error(err);
      this.status(`Training failed: ${err.message}`);
    } finally {
      this.running = false;
      this.$('t-start').disabled = false;
      this.$('t-stop').disabled = true;
      this.drawChart();
    }
  }

  drawChart() {
    const c = this.chart;
    const ctx = c.getContext('2d');
    const W = c.width, H = c.height, P = 28;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = '#243049';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#8d99b3';
    ctx.font = '20px system-ui, sans-serif';
    for (let i = 0; i <= 4; i++) {
      const y = P + ((H - 2 * P) * i) / 4;
      ctx.beginPath();
      ctx.moveTo(P, y);
      ctx.lineTo(W - 8, y);
      ctx.stroke();
    }
    ctx.fillText('100%', W - 64, P - 6);
    if (!this.history.length) {
      ctx.fillText('The chart fills in while training runs', P + 8, H / 2 + 6);
      return;
    }
    const total = Math.max(this.totalSteps || 1, this.history.at(-1).step);
    const X = (s) => P + ((W - P - 8) * s) / total;
    const Y = (v) => H - P - (H - 2 * P) * v;
    const line = (key, scale, color) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      this.history.forEach((h, i) => {
        const y = Y(Math.min(1, h[key] / scale));
        if (i) ctx.lineTo(X(h.step), y); else ctx.moveTo(X(h.step), y);
      });
      ctx.stroke();
    };
    line('loss', 2.5, '#f28a4c');
    line('acc', 1, '#fde9a0');
    ctx.fillStyle = '#6ee7b7';
    for (const t of this.tests) {
      ctx.beginPath();
      ctx.arc(X(t.step), Y(t.acc), 7, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
