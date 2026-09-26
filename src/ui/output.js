// The prediction card: the winning digit and one probability bar per class.
export class OutputBars {
  constructor({ bars, digit, conf }, { onSelect }) {
    this.digit = digit;
    this.conf = conf;
    this.rows = Array.from({ length: 10 }, (_, d) => {
      const row = document.createElement('div');
      row.className = 'bar-row';
      row.title = `Inspect the output neuron for ${d}`;
      row.innerHTML = `<span class="digit">${d}</span><div class="track"><div class="fill"></div></div><span class="pct"></span>`;
      row.addEventListener('click', () => onSelect(d));
      bars.append(row);
      return { row, fill: row.querySelector('.fill'), pct: row.querySelector('.pct') };
    });
  }

  update(res, blank) {
    this.rows.forEach(({ row, fill, pct }, d) => {
      const p = blank ? 0 : res.probs[d];
      fill.style.width = `${p * 100}%`;
      pct.textContent = blank ? '' : `${(p * 100).toFixed(1)}%`;
      row.classList.toggle('top', !blank && d === res.prediction);
    });
    if (blank) {
      this.digit.textContent = '–';
      this.conf.textContent = 'Draw something to start';
      return;
    }
    const p = res.probs[res.prediction];
    this.digit.textContent = res.prediction;
    this.conf.innerHTML = `<b>${(p * 100).toFixed(1)}%</b> confident<br>${p < 0.6 ? 'Not very sure: try drawing it bigger or clearer' : 'the tallest bar in the Output layer'}`;
  }
}
