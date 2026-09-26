// Loads the MNIST subsets written by scripts/prepare-data.mjs. Each split is a PNG with one
// 28x28 digit per 784-pixel row, plus a file with one label byte per row.
const cache = {};

export function loadMnist(split) {
  cache[split] ??= load(split);
  return cache[split];
}

async function load(split) {
  const base = import.meta.env.BASE_URL;
  const img = new Image();
  img.src = `${base}data/mnist-${split}.png`;
  const [labels] = await Promise.all([
    fetch(`${base}data/mnist-${split}-labels.bin`).then((r) => {
      if (!r.ok) throw new Error(`missing MNIST ${split} labels (run npm run data)`);
      return r.arrayBuffer();
    }),
    img.decode(),
  ]);

  const count = img.height;
  const canvas = document.createElement('canvas');
  canvas.width = 784;
  canvas.height = count;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const rgba = ctx.getImageData(0, 0, 784, count).data;
  const pixels = new Float32Array(count * 784);
  for (let i = 0; i < pixels.length; i++) pixels[i] = rgba[i * 4] / 255;

  return {
    count,
    pixels,
    labels: new Uint8Array(labels),
    image: (i) => pixels.subarray(i * 784, (i + 1) * 784),
  };
}
