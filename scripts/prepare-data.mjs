// Downloads MNIST (once) and produces:
//   .cache/mnist/*.gz                  - the original IDX files, used by scripts/train.mjs
//   public/data/mnist-{train,test}.png - sprites for the browser, one 28x28 digit per 784-px row
//   public/data/mnist-{train,test}-labels.bin - one uint8 label per row
//
// The browser only gets a subset (it trains from scratch in a minute or two and the files
// stay small). The Node training script uses the full 60k/10k split.
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const CACHE = path.join(ROOT, '.cache', 'mnist');
const OUT = path.join(ROOT, 'public', 'data');

const BROWSER_TRAIN = 12000;
const BROWSER_TEST = 2000;

const MIRRORS = [
  'https://storage.googleapis.com/cvdf-datasets/mnist/',
  'https://ossci-datasets.s3.amazonaws.com/mnist/',
];
const FILES = [
  'train-images-idx3-ubyte.gz',
  'train-labels-idx1-ubyte.gz',
  't10k-images-idx3-ubyte.gz',
  't10k-labels-idx1-ubyte.gz',
];

async function download(file) {
  const dest = path.join(CACHE, file);
  if (fs.existsSync(dest)) return dest;
  for (const mirror of MIRRORS) {
    try {
      process.stdout.write(`downloading ${mirror}${file} ... `);
      const res = await fetch(mirror + file);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      fs.writeFileSync(dest, Buffer.from(await res.arrayBuffer()));
      console.log('ok');
      return dest;
    } catch (err) {
      console.log(`failed (${err.message})`);
    }
  }
  throw new Error(`could not download ${file}`);
}

/** Parses a gzipped IDX file (the MNIST format). Returns { dims, data: Uint8Array }. */
export function readIdx(file) {
  const buf = zlib.gunzipSync(fs.readFileSync(file));
  const nDims = buf[3];
  const dims = [];
  for (let i = 0; i < nDims; i++) dims.push(buf.readUInt32BE(4 + 4 * i));
  const offset = 4 + 4 * nDims;
  return { dims, data: new Uint8Array(buf.buffer, buf.byteOffset + offset, buf.length - offset) };
}

// --- minimal 8-bit grayscale PNG encoder -------------------------------------------------
const CRC_TABLE = new Uint32Array(256).map((_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
function crc32(bytes) {
  let c = 0xffffffff;
  for (const b of bytes) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([len, body, crc]);
}
function encodeGrayPng(pixels, width, height) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 0; // colour type: grayscale
  const raw = Buffer.alloc((width + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (width + 1)] = 0; // filter: none
    raw.set(pixels.subarray(y * width, (y + 1) * width), y * (width + 1) + 1);
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', zlib.deflateSync(raw, { level: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

function writeSubset(name, images, labels, count) {
  const pixels = images.data.subarray(0, count * 784);
  fs.writeFileSync(path.join(OUT, `mnist-${name}.png`), encodeGrayPng(pixels, 784, count));
  fs.writeFileSync(path.join(OUT, `mnist-${name}-labels.bin`), labels.data.subarray(0, count));
  console.log(`wrote public/data/mnist-${name}.png (${count} digits)`);
}

export async function ensureMnist() {
  fs.mkdirSync(CACHE, { recursive: true });
  const [trainImages, trainLabels, testImages, testLabels] = await Promise.all(FILES.map(download));
  return {
    train: { images: readIdx(trainImages), labels: readIdx(trainLabels) },
    test: { images: readIdx(testImages), labels: readIdx(testLabels) },
  };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const mnist = await ensureMnist();
  fs.mkdirSync(OUT, { recursive: true });
  writeSubset('train', mnist.train.images, mnist.train.labels, BROWSER_TRAIN);
  writeSubset('test', mnist.test.images, mnist.test.labels, BROWSER_TEST);
}
