"""Trains the network on MNIST and writes public/model/weights.json, the pretrained
weights the web page loads on startup.

    python scripts/train.py            # 8 epochs, ~1 min on a laptop CPU
    python scripts/train.py --epochs 2

The architecture must match src/model/arch.js. Weights are exported in the same layout
TF.js uses (conv kernels [kh, kw, in, out], dense kernels [in, out], channels-last flatten),
so the browser can also load them into its TF.js model and keep training.
"""
import argparse
import datetime
import json
import os

import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'public', 'model', 'weights.json')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=8)
parser.add_argument('--seed', type=int, default=7)
args = parser.parse_args()
tf.keras.utils.set_random_seed(args.seed)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = (x_train[..., None] / 255.0).astype('float32')
x_test = (x_test[..., None] / 255.0).astype('float32')

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(8, 5, padding='same', activation='relu', name='conv1'),
    tf.keras.layers.MaxPooling2D(2, name='pool1'),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', name='conv2'),
    tf.keras.layers.MaxPooling2D(2, name='pool2'),
    tf.keras.layers.Flatten(name='flatten'),
    tf.keras.layers.Dense(64, activation='relu', name='dense1'),
    tf.keras.layers.Dense(10, activation='softmax', name='dense2'),
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Mild augmentation (outside the model, so the exported architecture stays identical).
# Hand-drawn digits in the browser are never as tidy as MNIST; small shifts, rotations and
# zooms make the model noticeably more forgiving.
augment = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.03, fill_mode='constant'),
    tf.keras.layers.RandomZoom(0.1, fill_mode='constant'),
    tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode='constant'),
])
train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(60000, seed=args.seed)
            .batch(64)
            .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))

model.fit(train_ds, epochs=args.epochs, validation_data=(x_test, y_test), verbose=2)
_, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'test accuracy: {test_acc:.4f}')


def rounded(a):
    return [round(float(v), 5) for v in np.asarray(a).ravel()]


layers = {}
for name in ['conv1', 'conv2', 'dense1', 'dense2']:
    kernel, bias = model.get_layer(name).get_weights()
    layers[name] = {'shape': list(kernel.shape), 'kernel': rounded(kernel), 'bias': rounded(bias)}

out = {
    'format': 'cnn-explorer/1',
    'meta': {
        'epochs': args.epochs,
        'trainSamples': int(len(x_train)),
        'testAccuracy': round(float(test_acc), 4),
        'trainedAt': datetime.date.today().isoformat(),
        'trainedWith': f'Keras {tf.__version__}',
    },
    'layers': layers,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'wrote {os.path.relpath(OUT, ROOT)} ({os.path.getsize(OUT) // 1024} KB)')
