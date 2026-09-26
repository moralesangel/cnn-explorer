// The network, in one place. Both the TF.js model (used for training) and the hand-written
// forward pass (used for visualization) are built from this description.
//
//   input 28x28x1
//   conv1  8 filters 5x5, same padding, ReLU  -> 28x28x8
//   pool1  2x2 max                            -> 14x14x8
//   conv2 16 filters 3x3, same padding, ReLU  -> 14x14x16
//   pool2  2x2 max                            ->  7x7x16  (= 784 numbers)
//   dense1 64 units, ReLU
//   dense2 10 units, softmax                  -> one probability per digit
//
// It is deliberately tiny (~52k parameters) so every neuron can be drawn, and it still
// reaches ~99% test accuracy.
export const ARCH = {
  inputSize: 28,
  conv1: { filters: 8, kernel: 5 },
  conv2: { filters: 16, kernel: 3 },
  dense1: { units: 64 },
  classes: 10,
};
