# Tutorial: Neural Network From Scratch (Step by Step)

This project builds and trains a neural network using **only NumPy** — every
part of the training loop is implemented by hand. The goal is to *understand*
what PyTorch and TensorFlow automate. By the end you'll understand the forward
pass, loss, backpropagation, and gradient descent well enough to explain them.

---

## The problem

We classify 2D points into **3 spiral-shaped classes**. This is deliberately
non-linear: a straight-line (linear) model tops out around ~50% accuracy on it.
Only a network with a **hidden layer and a non-linear activation** can bend the
decision boundaries to follow the spirals — so when our network hits ~99%, we've
*proven* the hidden layer is doing real work.

---

## Prerequisites

```bash
pip3 install numpy
```
That's it — one dependency.

## Run it

```bash
python3 src/neural_net.py
```
You'll watch loss fall and accuracy rise every 100 epochs.

---

## The five concepts (map each to the code)

### 1. Forward pass — making a prediction
Input → hidden layer → output probabilities. See `NeuralNet.forward()`:
- `z1 = X @ W1 + b1` — a linear combination (weights × inputs + bias)
- `a1 = relu(z1)` — the **non-linearity**; without it, stacking layers would
  collapse into a single linear model (this is the key insight)
- `z2 = a1 @ W2 + b2` then `softmax` — turns raw scores into probabilities that
  sum to 1 across the 3 classes

### 2. Loss — measuring how wrong we are
`cross_entropy(probs, y)` returns the average negative log-probability the model
assigned to the *correct* class. Lower loss = better predictions. This single
number is what training tries to minimize.

### 3. Backpropagation — computing gradients
`NeuralNet.backward()` computes how much each weight contributed to the loss (the
gradient), working backward from the output to the input using the chain rule:
- `dz2 = probs - onehot(y)` — a famously clean result of combining softmax +
  cross-entropy
- propagate back through `W2`, through the ReLU (`dz1 = da1 * (z1 > 0)`), to `W1`

### 4. Gradient descent — learning
Each weight is nudged *against* its gradient: `W -= lr * dW`. The learning rate
`lr` controls step size. Do this repeatedly and the loss goes down — that *is*
learning.

### 5. Epochs — repetition
One **epoch** = one pass over all the data. We repeat ~1000 times; each pass the
weights get a little better, which you see as falling loss and rising accuracy.

---

## Exercises

1. **Remove the non-linearity**: change `relu(x)` to just `return x`. Re-run —
   accuracy collapses toward ~50%. This proves why activations matter.
2. **Change the learning rate** (`lr`): try 0.01 (too slow) and 5.0 (may
   diverge). Observe the tradeoff.
3. **Change hidden size** (`hidden`): try 4 vs 128. More neurons = more capacity.
4. **Change epochs**: watch how accuracy plateaus.

---

## How this scales to real ML (next steps)

- **Frameworks** (PyTorch/TensorFlow) compute `backward()` *automatically* via
  autograd — you define only the forward pass. (That's the next project.)
- **Optimizers** like Adam improve on plain gradient descent.
- **Batching**: real training processes mini-batches, not the whole dataset at
  once.
- **Validation/Test split**: here we only measure training accuracy; real
  projects hold out data to detect overfitting.

Now that you've done it by hand, every one of those is just a convenience on top
of what you built here.
