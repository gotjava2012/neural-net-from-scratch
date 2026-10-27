"""
Neural Network From Scratch (NumPy) — a teaching implementation.

This trains a small feed-forward neural network using ONLY NumPy, implementing
every piece of the training loop by hand so the mechanics are visible:

    - FORWARD PASS   : inputs -> hidden layer -> output layer -> predictions
    - LOSS           : how wrong the predictions are (cross-entropy)
    - BACKPROP       : compute gradients of the loss w.r.t. every weight
    - GRADIENT DESCENT: nudge weights in the direction that reduces loss
    - repeat for many epochs until the network learns

No PyTorch, no TensorFlow — those frameworks automate exactly what you see here.
Understanding this makes every ML framework afterward make sense.

Task: classify points into 3 spiral-shaped classes (a classic non-linear
problem a single linear model CANNOT solve, which proves the hidden layer and
non-linear activation are doing real work).
"""

import numpy as np


# ----------------------------------------------------------------------------
# Dataset — generated in code (no download). A 2D spiral with 3 classes.
# ----------------------------------------------------------------------------
def make_spiral(points_per_class=100, num_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((points_per_class * num_classes, 2))
    y = np.zeros(points_per_class * num_classes, dtype=int)
    for c in range(num_classes):
        ix = range(points_per_class * c, points_per_class * (c + 1))
        r = np.linspace(0.0, 1.0, points_per_class)                 # radius
        t = np.linspace(c * 4, (c + 1) * 4, points_per_class) + rng.normal(0, 0.2, points_per_class)  # angle
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = c
    return X, y


# ----------------------------------------------------------------------------
# Activation + loss helpers
# ----------------------------------------------------------------------------
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    # subtract max for numerical stability
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy(probs, y):
    n = y.shape[0]
    # -log(probability the model assigned to the correct class), averaged
    return -np.mean(np.log(probs[range(n), y] + 1e-12))


# ----------------------------------------------------------------------------
# The network: 2 inputs -> hidden layer (ReLU) -> 3-class output (softmax)
# ----------------------------------------------------------------------------
class NeuralNet:
    def __init__(self, in_dim=2, hidden=64, out_dim=3, seed=0):
        rng = np.random.default_rng(seed)
        # small random weights; biases start at zero
        self.W1 = 0.1 * rng.standard_normal((in_dim, hidden))
        self.b1 = np.zeros((1, hidden))
        self.W2 = 0.1 * rng.standard_normal((hidden, out_dim))
        self.b2 = np.zeros((1, out_dim))

    def forward(self, X):
        # Layer 1: linear then ReLU non-linearity
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        # Layer 2: linear then softmax -> class probabilities
        self.z2 = self.a1 @ self.W2 + self.b2
        self.probs = softmax(self.z2)
        return self.probs

    def backward(self, X, y, lr):
        n = X.shape[0]
        # Gradient of cross-entropy + softmax w.r.t. z2 is simply (probs - onehot)
        dz2 = self.probs.copy()
        dz2[range(n), y] -= 1
        dz2 /= n

        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Backprop into the hidden layer
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)   # derivative of ReLU
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Gradient descent: step each parameter against its gradient
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


def accuracy(model, X, y):
    return float(np.mean(model.predict(X) == y))


def train(epochs=1000, lr=1.0, hidden=64, seed=0, verbose=True):
    X, y = make_spiral(seed=seed)
    model = NeuralNet(hidden=hidden, seed=seed)
    for epoch in range(epochs + 1):
        probs = model.forward(X)
        loss = cross_entropy(probs, y)
        model.backward(X, y, lr)
        if verbose and epoch % 100 == 0:
            print(f"  epoch {epoch:4d} | loss {loss:.4f} | accuracy {accuracy(model, X, y)*100:5.1f}%")
    return model, X, y


if __name__ == "__main__":
    print("Training a neural network from scratch on the 3-class spiral...\n")
    model, X, y = train()
    final = accuracy(model, X, y)
    print(f"\nFinal training accuracy: {final*100:.1f}%")
    print("A linear model maxes out near ~50% on this task; the hidden layer + ReLU")
    print("is what lets the network learn the non-linear spiral boundaries.")
