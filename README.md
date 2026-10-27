# Neural Network From Scratch (NumPy)

A feed-forward neural network trained with **only NumPy** — forward pass, loss,
**backpropagation**, and gradient descent all implemented by hand. The goal is to
understand exactly what PyTorch and TensorFlow automate.

## What it demonstrates
- Forward pass through a hidden layer with a ReLU non-linearity
- Cross-entropy loss and softmax
- Backpropagation (the chain rule) computed manually
- Gradient descent training loop
- Why a non-linearity is required: the net learns a 3-class **spiral** (a problem
  a linear model can't solve), reaching ~99% accuracy

## Quick start
```bash
pip3 install numpy
python3 src/neural_net.py      # watch loss fall and accuracy rise each 100 epochs
```

## Learn it step by step
See **[docs/TUTORIAL.md](docs/TUTORIAL.md)** — maps each ML concept to the code,
with exercises (remove the non-linearity, change the learning rate, etc.).

## Project layout
```
src/neural_net.py   # the network + training loop (heavily commented)
docs/TUTORIAL.md    # step-by-step tutorial
```

## Stack
Python · NumPy
