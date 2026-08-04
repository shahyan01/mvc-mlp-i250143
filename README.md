# Multilayer Perceptron from First Principles: Manual Backpropagation & NumPy MNIST Scaling

This repository contains the complete implementation, theoretical analysis, and mathematical verification of a Multilayer Perceptron (MLP) built from scratch. The project progresses from scalar hand-calculations and low-level C++ verification (2-2-2-1 architecture) to vectorized matrix operations in NumPy scaled to the MNIST handwritten digit dataset (784-128-64-10 architecture)].

---

**Repository Contents**

* `MVC-Report-i250143.pdf`: Comprehensive publication-style technical report detailing full backpropagation derivations, optimizer trajectories, and empirical findings.
* `notebook.ipynb`: Jupyter notebook containing the pure NumPy implementation of the MLP forward pass, backpropagation, and mini-batch training loop on MNIST.
* `mnist.npz`: Local offline dataset containing 60,000 training and 10,000 testing 28x28 grayscale images.
* `README.md`: Repository documentation.

---

**Project Structure & Key Milestones**

**1. Theoretical Mechanics & Math Audit (Tasks 1–3)**
* Hand-derived forward propagation, Sigmoid activations, and MSE loss for a 2-2-2-1 network topology across a 3-sample dataset.
* Constructed full upstream error signal computation graphs using the chain rule.
* **Discovered & Corrected Specification Typo:** Identified a weight swap error in the worksheet's printed backpropagation deltas ($\delta_1^{(1)}$ and $\delta_2^{(1)}$) during graph construction. Confirmed corrected equations ($\sum w_{kj} \delta_k$) with course staff and applied across all tasks.

**2. Low-Level Verification & Optimizer Benchmarking (Tasks 4–6)**
* Implemented scalar-level C++ engines enforcing deterministic 4-decimal-place rounding to mirror manual calculations.
* Analyzed optimization dynamics across:
  * Stochastic Gradient Descent (SGD)
  * Mini-Batch Gradient Descent (MBGD, batch size $B=2$)
  * SGD with Momentum ($\beta=0.9$)
  * Nesterov Accelerated Gradient (NAG)
* **Empirical Finding:** Observed that plain GD outperformed momentum-based methods over 10 iterations on a smooth, 3-sample loss surface due to early velocity scaling lag.

**3. Vectorized Scaling to MNIST (Task 7)**
* Translated scalar backpropagation equations directly into matrix-based vectorized NumPy functions without using PyTorch, TensorFlow, or Scikit-Learn.
* Scaled from 12 parameters to **108,938 parameters**.

---

**Task 7 Hyperparameters & Benchmark Results**

| Parameter | Configuration |
| :--- | :--- |
| **Architecture** | 784 (Input) $\rightarrow$ 128 (Hidden 1) $\rightarrow$ 64 (Hidden 2) $\rightarrow$ 10 (Output) |
| **Activation** | Sigmoid (all hidden & output layers) |
| **Loss Function** | Mean Squared Error (MSE) |
| **Batch Size ($B$)** | 32 |
| **Learning Rate ($\eta$)** | 0.1 |
| **Epochs** | 20 |
| **Weight Init** | Uniform $[ -0.5, 0.5 ]$ | Seed: 42 |

**Performance Metrics:**
* **Training Loss:** Decreased monotonically from **0.0296** (Epoch 1) to **0.0069** (Epoch 20).
* **Test Accuracy:** Achieved **95.54% classification accuracy** on 10,000 unseen test images.

---

**Execution & Environment**

To run the notebook locally without external deep learning frameworks:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/shahyan01/mvc-mlp-i250143.git](https://github.com/shahyan01/mvc-mlp-i250143.git)
   cd mvc-mlp-i250143
