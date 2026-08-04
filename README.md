# Multilayer Perceptron from First Principles: Manual Backpropagation & NumPy MNIST Scaling

This repository contains the complete implementation, theoretical analysis, and mathematical verification of a Multilayer Perceptron (MLP) built from scratch[cite: 6]. The project progresses from scalar hand-calculations and low-level C++ verification (2-2-2-1 architecture) to vectorized matrix operations in NumPy scaled to the MNIST handwritten digit dataset (784-128-64-10 architecture)[cite: 6].

---

**Repository Contents**

* `MVC-Report-i250143.pdf`: Comprehensive publication-style technical report detailing full backpropagation derivations, optimizer trajectories, and empirical findings[cite: 6].
* `notebook.ipynb`: Jupyter notebook containing the pure NumPy implementation of the MLP forward pass, backpropagation, and mini-batch training loop on MNIST[cite: 6].
* `mnist.npz`: Local offline dataset containing 60,000 training and 10,000 testing 28x28 grayscale images[cite: 6].
* `README.md`: Repository documentation.

---

**Project Structure & Key Milestones**

**1. Theoretical Mechanics & Math Audit (Tasks 1–3)**
* Hand-derived forward propagation, Sigmoid activations, and MSE loss for a 2-2-2-1 network topology across a 3-sample dataset[cite: 6].
* Constructed full upstream error signal computation graphs using the chain rule[cite: 6].
* **Discovered & Corrected Specification Typo:** Identified a weight swap error in the worksheet's printed backpropagation deltas ($\delta_1^{(1)}$ and $\delta_2^{(1)}$) during graph construction[cite: 6]. Confirmed corrected equations ($\sum w_{kj} \delta_k$) with course staff and applied across all tasks[cite: 6].

**2. Low-Level Verification & Optimizer Benchmarking (Tasks 4–6)**
* Implemented scalar-level C++ engines enforcing deterministic 4-decimal-place rounding to mirror manual calculations[cite: 6].
* Analyzed optimization dynamics across:
  * Stochastic Gradient Descent (SGD)[cite: 6]
  * Mini-Batch Gradient Descent (MBGD, batch size $B=2$)[cite: 6]
  * SGD with Momentum ($\beta=0.9$)[cite: 6]
  * Nesterov Accelerated Gradient (NAG)[cite: 6]
* **Empirical Finding:** Observed that plain GD outperformed momentum-based methods over 10 iterations on a smooth, 3-sample loss surface due to early velocity scaling lag[cite: 6].

**3. Vectorized Scaling to MNIST (Task 7)**
* Translated scalar backpropagation equations directly into matrix-based vectorized NumPy functions without using PyTorch, TensorFlow, or Scikit-Learn[cite: 6].
* Scaled from 12 parameters to **108,938 parameters**[cite: 6].

---

**Task 7 Hyperparameters & Benchmark Results**

| Parameter | Configuration[cite: 6] |
| :--- | :--- |
| **Architecture** | 784 (Input) $\rightarrow$ 128 (Hidden 1) $\rightarrow$ 64 (Hidden 2) $\rightarrow$ 10 (Output)[cite: 6] |
| **Activation** | Sigmoid (all hidden & output layers)[cite: 6] |
| **Loss Function** | Mean Squared Error (MSE)[cite: 6] |
| **Batch Size ($B$)** | 32[cite: 6] |
| **Learning Rate ($\eta$)** | 0.1[cite: 6] |
| **Epochs** | 20[cite: 6] |
| **Weight Init** | Uniform $[ -0.5, 0.5 ]$ | Seed: 42[cite: 6] |

**Performance Metrics:**
* **Training Loss:** Decreased monotonically from **0.0296** (Epoch 1) to **0.0069** (Epoch 20)[cite: 6].
* **Test Accuracy:** Achieved **95.54% classification accuracy** on 10,000 unseen test images[cite: 6].

---

**Execution & Environment**

To run the notebook locally without external deep learning frameworks:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/shahyan01/mvc-mlp-i250143.git](https://github.com/shahyan01/mvc-mlp-i250143.git)
   cd mvc-mlp-i250143
