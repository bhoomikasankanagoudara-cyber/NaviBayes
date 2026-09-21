# Naive Bayes Classifier implementation in Python

This project provides a clean, modular implementation of the **Naive Bayes Classifier** built from foundational probabilistic principles. Naive Bayes is a family of simple yet powerful supervised learning algorithms based on Bayes' Theorem, operating under the assumption that all input features are conditionally independent given the target class label.

---

## Technical Overview

### Probabilistic Foundation (Bayes' Theorem)
The model calculates the posterior probability $P(y \mid X)$ of a target class label $y$ given an input feature vector $X = (x_1, x_2, \dots, x_n)$ using Bayes' Theorem:

$$P(y \mid X) = \frac{P(X \mid y) \cdot P(y)}{P(X)}$$

Where:
* **$P(y \mid X)$**: Posterior probability of class $y$ given features $X$.
* **$P(y)$**: Prior probability of class $y$.
* **$P(X \mid y)$**: Likelihood of observing feature vector $X$ given class $y$.
* **$P(X)$**: Evidence probability (marginal likelihood).

### Naive Assumption (Feature Independence)
The algorithm assumes that each input feature $x_i$ is conditionally independent of any other feature $x_j$ given the class $y$:

$$P(X \mid y) = \prod_{i=1}^{n} P(x_i \mid y)$$

Substituting the independence assumption into Bayes' Theorem yields:

$$P(y \mid x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)$$

### Classification Rule
The final predicted class $\hat{y}$ is determined via the Maximum A Posteriori (MAP) decision rule:

$$\hat{y} = \arg\max_{y} \left( P(y) \prod_{i=1}^{n} P(x_i \mid y) \right)$$

---

## Features
* **Probability-based Classification**: Predicts target classes directly from posterior probability values.
* **High Efficiency**: Fast training and inference time due to the conditional independence assumption.
* **Scalable**: Handles multi-class classification problems seamlessly.

---

## Getting Started

### Prerequisites
* Python 3.8+
* `numpy`
* `scikit-learn` (for evaluation metrics)

### Installation
```bash
git clone [https://github.com/your-username/naive-bayes-classifier.git](https://github.com/your-username/naive-bayes-classifier.git)
cd naive-bayes-classifier
pip install -r requirements.txt
