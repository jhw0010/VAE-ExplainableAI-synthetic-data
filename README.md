# Interpretation for Variational Autoencoder Used to Generate Financial Synthetic Tabular Data

[![Paper](https://img.shields.io/badge/Paper-RBC%20Borealis%20AI-blue)](https://rbcborealis.com/publications/interpretation-for-variational-autoencoder-used-to-generate-financial-synthetic-tabular-data/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> Official TensorFlow 2.x implementation and gradient interpretability framework for Tabular Variational Autoencoders (TVAE). Featured by **RBC Borealis AI**.

---

## 📌 Abstract

Generative models such as Variational Autoencoders (VAEs) are widely adopted for generating high-fidelity synthetic tabular data. However, understanding feature importance and inter-feature interactions within non-linear latent spaces remains a significant challenge. This repository offers an end-to-end interpretability framework for TVAE models, featuring:

1. **Local & Global Feature Importance**: First-order Jacobian partial derivative aggregations.
2. **Feature Interaction Mapping**: Second-order partial derivative computations ($\frac{\partial^2 \mu}{\partial x_i \partial x_j}$) capturing latent variable interactions.

---

## 🚀 Quickstart

### Installation
```bash
git clone [https://github.com/jhw0010/jinhongwu.github.io.git](https://github.com/jhw0010/jinhongwu.github.io.git)
cd jinhongwu.github.io
pip install -r requirements.txt
