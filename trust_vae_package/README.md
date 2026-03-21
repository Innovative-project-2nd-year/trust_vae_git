# TRUST-VAE

**Task-Aware, Uncertainty-Guided and Interpretable Variational Autoencoder for Trustworthy Synthetic Data Generation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Overview

TRUST-VAE is a research project that transforms the latent space of a Variational Autoencoder (VAE) into a **structured, interpretable, and uncertainty-aware decision interface** for trustworthy synthetic data generation.

## 🎯 Key Features

- ✅ **Structured Latent Space** - Hierarchical 128-dimensional representation
- ✅ **Latent Interpretability** - Meaningful latent traversal and control
- ✅ **Uncertainty Estimation** - Built-in confidence metrics for generated samples
- ✅ **Task-Aware Training** - Joint optimization with downstream classifier
- ✅ **Human-in-the-Loop** - Interactive Gradio interface for latent manipulation

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Reconstruction SSIM** | 0.9547 |
| **PSNR** | 28.72 dB |
| **Classification Accuracy** | 100% |
| **Latent Dimensions** | 128 |
| **Training Epochs** | 60 |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/trust-vae.git
cd trust-vae

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .