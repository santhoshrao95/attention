# Self-Attention Implementation

A clean, educational implementation of self-attention and masked self-attention mechanisms from scratch using PyTorch. This repository provides a clear understanding of how attention works in transformer models with detailed debug output.

## Overview

This repository contains two main implementations:

1. **Self-Attention** (`selfAttention`) - Basic self-attention mechanism with comprehensive debugging
2. **Masked Self-Attention** (`MaskedSelfAttention`) - Self-attention with causal masking for autoregressive tasks

Both implementations are designed for educational purposes and `selfAttention` include step-by-step mathematical validation.

## Features

- **Pure PyTorch implementation** - No external dependencies beyond torch
- **Educational focus** - Clear, readable code with detailed comments
- **Debug output** - Comprehensive logging of all intermediate steps
- **Mathematical validation** - Manual calculations to verify each step
- **Causal masking** - Support for autoregressive attention patterns

## Files

- `self_attention.py` - Basic self-attention with detailed debug prints
- `masked_self_attention.py` - Masked self-attention implementation
- `working.ipynb` - Jupyter notebook demonstrating both implementations
- `requirements.txt` - Required libraries

## Self-Attention Mechanism

The self-attention implementation follows the standard scaled dot-product attention:

```
Attention(Q,K,V) = softmax(QK^T / √d_k)V
```

### Key Components:
- **Query (Q)**, **Key (K)**, **Value (V)** linear projections
- **Scaled dot-product attention** with √d_k normalization
- **Softmax normalization** for attention weights
- **Debug output** showing all intermediate calculations

## Masked Self-Attention

The masked version adds causal masking to prevent attention to future tokens:

- **Upper triangular mask** to hide future positions
- **Configurable masking** via boolean parameter
- **Same core attention mechanism** as basic version

### Interactive Demo
Open `working_nb.ipynb` in Jupyter to see both implementations in action.

## Input Format

- **Example**: 3 tokens with 2-dimensional embeddings
```python
token_encodings = torch.tensor([[1.16, 0.26],
                               [0.57, 1.36],
                               [4.41, -2.16]])
```

## Mathematical Details

### Attention Computation Steps:
1. **Linear transformations**: Q = XW_q, K = XW_k, V = XW_v
2. **Similarity scores**: QK^T
3. **Scaling**: QK^T / √d_k
4. **Masking** (if enabled): Set future positions to -∞
5. **Normalization**: softmax along sequence dimension
6. **Weighted sum**: Attention weights × V

### Debug Output Includes:
- Weight matrices (W_q, W_k, W_v)
- Query, Key, Value tensors
- Unscaled and scaled attention scores
- Attention probabilities (post-softmax)
- Manual validation of each computation step

## Educational Value

This implementation is designed to help understand:
- How self-attention works mathematically
- The role of Q, K, V transformations
- Why scaling by √d_k is important
- How masking enables causal attention
- The complete attention computation pipeline
