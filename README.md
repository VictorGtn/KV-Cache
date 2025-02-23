# Transformer Decoder with KV Cache Implementation

A PyTorch implementation of a Transformer decoder model featuring key-value caching for efficient text generation.

## Overview

This implementation includes:
- Transformer decoder architecture with configurable parameters
- Key-Value (KV) cache for optimized token generation
- Comparison utilities for cached vs. non-cached generation
- Detailed performance monitoring and metrics

## Components

### Main Classes

- `MultiHeadAttention`: Implements multi-head attention with KV caching
- `TransformerDecoderLayer`: Single transformer decoder layer
- `TransformerDecoder`: Complete decoder model with embedding layers
- `Config`: Configuration class for model parameters

### Key Features

- Configurable architecture parameters
- Position and token embeddings
- KV cache with automatic management
- Generation utilities with performance tracking

## Configuration

Default model parameters:
```python
vocab_size = 50257  # GPT-2 vocabulary size
max_seq_len = 1024
embed_dim = 768
num_heads = 1
num_layers = 1
ffn_dim = 3072
max_cache_len = 2048
```

## Usage

### Basic Model Creation

```python
config = Config()
model = TransformerDecoder(config)
```

### Text Generation

```python
# Generate tokens
input_ids = torch.tensor([[1]])  # Your input tokens
max_new_tokens = 50
output_ids = model.generate(input_ids, max_new_tokens)
```

### Performance Testing

```python
# Run comparison between cached and non-cached generation
run_simulation()
```

## Performance Metrics

The implementation includes detailed performance tracking:
- Generation time per token
- Memory usage
- Output verification
- Comparison between cached and non-cached approaches

## Cache Management

The KV cache implementation includes:
- Automatic cache size management
- Cache rotation for handling long sequences
- Manual cache reset functionality

## Requirements

- PyTorch
- NumPy
- CUDA (optional, for GPU support)

## Features

- Deterministic execution with seed setting
- Memory usage tracking
- Generation time profiling
- Output verification between cached and non-cached versions

## Notes

- The implementation focuses on demonstrating KV caching mechanisms
- The model architecture is simplified for educational purposes
- Performance metrics are included for comparison and verification

## Example Output

The simulation provides detailed metrics including:
- Initial forward pass times
- Token generation times
- Memory usage
- Output verification between cached and non-cached versions
