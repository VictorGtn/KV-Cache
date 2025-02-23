import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import time
from torch.profiler import profile, record_function, ProfilerActivity
import random
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_size = config.embed_dim // config.num_heads
        self.embed_dim = config.embed_dim
        
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
        
        self.max_cache_len = config.max_cache_len
        self.cache_size = 0
        self.k_cache = torch.zeros(1, self.num_heads, self.max_cache_len, self.head_size)
        self.v_cache = torch.zeros(1, self.num_heads, self.max_cache_len, self.head_size)
        
    def forward(self, x, mask=None, use_cache=False):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        
        print(f"\nInput shape: {x.shape}")
        print(f"Q shape before cache: {q.shape}")
        print(f"K shape before cache: {k.shape}")
        
        if use_cache:
            if self.cache_size + seq_len > self.max_cache_len:
                shift_amount = seq_len
                self.k_cache = torch.roll(self.k_cache, shifts=-shift_amount, dims=2)
                self.v_cache = torch.roll(self.v_cache, shifts=-shift_amount, dims=2)
                
                self.k_cache[:, :, -seq_len:] = k
                self.v_cache[:, :, -seq_len:] = v
                self.cache_size = self.max_cache_len
            else:
                self.k_cache[:, :, self.cache_size:self.cache_size + seq_len] = k
                self.v_cache[:, :, self.cache_size:self.cache_size + seq_len] = v
                self.cache_size += seq_len
            
            k = self.k_cache[:, :, :self.cache_size]
            v = self.v_cache[:, :, :self.cache_size]
            print(f"Cache size: {self.cache_size}")
            print(f"K shape after cache: {k.shape}")
        
        scale = math.sqrt(self.head_size)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        print(f"Attention scores shape: {scores.shape}")
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output

    def reset_cache(self):
        self.cache_size = 0
        self.k_cache.zero_()
        self.v_cache.zero_()

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.ffn_dim),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.embed_dim)
        )
        self.norm2 = nn.LayerNorm(config.embed_dim)
        
    def forward(self, x, mask=None, use_cache=False):
        # Self attention with residual connection
        attn_output = self.self_attn(x, mask=mask, use_cache=use_cache)
        x = self.norm1(x + attn_output)
        
        # FFN with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config)
            for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.embed_dim)
        self.output_proj = nn.Linear(config.embed_dim, config.vocab_size)
        
        self.config = config  # Save config for later use
        
    def forward(self, input_ids, mask=None, use_cache=False):
        seq_len = input_ids.shape[1]
        device = input_ids.device
        
        # Get embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)
        
        x = token_emb + pos_emb
        
        # Process through decoder layers
        for layer in self.layers:
            x = layer(x, mask=mask, use_cache=use_cache)
            
        x = self.norm(x)
        logits = self.output_proj(x)
        
        return logits
    
    def generate(self, input_ids, max_new_tokens):
        """
        Generate new tokens given initial input_ids
        Args:
            input_ids: Starting token ids (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
        """
        self.eval()  # Set to evaluation mode
        
        # First forward pass with the whole prompt
        logits = self(input_ids, use_cache=True)
        
        for _ in range(max_new_tokens):
            # For subsequent passes, only use the last token
            last_token = input_ids[:, -1:]
            
            # Get next token logits and select most probable token
            logits = self(last_token, use_cache=True)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Append new token to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
                
        return input_ids

    def reset_cache(self):
        """Reset the KV cache in all attention layers"""
        for layer in self.layers:
            layer.self_attn.reset_cache()

class Config:
    def __init__(self):
        self.vocab_size = 50257  # GPT-2 vocabulary size
        self.max_seq_len = 1024
        self.embed_dim = 768
        self.num_heads = 1
        self.num_layers = 1
        self.ffn_dim = 3072
        self.eos_token_id = 50256  # GPT-2 end of sequence token
        self.pad_token_id = 50257  # Padding token
        self.max_cache_len = 2048  # Maximum number of tokens to keep in cache
        



def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def simulate_generation(model, input_ids, max_new_tokens, use_cache):
    """
    Simulate text generation with or without KV cache
    """
    model.eval()
    if use_cache:
        model.reset_cache()  # Start with fresh cache
    
    generation_times = []
    memory_usage = []
    
    print(f"\nStarting generation with {'cache' if use_cache else 'no cache'}")
    # Initial forward pass
    start_time = time.time()
    with torch.no_grad():
        logits = model(input_ids, use_cache=use_cache)
        print(f"Initial logits shape: {logits.shape}")
        print(f"Initial next token: {logits[:, -1, :].argmax(dim=-1)}")
    initial_time = time.time() - start_time
    
    generated_ids = input_ids.clone()
    
    # Generate tokens one by one
    for i in range(max_new_tokens):
        start_time = time.time()
        
        with torch.no_grad():
            if use_cache:
                # With cache: only process the last token
                last_token = generated_ids[:, -1:]
                logits = model(last_token, use_cache=True)
            else:
                # Without cache: process the entire sequence each time
                logits = model(generated_ids, use_cache=False)
            
            print(f"\nStep {i+1}")
        
        # Simply take the most probable token
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        
        # Record metrics
        generation_times.append(time.time() - start_time)
        memory_usage.append(torch.cuda.memory_allocated() if torch.cuda.is_available() else 0)
        
        # Append new token
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
    
    return {
        'initial_time': initial_time,
        'generation_times': generation_times,
        'memory_usage': memory_usage,
        'generated_ids': generated_ids
    }

def run_simulation():
    # Set seed for reproducibility
    SEED = 42
    set_seed(SEED)
    
    # Initialize model and move to GPU if available
    config = Config()
    model = TransformerDecoder(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create input sequence
    input_ids = torch.tensor([[1]], device=device)  # Sample input sequence
    max_new_tokens = 4  # Generate 50 new tokens
    
    print(f"Running simulation on {device}")
    print(f"Random seed: {SEED}")
    print(f"Input sequence length: {input_ids.shape[1]}")
    print(f"Generating {max_new_tokens} new tokens\n")
    
    # Run with cache
    print("Running with KV cache...")
    set_seed(SEED)  # Reset seed before each run
    cached_results = simulate_generation(model, input_ids, max_new_tokens, use_cache=True)
    
    # Run without cache
    print("Running without KV cache...")
    set_seed(SEED)  # Reset seed before each run
    uncached_results = simulate_generation(model, input_ids, max_new_tokens, use_cache=False)
    
    # Print results
    print("\nResults:")
    print(f"Initial forward pass:")
    print(f"  With cache: {cached_results['initial_time']:.4f}s")
    print(f"  Without cache: {uncached_results['initial_time']:.4f}s")

    print("total generation times:")
    print(f"  With cache: {sum(cached_results['generation_times'])}")
    print(f"  Without cache: {sum(uncached_results['generation_times'])}")
    
    print(f"\nAverage token generation time:")
    print(f"  With cache: {sum(cached_results['generation_times'])/len(cached_results['generation_times']):.4f}s")
    print(f"  Without cache: {sum(uncached_results['generation_times'])/len(uncached_results['generation_times']):.4f}s")
    
    print(f"\nPeak memory usage:")
    print(f"  With cache: {max(cached_results['memory_usage']) / 1024**2:.2f} MB")
    print(f"  Without cache: {max(uncached_results['memory_usage']) / 1024**2:.2f} MB")

    
    # Verify outputs match
    outputs_match = torch.equal(cached_results['generated_ids'], uncached_results['generated_ids'])
    print(f"\nOutputs match: {outputs_match}")
    
    if not outputs_match:
        print("\nWarning: Outputs don't match! Showing first few tokens:")
        print("Cached output:", cached_results['generated_ids'][0][:10].tolist())
        print("Uncached output:", uncached_results['generated_ids'][0][:10].tolist())

if __name__ == "__main__":
    run_simulation()
