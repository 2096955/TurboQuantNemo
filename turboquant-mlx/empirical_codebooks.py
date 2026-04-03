import mlx.core as mx
import numpy as np
import os
import argparse
from scipy.cluster.vq import kmeans2
from mlx_lm import load
from validate_real_kv import capture_kv_from_attention

def compute_empirical_codebooks(model_path="mlx-community/Qwen2.5-1.5B-Instruct-4bit", prompt_tokens=512, codebook_dir="codebooks_empirical"):
    model, tokenizer = load(model_path)
    config = getattr(model, "args", getattr(model, "config", None))
    n_q = getattr(config, "num_attention_heads", 32)
    n_kv = getattr(config, "num_key_value_heads", n_q)
    head_dim = getattr(config, "head_dim", getattr(config, "hidden_size", 4096) // n_q)
    
    base_prompt = "The quick brown fox jumps over the lazy dog. " * 500
    tokens_list = tokenizer.encode(base_prompt)[:prompt_tokens]
    prompt = tokenizer.decode(tokens_list)
    
    captured, _ = capture_kv_from_attention(model, tokenizer, prompt, max_tokens=prompt_tokens)
    
    all_scalars = []
    
    for layer_idx in captured.keys():
        if not captured[layer_idx]: continue
        keys, _ = captured[layer_idx][0]
        
        for h in range(keys.shape[1]):
            k_head = keys[0, h, :, :]
            
            rng = np.random.default_rng(42 + layer_idx * 1000 + h)
            A = rng.normal(size=(head_dim, head_dim)).astype(np.float32)
            Q, _ = np.linalg.qr(A)
            Q_mx = mx.array(Q, dtype=mx.float32)
            Q_T = mx.transpose(Q_mx)
            
            x_f32 = k_head.astype(mx.float32)
            x_norm = mx.linalg.norm(x_f32, axis=-1, keepdims=True)
            x_unit = x_f32 / mx.maximum(x_norm, mx.array(1e-8))
            
            x_rot = mx.matmul(x_unit, Q_T)
            all_scalars.append(np.array(x_rot).flatten())
            
    all_scalars = np.concatenate(all_scalars)
    
    # Subsample for faster K-means if too large
    if len(all_scalars) > 1000000:
        all_scalars = np.random.choice(all_scalars, 1000000, replace=False)
        
    print(f"Collected {len(all_scalars)} empirical scalars.")
    
    os.makedirs(codebook_dir, exist_ok=True)
    
    for bits in [1, 2, 3, 4]:
        num_levels = 2 ** bits
        print(f"Computing {bits}-bit empirical codebook...")
        
        # scipy kmeans2 for 1D
        # initial centroids evenly spaced
        initial_centroids = np.linspace(np.min(all_scalars), np.max(all_scalars), num_levels)
        centroids, labels = kmeans2(all_scalars, initial_centroids, minit='matrix', iter=100)
        
        centroids = np.sort(centroids)
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        
        filepath = os.path.join(codebook_dir, f"dim_{head_dim}_{bits}bit.npz")
        np.savez(
            filepath,
            centroids=centroids,
            boundaries=boundaries,
            sigma=np.array([np.std(all_scalars)]),
            mse=np.array([0.0]), # dummy
            head_dim=np.array([head_dim]),
            bits=np.array([bits])
        )
        print(f"Saved {filepath} with centroids: {centroids}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--codebook-dir", type=str, default="codebooks_empirical")
    args = parser.parse_args()
    compute_empirical_codebooks(args.model, args.prompt_tokens, args.codebook_dir)
