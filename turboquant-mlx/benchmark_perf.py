import argparse
import gc
import time

import mlx.core as mx
from mlx_lm import generate, load

def benchmark(model, tokenizer, model_path, kv_type, prompt_tokens, gen_tokens, bits=3):
    prompt = "The quick brown fox jumps over the lazy dog. " * (prompt_tokens // 10)
    
    print(f"--- Benchmarking {kv_type} on {model_path} with {prompt_tokens} prompt tokens ---")
    mx.eval(model.parameters())
    mx.metal.clear_cache()
    gc.collect()
    time.sleep(0.5)
    mx.metal.reset_peak_memory()
    
    start_mem = mx.metal.get_active_memory() / 1024**2
    print(f"Base Model Memory: {start_mem:.2f} MB")

    tic = time.time()
    
    response = generate(
        model, 
        tokenizer, 
        prompt, 
        max_tokens=gen_tokens, 
        kv_cache_type=kv_type,
        verbose=False
    )
    
    toc = time.time()
    
    mx.eval(model.parameters()) # ensure everything is finished
    end_mem = mx.metal.get_active_memory() / 1024**2
    peak_mem = mx.metal.get_peak_memory() / 1024**2
    
    cache_overhead = peak_mem - start_mem
    
    config = getattr(model, "args", None) or getattr(model, "config", None)
    n_layers = config.num_hidden_layers
    n_kv = getattr(config, "num_key_value_heads", config.num_attention_heads)
    d = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    total_tokens = prompt_tokens + gen_tokens
    
    # Only the dense baseline has a straightforward closed-form storage model
    # here. The current TurboQuant Python path still stores dense values and
    # dense residual-sign tensors, so a packed-size estimate would be misleading.
    dense_kv_size_mb = (n_layers * 2 * total_tokens * n_kv * d * 2) / (1024**2)
    
    print(f"KV Type: {kv_type} ({bits}-bit)")
    print(f"Tokens generated: {gen_tokens}")
    print(f"Total Time: {toc - tic:.2f}s")
    print(f"Tok/s (generation): {gen_tokens / (toc - tic):.2f}")
    print(f"Peak Memory: {peak_mem:.2f} MB")
    print(f"Active Memory at End: {end_mem:.2f} MB")
    print(f"Estimated Peak Overhead: {cache_overhead:.2f} MB")
    print(f"Dense KV Lower Bound: {dense_kv_size_mb:.2f} MB")
    if kv_type == "turboquant":
        print(
            "TurboQuant note: current implementation packs key indices only; "
            "values and residual-sign tensors are still stored densely."
        )
    print(f"--------------------------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["mlx-community/Qwen2.5-1.5B-Instruct-4bit"])
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--gen-tokens", type=int, default=32)
    args = parser.parse_args()
    
    for model_path in args.models:
        model, tokenizer = load(model_path)
        for pt in args.prompt_tokens:
            benchmark(model, tokenizer, model_path, "default", prompt_tokens=pt, gen_tokens=args.gen_tokens)
            benchmark(model, tokenizer, model_path, "turboquant", prompt_tokens=pt, gen_tokens=args.gen_tokens, bits=3)
