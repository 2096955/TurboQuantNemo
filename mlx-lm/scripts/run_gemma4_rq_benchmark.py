import mlx.core as mx
import time
import psutil
import os
import json
from mlx_lm import utils, generate

model_path = "../gemma-4-26b-a4b-it-4bit"
out_file = "../results/gemma4_rotorquant_benchmark.json"

def print_memory_stats(prefix=""):
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss / (1024 * 1024 * 1024)
    try:
        peak = mx.metal.get_peak_memory() / (1024 * 1024 * 1024)
    except Exception:
        peak = 0.0
    print(f"{prefix} RSS: {rss:.2f} GB, MLX Peak: {peak:.2f} GB")
    return {"rss_gb": rss, "peak_gb": peak}

print(f"Loading model {model_path} with kv_cache_type=rotorquant...")
# Override CLI args directly since utils.load reads from it
import argparse
args = argparse.Namespace(kv_cache_type="rotorquant", expert_offload=True)

model, tokenizer = utils.load(model_path, model_config={"kv_cache_type": "rotorquant", "expert_offload": True})
model.expert_offload_manager.max_resident_experts = 64

# Force the response generator CLI args as well if needed
from mlx_lm.server import ModelProvider
provider = ModelProvider(args)
provider.model = model
provider.tokenizer = tokenizer

stats = print_memory_stats("Post-load")

results = {
    "model": model_path,
    "kv_cache_type": "rotorquant",
    "post_load_memory": stats,
    "runs": []
}

prompts = [
    ("Short (128 tokens)", " ".join(["Hello world"] * 64)),
    ("Medium (512 tokens)", " ".join(["Hello world"] * 256)),
    ("Long (1024 tokens)", " ".join(["Hello world"] * 512)),
    ("Extra Long (2048 tokens)", " ".join(["Hello world"] * 1024)),
]

for name, text in prompts:
    print(f"\n--- Profiling {name} ---")
    
    try:
        mx.metal.reset_peak_memory()
    except Exception:
        pass

    prompt_len = len(tokenizer.encode(text))
    print(f"Prompt length: {prompt_len} tokens")
    
    t0 = time.time()
    
    # We will measure prefill manually by passing a single token generate request
    output = generate(model, tokenizer, prompt=text, max_tokens=10, verbose=False)
            
    t1 = time.time()
    duration = t1 - t0
    
    print(f"Generated tokens in {duration:.2f}s")
    run_stats = print_memory_stats(f"After {name}")
    
    results["runs"].append({
        "name": name,
        "prompt_tokens": prompt_len,
        "duration_s": duration,
        "memory": run_stats,
    })

with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"Wrote benchmark results to {out_file}")
