"""E2E smoke test: NPT=8 tiled path on Gemma 4 (head_dim=256).

Runs IsoQuant paths (3-kernel, NPT=8 v1, NPT=8 tiled) plus default KV
baseline. The tiled path requires T >= 512 in the cache at decode time.
LONG_PROMPT must tokenize to >= 512 tokens so the first decode step
already sees T=512+ and triggers tiled dispatch.

After the subprocess runs, we verify prompt token count >= 512 using
the model's tokenizer. If below threshold, the tiled label is flagged.
"""

import json
import os
import subprocess
import sys
import time

MODEL = "gemma-4-26b-a4b-it-4bit"
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), MODEL)
MAX_TOKENS = 64

# ~700 tokens with Gemma 4 tokenizer (verified: 383 was too short at rev 1).
# Threshold is 512; we target well above to avoid tokenizer variance.
LONG_PROMPT = (
    "You are a technical writer. Write a detailed, multi-paragraph explanation "
    "of how attention mechanisms work in transformer architectures. Cover the "
    "following topics in order: 1) The original scaled dot-product attention "
    "formula and its mathematical derivation from query, key, and value matrices. "
    "2) Multi-head attention and why splitting into multiple heads helps capture "
    "different relationship patterns. 3) The computational complexity of attention "
    "O(n^2) and why this becomes a bottleneck for long sequences. 4) Key-value "
    "caching during autoregressive generation and how it reduces redundant "
    "computation. 5) Grouped query attention (GQA) where multiple query heads "
    "share fewer key-value heads, reducing memory bandwidth while preserving "
    "quality. 6) Flash attention and its tiling strategy to avoid materializing "
    "the full attention matrix in HBM. 7) Quantized KV caches that compress "
    "stored keys and values to lower precision (e.g., 3-bit or 4-bit) to fit "
    "longer contexts in limited GPU/accelerator memory. 8) The tradeoffs between "
    "compression ratio, decode throughput, and output quality when using "
    "quantized caches on consumer hardware like Apple Silicon. Be thorough, "
    "use precise mathematical notation where appropriate, and include concrete "
    "examples of tensor shapes for a model with hidden_size=2816, num_heads=16, "
    "num_kv_heads=8, and head_dim=256. Explain how each technique builds on "
    "the previous one to enable running large language models on devices with "
    "limited memory. Also explain the role of rotary position embeddings (RoPE) "
    "in encoding positional information and how they interact with KV caching. "
    "Discuss the difference between dense attention and sparse/MoE attention "
    "patterns. Describe how online softmax works in the context of "
    "tiled attention kernels and the FA2-style merge used to combine partial "
    "results from different tiles into a numerically stable final output. "
    "9) Explain the concept of sliding window attention and how it limits the "
    "effective receptive field to reduce memory usage while maintaining local "
    "context awareness. 10) Describe multi-query attention (MQA) as a special "
    "case of GQA with num_kv_heads=1, its memory bandwidth savings during "
    "autoregressive decoding, and why it sometimes degrades quality compared "
    "to full multi-head attention. 11) Explain the role of the softmax "
    "temperature (scaling factor 1/sqrt(d_k)) in preventing attention score "
    "saturation and how it interacts with quantized key representations. "
    "12) Describe the memory hierarchy on Apple M-series chips — unified "
    "memory architecture, the difference between bandwidth to GPU cores vs "
    "Neural Engine, and why KV cache compression is especially important "
    "when the entire model and cache compete for the same memory pool. "
    "13) Explain how IsoQuant uses quaternion-inspired 4D rotations (SO(4) "
    "block matrices) to decorrelate key/value dimensions before quantization, "
    "why this improves codebook utilization compared to direct per-dimension "
    "quantization, and how the inverse rotation is applied after attention "
    "output accumulation to recover the original representation space. "
    "14) Discuss the NPT=8 (values per thread) kernel design where 32 SIMD "
    "lanes each process 8 dimensions for a total of head_dim=256, and how "
    "this maps to Metal threadgroup execution on Apple GPUs."
)

SHORT_PROMPT = "Explain what makes the number 42 special in exactly three sentences."


def run_generate(prompt, env_overrides, label):
    env = os.environ.copy()
    env["ISOQUANT_BITS"] = "3"
    env.update(env_overrides)

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.generate",
        "--model",
        MODEL_PATH,
        "--prompt",
        prompt,
        "--max-tokens",
        str(MAX_TOKENS),
        "--kv-cache-type",
        env_overrides.get("_kv_type", "isoquant"),
    ]
    if "_kv_type" in env_overrides:
        del env["_kv_type"]

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.dirname(os.path.dirname(__file__)) + "/mlx-lm",
        timeout=300,
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        print(f"  stderr: {result.stderr[-500:]}")
        return None

    output = result.stdout
    lines = output.strip().split("\n")

    stats = {}
    for line in lines:
        if "Prompt:" in line:
            try:
                stats["prompt_tps"] = float(
                    line.split("tokens-per-sec")[0].split(",")[-1].strip()
                )
            except (ValueError, IndexError):
                pass
        if "Generation:" in line:
            try:
                stats["gen_tps"] = float(
                    line.split("tokens-per-sec")[0].split(",")[-1].strip()
                )
            except (ValueError, IndexError):
                pass
        if "Peak memory:" in line:
            try:
                stats["peak_mem_gb"] = float(
                    line.split(":")[-1].strip().replace(" GB", "")
                )
            except (ValueError, IndexError):
                pass

    text_lines = [
        l
        for l in lines
        if not l.startswith("=")
        and "tokens-per-sec" not in l
        and "Peak memory" not in l
        and l.strip()
    ]
    generated_text = "\n".join(text_lines[-5:]) if text_lines else "(no output)"

    print(f"  Time: {elapsed:.1f}s")
    print(f"  Prompt tok/s: {stats.get('prompt_tps', '?')}")
    print(f"  Gen tok/s: {stats.get('gen_tps', '?')}")
    print(f"  Peak mem: {stats.get('peak_mem_gb', '?')} GB")
    print("  Output (last 5 lines):")
    for l in generated_text.split("\n"):
        print(f"    {l[:100]}")

    stats["label"] = label
    stats["elapsed"] = elapsed
    stats["text_snippet"] = generated_text[:200]
    return stats


def verify_prompt_tokens(prompt, min_tokens=512):
    """Verify prompt tokenizes to >= min_tokens using the model's tokenizer."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        ids = tok(prompt, add_special_tokens=True)["input_ids"]
        n = len(ids)
        ok = n >= min_tokens
        print(
            f"  Prompt tokens: {n} (threshold: {min_tokens}) {'OK' if ok else 'BELOW THRESHOLD'}"
        )
        return n, ok
    except Exception as e:
        print(f"  Tokenizer check skipped: {e}")
        return -1, False


def main():
    if not os.path.isdir(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)

    print("Verifying LONG_PROMPT token count...")
    n_tokens, long_ok = verify_prompt_tokens(LONG_PROMPT, min_tokens=512)
    if not long_ok:
        print(
            f"  FATAL: LONG_PROMPT is only {n_tokens} tokens; tiled path needs >= 512."
        )
        print("  Extend LONG_PROMPT and rerun.")
        sys.exit(1)

    results = []

    r = run_generate(
        SHORT_PROMPT,
        {"ISOQUANT_USE_NPT8_FUSED": "0", "_kv_type": "isoquant"},
        "IsoQuant 3-kernel (short context)",
    )
    if r:
        results.append(r)

    r = run_generate(
        SHORT_PROMPT,
        {"ISOQUANT_USE_NPT8_FUSED": "1", "_kv_type": "isoquant"},
        "IsoQuant NPT=8 v1 (short context, T < 512)",
    )
    if r:
        results.append(r)

    r = run_generate(
        LONG_PROMPT,
        {"ISOQUANT_USE_NPT8_FUSED": "1", "_kv_type": "isoquant"},
        f"IsoQuant NPT=8 tiled (prompt={n_tokens} tokens, T >= 512)",
    )
    if r:
        results.append(r)

    r = run_generate(
        LONG_PROMPT,
        {"ISOQUANT_USE_NPT8_FUSED": "0", "_kv_type": "isoquant"},
        f"IsoQuant 3-kernel (prompt={n_tokens} tokens)",
    )
    if r:
        results.append(r)

    r = run_generate(
        SHORT_PROMPT, {"_kv_type": "default"}, "Default KV baseline (short)"
    )
    if r:
        results.append(r)

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        print(
            f"  {r['label']:55s} gen={r.get('gen_tps', '?'):>8} tok/s  prompt={r.get('prompt_tps', '?'):>8} tok/s"
        )

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "artifacts", "phase3b-tiled-smoke"
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "smoke_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_dir}/smoke_results.json")


if __name__ == "__main__":
    main()
