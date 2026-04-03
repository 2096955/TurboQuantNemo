import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import numpy as np
from tqdm import tqdm

from mlx_turboquant import TurboQuantKVCache
from validate_utils import cosine_similarity_flat, top_k_match


def capture_kv_from_attention(model, tokenizer, prompt: str, max_tokens: int = 2048):
    """Run forward pass and hook into attention layers to capture raw pre-cache KV."""

    cache = make_prompt_cache(model)

    tokens = mx.array(tokenizer.encode(prompt)[:max_tokens])
    captured_kv = {}

    original_modules = {}

    def make_wrapper(layer_idx, orig_module):
        class Wrapper:
            def __call__(self, x, mask=None, cache=None, **kwargs):
                B, L, D = x.shape

                keys, values = orig_module.k_proj(x), orig_module.v_proj(x)
                queries_raw = orig_module.q_proj(x)

                n_kv = getattr(
                    orig_module,
                    "num_key_value_heads",
                    getattr(orig_module, "n_kv_heads", 2),
                )
                n_q = getattr(
                    orig_module,
                    "num_attention_heads",
                    getattr(orig_module, "n_heads", 32),
                )

                keys = keys.reshape(B, L, n_kv, -1)
                values = values.reshape(B, L, n_kv, -1)

                if hasattr(orig_module, "q_norm"):
                    # Qwen3Next style
                    queries, _ = mx.split(
                        queries_raw.reshape(B, L, n_q, -1), 2, axis=-1
                    )
                    queries = orig_module.q_norm(queries)
                else:
                    queries = queries_raw.reshape(B, L, n_q, -1)

                if hasattr(orig_module, "k_norm"):
                    keys = orig_module.k_norm(keys)

                keys = keys.transpose(0, 2, 1, 3)
                values = values.transpose(0, 2, 1, 3)
                queries = queries.transpose(0, 2, 1, 3)

                if cache is not None:
                    if hasattr(orig_module, "rope"):
                        keys = orig_module.rope(keys, offset=cache.offset)
                        queries = orig_module.rope(queries, offset=cache.offset)
                    captured_kv[layer_idx] = [(queries, keys, values)]

                return orig_module(x, mask=mask, cache=cache, **kwargs)

            def __getattr__(self, name):
                if (
                    name in orig_module.__dict__
                    or hasattr(orig_module.__class__, name)
                    or name in orig_module.keys()
                ):
                    return getattr(orig_module, name)
                raise AttributeError(
                    f"'{orig_module.__class__.__name__}' object has no attribute '{name}'"
                )

        return Wrapper()

    for i, layer in enumerate(model.layers):
        if hasattr(layer, "self_attn"):
            original_modules[i] = layer.self_attn
            layer.self_attn = make_wrapper(i, layer.self_attn)
        elif hasattr(layer, "attention"):
            original_modules[i] = layer.attention
            layer.attention = make_wrapper(i, layer.attention)

    logits = model(tokens[None], cache=cache)
    mx.eval(logits)

    for i, orig in original_modules.items():
        if hasattr(model.layers[i], "self_attn"):
            model.layers[i].self_attn = orig
        else:
            model.layers[i].attention = orig

    return captured_kv, cache


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--codebook-dir", type=str, default="codebooks")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model, lazy=True)

    prompt = "A very long prompt..." * 200
    print("Capturing KV from attention...")
    captured, _ = capture_kv_from_attention(
        model, tokenizer, prompt, max_tokens=args.prompt_tokens
    )

    results = []
    print(
        f"\nEvaluating {args.bits}-bit compression with codebooks from '{args.codebook_dir}'..."
    )

    config = getattr(model, "args", getattr(model, "config", None))
    n_q = getattr(config, "num_attention_heads", 32)
    n_kv = getattr(config, "num_key_value_heads", n_q)
    head_dim = getattr(config, "head_dim", getattr(config, "hidden_size", 4096) // n_q)

    for layer_idx, kv_pairs in tqdm(captured.items()):
        if not kv_pairs:
            continue
        queries, keys, values = kv_pairs[0]

        tq_cache = TurboQuantKVCache(
            num_heads=n_kv,
            head_dim=head_dim,
            bit_width=args.bits,
            layer_idx=layer_idx,
            codebook_dir=args.codebook_dir,
            seed=42,
        )

        tq_cache.update_and_fetch(keys, values)

        # Real queries
        q_sample = queries  # (batch, n_q, seq_len, head_dim)

        # Baseline scores + shared broadcast shapes
        scale = 1.0 / mx.sqrt(float(head_dim))
        heads_per_kv = n_q // n_kv

        # We need to broadcast keys to match q_sample's query heads
        q_sample_reshaped = q_sample.reshape(
            1, n_kv, heads_per_kv, q_sample.shape[2], head_dim
        )
        keys_reshaped = keys[:, :, None, :, :]  # (1, n_kv, 1, seq_len, head_dim)

        # Cast to float32 to prevent FP16 overflow during matmul
        q_sample_f32 = q_sample_reshaped.astype(mx.float32)
        keys_f32 = keys_reshaped.astype(mx.float32)

        baseline_scores = (
            mx.matmul(q_sample_f32, keys_f32.transpose(0, 1, 2, 4, 3)) * scale
        )
        baseline_scores = baseline_scores.reshape(1, n_q, q_sample.shape[2], -1)

        # TQ scores
        # Primary path: fused asymmetric attention scoring (requires residual_signs in compressed payload).
        # Fallback path: reconstruction-based scoring (works with the current KV cache storage format).
        try:
            tq_scores = tq_cache.asymmetric_attention_scores(
                q_sample
            )  # (1, num_q_heads, seq_len, seq_len)
            tq_scores = tq_scores.astype(mx.float32) * scale
        except KeyError as e:
            if "residual_signs" not in str(e):
                raise

            keys_recon = tq_cache.reconstruct_keys().astype(
                mx.float32
            )  # (1, n_kv, seq_len, head_dim)
            keys_recon_reshaped = keys_recon[
                :, :, None, :, :
            ]  # (1, n_kv, 1, seq_len, head_dim)

            tq_scores = (
                mx.matmul(q_sample_f32, keys_recon_reshaped.transpose(0, 1, 2, 4, 3))
                * scale
            )
            tq_scores = tq_scores.reshape(1, n_q, q_sample.shape[2], -1)

        mx.eval(tq_scores, baseline_scores)

        for h_q in range(n_q):
            tq_2d = tq_scores[0, h_q]  # (seq_len, seq_len)
            base_2d = baseline_scores[0, h_q]  # (seq_len, seq_len)

            if mx.isnan(tq_2d).any() or mx.isnan(base_2d).any():
                print(f"NaN detected at layer {layer_idx}, head {h_q}")
                continue

            cos_sim = cosine_similarity_flat(tq_2d, base_2d)
            top1_match = top_k_match(tq_2d, base_2d, k=1)
            top5_match = top_k_match(tq_2d, base_2d, k=5)

            results.append(
                {
                    "layer": layer_idx,
                    "head": h_q,
                    "cosine_sim": float(cos_sim),
                    "top1_match": float(top1_match),
                    "top5_match": float(top5_match),
                }
            )

    if not results:
        print("No valid results found (all NaNs?)")
    else:
        avg_cos = np.mean([r["cosine_sim"] for r in results])
        avg_top1 = np.mean([r["top1_match"] for r in results]) * 100
        avg_top5 = np.mean([r["top5_match"] for r in results]) * 100

        print(f"\nAverage cosine similarity: {avg_cos:.5f}")
        print(f"Top-1 Match: {avg_top1:.1f}%")
        print(f"Top-5 Match: {avg_top5:.1f}%")
        if avg_cos < 0.99 and args.bits == 3:
            print(
                "WARNING: Below fidelity gate (0.99) — consider empirical codebook recompute"
            )
