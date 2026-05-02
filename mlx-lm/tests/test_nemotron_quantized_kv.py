from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.mlx_turboquant import TurboQuantKVCache
from mlx_lm.models.nemotron_h import Model, ModelArgs, NemotronHAttention


def _make_args(**overrides):
    defaults = dict(
        model_type="nemotron_h",
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=4,
        max_position_embeddings=256,
        num_attention_heads=8,
        num_key_value_heads=4,
        attention_bias=False,
        mamba_num_heads=8,
        mamba_head_dim=64,
        mamba_proj_bias=False,
        ssm_state_size=128,
        conv_kernel=3,
        n_groups=4,
        mlp_bias=False,
        layer_norm_epsilon=1e-4,
        use_bias=True,
        use_conv_bias=True,
        hybrid_override_pattern=["*", "M", "*", "M"],
    )
    defaults.update(overrides)
    return ModelArgs(**defaults)


class TestNemotronHybridQuantizedKV:
    def test_make_prompt_cache_preserves_mamba_caches_and_wraps_attention(self):
        model = Model(_make_args())
        caches = make_prompt_cache(model, kv_cache_type="turboquant")

        assert len(caches) == 4
        # Default skip_layers=2 keeps the first attention layer native.
        assert isinstance(caches[0], KVCache)
        assert isinstance(caches[1], ArraysCache)
        # The second attention layer should be quantized (or wrapped with a
        # TurboQuantKVCache object that can safely fall back internally).
        assert isinstance(caches[2], TurboQuantKVCache)
        assert isinstance(caches[3], ArraysCache)

    def test_attention_source_handles_turboquant_cache(self):
        src = NemotronHAttention.__call__.__code__.co_names
        assert "TurboQuantKVCache" in src
        assert "reconstruct_keys" in src
        assert "get_values" in src
