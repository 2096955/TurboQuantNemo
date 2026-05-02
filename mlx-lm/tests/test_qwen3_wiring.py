import pytest
import mlx.core as mx
from mlx_lm.expert_offload import ExpertOffloadManager, is_expert_weight_key
from mlx_lm.repack_experts import _build_repacked_weights_for_qwen3_shard

def test_qwen3_repack_shard():
    weights = {
        "model.layers.4.mlp.switch_mlp.up_proj.weight": mx.zeros((128, 1024, 512)),
        "model.layers.4.mlp.gate.weight": mx.zeros((128, 512)),
    }
    repacked = _build_repacked_weights_for_qwen3_shard(
        shard="shard1",
        keys=list(weights.keys()),
        weights=weights,
        num_experts=128,
    )
    
    assert "model.layers.4.mlp.gate.weight" in repacked
    assert "model.layers.4.mlp.experts.127.up_proj.weight" in repacked
    assert repacked["model.layers.4.mlp.experts.127.up_proj.weight"].shape == (1024, 512)

def test_qwen3_expert_key_regex():
    assert is_expert_weight_key("model.layers.1.mlp.experts.0.gate_proj.weight", "qwen3_moe")