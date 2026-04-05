from mlx_lm.expert_offload import AttnResExpertPredictor
import mlx.core as mx

def test_attn_res_predictor():
    predictor = AttnResExpertPredictor(num_blocks=3, num_experts=128, num_layers=30)
    alpha = mx.array([[[0.1]], [[0.8]], [[0.1]]])  # shape (3, 1, 1)
    expert_ids = mx.array([5, 10, 42])
    predictor.record_activation(layer_idx=15, block_attention_weights=alpha, expert_ids=expert_ids)
    predicted = predictor.predict_experts(layer_idx=15, block_attention_weights=alpha, top_k=2)
    assert len(predicted) == 2
