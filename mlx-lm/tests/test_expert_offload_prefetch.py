import mlx.core as mx
import tempfile
from mlx_lm.expert_offload import ExpertOffloadManager
import time

def test_prefetch():
    with tempfile.TemporaryDirectory() as tmpdir:
        weight_map = {"backbone.layers.0.mixer.experts.0.up_proj.weight": "model-001.safetensors"}
        expert_key_table = {(0, 0): {"fc1_weight": "backbone.layers.0.mixer.experts.0.up_proj.weight"}}
        
        manager = ExpertOffloadManager(
            base_path=tmpdir,
            weight_map=weight_map,
            expert_key_table=expert_key_table,
            max_resident_experts=2,
            max_cached_shards=1
        )
        
        # Mock _load_expert_pair_tensors to just return a dummy tensor
        def mock_load(spec):
            time.sleep(0.1) # Simulate IO
            return {"fc1_weight": mx.zeros((10, 10))}
            
        manager._load_expert_pair_tensors = mock_load
        
        manager.prefetch(layer_idx=0, expert_ids=[0])
        
        # Wait for prefetch
        time.sleep(0.2)
        
        # Check if it's loaded
        assert (0, 0) in manager._cache
