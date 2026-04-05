from mlx_lm.expert_offload import ExpertOffloadManager
import tempfile

def test_importance_eviction():
    with tempfile.TemporaryDirectory() as tmpdir:
        weight_map = {}
        expert_key_table = {}
        
        manager = ExpertOffloadManager(
            base_path=tmpdir,
            weight_map=weight_map,
            expert_key_table=expert_key_table,
            max_resident_experts=2,
            max_cached_shards=1
        )
        
        # Manually populate cache and lru
        manager._cache[(0, 1)] = {}
        manager._lru[(0, 1)] = None
        
        manager._cache[(0, 2)] = {}
        manager._lru[(0, 2)] = None
        
        manager.update_expert_importance({
            (0, 1): 0.8,
            (0, 2): 0.2
        })
        
        manager._evict_one_unpinned(set())
        
        # (0, 2) has lower importance, should be evicted
        assert (0, 2) not in manager._cache
        assert (0, 1) in manager._cache
