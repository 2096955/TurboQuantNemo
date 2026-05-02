from mlx_lm.expert_offload import ExpertOffloadManager, DedeKimiObserver


def test_dedekimi_observer_ema():
    observer = DedeKimiObserver(num_layers=10, num_experts=128)
    observer.record_activation(layer=0, expert_ids=[1, 2, 3])

    entropy = observer.get_layer_entropy(layer=0)
    assert entropy > 0


def test_task_aware_pinning():
    manager = ExpertOffloadManager(
        "test_model", weight_map={}, expert_key_table={}, max_resident_experts=32
    )
    # Mocking task clique
    manager.pre_populate_task_clique("Rust_Refactor", layer_expert_map={0: [1, 2, 3]})
    assert manager._pinned_tasks["Rust_Refactor"] is not None


def test_dedekimi_health_summary_and_collapse():
    obs = DedeKimiObserver(num_layers=4, num_experts=8, decay=0.5)
    for _ in range(20):
        obs.record_activation(0, [0, 0, 0, 0])  # collapse toward expert 0
    h = obs.health_summary(min_entropy=2.0)
    assert "collapse_risk" in h
    assert h["mean_entropy"] >= 0
