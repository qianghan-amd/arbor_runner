from arbor_ab import config


def test_matrix_cells():
    m = config.load_matrix()
    active = config.cells(m)
    assert len(active) == 12                                    # 6 active models x 2 (design.md 2.1)
    assert len(config.cells(m, include_placeholder=True)) == 14  # + qwen38 placeholder
    assert ("kimik3", "c32_tput") in active and ("qwen38", "c1_p90") not in active


def test_every_model_has_a_model_file():
    m = config.load_matrix()
    for key in m["models"]:
        assert config.load_model(key)["key"] == key


def test_target_spec_matches_design_example():
    t = config.target_spec(config.load_matrix(), "dsv41flash", "c32_tput")
    assert t == {"key": "c32_tput", "conc": 32, "direction": "max",
                 "primary_metric_path": "metrics.total_tput_tps",
                 "primary_metric_per_gpu_path": "metrics.tput_per_gpu",
                 "secondary_metric_path": "metrics.e2e_norm_intvty.p50"}


def test_default_tolerance():
    assert config.tolerance_pct(config.load_matrix(), "glm52", "c1_p90") == 3.0


def test_cluster_and_local_example():
    assert "crsuse2-slog-006.crusoe.amd.com" in config.load_cluster()["login_hosts"]
    ex = config.load_local(config.REPO_ROOT / "local.yaml.example")
    assert ex["github"]["token_source"] == "relay_gh_hosts"
