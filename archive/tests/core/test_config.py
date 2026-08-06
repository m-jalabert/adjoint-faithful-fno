from bire_repro.core.config import config_sha256, experiment, load_config


def test_locked_config_invariants():
    config = load_config()
    assert sum(config["grid"]["del_r_m"]) == 1800.0
    assert experiment(config, "control")["id"] == 3
    assert experiment(config, 5)["tau0_n_m2"] == 0.125
    assert len(config_sha256(config)) == 64
