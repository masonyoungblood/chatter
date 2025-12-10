def test_public_imports():
    # Smoke test that the public API imports without optional heavy deps
    from chatter import (
        make_config,
    )  # noqa: F401

    # Ensure helper functions return a config dict with expected keys
    cfg = make_config()
    assert isinstance(cfg, dict)
    assert "sr" in cfg
